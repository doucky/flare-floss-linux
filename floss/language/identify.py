# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
from enum import Enum
from typing import Tuple, Iterable
from pathlib import Path

import pefile

import floss.logging_
from floss.results import StaticString
from floss.language.binary import Format, BinaryFile, load_binary
from floss.language.rust.rust_version_database import rust_commit_hash

logger = floss.logging_.getLogger(__name__)


VERSION_UNKNOWN_OR_NA = "version unknown"


class Language(Enum):
    GO = "go"
    RUST = "rust"
    DOTNET = "dotnet"
    UNKNOWN = "unknown"
    DISABLED = "none"


def identify_language_and_version(sample: Path, static_strings: Iterable[StaticString]) -> Tuple[Language, str]:
    is_rust, version = get_if_rust_and_version(static_strings)
    if is_rust:
        logger.info("Rust binary found with version: %s", version)
        return Language.RUST, version

    # load the sample through the format abstraction so that Go detection works
    # for both PE and ELF binaries.
    try:
        bf = load_binary(sample)
    except Exception as err:
        logger.debug(
            "FLOSS currently only detects if PE or ELF files were written in Go (and PE files in .NET). "
            "Could not parse this file: %s",
            err,
        )
        return Language.UNKNOWN, VERSION_UNKNOWN_OR_NA

    is_go, version = get_if_go_and_version(bf)
    if is_go:
        logger.info("Go binary found with version %s", version)
        return Language.GO, version
    elif bf.format == Format.PE and is_dotnet_bin(bf.pe):
        return Language.DOTNET, VERSION_UNKNOWN_OR_NA
    else:
        return Language.UNKNOWN, VERSION_UNKNOWN_OR_NA


def get_if_rust_and_version(static_strings: Iterable[StaticString]) -> Tuple[bool, str]:
    """
    Return if the binary given is compiled with Rust compiler and its version
    reference: https://github.com/mandiant/flare-floss/issues/766
    """

    # Check if the binary contains the rustc/commit-hash string

    # matches strings like "rustc/commit-hash[40 characters]/library" e.g. "rustc/59eed8a2aac0230a8b53e89d4e99d55912ba6b35/library"
    regex_hash = re.compile(r"rustc/(?P<hash>[a-z0-9]{40})[\\\/]library")

    # matches strings like "rustc/version/library" e.g. "rustc/1.54.0/library"
    regex_version = re.compile(r"rustc/(?P<version>[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})")

    for static_string_obj in static_strings:
        string = static_string_obj.string

        match = regex_version.search(string)
        if match:
            return True, match["version"]

        matches = regex_hash.search(string)
        if matches:
            if matches["hash"] in rust_commit_hash.keys():
                version = rust_commit_hash[matches["hash"]]
                return True, version
            else:
                logger.debug("hash %s not found in Rust commit hash database", matches["hash"])
                return True, VERSION_UNKNOWN_OR_NA

    return False, VERSION_UNKNOWN_OR_NA


# magic header of the Go pclntab structure (-pcHeader-); the magic varies by version
GO_MAGIC = [
    b"\xf0\xff\xff\xff\x00\x00",
    b"\xfb\xff\xff\xff\x00\x00",
    b"\xfa\xff\xff\xff\x00\x00",
    b"\xf1\xff\xff\xff\x00\x00",
]

# common Go runtime function names, present in all Go samples including obfuscated ones
GO_FUNCTIONS = [
    b"runtime.main",
    b"main.main",
    b"runtime.gcWork",
    b"runtime.morestack",
    b"runtime.morestack_noctxt",
    b"runtime.newproc",
    b"runtime.gcWriteBarrier",
    b"runtime.Gosched",
]


def _go_priority_sections(bf: BinaryFile):
    """
    yield the sections most likely to contain the Go pclntab / runtime metadata,
    most-likely first, without duplicates.

    PE   -> .rdata
    ELF  -> .gopclntab (older toolchains), .rodata (newer toolchains)
    """
    seen = set()

    try:
        ro = bf.get_read_only_data_section()
        seen.add(ro.name)
        yield ro
    except ValueError:
        pass

    for section in bf.sections:
        if section.name == ".gopclntab" and section.name not in seen:
            seen.add(section.name)
            yield section

    for section in bf.sections:
        if section.name not in seen:
            yield section


def get_if_go_and_version(bf: BinaryFile) -> Tuple[bool, str]:
    """
    Return if the binary given is compiled with the Go compiler and its version.

    this checks the magic header of the pclntab structure -pcHeader-
    the magic values varies through the version
    reference:
    https://github.com/0xjiayu/go_parser/blob/865359c297257e00165beb1683ef6a679edc2c7f/pclntbl.py#L46

    works for both PE (.rdata / sections) and ELF (.gopclntab / .rodata / sections).
    """
    sections = list(_go_priority_sections(bf))

    # 1) look for the pclntab magic header
    for section in sections:
        data = section.data
        for magic in GO_MAGIC:
            idx = data.find(magic)
            if idx != -1 and verify_pclntab(data, idx):
                return True, get_go_version(magic)

    # 2) the magic bytes may have been patched: search for common Go function names
    for section in sections:
        data = section.data
        for go_function in GO_FUNCTIONS:
            if go_function in data:
                logger.info("Go binary found, function name %s", go_function)
                return True, VERSION_UNKNOWN_OR_NA

    return False, VERSION_UNKNOWN_OR_NA


def get_go_version(magic):
    """get the version of the go compiler used to compile the binary"""

    MAGIC_112 = b"\xfb\xff\xff\xff\x00\x00"  # Magic Number from version 1.12
    MAGIC_116 = b"\xfa\xff\xff\xff\x00\x00"  # Magic Number from version 1.16
    MAGIC_118 = b"\xf0\xff\xff\xff\x00\x00"  # Magic Number from version 1.18
    MAGIC_120 = b"\xf1\xff\xff\xff\x00\x00"  # Magic Number from version 1.20

    if magic == MAGIC_112:
        return "1.12"
    elif magic == MAGIC_116:
        return "1.16"
    elif magic == MAGIC_118:
        return "1.18"
    elif magic == MAGIC_120:
        return "1.20"
    else:
        return VERSION_UNKNOWN_OR_NA


def verify_pclntab(section_data: bytes, pclntab_offset: int) -> bool:
    """
    Parse headers of pclntab to verify it is legit
    used in go parser itself https://go.dev/src/debug/gosym/pclntab.go

    ``pclntab_offset`` is the offset of the magic header within ``section_data``.
    """
    try:
        pc_quanum = section_data[pclntab_offset + 6]
        pointer_size = section_data[pclntab_offset + 7]
    except IndexError:
        logger.error("Error parsing pclntab header")
        return False
    return True if pc_quanum in {1, 2, 4} and pointer_size in {4, 8} else False


def is_dotnet_bin(pe: pefile.PE) -> bool:
    """
    Check if the binary is .net or not
    Checks the IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR entry in the OPTIONAL_HEADER of the file.
    If the entry is not found, or if its size is 0, the file is not a .net file.

    .NET assemblies are always PE files, so this check is PE-only.
    """
    try:
        directory_index = pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR"]
        dir_entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[directory_index]
    except IndexError:
        return False

    return dir_entry.Size != 0 and dir_entry.VirtualAddress != 0

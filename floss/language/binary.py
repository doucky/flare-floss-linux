# Copyright 2024 Google LLC
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

"""
Executable format abstraction layer used by the language-specific string extractors.

Historically, FLOSS' language-specific string extraction (Go, Rust) was written
directly against ``pefile`` and therefore only supported Windows PE files. This
module introduces a thin, format-agnostic interface, :class:`BinaryFile`, that
exposes everything the extractors need (sections, image base, architecture,
virtual-address <-> file-offset translation) for both PE *and* ELF binaries.

Two implementations are provided:

* :class:`PeBinary`  - backed by ``pefile`` (Windows PE/PE+).
* :class:`ElfBinary` - backed by ``pyelftools`` (System V ELF).

Use :func:`load_binary` to auto-detect the format from the file magic.

Design notes
------------
All addresses exposed by this interface are *absolute virtual addresses* (VA),
i.e. ``image_base`` is already added. This matches the convention the Go/Rust
extractors already used for PE (``image_base + section.VirtualAddress``) and lets
the same arithmetic work unchanged for ELF, where section ``sh_addr`` values are
already absolute.

For position-independent ELF binaries (PIE executables and shared objects), the
pointers that the Go/Rust string heuristics rely on are not stored literally in
the file; instead they are materialized at load time by ``R_*_RELATIVE``
relocations. :class:`ElfBinary` applies these relative relocations to an in-memory
copy of the section data so that the pointer-scanning heuristics see the same
bytes a loader would, exactly like they do for PE.
"""

import abc
import struct
import logging
from enum import Enum
from typing import List, Tuple, Union, Optional
from pathlib import Path
from dataclasses import dataclass

from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

VA: TypeAlias = int

# file magic bytes, duplicated here to keep this module dependency-light
MAGIC_PE = b"MZ"
MAGIC_ELF = b"\x7fELF"


class Arch(Enum):
    """processor architecture, normalized across PE and ELF."""

    I386 = "i386"
    AMD64 = "amd64"
    UNKNOWN = "unknown"


class Format(Enum):
    PE = "pe"
    ELF = "elf"


@dataclass
class Section:
    """
    a normalized executable section.

    addresses are absolute virtual addresses (image base already applied).
    ``data`` contains the bytes as they would appear in memory after the loader
    applied relative relocations (relevant for PIE ELF binaries); for PE this is
    simply the raw section data.
    """

    name: str
    # absolute virtual address where the section is mapped
    virtual_address: VA
    # size of the section as mapped in memory
    virtual_size: int
    # file offset of the section's raw data
    raw_offset: int
    # size of the section's raw data on disk
    raw_size: int
    # section bytes (relocations applied for ELF)
    data: bytes
    is_executable: bool
    is_readable: bool

    @property
    def virtual_end(self) -> VA:
        # use the raw size for the end boundary: this matches the original PE
        # behavior (SizeOfRawData) and avoids walking virtual padding / .bss.
        return self.virtual_address + self.raw_size


class BinaryFile(abc.ABC):
    """format-agnostic view over an executable, used by the language extractors."""

    format: Format
    arch: Arch
    image_base: VA
    sections: List[Section]

    @property
    def image_range(self) -> Tuple[VA, VA]:
        """return the (low, high) range of the image in memory."""
        raise NotImplementedError

    def get_section_by_va(self, va: VA) -> Optional[Section]:
        for section in self.sections:
            if section.virtual_address <= va < section.virtual_end:
                return section
        return None

    def get_sections_by_name(self, *names: str) -> List[Section]:
        wanted = set(names)
        return [s for s in self.sections if s.name in wanted]

    def get_read_only_data_section(self) -> Section:
        """
        return the primary read-only data section (PE ``.rdata`` / ELF ``.rodata``).

        raises ValueError if no such section exists.
        """
        raise NotImplementedError

    def va_to_offset(self, va: VA) -> int:
        """translate an absolute virtual address to a file offset."""
        section = self.get_section_by_va(va)
        if section is None:
            raise ValueError("address 0x%x is not mapped to any section" % va)
        return section.raw_offset + (va - section.virtual_address)

    def get_data(self, va: VA, size: int) -> bytes:
        """read ``size`` bytes starting at the given absolute virtual address."""
        section = self.get_section_by_va(va)
        if section is None:
            raise ValueError("address 0x%x is not mapped to any section" % va)
        start = va - section.virtual_address
        return bytes(section.data[start : start + size])

    @property
    def max_section_size(self) -> int:
        return max((s.raw_size for s in self.sections), default=0)


class PeBinary(BinaryFile):
    """PE/PE+ implementation backed by ``pefile``.

    This delegates address translation to the underlying ``pefile.PE`` object so
    that behavior is byte-for-byte identical to FLOSS' historical PE handling.
    """

    format = Format.PE

    def __init__(self, buf: bytes):
        import pefile

        self.pe = pefile.PE(data=buf, fast_load=True)
        self.image_base = self.pe.OPTIONAL_HEADER.ImageBase

        machine = self.pe.FILE_HEADER.Machine
        if machine == pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
            self.arch = Arch.AMD64
        elif machine == pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_I386"]:
            self.arch = Arch.I386
        else:
            self.arch = Arch.UNKNOWN

        self.sections = []
        for section in self.pe.sections:
            name = section.Name.partition(b"\x00")[0].decode("utf-8", "ignore")
            self.sections.append(
                Section(
                    name=name,
                    virtual_address=self.image_base + section.VirtualAddress,
                    virtual_size=section.Misc_VirtualSize,
                    raw_offset=section.PointerToRawData,
                    raw_size=section.SizeOfRawData,
                    data=section.get_data(),
                    is_executable=bool(section.IMAGE_SCN_MEM_EXECUTE),
                    is_readable=bool(section.IMAGE_SCN_MEM_READ),
                )
            )

    @property
    def image_range(self) -> Tuple[VA, VA]:
        return self.image_base, self.image_base + self.pe.OPTIONAL_HEADER.SizeOfImage

    def get_read_only_data_section(self) -> Section:
        for section in self.sections:
            if section.name == ".rdata":
                return section
        raise ValueError("no .rdata section found")

    def va_to_offset(self, va: VA) -> int:
        # defer to pefile to preserve exact historical behavior
        return self.pe.get_offset_from_rva(va - self.image_base)

    def get_data(self, va: VA, size: int) -> bytes:
        return self.pe.get_data(va - self.image_base, size)


# ELF constants we need (avoid importing private pyelftools enums)
_SHF_ALLOC = 0x2
_SHF_EXECINSTR = 0x4
_PT_LOAD = "PT_LOAD"


class ElfBinary(BinaryFile):
    """ELF implementation backed by ``pyelftools``."""

    format = Format.ELF

    def __init__(self, buf: bytes):
        import io

        from elftools.elf.elffile import ELFFile

        self.elf = ELFFile(io.BytesIO(buf))
        self._buf = buf

        # ELF section addresses are already absolute; we keep image_base at 0 and
        # operate on absolute VAs throughout, like PeBinary does after adding the
        # PE image base.
        self.image_base = 0

        machine = self.elf.header["e_machine"]
        if machine == "EM_X86_64":
            self.arch = Arch.AMD64
        elif machine == "EM_386":
            self.arch = Arch.I386
        else:
            self.arch = Arch.UNKNOWN

        self._ptr_size = 8 if self.elf.elfclass == 64 else 4
        relocations = self._collect_relative_relocations()

        self.sections = []
        for section in self.elf.iter_sections():
            sh_flags = section["sh_flags"]
            if not (sh_flags & _SHF_ALLOC):
                # only mapped sections matter for string extraction
                continue

            sh_addr = section["sh_addr"]
            sh_size = section["sh_size"]
            sh_offset = section["sh_offset"]

            if section["sh_type"] == "SHT_NOBITS":
                # e.g. .bss - no file bytes
                data = b"\x00" * sh_size
                raw_size = 0
            else:
                data = bytearray(section.data())
                raw_size = len(data)
                # apply relative relocations that fall inside this section so the
                # pointer heuristics see loader-resolved pointers (PIE / .so).
                self._apply_relocations(data, sh_addr, raw_size, relocations, self._ptr_size)
                data = bytes(data)

            self.sections.append(
                Section(
                    name=section.name,
                    virtual_address=sh_addr,
                    virtual_size=sh_size,
                    raw_offset=sh_offset,
                    raw_size=raw_size,
                    data=data,
                    is_executable=bool(sh_flags & _SHF_EXECINSTR),
                    is_readable=bool(sh_flags & _SHF_ALLOC),
                )
            )

    def _collect_relative_relocations(self) -> List[Tuple[int, int]]:
        """
        collect (offset, value) pairs for R_*_RELATIVE relocations.

        For PIE executables and shared objects, pointers stored in read-only data
        (e.g. Go's struct String table) are emitted as relative relocations whose
        addend is the target virtual address. We materialize them so the pointer
        heuristics work just like they do on non-PIE / PE binaries.
        """
        from elftools.elf.relocation import RelocationSection

        results: List[Tuple[int, int]] = []
        is64 = self.elf.elfclass == 64
        # R_X86_64_RELATIVE == 8, R_386_RELATIVE == 8
        relative_types = {8}

        for section in self.elf.iter_sections():
            if not isinstance(section, RelocationSection):
                continue
            try:
                entries = list(section.iter_relocations())
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("failed to parse relocation section %s: %s", section.name, e)
                continue
            for reloc in entries:
                if reloc["r_info_type"] not in relative_types:
                    continue
                offset = reloc["r_offset"]
                if section.is_RELA():
                    value = reloc["r_addend"]
                else:
                    # REL: the addend is stored in place; leave the bytes as-is.
                    # reading the current value would require resolving the
                    # section, and the in-place value is already correct.
                    continue
                results.append((offset, value))

        return results

    @staticmethod
    def _apply_relocations(
        data: bytearray, sh_addr: int, sh_size: int, relocations: List[Tuple[int, int]], ptr_size: int
    ) -> None:
        section_end = sh_addr + sh_size
        fmt = "<Q" if ptr_size == 8 else "<I"
        mask = (1 << (ptr_size * 8)) - 1
        for offset, value in relocations:
            if not (sh_addr <= offset < section_end):
                continue
            local = offset - sh_addr
            if local + ptr_size <= len(data):
                data[local : local + ptr_size] = struct.pack(fmt, value & mask)

    @property
    def image_range(self) -> Tuple[VA, VA]:
        low = None
        high = None
        for segment in self.elf.iter_segments():
            if segment["p_type"] != _PT_LOAD:
                continue
            start = segment["p_vaddr"]
            end = start + segment["p_memsz"]
            low = start if low is None else min(low, start)
            high = end if high is None else max(high, end)

        if low is None or high is None:
            # fall back to allocated sections
            if not self.sections:
                return 0, 0
            low = min(s.virtual_address for s in self.sections)
            high = max(s.virtual_end for s in self.sections)

        return low, high

    def get_read_only_data_section(self) -> Section:
        for section in self.sections:
            if section.name == ".rodata":
                return section
        raise ValueError("no .rodata section found")


def load_binary(sample: Union[str, Path, bytes]) -> BinaryFile:
    """
    load a sample as a :class:`BinaryFile`, auto-detecting PE vs ELF from the magic.

    raises ValueError for unsupported formats.
    """
    if isinstance(sample, (str, Path)):
        buf = Path(sample).read_bytes()
    else:
        buf = sample

    if buf[:4] == MAGIC_ELF:
        return ElfBinary(buf)
    elif buf[:2] == MAGIC_PE:
        return PeBinary(buf)
    else:
        raise ValueError("unsupported file format (expected PE or ELF)")

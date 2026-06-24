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
Tests for the executable-format abstraction layer (floss/language/binary.py) and
the ELF code path of the language-specific string extractors.

The end-to-end ELF tests build tiny sample binaries with the system C compiler.
They are skipped automatically when no suitable compiler/toolchain is available
(e.g. on Windows CI), so they never fail spuriously.
"""

import shutil
import textwrap
import subprocess

import pytest

from floss.language.binary import Arch, Format, ElfBinary, load_binary
from floss.language.identify import get_if_go_and_version

C_SOURCE = textwrap.dedent(
    """
    #include <stdio.h>
    const char *msg1 = "FLOSS_ELF_ABSTRACTION_TEST_STRING";
    const char *msg2 = "another_readonly_string_for_rodata";
    int main(void) { printf("%s %s\\n", msg1, msg2); return 0; }
    """
)

NEEDLE_1 = "FLOSS_ELF_ABSTRACTION_TEST_STRING"
NEEDLE_2 = "another_readonly_string_for_rodata"


def _have_cc() -> bool:
    return shutil.which("gcc") is not None or shutil.which("cc") is not None


def _build_elf(tmp_path, pie: bool):
    cc = shutil.which("gcc") or shutil.which("cc")
    src = tmp_path / "sample.c"
    src.write_text(C_SOURCE)
    out = tmp_path / ("sample_pie" if pie else "sample_nopie")
    flags = ["-O2", "-fPIE", "-pie"] if pie else ["-O2", "-no-pie"]
    try:
        subprocess.run([cc, *flags, str(src), "-o", str(out)], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover - toolchain dependent
        pytest.skip(f"C toolchain cannot build sample: {e.stderr.decode(errors='ignore')[:200]}")
    return out


def test_load_binary_rejects_garbage():
    with pytest.raises(ValueError):
        load_binary(b"not an executable at all")


def test_load_binary_detects_elf(tmp_path):
    if not _have_cc():
        pytest.skip("no C compiler available")
    path = _build_elf(tmp_path, pie=False)
    bf = load_binary(path)
    assert bf.format == Format.ELF
    assert isinstance(bf, ElfBinary)
    assert bf.arch in (Arch.AMD64, Arch.I386)


@pytest.mark.parametrize("pie", [False, True])
def test_elf_section_and_offset_translation(tmp_path, pie):
    if not _have_cc():
        pytest.skip("no C compiler available")
    path = _build_elf(tmp_path, pie=pie)
    bf = load_binary(path)

    rodata = bf.get_read_only_data_section()
    assert rodata.name == ".rodata"

    idx = rodata.data.find(NEEDLE_1.encode())
    assert idx != -1, "test string should be present in .rodata"

    va = rodata.virtual_address + idx
    # the bytes read by virtual address must match the literal string
    assert bf.get_data(va, len(NEEDLE_1)) == NEEDLE_1.encode()

    # va -> file offset must point at the same bytes on disk
    off = bf.va_to_offset(va)
    disk = path.read_bytes()[off : off + len(NEEDLE_1)]
    assert disk == NEEDLE_1.encode()


@pytest.mark.parametrize("pie", [False, True])
def test_rust_extractor_runs_on_elf(tmp_path, pie):
    if not _have_cc():
        pytest.skip("no C compiler available")
    from floss.language.rust.extract import extract_rust_strings

    path = _build_elf(tmp_path, pie=pie)
    strings = extract_rust_strings(path, min_length=6)
    values = {s.string for s in strings}
    # the Rust .rodata blob algorithm should recover our read-only strings
    assert NEEDLE_1 in values
    assert NEEDLE_2 in values

    # every reported offset must correspond to the real on-disk bytes
    disk = path.read_bytes()
    for s in strings:
        if s.string in (NEEDLE_1, NEEDLE_2):
            assert disk[s.offset : s.offset + len(s.string)] == s.string.encode()


@pytest.mark.parametrize("pie", [False, True])
def test_go_detection_negative_on_c_binary(tmp_path, pie):
    if not _have_cc():
        pytest.skip("no C compiler available")
    path = _build_elf(tmp_path, pie=pie)
    bf = load_binary(path)
    is_go, _ = get_if_go_and_version(bf)
    assert is_go is False

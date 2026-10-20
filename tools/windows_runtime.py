"""Windows release bundle runtime policy and a minimal PE reader.

The PE reader runs on any host, so the merged release bundle is checked on
the macOS packaging runner as well as on the Windows build runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import mmap
from pathlib import Path
import struct

MACHINES = {0x8664: "x64", 0xAA64: "arm64", 0x014C: "x86"}

# Libraries every supported Windows 10+ host provides, plus the GPU loaders
# that drivers install (vulkan-1.dll, nvcuda.dll, OpenCL.dll). Bundles never
# ship these.
OS_PROVIDED_DLLS = frozenset({
    "advapi32.dll",
    "bcrypt.dll",
    "cfgmgr32.dll",
    "combase.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d3d12.dll",
    "dbghelp.dll",
    "dxgi.dll",
    "gdi32.dll",
    "gdi32full.dll",
    "imm32.dll",
    "kernel32.dll",
    "kernelbase.dll",
    "msvcrt.dll",
    "ntdll.dll",
    "nvcuda.dll",
    "ole32.dll",
    "oleaut32.dll",
    "opencl.dll",
    "powrprof.dll",
    "psapi.dll",
    "rpcrt4.dll",
    "sechost.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "ucrtbase.dll",
    "user32.dll",
    "version.dll",
    "vulkan-1.dll",
    "winmm.dll",
    "ws2_32.dll",
})
OS_PROVIDED_PREFIXES = ("api-ms-win-", "ext-ms-")

# Visual C++ v14 runtime and OpenMP DLLs. Release bundles ship them app-locally
# from the build toolset's redistributable folder (VCToolsRedistDir), never
# from System32 or a debug_nonredist folder.
VC_RUNTIME_PATTERNS = (
    "msvcp140.dll",
    "msvcp140_*.dll",
    "vcruntime140.dll",
    "vcruntime140_*.dll",
    "concrt140.dll",
    "vccorlib140.dll",
)
OPENMP_RUNTIME_PATTERNS = ("vcomp140.dll", "libomp140.*.dll")
VC_RUNTIME_FOLDER = "Microsoft.VC*.CRT"
OPENMP_RUNTIME_FOLDER = "Microsoft.VC*.OpenMP*"


def is_os_provided(name: str) -> bool:
    lowered = name.lower()
    return lowered in OS_PROVIDED_DLLS or lowered.startswith(OS_PROVIDED_PREFIXES)


def is_vc_runtime(name: str) -> bool:
    return any(fnmatch(name.lower(), pattern) for pattern in VC_RUNTIME_PATTERNS)


def is_vc_redistributable(name: str) -> bool:
    lowered = name.lower()
    return is_vc_runtime(lowered) or any(fnmatch(lowered, pattern) for pattern in OPENMP_RUNTIME_PATTERNS)


def find_vc_redistributable(name: str, arch: str, redist_dir: Path) -> Path | None:
    """The desktop redistributable copy of `name` for `arch`, if the toolset has one."""
    folder = VC_RUNTIME_FOLDER if is_vc_runtime(name) else OPENMP_RUNTIME_FOLDER
    lowered = name.lower()
    matches = sorted(
        candidate
        for candidate in (redist_dir / arch).glob(f"{folder}/*.dll")
        if candidate.name.lower() == lowered
    )
    return matches[-1] if matches else None


class PeError(ValueError):
    pass


@dataclass(frozen=True)
class PeImage:
    machine: str
    linker_version: tuple[int, int]
    imports: dict[str, frozenset[str]]
    delay_imports: dict[str, frozenset[str]]
    exports: frozenset[str]
    file_version: tuple[int, int, int, int] | None

    def all_imports(self) -> dict[str, frozenset[str]]:
        merged = {name: set(symbols) for name, symbols in self.imports.items()}
        for name, symbols in self.delay_imports.items():
            merged.setdefault(name, set()).update(symbols)
        return {name: frozenset(symbols) for name, symbols in merged.items()}


class _Reader:
    def __init__(self, data) -> None:
        self.data = data
        if data[:2] != b"MZ":
            raise PeError("missing MZ header")
        (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
        if data[pe_offset:pe_offset + 4] != b"PE\0\0":
            raise PeError("missing PE signature")
        coff = pe_offset + 4
        machine, section_count = struct.unpack_from("<HH", data, coff)
        (optional_size,) = struct.unpack_from("<H", data, coff + 16)
        optional = coff + 20
        magic, major, minor = struct.unpack_from("<HBB", data, optional)
        if magic == 0x20B:
            self.pe32_plus = True
            (self.image_base,) = struct.unpack_from("<Q", data, optional + 24)
            directories = optional + 112
        elif magic == 0x10B:
            self.pe32_plus = False
            (self.image_base,) = struct.unpack_from("<I", data, optional + 28)
            directories = optional + 96
        else:
            raise PeError(f"unknown optional header magic 0x{magic:x}")
        (directory_count,) = struct.unpack_from("<I", data, directories - 4)
        self.directories = [
            struct.unpack_from("<II", data, directories + 8 * index)
            for index in range(min(directory_count, 16))
        ]
        self.machine = MACHINES.get(machine, f"0x{machine:04x}")
        self.linker_version = (major, minor)
        self.sections = []
        for index in range(section_count):
            entry = optional + optional_size + 40 * index
            virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from("<IIII", data, entry + 8)
            self.sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def offset(self, rva: int) -> int:
        for virtual_address, size, raw_offset in self.sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + rva - virtual_address
        raise PeError(f"RVA 0x{rva:x} is outside every section")

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def u32(self, rva: int) -> int:
        return struct.unpack_from("<I", self.data, self.offset(rva))[0]

    def string(self, rva: int) -> str:
        start = self.offset(rva)
        end = self.data.find(b"\0", start)
        if end < 0:
            raise PeError(f"unterminated string at RVA 0x{rva:x}")
        return self.data[start:end].decode("ascii", errors="replace")

    def thunk_names(self, rva: int) -> frozenset[str]:
        names = set()
        width, ordinal_flag = (8, 1 << 63) if self.pe32_plus else (4, 1 << 31)
        while rva:
            (thunk,) = struct.unpack_from("<Q" if self.pe32_plus else "<I", self.data, self.offset(rva))
            if thunk == 0:
                break
            names.add(f"#{thunk & 0xFFFF}" if thunk & ordinal_flag else self.string(thunk + 2))
            rva += width
        return frozenset(names)

    def imports(self) -> dict[str, frozenset[str]]:
        rva, size = self.directory(1)
        result: dict[str, frozenset[str]] = {}
        while rva and size:
            lookup, _, _, name, address = struct.unpack_from("<IIIII", self.data, self.offset(rva))
            if name == 0:
                break
            result[self.string(name).lower()] = self.thunk_names(lookup or address)
            rva += 20
        return result

    def delay_imports(self) -> dict[str, frozenset[str]]:
        rva, size = self.directory(13)
        result: dict[str, frozenset[str]] = {}
        while rva and size:
            attributes, name, _, _, name_table = struct.unpack_from("<IIIII", self.data, self.offset(rva))
            if name == 0:
                break
            base = 0 if attributes & 1 else self.image_base
            result[self.string(name - base).lower()] = self.thunk_names(name_table - base if name_table else 0)
            rva += 32
        return result

    def exports(self) -> frozenset[str]:
        rva, size = self.directory(0)
        if not rva or not size:
            return frozenset()
        base_offset = self.offset(rva)
        name_count, _, names = struct.unpack_from("<III", self.data, base_offset + 24)
        return frozenset(self.string(self.u32(names + 4 * index)) for index in range(name_count))

    def file_version(self) -> tuple[int, int, int, int] | None:
        rva, size = self.directory(2)
        if not rva or not size:
            return None
        for data_rva, data_size in self._resource_leaves(rva, rva, 16, depth=0):
            start = self.offset(data_rva)
            blob = self.data[start:start + data_size]
            signature = blob.find(struct.pack("<I", 0xFEEF04BD))
            if signature < 0:
                continue
            most, least = struct.unpack_from("<II", blob, signature + 8)
            return most >> 16, most & 0xFFFF, least >> 16, least & 0xFFFF
        return None

    def _resource_leaves(self, root: int, directory: int, type_id: int | None, depth: int):
        start = self.offset(directory)
        named, numbered = struct.unpack_from("<HH", self.data, start + 12)
        for index in range(named + numbered):
            name, target = struct.unpack_from("<II", self.data, start + 16 + 8 * index)
            if depth == 0 and (name & 0x80000000 or name != type_id):
                continue
            if target & 0x80000000:
                yield from self._resource_leaves(root, root + (target & 0x7FFFFFFF), None, depth + 1)
            else:
                yield struct.unpack_from("<II", self.data, self.offset(root + target))


def read_pe(path: Path) -> PeImage:
    with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        try:
            reader = _Reader(data)
            return PeImage(
                machine=reader.machine,
                linker_version=reader.linker_version,
                imports=reader.imports(),
                delay_imports=reader.delay_imports(),
                exports=reader.exports(),
                file_version=reader.file_version(),
            )
        except (struct.error, ValueError) as error:
            raise PeError(f"{path.name}: malformed PE image: {error}") from error

"""Verify retained statics across mixed GCC/LLVM Linux linkers."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LinuxStaticDestructorsTest(unittest.TestCase):
    def test_gcc_object_survives_llvm_link_and_preserves_host_exit(self) -> None:
        gcc = shutil.which("gcc")
        clang = shutil.which(os.environ.get("LLAMADART_TEST_CLANG", "clang++"))
        cmake = shutil.which("cmake")
        readelf = shutil.which("readelf")
        if not all((gcc, clang, cmake, readelf)) or sys.platform != "linux":
            self.skipTest("requires Linux, GCC, Clang, CMake and readelf")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "owned.cpp").write_text(
                '#include <cstdio>\n'
                'struct Owned { ~Owned() { std::puts("OWNED_DESTROYED"); } };\n'
                'static Owned owned;\n'
                'extern "C" int probe() { return 7; }\n'
            )
            (source / "host.cpp").write_text(
                '#include <cstdio>\n#include <cstdlib>\n#include <dlfcn.h>\n'
                'int main(int argc, char **argv) {\n'
                ' if (argc != 2 || !dlopen(argv[1], RTLD_NOW | RTLD_LOCAL)) return 1;\n'
                ' atexit([] { std::puts("HOST_CALLBACK"); });\n'
                ' std::printf("C_BUFFERED_OUTPUT\\n");\n'
                ' std::exit(23);\n}\n'
            )
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.23)\n'
                'project(retention LANGUAGES C CXX)\n'
                'set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n'
                'set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)\n'
                f'include("{ROOT / "cmake/linux_static_destructors.cmake"}")\n'
                'add_library(owned MODULE owned.cpp)\n'
                'set_property(TARGET owned PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)\n'
                'target_link_options(owned PRIVATE -fuse-ld=lld)\n'
                'llamadart_drop_static_destructors("${CMAKE_CURRENT_SOURCE_DIR}")\n'
                'if(TEST_BAD_LTO)\n'
                ' set_property(TARGET llamadart_static_destructors PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)\n'
                'endif()\n'
                'add_executable(host host.cpp)\n'
                'set_property(TARGET host PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)\n'
                'target_link_libraries(host PRIVATE dl)\n'
            )
            for bad_lto in (True, False):
                with self.subTest(gcc_lto=bad_lto):
                    build = root / ("baseline" if bad_lto else "repaired")
                    for command in (
                        [cmake, "-S", str(source), "-B", str(build),
                         f"-DCMAKE_C_COMPILER={gcc}", f"-DCMAKE_CXX_COMPILER={clang}",
                         "-DCMAKE_BUILD_TYPE=Release", f"-DTEST_BAD_LTO={'ON' if bad_lto else 'OFF'}"],
                        [cmake, "--build", str(build)],
                    ):
                        result = subprocess.run(command, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    library = build / "libowned.so"
                    symbols = subprocess.run(
                        [readelf, "--dyn-syms", "-W", str(library)],
                        capture_output=True, text=True, check=True,
                    ).stdout
                    imports = [line for line in symbols.splitlines()
                               if " UND " in line and "__cxa_atexit" in line]
                    result = subprocess.run(
                        [str(build / "host"), str(library)], capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                    self.assertIn("C_BUFFERED_OUTPUT", result.stdout)
                    self.assertIn("HOST_CALLBACK", result.stdout)
                    if bad_lto:
                        self.assertTrue(imports)
                        self.assertIn("OWNED_DESTROYED", result.stdout)
                    else:
                        self.assertFalse(imports)
                        self.assertNotIn("OWNED_DESTROYED", result.stdout)


if __name__ == "__main__":
    unittest.main()

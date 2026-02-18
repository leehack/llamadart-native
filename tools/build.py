#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
OPENCL_HEADERS_DIR = THIRD_PARTY_DIR / "OpenCL-Headers"
OPENCL_LOADER_DIR = THIRD_PARTY_DIR / "OpenCL-ICD-Loader"
OPENCL_STUB_DIR = THIRD_PARTY_DIR / "opencl-stubs"
BIN_DIR = REPO_ROOT / "bin"
BUILD_ROOT = REPO_ROOT / "build"


APPLE_TARGETS = {
    "macos-arm64": ("macos-arm64", BIN_DIR / "macos/arm64"),
    "macos-x86_64": ("macos-x86_64", BIN_DIR / "macos/x86_64"),
    "macos-x64": ("macos-x86_64", BIN_DIR / "macos/x86_64"),
    "ios-device-arm64": ("ios-device-arm64", BIN_DIR / "ios/arm64"),
    "ios-sim-arm64": ("ios-sim-arm64", BIN_DIR / "ios/arm64-sim"),
    "ios-sim-x86_64": ("ios-sim-x86_64", BIN_DIR / "ios/x86_64-sim"),
    "ios-sim-x64": ("ios-sim-x86_64", BIN_DIR / "ios/x86_64-sim"),
}

ANDROID_ABI_ALIASES = {
    "arm64-v8a": "arm64-v8a",
    "arm64": "arm64-v8a",
    "x86_64": "x86_64",
    "x64": "x86_64",
}

ANDROID_OUT_ARCH = {"arm64-v8a": "arm64", "x86_64": "x64"}
ANDROID_ARCH_PATH = {"arm64-v8a": "aarch64-linux-android", "x86_64": "x86_64-linux-android"}
ANDROID_PRAGMA_WARN_SUPPRESS = "-Wno-#pragma-messages"
ANDROID_OPENCL_LOADER_WARN_SUPPRESS = "-Wno-#pragma-messages -Wno-typedef-redefinition"
WINDOWS_VCPKG_TRIPLETS = {"x64": "x64-windows", "arm64": "arm64-windows"}
ANDROID_BACKENDS = ("full", "vulkan", "opencl")
LINUX_BACKENDS = ("full", "vulkan", "cuda", "blas")
WINDOWS_BACKENDS = ("full", "vulkan", "cuda", "blas")


def fail(message: str) -> None:
    raise SystemExit(message)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    effective_env = (env.copy() if env is not None else os.environ.copy())
    effective_env.setdefault("CCACHE_DIR", str(REPO_ROOT / ".ccache"))
    subprocess.run(cmd, cwd=REPO_ROOT, env=effective_env, check=True)


@lru_cache(maxsize=1)
def load_cmake_presets() -> dict:
    presets_path = REPO_ROOT / "CMakePresets.json"
    try:
        return json.loads(presets_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Failed to read {presets_path}: {exc}")


def resolve_build_dir_for_preset(preset: str) -> Path:
    data = load_cmake_presets()
    for cfg in data.get("configurePresets", []):
        if cfg.get("name") != preset:
            continue
        binary_dir = cfg.get("binaryDir")
        if not binary_dir:
            return BUILD_ROOT / preset
        resolved = binary_dir.replace("${sourceDir}", str(REPO_ROOT)).replace("${presetName}", preset)
        return Path(resolved)
    return BUILD_ROOT / preset


def ensure_submodule(path: Path, error: str) -> None:
    if not path.is_file():
        fail(error)


def patch_llama_zendnn_install_target() -> bool:
    """Patch ZenDNN ExternalProject install target for current llama.cpp revision.

    Some pinned llama.cpp revisions configure ZenDNN without a top-level "install"
    target, so "--target install" fails. The "zendnnl" target already performs
    dependency install + library staging and is safe for the install step.
    """

    cmake_file = THIRD_PARTY_DIR / "llama.cpp/ggml/src/ggml-zendnn/CMakeLists.txt"
    ensure_submodule(
        cmake_file,
        "Missing ggml-zendnn CMake file in llama.cpp submodule. Run: git submodule update --init --recursive",
    )

    old = "INSTALL_COMMAND ${CMAKE_COMMAND} --build ${ZENDNN_BUILD_DIR} --target install"
    new = "INSTALL_COMMAND ${CMAKE_COMMAND} --build ${ZENDNN_BUILD_DIR} --target zendnnl"

    text = cmake_file.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        fail(
            "Could not apply ZenDNN install target patch: expected install command not found in "
            f"{cmake_file}"
        )

    cmake_file.write_text(text.replace(old, new), encoding="utf-8")
    print("Patched llama.cpp ggml-zendnn install target to zendnnl")
    return True


def restore_llama_zendnn_install_target() -> None:
    cmake_file = THIRD_PARTY_DIR / "llama.cpp/ggml/src/ggml-zendnn/CMakeLists.txt"
    old = "INSTALL_COMMAND ${CMAKE_COMMAND} --build ${ZENDNN_BUILD_DIR} --target install"
    new = "INSTALL_COMMAND ${CMAKE_COMMAND} --build ${ZENDNN_BUILD_DIR} --target zendnnl"

    if not cmake_file.is_file():
        return

    text = cmake_file.read_text(encoding="utf-8")
    if new not in text:
        return

    cmake_file.write_text(text.replace(new, old), encoding="utf-8")
    print("Restored llama.cpp ggml-zendnn install target to install")


def clean_build_dir(preset: str, clean: bool) -> Path:
    build_dir = resolve_build_dir_for_preset(preset)
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    return build_dir


def configure_and_build(
    preset: str,
    *,
    jobs: int | None,
    extra_cmake_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    configure_cmd = ["cmake", "--preset", preset]
    if extra_cmake_args:
        configure_cmd.extend(extra_cmake_args)
    run(configure_cmd, env=env)

    build_cmd = ["cmake", "--build", "--preset", preset]
    if jobs and jobs > 0:
        build_cmd.extend(["--parallel", str(jobs)])
    run(build_cmd, env=env)
    return resolve_build_dir_for_preset(preset)


def detect_linux_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    fail(f"Unsupported host architecture '{machine}' for Linux builds")


def cmake_cache_args(cache_vars: dict[str, str]) -> list[str]:
    return [f"-D{key}={value}" for key, value in cache_vars.items()]


def linux_backend_cache_vars(arch: str, backend: str) -> dict[str, str]:
    if backend == "cuda" and arch != "x64":
        fail("Linux cuda backend build is only available for x64")

    cache_vars: dict[str, str] = {
        "GGML_VULKAN": "OFF",
        "GGML_OPENCL": "OFF",
        "GGML_CUDA": "OFF",
        "GGML_BLAS": "OFF",
        "GGML_ZENDNN": "OFF",
        "GGML_CPU_KLEIDIAI": "ON" if arch == "arm64" else "OFF",
    }

    if backend in ("full", "vulkan"):
        cache_vars["GGML_VULKAN"] = "ON"
    if backend in ("full", "cuda"):
        cache_vars["GGML_CUDA"] = "ON"
    if backend in ("full", "blas"):
        cache_vars["GGML_BLAS"] = "ON"
        cache_vars["GGML_BLAS_VENDOR"] = "OpenBLAS"

    return cache_vars


def android_backend_cache_vars(abi: str, backend: str) -> dict[str, str]:
    cache_vars: dict[str, str] = {
        "GGML_VULKAN": "OFF",
        "GGML_OPENCL": "OFF",
        "GGML_CPU_KLEIDIAI": "ON" if abi == "arm64-v8a" else "OFF",
    }

    if backend in ("full", "vulkan"):
        cache_vars["GGML_VULKAN"] = "ON"
    if backend in ("full", "opencl"):
        cache_vars["GGML_OPENCL"] = "ON"

    return cache_vars


def windows_backend_cache_vars(arch: str, backend: str) -> dict[str, str]:
    if backend == "cuda" and arch != "x64":
        fail("Windows cuda backend build is only available for x64")

    cache_vars: dict[str, str] = {
        "GGML_VULKAN": "OFF",
        "GGML_OPENCL": "OFF",
        "GGML_CUDA": "OFF",
        "GGML_BLAS": "OFF",
        "GGML_CPU_KLEIDIAI": "ON" if arch == "arm64" else "OFF",
    }

    if backend in ("full", "vulkan"):
        cache_vars["GGML_VULKAN"] = "ON"
    if backend in ("full", "cuda"):
        cache_vars["GGML_CUDA"] = "ON"
    if backend in ("full", "blas"):
        cache_vars["GGML_BLAS"] = "ON"
        cache_vars["GGML_BLAS_VENDOR"] = "OpenBLAS"

    return cache_vars


def detect_android_ndk() -> Path | None:
    env_ndk = os.environ.get("ANDROID_NDK_HOME")
    if env_ndk:
        p = Path(env_ndk).expanduser()
        if p.exists():
            return p

    sdk_roots = []
    for key in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        value = os.environ.get(key)
        if value:
            sdk_roots.append(Path(value).expanduser())
    sdk_roots.extend(
        [
            Path("~/Library/Android/sdk").expanduser(),
            Path("~/Android/Sdk").expanduser(),
            Path("/usr/local/lib/android/sdk"),
        ]
    )

    for root in sdk_roots:
        ndk_root = root / "ndk"
        if ndk_root.is_dir():
            versions = sorted([p for p in ndk_root.iterdir() if p.is_dir()], reverse=True)
            if versions:
                return versions[0]

    legacy = Path("/usr/local/lib/android/sdk/ndk-bundle")
    if legacy.is_dir():
        return legacy
    return None


def find_file_with_suffix(root: Path, suffix: str, *, contains: str | None = None) -> Path | None:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if not str(p).endswith(suffix):
            continue
        if contains and contains not in str(p):
            continue
        return p
    return None


def infer_vulkan_sdk(path_hint: str | None) -> Path | None:
    if path_hint:
        p = Path(path_hint).expanduser()
        if p.exists():
            return p

    env_sdk = os.environ.get("VULKAN_SDK")
    if env_sdk:
        p = Path(env_sdk).expanduser()
        if p.exists():
            return p

    glslc = shutil.which("glslc") or shutil.which("glslc.exe")
    if glslc:
        glslc_path = Path(glslc).resolve()
        for parent in (glslc_path.parent, glslc_path.parent.parent):
            include_dir = parent / "Include" / "vulkan" / "vulkan.h"
            lib_dir = parent / "Lib"
            if include_dir.exists() or lib_dir.exists():
                return parent
    return None


def detect_vcpkg_root() -> Path | None:
    for key in ("VCPKG_ROOT", "VCPKG_INSTALLATION_ROOT"):
        value = os.environ.get(key)
        if not value:
            continue
        root = Path(value).expanduser()
        if root.is_dir():
            return root
    for candidate in (Path("C:/vcpkg"), Path("C:/tools/vcpkg")):
        if candidate.is_dir():
            return candidate
    return None


def abi_env_key(abi: str) -> str:
    return abi.upper().replace("-", "_")


def build_android_opencl_loader(abi: str, ndk: Path, build_dir: Path, env: dict[str, str], jobs: int | None) -> Path | None:
    if not OPENCL_LOADER_DIR.joinpath("CMakeLists.txt").is_file():
        return None
    if not OPENCL_HEADERS_DIR.joinpath("CL/cl.h").is_file():
        return None

    loader_build = build_dir / "opencl-loader"
    loader_build.mkdir(parents=True, exist_ok=True)

    toolchain_file = ndk / "build/cmake/android.toolchain.cmake"
    configure_cmd = [
        "cmake",
        "-S",
        str(OPENCL_LOADER_DIR),
        "-B",
        str(loader_build),
        "-G",
        "Ninja",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
        f"-DANDROID_ABI={abi}",
        "-DANDROID_PLATFORM=android-28",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_FLAGS={ANDROID_OPENCL_LOADER_WARN_SUPPRESS}",
        "-DENABLE_OPENCL_LAYERS=OFF",
        "-DENABLE_OPENCL_LAYERINFO=OFF",
        f"-DOPENCL_ICD_LOADER_HEADERS_DIR={OPENCL_HEADERS_DIR}",
        "-DOPENCL_ICD_LOADER_BUILD_TESTING=OFF",
        "-DBUILD_TESTING=OFF",
    ]
    run(configure_cmd, env=env)

    build_cmd = ["cmake", "--build", str(loader_build), "--config", "Release"]
    if jobs and jobs > 0:
        build_cmd.extend(["--parallel", str(jobs)])
    run(build_cmd, env=env)

    return find_file_with_suffix(loader_build, "libOpenCL.so")


def resolve_android_opencl(abi: str, ndk: Path, build_dir: Path, env: dict[str, str], jobs: int | None) -> tuple[Path, Path]:
    env_key = abi_env_key(abi)
    include_env = os.environ.get("OPENCL_INCLUDE_DIR")
    lib_env = os.environ.get(f"OPENCL_LIBRARY_ANDROID_{env_key}")

    include_candidates: list[Path] = []
    if include_env:
        include_candidates.append(Path(include_env).expanduser())
    include_candidates.extend([OPENCL_HEADERS_DIR, OPENCL_STUB_DIR / "include"])

    opencl_include: Path | None = None
    for include_dir in include_candidates:
        if include_dir.joinpath("CL/cl.h").is_file():
            opencl_include = include_dir
            break

    if lib_env:
        opencl_lib = Path(lib_env).expanduser()
        if not opencl_lib.is_file():
            fail(f"OPENCL_LIBRARY_ANDROID_{env_key} is set but file does not exist: {opencl_lib}")
    else:
        arch_path = ANDROID_ARCH_PATH[abi]
        opencl_lib = find_file_with_suffix(ndk, "libOpenCL.so", contains=f"/{arch_path}/")
        if not opencl_lib:
            prebuilt = OPENCL_STUB_DIR / "android" / abi / "libOpenCL.so"
            if prebuilt.is_file():
                opencl_lib = prebuilt
        if not opencl_lib:
            opencl_lib = build_android_opencl_loader(abi, ndk, build_dir, env, jobs)

        if not opencl_lib:
            fail(
                "Could not resolve Android OpenCL library.\n"
                "Provide one of:\n"
                f"- env OPENCL_LIBRARY_ANDROID_{env_key}=/path/to/libOpenCL.so\n"
                "- repo path third_party/opencl-stubs/android/<abi>/libOpenCL.so\n"
                "- third_party/OpenCL-ICD-Loader + third_party/OpenCL-Headers submodules for auto-build"
            )

    if not opencl_include:
        fail(
            "Could not resolve OpenCL headers (missing CL/cl.h).\n"
            "Provide one of:\n"
            "- env OPENCL_INCLUDE_DIR=/path/to/opencl-headers\n"
            "- third_party/OpenCL-Headers submodule\n"
            "- third_party/opencl-stubs/include"
        )

    return opencl_include, opencl_lib


def is_runtime_library(path: Path) -> bool:
    name = path.name.lower()
    if not path.is_file():
        return False

    # Keep canonical runtime filenames only (drop Linux SONAME aliases like libfoo.so.0 / libfoo.so.0.0.0).
    if not (name.endswith(".dll") or name.endswith(".dylib") or name.endswith(".so")):
        return False

    prefixes = (
        "llamadart",
        "llama",
        "ggml",
        "mtmd",
        "libllamadart",
        "libllama",
        "libggml",
        "libmtmd",
    )
    return any(name.startswith(p) for p in prefixes)


def collect_runtime_libraries(build_dir: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    for p in sorted(build_dir.rglob("*")):
        if not is_runtime_library(p):
            continue
        selected[p.name] = p

    if not selected:
        fail(f"No runtime libraries found under {build_dir}")

    return [selected[k] for k in sorted(selected)]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_output(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Built: {dst.relative_to(REPO_ROOT)}")


def copy_runtime_libraries(build_dir: Path, out_dir: Path) -> None:
    libs = collect_runtime_libraries(build_dir)
    reset_dir(out_dir)

    for src in libs:
        copy_output(src, out_dir / src.name)

    print(f"Copied {len(libs)} runtime libraries to {out_dir.relative_to(REPO_ROOT)}")


def build_apple(args: argparse.Namespace) -> None:
    if platform.system().lower() != "darwin":
        fail("Apple builds must be run on macOS hosts")
    ensure_submodule(
        THIRD_PARTY_DIR / "llama.cpp/CMakeLists.txt",
        "Missing submodule: third_party/llama.cpp. Run: git submodule update --init --recursive",
    )

    normalized, out_dir = APPLE_TARGETS[args.target]
    preset = f"{normalized}-full"
    clean_build_dir(preset, args.clean)
    build_dir = configure_and_build(preset, jobs=args.jobs)
    copy_runtime_libraries(build_dir, out_dir)


def build_linux(args: argparse.Namespace) -> None:
    if platform.system().lower() != "linux":
        fail("Linux builds must be run on Linux hosts")
    ensure_submodule(
        THIRD_PARTY_DIR / "llama.cpp/CMakeLists.txt",
        "Missing submodule: third_party/llama.cpp. Run: git submodule update --init --recursive",
    )

    arch = args.arch or detect_linux_arch()
    backend = args.backend
    preset = f"linux-{arch}-full"
    clean_build_dir(preset, args.clean)

    cache_vars = linux_backend_cache_vars(arch, backend)
    extra_args = cmake_cache_args(cache_vars)
    host_arch = detect_linux_arch()
    if arch == "arm64" and host_arch != "arm64":
        cc = shutil.which("aarch64-linux-gnu-gcc")
        cxx = shutil.which("aarch64-linux-gnu-g++")
        if not cc or not cxx:
            fail("Cross compiler not found. Install aarch64-linux-gnu-gcc and aarch64-linux-gnu-g++")
        extra_args.extend(
            [
                f"-DCMAKE_C_COMPILER={cc}",
                f"-DCMAKE_CXX_COMPILER={cxx}",
                "-DCMAKE_SYSTEM_NAME=Linux",
                "-DCMAKE_SYSTEM_PROCESSOR=aarch64",
            ]
        )

    if cache_vars["GGML_CUDA"] == "ON" and not (shutil.which("nvcc") or shutil.which("nvcc.exe")):
        fail("Linux CUDA backend build requires CUDA (nvcc not found in PATH)")

    zendnn_patch_applied = False
    if arch == "x64" and cache_vars["GGML_ZENDNN"] == "ON":
        zendnn_patch_applied = patch_llama_zendnn_install_target()

    try:
        build_dir = configure_and_build(preset, jobs=args.jobs, extra_cmake_args=extra_args)
        copy_runtime_libraries(build_dir, BIN_DIR / f"linux/{arch}")
    finally:
        if zendnn_patch_applied:
            restore_llama_zendnn_install_target()


def write_android_host_toolchain(path: Path) -> None:
    make_program = shutil.which("ninja") or shutil.which("make")
    lines = []
    if make_program:
        lines.append(f'set(CMAKE_MAKE_PROGRAM "{make_program}" CACHE STRING "make program" FORCE)')
    lines.append(f'set(CMAKE_SYSTEM_NAME "{platform.system()}")')
    lines.append("set(Threads_FOUND TRUE)")
    lines.append('set(CMAKE_THREAD_LIBS_INIT "-pthread")')
    lines.append("set(CMAKE_USE_PTHREADS_INIT TRUE)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_android_abi(abi: str, args: argparse.Namespace, env: dict[str, str]) -> None:
    preset = f"android-{abi}-full"
    build_dir = clean_build_dir(preset, args.clean)
    backend = args.backend

    ndk = Path(env["ANDROID_NDK_HOME"])
    cache_vars = android_backend_cache_vars(abi, backend)
    extra_args = cmake_cache_args(cache_vars)
    extra_args.extend(
        [
            f"-DCMAKE_C_FLAGS={ANDROID_PRAGMA_WARN_SUPPRESS}",
            f"-DCMAKE_CXX_FLAGS={ANDROID_PRAGMA_WARN_SUPPRESS}",
        ]
    )

    if cache_vars["GGML_VULKAN"] == "ON":
        ensure_submodule(
            THIRD_PARTY_DIR / "Vulkan-Headers/include/vulkan/vulkan.h",
            "Missing submodule: third_party/Vulkan-Headers. Run: git submodule update --init --recursive",
        )
        toolchain = build_dir / "android-host-toolchain.cmake"
        write_android_host_toolchain(toolchain)
        extra_args.append(f"-DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN={toolchain}")

        glslc = find_file_with_suffix(ndk, "glslc") or find_file_with_suffix(ndk, "glslc.exe")
        if glslc:
            extra_args.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc}")

        arch_path = ANDROID_ARCH_PATH[abi]
        vulkan_lib = find_file_with_suffix(ndk, "libvulkan.so", contains=f"/{arch_path}/28/")
        if not vulkan_lib:
            vulkan_lib = find_file_with_suffix(ndk, "libvulkan.so", contains=f"/{arch_path}/")
        if not vulkan_lib:
            fail(f"Could not find libvulkan.so in NDK for ABI {abi}")

        extra_args.extend(
            [
                f"-DVulkan_LIBRARY={vulkan_lib}",
                f"-DVulkan_INCLUDE_DIR={THIRD_PARTY_DIR / 'Vulkan-Headers/include'}",
            ]
        )

    if cache_vars["GGML_OPENCL"] == "ON":
        opencl_include, opencl_lib = resolve_android_opencl(abi, ndk, build_dir, env, args.jobs)
        extra_args.extend(
            [
                f"-DOpenCL_INCLUDE_DIR={opencl_include}",
                f"-DOpenCL_LIBRARY={opencl_lib}",
            ]
        )

    built_dir = configure_and_build(preset, jobs=args.jobs, extra_cmake_args=extra_args, env=env)
    out_arch = ANDROID_OUT_ARCH[abi]
    copy_runtime_libraries(built_dir, BIN_DIR / f"android/{out_arch}")


def build_android(args: argparse.Namespace) -> None:
    ensure_submodule(
        THIRD_PARTY_DIR / "llama.cpp/CMakeLists.txt",
        "Missing submodule: third_party/llama.cpp. Run: git submodule update --init --recursive",
    )

    ndk = detect_android_ndk()
    if not ndk:
        fail("ANDROID_NDK_HOME is not set and no NDK installation was detected")

    env = os.environ.copy()
    env["ANDROID_NDK_HOME"] = str(ndk)
    print(f"Using NDK: {ndk}")

    if args.abi == "all":
        abis = ["arm64-v8a", "x86_64"]
    else:
        abis = [ANDROID_ABI_ALIASES[args.abi]]

    for abi in abis:
        print(f"Building Android ABI={abi} backend={args.backend}")
        build_android_abi(abi, args, env)


def build_windows(args: argparse.Namespace) -> None:
    if platform.system().lower() != "windows":
        fail("Windows builds must be run on Windows hosts")
    ensure_submodule(
        THIRD_PARTY_DIR / "llama.cpp/CMakeLists.txt",
        "Missing submodule: third_party/llama.cpp. Run: git submodule update --init --recursive",
    )

    arch = args.arch
    backend = args.backend
    cache_vars = windows_backend_cache_vars(arch, backend)

    if cache_vars["GGML_CUDA"] == "ON" and not (shutil.which("nvcc") or shutil.which("nvcc.exe")):
        fail("Windows CUDA backend build requires CUDA (nvcc not found in PATH)")

    preset = f"windows-{arch}-full"
    clean_build_dir(preset, args.clean)

    extra_args = cmake_cache_args(cache_vars)
    vcpkg_root = detect_vcpkg_root()
    if vcpkg_root and cache_vars["GGML_BLAS"] == "ON":
        toolchain = vcpkg_root / "scripts/buildsystems/vcpkg.cmake"
        if toolchain.is_file():
            extra_args.extend(
                [
                    f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}",
                    f"-DVCPKG_TARGET_TRIPLET={WINDOWS_VCPKG_TRIPLETS[arch]}",
                ]
            )

    sdk = infer_vulkan_sdk(args.vulkan_sdk) if cache_vars["GGML_VULKAN"] == "ON" else None
    if sdk:
        if arch == "x64":
            extra_args.extend(
                [
                    f"-DVulkan_ROOT={sdk.as_posix()}",
                    f"-DVulkan_INCLUDE_DIR={(sdk / 'Include').as_posix()}",
                ]
            )
            vulkan_lib = find_file_with_suffix(sdk, "vulkan-1.lib")
            if vulkan_lib:
                extra_args.append(f"-DVulkan_LIBRARY={vulkan_lib.as_posix()}")
        glslc = find_file_with_suffix(sdk, "glslc.exe")
        if glslc:
            extra_args.append(f"-DVulkan_GLSLC_EXECUTABLE={glslc.as_posix()}")

    build_dir = configure_and_build(preset, jobs=args.jobs, extra_cmake_args=extra_args)
    copy_runtime_libraries(build_dir, BIN_DIR / f"windows/{arch}")


def print_presets() -> None:
    presets = [
        "apple: target=macos-arm64|macos-x86_64|ios-device-arm64|ios-sim-arm64|ios-sim-x86_64 (consolidated: metal+cpu in one dylib)",
        "linux: arch=x64|arm64 backend=full|vulkan|cuda|blas (x64 full=vulkan+cuda+blas+cpu, arm64 full=vulkan+blas+kleidi+cpu)",
        "android: abi=arm64-v8a|x86_64|all backend=full|vulkan|opencl (arm64 full=vulkan+opencl+kleidi+cpu, x86_64 full=vulkan+opencl+cpu)",
        "windows: arch=x64|arm64 backend=full|vulkan|cuda|blas (x64 full=vulkan+cuda+blas+cpu, arm64 full=vulkan+blas+kleidi+cpu)",
    ]
    for p in presets:
        print(p)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clean", action="store_true", help="Delete preset build directory before configure")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel job count passed to cmake --build")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build native llamadart binaries via CMake presets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List supported platform/target combinations")
    list_parser.set_defaults(func=lambda _: print_presets())

    apple = subparsers.add_parser("apple", help="Build Apple targets")
    apple.add_argument("--target", required=True, choices=sorted(APPLE_TARGETS.keys()))
    add_common_args(apple)
    apple.set_defaults(func=build_apple)

    linux = subparsers.add_parser("linux", help="Build Linux shared libraries")
    linux.add_argument("--arch", choices=["x64", "arm64"], default=None)
    linux.add_argument("--backend", choices=LINUX_BACKENDS, default="full")
    add_common_args(linux)
    linux.set_defaults(func=build_linux)

    android = subparsers.add_parser("android", help="Build Android shared libraries")
    android.add_argument("--abi", default="arm64-v8a", choices=["arm64-v8a", "arm64", "x86_64", "x64", "all"])
    android.add_argument("--backend", choices=ANDROID_BACKENDS, default="full")
    add_common_args(android)
    android.set_defaults(func=build_android)

    windows = subparsers.add_parser("windows", help="Build Windows shared libraries")
    windows.add_argument("--arch", choices=["x64", "arm64"], default="x64")
    windows.add_argument("--backend", choices=WINDOWS_BACKENDS, default="full")
    windows.add_argument("--vulkan-sdk", default=None, help="Optional explicit Vulkan SDK root")
    add_common_args(windows)
    windows.set_defaults(func=build_windows)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        fail(f"Command failed with exit code {exc.returncode}")

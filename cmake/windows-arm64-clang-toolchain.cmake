set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

# Use the x64-hosted LLVM compiler from Visual Studio to produce generic
# Windows ARM64 binaries. CPU-specific architecture flags belong in runtime
# dispatched variants, not in this baseline package.
set(_llamadart_arm64_target arm64-pc-windows-msvc)

if(DEFINED ENV{VSINSTALLDIR})
    file(TO_CMAKE_PATH "$ENV{VSINSTALLDIR}" _llamadart_vs_install_dir)
    set(_llamadart_llvm_bin "${_llamadart_vs_install_dir}/VC/Tools/Llvm/x64/bin")
    find_program(
        _llamadart_clang
        NAMES clang.exe clang
        PATHS "${_llamadart_llvm_bin}"
        NO_DEFAULT_PATH
    )
    find_program(
        _llamadart_clangxx
        NAMES clang++.exe clang++
        PATHS "${_llamadart_llvm_bin}"
        NO_DEFAULT_PATH
    )
endif()

if(NOT _llamadart_clang)
    find_program(_llamadart_clang NAMES clang.exe clang REQUIRED)
endif()
if(NOT _llamadart_clangxx)
    find_program(_llamadart_clangxx NAMES clang++.exe clang++ REQUIRED)
endif()

set(CMAKE_C_COMPILER "${_llamadart_clang}")
set(CMAKE_CXX_COMPILER "${_llamadart_clangxx}")
set(CMAKE_C_COMPILER_TARGET "${_llamadart_arm64_target}")
set(CMAKE_CXX_COMPILER_TARGET "${_llamadart_arm64_target}")

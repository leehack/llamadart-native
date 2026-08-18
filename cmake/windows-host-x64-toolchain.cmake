set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

# Build vulkan-shaders-gen as a host x64 executable, even when the parent
# environment is configured for ARM64 cross-compilation.
set(_llamadart_host_target x86_64-pc-windows-msvc)

if(DEFINED ENV{VSINSTALLDIR})
    file(TO_CMAKE_PATH "$ENV{VSINSTALLDIR}" _llamadart_vs_install_dir)
    set(_llamadart_llvm_bin "${_llamadart_vs_install_dir}/VC/Tools/Llvm/x64/bin")
    find_program(
        _llamadart_host_clang
        NAMES clang.exe clang
        PATHS "${_llamadart_llvm_bin}"
        NO_DEFAULT_PATH
    )
    find_program(
        _llamadart_host_clangxx
        NAMES clang++.exe clang++
        PATHS "${_llamadart_llvm_bin}"
        NO_DEFAULT_PATH
    )
endif()

if(NOT _llamadart_host_clang)
    find_program(_llamadart_host_clang NAMES clang.exe clang REQUIRED)
endif()
if(NOT _llamadart_host_clangxx)
    find_program(_llamadart_host_clangxx NAMES clang++.exe clang++ REQUIRED)
endif()

set(CMAKE_C_COMPILER "${_llamadart_host_clang}")
set(CMAKE_CXX_COMPILER "${_llamadart_host_clangxx}")
set(CMAKE_C_COMPILER_TARGET "${_llamadart_host_target}")
set(CMAKE_CXX_COMPILER_TARGET "${_llamadart_host_target}")

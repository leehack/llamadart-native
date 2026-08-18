set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

# Build vulkan-shaders-gen as a host x64 executable, even when the parent
# environment is configured for ARM64 cross-compilation.
set(_llamadart_host_target x86_64-pc-windows-msvc)

if(NOT DEFINED ENV{VCToolsInstallDir}
   OR NOT DEFINED ENV{WindowsSdkDir}
   OR NOT DEFINED ENV{WindowsSDKVersion})
    message(FATAL_ERROR "MSVC and Windows SDK environment is required for the host x64 toolchain")
endif()

file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _llamadart_vc_tools_dir)
file(TO_CMAKE_PATH "$ENV{WindowsSdkDir}" _llamadart_windows_sdk_dir)
file(TO_CMAKE_PATH "$ENV{WindowsSDKVersion}" _llamadart_windows_sdk_version)
string(REGEX REPLACE "/+$" "" _llamadart_windows_sdk_version "${_llamadart_windows_sdk_version}")
set(
    ENV{LIB}
    "${_llamadart_vc_tools_dir}/lib/x64;${_llamadart_windows_sdk_dir}/Lib/${_llamadart_windows_sdk_version}/ucrt/x64;${_llamadart_windows_sdk_dir}/Lib/${_llamadart_windows_sdk_version}/um/x64"
)
set(
    _llamadart_host_library_flags
    "-L\"${_llamadart_vc_tools_dir}/lib/x64\" -L\"${_llamadart_windows_sdk_dir}/Lib/${_llamadart_windows_sdk_version}/ucrt/x64\" -L\"${_llamadart_windows_sdk_dir}/Lib/${_llamadart_windows_sdk_version}/um/x64\""
)
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_llamadart_host_library_flags}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_llamadart_host_library_flags}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_llamadart_host_library_flags}")

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

set(CMAKE_SYSTEM_NAME Windows)

# Build vulkan-shaders-gen as a host (x64) executable, even when target preset is ARM64.
if(DEFINED ENV{VCToolsInstallDir})
    file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _vc_tools_install_dir)
    set(_host_x64_cl "${_vc_tools_install_dir}/bin/Hostx64/x64/cl.exe")
endif()

if(NOT DEFINED _host_x64_cl OR NOT EXISTS "${_host_x64_cl}")
    find_program(_host_x64_cl NAMES cl PATHS ENV PATH)
endif()

if(NOT _host_x64_cl)
    message(FATAL_ERROR "Unable to locate host x64 cl.exe for GGML_VULKAN_SHADERS_GEN_TOOLCHAIN")
endif()

set(CMAKE_C_COMPILER "${_host_x64_cl}")
set(CMAKE_CXX_COMPILER "${_host_x64_cl}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)

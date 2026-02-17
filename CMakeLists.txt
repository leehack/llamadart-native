cmake_minimum_required(VERSION 3.14)
project(llamadart_native)

find_package(Threads REQUIRED)

# Keep all targets PIC-safe for shared linking.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apple targets use a single consolidated dylib by default.
if (APPLE)
    set(LLAMADART_CONSOLIDATE ON CACHE BOOL "Consolidate llama.cpp + ggml into libllamadart on Apple")
else()
    set(LLAMADART_CONSOLIDATE OFF CACHE BOOL "Consolidate llama.cpp + ggml into libllamadart on Apple")
endif()

if (APPLE AND LLAMADART_CONSOLIDATE)
    # Consolidated Apple strategy: static deps + one shared wrapper dylib.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "Build ggml backends as dynamic libraries" FORCE)
else()
    # Non-Apple strategy: shared core + dynamic backend modules.
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "Build ggml backends as dynamic libraries" FORCE)
endif()

# Add llama.cpp as a subdirectory
set(LLAMA_BUILD_NUMBER 0 CACHE STRING "Llama build number" FORCE)
set(LLAMA_INSTALL_VERSION 0.0.0 CACHE STRING "Llama install version" FORCE)

# Common llama.cpp options
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Build llama.cpp common library" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build llama.cpp tools" FORCE)
set(LLAMA_HTTPLIB OFF CACHE BOOL "Disable httplib" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "Disable OpenSSL" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Disable native CPU optimizations for portability" FORCE)

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "Missing submodule: third_party/llama.cpp. Run: git submodule update --init --recursive")
endif()

# Set defaults for Release builds
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if (APPLE)
    # Default Metal options for macOS/iOS (allow presets/CLI overrides)
    if (NOT DEFINED GGML_BLAS)
        set(GGML_BLAS OFF CACHE BOOL "Disable BLAS backend on Apple by default")
    endif()
    if (NOT DEFINED GGML_CPU_KLEIDIAI)
        set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "Disable KleidiAI on Apple by default")
    endif()
    if (NOT DEFINED GGML_METAL)
        set(GGML_METAL ON CACHE BOOL "Enable Metal")
    endif()
    if (NOT DEFINED GGML_METAL_EMBED_LIBRARY)
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal library")
    endif()
    if (NOT DEFINED GGML_METAL_USE_BF16)
        set(GGML_METAL_USE_BF16 OFF CACHE BOOL "Use BF16 in Metal")
    endif()
endif()

add_subdirectory(third_party/llama.cpp/tools/mtmd EXCLUDE_FROM_ALL)
add_subdirectory(third_party/llama.cpp)

add_library(llamadart_lib SHARED "src/llama_dart_wrapper.cpp")

if (MINGW)
    target_link_options(llamadart_lib PRIVATE -Wl,--export-all-symbols)
endif()

if (APPLE AND LLAMADART_CONSOLIDATE)
    # Ensure static libs are fully embedded into libllamadart.dylib.
    target_link_libraries(llamadart_lib PUBLIC -Wl,-force_load llama -Wl,-force_load mtmd)
else()
    # Thin wrapper library; llama.cpp / ggml remain separate dynamic libraries.
    target_link_libraries(llamadart_lib PUBLIC llama mtmd)
endif()

set_target_properties(llamadart_lib PROPERTIES
    OUTPUT_NAME llamadart
    LIBRARY_OUTPUT_NAME llamadart
)

target_include_directories(llamadart_lib PRIVATE
    third_party/llama.cpp/include
    third_party/llama.cpp/src
    src
)

cmake_minimum_required(VERSION 3.14)
project(llamadart_native)

find_package(Threads REQUIRED)

option(LLAMADART_BUILD_TESTS "Build libllamadart wrapper tests" OFF)
option(LLAMADART_BUILD_TTS_SMOKE "Build the local Qwen3-TTS smoke tool" OFF)
option(LLAMADART_BUILD_KLEIDIAI_TESTS "Build standalone KleidiAI dispatch and compute qualification" OFF)

# Keep all targets PIC-safe for shared linking.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apple targets use a single consolidated dylib by default.
if (APPLE)
    set(LLAMADART_CONSOLIDATE ON CACHE BOOL "Consolidate llama.cpp + ggml into libllamadart on Apple")
else()
    set(LLAMADART_CONSOLIDATE OFF CACHE BOOL "Consolidate llama.cpp + ggml into libllamadart on Apple")
endif()

set(LLAMADART_CPU_REQUIRED_FEATURES "" CACHE STRING
    "Semicolon-separated GGML_USE_* defines for Android arm64 CPU backend scoring")
set(LLAMADART_MSVC_LLAMA_COMMON_IPO OFF CACHE BOOL
    "Enable MSVC IPO/LTCG for llama-common")
set(LLAMADART_MSVC_MTMD_IPO OFF CACHE BOOL
    "Enable MSVC IPO/LTCG for mtmd")

function(llamadart_disable_ipo)
    foreach(target_name ${ARGN})
        if (TARGET ${target_name})
            set_target_properties(${target_name} PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION OFF
                INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF
                INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL OFF
            )
        endif()
    endforeach()
endfunction()

function(llamadart_attach_android_cpu_backend_score target_name)
    if (NOT ANDROID OR NOT TARGET ${target_name} OR NOT GGML_CPU_ARM_ARCH)
        return()
    endif()

    set(score_target ${target_name}-llamadart-feats)
    add_library(${score_target} OBJECT third_party/llama.cpp/ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp)
    target_include_directories(${score_target} PRIVATE
        third_party/llama.cpp/ggml/src
        third_party/llama.cpp/ggml/include
    )
    target_compile_definitions(${score_target} PRIVATE
        GGML_BACKEND_DL
        GGML_BACKEND_BUILD
        GGML_BACKEND_SHARED
        ${LLAMADART_CPU_REQUIRED_FEATURES}
    )
    set_target_properties(${score_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${score_target} PRIVATE -fno-lto)
    target_sources(${target_name} PRIVATE $<TARGET_OBJECTS:${score_target}>)
endfunction()

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
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common library" FORCE)
# Let llama.cpp add mtmd from its own directory scope so its version variables
# resolve before CMake assigns the shared-library VERSION and SOVERSION.
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build llama.cpp mtmd library" FORCE)
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

if (GGML_VULKAN)
    set(LLAMADART_SPIRV_HEADERS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/SPIRV-Headers/include")
    if (NOT EXISTS "${LLAMADART_SPIRV_HEADERS_INCLUDE}/spirv/unified1/spirv.hpp")
        message(FATAL_ERROR "Missing submodule: third_party/SPIRV-Headers. Run: git submodule update --init --recursive")
    endif()

    # Newer llama.cpp Vulkan builds require find_package(SPIRV-Headers CONFIG).
    # Provide a tiny package config backed by the vendored submodule so release
    # builds do not depend on host-specific SPIR-V header packages.
    set(LLAMADART_SPIRV_HEADERS_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/cmake/SPIRV-Headers")
    file(MAKE_DIRECTORY "${LLAMADART_SPIRV_HEADERS_CONFIG_DIR}")
    file(TO_CMAKE_PATH "${LLAMADART_SPIRV_HEADERS_INCLUDE}" LLAMADART_SPIRV_HEADERS_INCLUDE_CMAKE)
    file(WRITE "${LLAMADART_SPIRV_HEADERS_CONFIG_DIR}/SPIRV-HeadersConfig.cmake"
        "if (NOT TARGET SPIRV-Headers::SPIRV-Headers)\n"
        "  add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)\n"
        "  set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES\n"
        "    INTERFACE_INCLUDE_DIRECTORIES \"${LLAMADART_SPIRV_HEADERS_INCLUDE_CMAKE}\"\n"
        "  )\n"
        "endif()\n"
        "set(SPIRV-Headers_FOUND TRUE)\n"
    )
    set(SPIRV-Headers_DIR "${LLAMADART_SPIRV_HEADERS_CONFIG_DIR}" CACHE PATH "Vendored SPIRV-Headers package config" FORCE)
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

add_subdirectory(third_party/llama.cpp)
include(cmake/kleidiai_windows.cmake)
llamadart_configure_kleidiai_windows_assembly()

# Probe declarations without calling upstream symbols or running target code.
# Recheck after switching upstream revisions in an existing build.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd")
unset(LLAMADART_MTMD_HELPER_HAS_OPTIONS CACHE)
check_cxx_source_compiles("#include <mtmd-helper.h>
int main() { return sizeof(decltype(mtmd_helper_init_opt_default())) == 0; }"
    LLAMADART_MTMD_HELPER_HAS_OPTIONS)
cmake_pop_check_state()
if (TARGET ggml-vulkan AND LLAMADART_SPIRV_HEADERS_INCLUDE)
    target_include_directories(ggml-vulkan SYSTEM PRIVATE "${LLAMADART_SPIRV_HEADERS_INCLUDE}")
endif()

if (MSVC AND NOT LLAMADART_MSVC_LLAMA_COMMON_IPO)
    # MSVC link.exe can access-violate while linking llama-common.dll with
    # LTCG. Keep IPO elsewhere, but compile this utility stack without LTCG.
    llamadart_disable_ipo(llama-common llama-common-base cpp-httplib)
endif()

if (MSVC AND NOT LLAMADART_MSVC_MTMD_IPO)
    # CMake's WINDOWS_EXPORT_ALL_SYMBOLS scanner cannot read MSVC LTCG object
    # files when generating mtmd.dll's exports. Keep IPO enabled globally, but
    # emit ordinary COFF objects for this shared library.
    llamadart_disable_ipo(mtmd)
endif()

llamadart_attach_android_cpu_backend_score(ggml-cpu)

add_library(llamadart_lib SHARED "src/llama_dart_wrapper.cpp")
target_compile_definitions(llamadart_lib PRIVATE
    LLAMADART_MTMD_HELPER_HAS_OPTIONS=$<BOOL:${LLAMADART_MTMD_HELPER_HAS_OPTIONS}>)

if (MINGW)
    target_link_options(llamadart_lib PRIVATE -Wl,--export-all-symbols)
endif()

if (APPLE AND LLAMADART_CONSOLIDATE)
    # Ensure static libs are fully embedded into libllamadart.dylib.
    target_link_libraries(llamadart_lib PUBLIC
        -Wl,-force_load llama
        -Wl,-force_load mtmd
        -Wl,-force_load llama-common)
else()
    # Thin wrapper library; llama.cpp / ggml remain separate dynamic libraries.
    target_link_libraries(llamadart_lib PUBLIC llama mtmd llama-common)
endif()

target_compile_features(llamadart_lib PRIVATE cxx_std_17)

set_target_properties(llamadart_lib PROPERTIES
    OUTPUT_NAME llamadart
    LIBRARY_OUTPUT_NAME llamadart
)

target_include_directories(llamadart_lib PRIVATE
    third_party/llama.cpp/include
    third_party/llama.cpp/src
    third_party/llama.cpp/common
    src
)

if (LLAMADART_BUILD_TESTS)
    enable_testing()
    add_executable(llamadart_speculative_api_test tests/speculative_api_test.cpp)
    target_compile_features(llamadart_speculative_api_test PRIVATE cxx_std_17)
    target_link_libraries(llamadart_speculative_api_test PRIVATE llamadart_lib)
    target_include_directories(llamadart_speculative_api_test PRIVATE
        third_party/llama.cpp/common
        src
    )
    add_test(NAME llamadart_speculative_api_test COMMAND llamadart_speculative_api_test)

    add_executable(llamadart_tts_api_test tests/tts_api_test.c)
    target_link_libraries(llamadart_tts_api_test PRIVATE llamadart_lib)
    target_include_directories(llamadart_tts_api_test PRIVATE src)
    add_test(NAME llamadart_tts_api_test COMMAND llamadart_tts_api_test)

    add_executable(llamadart_mtmd_compat_test tests/mtmd_compat_test.cpp)
    target_compile_features(llamadart_mtmd_compat_test PRIVATE cxx_std_17)
    target_include_directories(llamadart_mtmd_compat_test PRIVATE src)
    target_compile_definitions(llamadart_mtmd_compat_test PRIVATE
        LLAMADART_MTMD_HELPER_HAS_OPTIONS=$<BOOL:${LLAMADART_MTMD_HELPER_HAS_OPTIONS}>)
    target_link_libraries(llamadart_mtmd_compat_test PRIVATE mtmd)
    add_test(NAME llamadart_mtmd_compat_test COMMAND llamadart_mtmd_compat_test)
endif()

if (LLAMADART_BUILD_KLEIDIAI_TESTS)
    if (NOT TARGET kleidiai OR NOT TARGET ggml-cpu)
        message(FATAL_ERROR "KleidiAI qualification requires upstream standalone kleidiai and ggml-cpu targets")
    endif()
    enable_testing()
    add_executable(llamadart_kleidiai_dispatch_test tests/kleidiai_dispatch_test.cpp)
    target_compile_features(llamadart_kleidiai_dispatch_test PRIVATE cxx_std_17)
    target_include_directories(llamadart_kleidiai_dispatch_test PRIVATE
        third_party/llama.cpp/ggml/include
        third_party/llama.cpp/ggml/src
        third_party/llama.cpp/ggml/src/ggml-cpu/kleidiai)
    set_target_properties(llamadart_kleidiai_dispatch_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    # ggml-cpu is a loadable MODULE on Android. Link its actual file to test
    # the shipped selectors, not a second test-only copy of their source.
    target_link_libraries(llamadart_kleidiai_dispatch_test PRIVATE
        "$<TARGET_FILE:ggml-cpu>" ggml ggml-base)
    add_dependencies(llamadart_kleidiai_dispatch_test ggml-cpu)
    add_test(NAME llamadart_kleidiai_dispatch_test COMMAND llamadart_kleidiai_dispatch_test)
endif()

if (LLAMADART_BUILD_TTS_SMOKE)
    add_executable(llamadart_tts_smoke tools/tts_smoke.cpp)
    target_compile_features(llamadart_tts_smoke PRIVATE cxx_std_17)
    target_link_libraries(llamadart_tts_smoke PRIVATE llamadart_lib)
    target_include_directories(llamadart_tts_smoke PRIVATE
        third_party/llama.cpp/include
        third_party/llama.cpp/tools/mtmd
        src
    )
endif()

cmake_minimum_required(VERSION 3.22)
project(kleidiai_windows_metadata NONE)

# These tests check CMake property scope and branch behavior on every host.
# Only a real Visual Studio build can validate the generated assembly pipeline.
include("${HELPER}")
if (NOT CASE STREQUAL "missing_target")
    add_subdirectory(kleidiai)
endif()
set(MSVC TRUE)
set(CMAKE_C_COMPILER_ID Clang)
set(CMAKE_GENERATOR "Visual Studio 17 2022")
if (CASE STREQUAL "not_msvc")
    set(MSVC FALSE)
elseif (CASE STREQUAL "not_clang")
    set(CMAKE_C_COMPILER_ID MSVC)
elseif (CASE STREQUAL "not_visual_studio")
    set(CMAKE_GENERATOR Ninja)
endif()

llamadart_configure_kleidiai_windows_assembly()
if (TARGET kleidiai)
    foreach(source kernel.S ordinary.c generic.S)
        get_source_file_property(settings "${CMAKE_CURRENT_SOURCE_DIR}/kleidiai/${source}"
            TARGET_DIRECTORY kleidiai VS_SETTINGS)
        set(expected "ExistingMetadata=retained")
        if (CASE STREQUAL "enabled" AND source STREQUAL "kernel.S")
            list(APPEND expected "PreprocessSuppressLineNumbers=true")
        endif()
        if (NOT settings STREQUAL expected)
            message(FATAL_ERROR "${source}: expected '${expected}', got '${settings}'")
        endif()
    endforeach()
    get_target_property(sources kleidiai SOURCES)
    if (NOT sources STREQUAL "kernel.S;ordinary.c;generic.S")
        message(FATAL_ERROR "Kernel source list changed: ${sources}")
    endif()
endif()

# Stop before generation: the fixture intentionally models a Windows compiler
# and source language without requiring a Windows toolchain on this host.
message(FATAL_ERROR "METADATA_CONTRACT_PASSED")

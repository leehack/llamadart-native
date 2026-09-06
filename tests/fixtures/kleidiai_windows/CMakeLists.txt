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
            list(APPEND expected
                "UndefinePreprocessorDefinitions=__clang__\;%(UndefinePreprocessorDefinitions)")
        endif()
        if (NOT settings STREQUAL expected)
            message(FATAL_ERROR "${source}: expected '${expected}', got '${settings}'")
        endif()
    endforeach()
    get_target_property(sources kleidiai SOURCES)
    if (CASE STREQUAL "enabled")
        list(SUBLIST sources 0 3 original_sources)
        list(SUBLIST sources 3 -1 added_sources)
        set(expected_names
            kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
            kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c
            kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c
            kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c
            kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c
            kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c
            kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
            kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c
            kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c
            kai_lhs_pack_bf16p2vlx2_f32_sme.c
            kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c)
        foreach(source IN LISTS added_sources)
            get_filename_component(name "${source}" NAME)
            list(APPEND names "${name}")
            get_source_file_property(options "${source}"
                TARGET_DIRECTORY kleidiai COMPILE_OPTIONS)
            if (name MATCHES "neon_dotprod")
                set(expected_options /clang:-march=armv8.2-a+dotprod)
            elseif (name MATCHES "neon_i8mm")
                set(expected_options /clang:-march=armv8.2-a+i8mm)
            else()
                set(expected_options /clang:-march=armv8.2-a+sve+sve2
                    /clang:-fno-tree-vectorize /clang:-fno-tree-slp-vectorize)
            endif()
            if (NOT options STREQUAL expected_options)
                message(FATAL_ERROR "Incorrect source ISA options for ${name}: ${options}")
            endif()
        endforeach()
        if (NOT names STREQUAL expected_names)
            message(FATAL_ERROR "Unexpected kernel inventory: ${names}")
        endif()
        llamadart_add_kleidiai_clangcl_kernels()
        get_target_property(repeated_sources kleidiai SOURCES)
        if (NOT sources STREQUAL repeated_sources)
            message(FATAL_ERROR "Duplicate kernels added on repeated configuration")
        endif()
    else()
        set(original_sources "${sources}")
    endif()
    if (NOT original_sources STREQUAL "kernel.S;ordinary.c;generic.S")
        message(FATAL_ERROR "Original kernel source list changed: ${original_sources}")
    endif()
endif()

# Stop before generation: the fixture intentionally models a Windows compiler
# and source language without requiring a Windows toolchain on this host.
message(FATAL_ERROR "METADATA_CONTRACT_PASSED")

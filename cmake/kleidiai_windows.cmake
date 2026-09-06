# ClangCL's preprocessor emits GNU-style line markers, which armasm64 rejects
# with A2230. Some kernels also select GNU assembly syntax when __clang__ is
# defined, even though MSBuild invokes armasm64, not Clang's assembler. Configure
# only the MARMASM preprocessing task: omit line markers and select upstream's
# existing MSVC assembly dialect. C/C++ compilation remains unchanged.
function(llamadart_configure_kleidiai_windows_assembly)
    if (NOT MSVC OR NOT CMAKE_C_COMPILER_ID STREQUAL "Clang" OR
        NOT CMAKE_GENERATOR MATCHES "^Visual Studio" OR NOT TARGET kleidiai)
        return()
    endif()

    # VS_SETTINGS applies to known source types starting with CMake 3.22.
    if (CMAKE_VERSION VERSION_LESS 3.22)
        message(FATAL_ERROR "ClangCL KleidiAI assembly requires CMake >= 3.22")
    endif()

    get_target_property(kleidiai_source_dir kleidiai SOURCE_DIR)
    get_target_property(kleidiai_sources kleidiai SOURCES)
    foreach(source IN LISTS kleidiai_sources)
        if (NOT IS_ABSOLUTE "${source}")
            set(source "${kleidiai_source_dir}/${source}")
        endif()
        get_source_file_property(language "${source}"
            TARGET_DIRECTORY kleidiai LANGUAGE)
        if (language STREQUAL "ASM_MARMASM")
            set_property(SOURCE "${source}" TARGET_DIRECTORY kleidiai APPEND
                PROPERTY VS_SETTINGS "PreprocessSuppressLineNumbers=true"
                "UndefinePreprocessorDefinitions=__clang__\;%(UndefinePreprocessorDefinitions)")
        endif()
    endforeach()

    llamadart_add_kleidiai_clangcl_kernels()
endfunction()

# KleidiAI 1.24's MSVC list excludes GNU inline-assembly C kernels that ClangCL
# supports and ggml's runtime-dispatched kernel table references. Restore only
# those exact kernels, with the same per-source ISA and SME vectorization policy
# as KleidiAI's non-MSVC build. Never raise the whole library's baseline ISA.
function(llamadart_add_kleidiai_clangcl_kernels)
    get_target_property(source_dir kleidiai SOURCE_DIR)
    set(dotprod_sources
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c
        kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c
        kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c
        kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c)
    set(i8mm_sources
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
        kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c)
    set(sme_sources
        kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c
        kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.c
        kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c)
    foreach(family dotprod i8mm sme)
        if (family STREQUAL "sme")
            set(options /clang:-march=armv8.2-a+sve+sve2
                /clang:-fno-tree-vectorize /clang:-fno-tree-slp-vectorize)
        else()
            set(options "/clang:-march=armv8.2-a+${family}")
        endif()
        foreach(source IN LISTS ${family}_sources)
            get_target_property(existing_sources kleidiai SOURCES)
            set(absolute_source "${source_dir}/${source}")
            if (NOT source IN_LIST existing_sources AND
                NOT absolute_source IN_LIST existing_sources)
                target_sources(kleidiai PRIVATE "${absolute_source}")
            endif()
            set_property(SOURCE "${absolute_source}" TARGET_DIRECTORY kleidiai
                PROPERTY COMPILE_OPTIONS "${options}")
        endforeach()
    endforeach()
endfunction()

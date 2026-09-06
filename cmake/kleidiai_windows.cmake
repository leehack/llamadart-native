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
endfunction()

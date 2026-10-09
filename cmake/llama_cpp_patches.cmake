# Carried llama.cpp patches (patches/llama.cpp/README.md). The submodule stays
# as upstream published it: each patched source is written to the build tree
# and the upstream target that compiles the original compiles the copy instead.

function(llamadart_collect_targets directory out_var)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        llamadart_collect_targets("${subdirectory}" nested)
        list(APPEND targets ${nested})
    endforeach()
    set(${out_var} "${targets}" PARENT_SCOPE)
endfunction()

function(llamadart_apply_llama_cpp_patches)
    set(patches_dir "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama.cpp")
    set(upstream_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp-patched")

    file(GLOB patch_files "${patches_dir}/*.patch")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${patches_dir}/series.json" ${patch_files}
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/llama_cpp_patches.py")

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    # A patch that does not fit this llama.cpp is skipped: lanes here build
    # other upstream commits than the pinned one. What keeps a release from
    # losing a patch is tools/validate_android_artifacts.py on the bundle.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/llama_cpp_patches.py"
            --patches "${patches_dir}"
            apply --upstream "${upstream_dir}" --output "${output_dir}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE patched_sources
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "Could not apply patches/llama.cpp:\n${error}")
    endif()
    if (error)
        message(WARNING "patches/llama.cpp does not fit this llama.cpp:\n${error}")
    endif()
    if (NOT patched_sources)
        return()
    endif()

    llamadart_collect_targets("${upstream_dir}" upstream_targets)
    foreach(relative IN LISTS patched_sources)
        set(original "${upstream_dir}/${relative}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${original}")
        get_filename_component(original_dir "${original}" DIRECTORY)
        foreach(target IN LISTS upstream_targets)
            get_target_property(type ${target} TYPE)
            if (type STREQUAL "INTERFACE_LIBRARY")
                continue()
            endif()
            get_target_property(sources ${target} SOURCES)
            get_target_property(source_dir ${target} SOURCE_DIR)
            if (NOT sources)
                continue()
            endif()
            set(replaced OFF)
            set(new_sources "")
            foreach(source IN LISTS sources)
                if (NOT source MATCHES "^\\$<")
                    get_filename_component(absolute "${source}" ABSOLUTE BASE_DIR "${source_dir}")
                    if (absolute STREQUAL original)
                        set(source "${output_dir}/${relative}")
                        set(replaced ON)
                    endif()
                endif()
                list(APPEND new_sources "${source}")
            endforeach()
            if (replaced)
                set_property(TARGET ${target} PROPERTY SOURCES "${new_sources}")
                # The copy includes its neighbours by name, which resolved
                # next to the original.
                target_include_directories(${target} BEFORE PRIVATE "${original_dir}")
                message(STATUS "llama.cpp patches: ${target} compiles patched ${relative}")
            endif()
        endforeach()
    endforeach()
endfunction()

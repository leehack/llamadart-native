# The retention shim must be a native object: GCC LTO cannot be consumed by
# the LLVM linker used for HIP backends, which otherwise drops its definition.
add_library(llamadart_static_destructors OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/../src/llama_dart_static_destructors.c")
set_target_properties(llamadart_static_destructors PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF)

function(llamadart_drop_static_destructors directory)
    get_property(target_names DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target_name IN LISTS target_names)
        get_target_property(target_type ${target_name} TYPE)
        if (target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "MODULE_LIBRARY")
            target_sources(${target_name} PRIVATE
                $<TARGET_OBJECTS:llamadart_static_destructors>)
        endif()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        llamadart_drop_static_destructors("${subdirectory}")
    endforeach()
endfunction()

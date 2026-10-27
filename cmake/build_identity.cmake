# The identity libllamadart reports (docs/build_identity.md). The git part is
# stamped on every build, so a commit made after configuring still changes it.

function(llamadart_add_build_identity target)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    llamadart_collect_targets(
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml" ggml_targets)
    set(backends "")
    foreach(ggml_target IN LISTS ggml_targets)
        if (NOT ggml_target MATCHES "^ggml-(.+)$" OR ggml_target STREQUAL "ggml-base")
            continue()
        endif()
        set(backend "${CMAKE_MATCH_1}")
        get_target_property(type ${ggml_target} TYPE)
        # ggml-cpu's feature probe is an OBJECT library, not a backend.
        if (type MATCHES "^(MODULE|SHARED|STATIC)_LIBRARY$")
            list(APPEND backends "${backend}")
        endif()
    endforeach()
    list(SORT backends)
    list(JOIN backends "," backends)

    # Every ggml switch, not only the backend ones: some change the kernels.
    get_cmake_property(cache_variables CACHE_VARIABLES)
    set(options "LLAMADART_CPU_REQUIRED_FEATURES=${LLAMADART_CPU_REQUIRED_FEATURES}")
    foreach(name IN LISTS cache_variables)
        if (NOT name MATCHES "^GGML_")
            continue()
        endif()
        get_property(type CACHE ${name} PROPERTY TYPE)
        if (type STREQUAL "BOOL")
            if (${name})
                list(APPEND options "${name}=ON")
            else()
                list(APPEND options "${name}=OFF")
            endif()
        endif()
    endforeach()
    list(SORT options)
    list(JOIN options "\n" options)
    string(SHA256 options "${options}")
    string(SUBSTRING "${options}" 0 16 options)

    set(compiler
        "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION},${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION}")

    set(target_parts "${CMAKE_SYSTEM_NAME}" "${CMAKE_SYSTEM_PROCESSOR}")
    if (APPLE)
        string(REPLACE ";" "+" architectures "${CMAKE_OSX_ARCHITECTURES}")
        list(APPEND target_parts "${architectures}")
        # The SDK's platform, not where it is installed.
        get_filename_component(sysroot "${CMAKE_OSX_SYSROOT}" NAME)
        string(TOLOWER "${sysroot}" sysroot)
        if (sysroot MATCHES "^([a-z]+)")
            list(APPEND target_parts "${CMAKE_MATCH_1}")
        endif()
    endif()
    if (ANDROID)
        list(APPEND target_parts "${CMAKE_ANDROID_ARCH_ABI}")
    endif()
    list(APPEND target_parts
        "${CMAKE_CXX_COMPILER_TARGET}" "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
    list(FILTER target_parts EXCLUDE REGEX "^$")
    list(JOIN target_parts "-" target_value)

    set(header_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(header "${header_dir}/llama_dart_build_identity.h")
    add_custom_target(llamadart_build_identity
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_identity.py"
            --source "${CMAKE_CURRENT_SOURCE_DIR}"
            --output "${header}"
            --config "$<CONFIG>"
            --compiler "${compiler}"
            --target "${target_value}"
            --backends "${backends}"
            --options "${options}"
        BYPRODUCTS "${header}"
        COMMENT "Stamping the libllamadart build identity"
        VERBATIM)
    add_dependencies(${target} llamadart_build_identity)
    target_include_directories(${target} PRIVATE "${header_dir}")
    set(LLAMADART_BUILD_IDENTITY_INCLUDE "${header_dir}" PARENT_SCOPE)
endfunction()

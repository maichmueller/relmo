cmake_minimum_required(VERSION 3.24)
project(relmo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
endif ()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(RELM_MP_OPS "AUTO" CACHE STRING "Build non-stable ABI mp custom ops: AUTO, ON, OFF.")
set_property(CACHE RELM_MP_OPS PROPERTY STRINGS AUTO ON OFF)
set(RELM_RPATH_MODE "dev" CACHE STRING "Runtime path mode: dev or wheel")
set_property(CACHE RELM_RPATH_MODE PROPERTY STRINGS dev wheel)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

set(_relm_mp_enabled FALSE)
set(_relm_torch_version "")
set(_relm_torch_cuda_raw "cpu")
set(_relm_torch_cuda_tag "cpu")
set(_relm_torch_abi_target "unknown")
set(_relm_torch_link_target "")
set(_relm_pkg_dir "${CMAKE_SOURCE_DIR}/src/relmo")
set(_relm_legacy_mp_artifact "${_relm_pkg_dir}/relm_relmp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}")
string(TOUPPER "${RELM_MP_OPS}" _relm_mp_mode)

set(_relm_mp_modes AUTO ON OFF)
list(FIND _relm_mp_modes "${_relm_mp_mode}" _relm_mp_mode_idx)
if (_relm_mp_mode_idx EQUAL -1)
    message(FATAL_ERROR "Invalid RELM_MP_OPS=${RELM_MP_OPS}. Use AUTO, ON, or OFF.")
endif ()

if (NOT _relm_mp_mode STREQUAL "OFF")
    execute_process(
            COMMAND "${Python_EXECUTABLE}" "-c" "import torch; print(torch.utils.cmake_prefix_path); print(torch.__version__); print(torch.version.cuda or 'cpu')"
            RESULT_VARIABLE _relm_torch_probe_result
            OUTPUT_VARIABLE _relm_torch_probe_output
            ERROR_VARIABLE _relm_torch_probe_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (_relm_torch_probe_result EQUAL 0)
        string(REPLACE "\r\n" "\n" _relm_torch_probe_output "${_relm_torch_probe_output}")
        string(REPLACE "\n" ";" _relm_torch_probe_lines "${_relm_torch_probe_output}")
        list(LENGTH _relm_torch_probe_lines _relm_torch_probe_len)
        if (_relm_torch_probe_len GREATER 0)
            list(GET _relm_torch_probe_lines 0 _relm_torch_prefix_path)
        endif ()
        if (_relm_torch_probe_len GREATER 1)
            list(GET _relm_torch_probe_lines 1 _relm_torch_version)
        endif ()
        if (_relm_torch_probe_len GREATER 2)
            list(GET _relm_torch_probe_lines 2 _relm_torch_cuda_raw)
        endif ()

        if (_relm_torch_prefix_path AND EXISTS "${_relm_torch_prefix_path}")
            list(PREPEND CMAKE_PREFIX_PATH "${_relm_torch_prefix_path}")
        endif ()

        find_package(Torch CONFIG QUIET)
        if (Torch_FOUND)
            if (TARGET Torch::Torch)
                set(_relm_torch_link_target Torch::Torch)
            elseif (TARGET torch)
                set(_relm_torch_link_target torch)
            endif ()
            if (NOT _relm_torch_link_target STREQUAL "" OR TORCH_LIBRARIES)
                set(_relm_mp_enabled TRUE)
            endif ()
        endif ()
    endif ()

    if (_relm_mp_enabled)
        if (_relm_torch_cuda_raw STREQUAL "" OR _relm_torch_cuda_raw STREQUAL "cpu" OR _relm_torch_cuda_raw STREQUAL "None")
            set(_relm_torch_cuda_tag "cpu")
        else ()
            string(REPLACE "." "" _relm_torch_cuda_digits "${_relm_torch_cuda_raw}")
            set(_relm_torch_cuda_tag "cu${_relm_torch_cuda_digits}")
        endif ()

        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _relm_torch_abi_target "${_relm_torch_version}")
        if (_relm_torch_abi_target STREQUAL "")
            set(_relm_torch_abi_target "unknown")
        endif ()
        message(STATUS "RELM mp ops enabled (torch=${_relm_torch_version}, cuda_tag=${_relm_torch_cuda_tag}, abi=nonstable).")
    elseif (_relm_mp_mode STREQUAL "ON")
        message(FATAL_ERROR "RELM_MP_OPS=ON but pip torch could not be resolved for CMake.")
    else ()
        message(STATUS "RELM mp ops disabled (AUTO): pip torch probe failed or Torch CMake package unavailable.")
        if (NOT _relm_torch_probe_result EQUAL 0 AND NOT _relm_torch_probe_error STREQUAL "")
            message(STATUS "Torch probe error: ${_relm_torch_probe_error}")
        endif ()
    endif ()
endif ()

if (_relm_mp_enabled)
    set(_relm_mp_sources "_ops/mp/mp_ops.cpp")
    set(_relm_mp_cuda_enabled FALSE)

    if (NOT _relm_torch_cuda_tag STREQUAL "cpu")
        if (CMAKE_CUDA_COMPILER)
            list(APPEND _relm_mp_sources "_ops/mp/mp_cuda_kernels.cu")
            set(_relm_mp_cuda_enabled TRUE)
            message(STATUS "RELM mp CUDA kernels enabled.")
        else ()
            message(WARNING "RELM mp CUDA kernels disabled: torch CUDA detected but CMAKE_CUDA_COMPILER unavailable.")
        endif ()
    endif ()

    add_library(relm_mp_ops SHARED ${_relm_mp_sources})
    set_target_properties(relm_mp_ops PROPERTIES OUTPUT_NAME "relm_mp_ops" PREFIX "" POSITION_INDEPENDENT_CODE ON)

    if (TORCH_INCLUDE_DIRS)
        target_include_directories(relm_mp_ops PRIVATE ${TORCH_INCLUDE_DIRS})
    endif ()

    if (NOT _relm_torch_link_target STREQUAL "")
        target_link_libraries(relm_mp_ops PRIVATE "${_relm_torch_link_target}")
    elseif (TORCH_LIBRARIES)
        target_link_libraries(relm_mp_ops PRIVATE ${TORCH_LIBRARIES})
    else ()
        message(FATAL_ERROR "Torch was found but no link libraries were exposed for relm_mp_ops.")
    endif ()

    if (TORCH_CXX_FLAGS)
        separate_arguments(_relm_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(relm_mp_ops PRIVATE ${_relm_torch_cxx_flags})
        unset(_relm_torch_cxx_flags)
    endif ()

    target_compile_definitions(
            relm_mp_ops
            PRIVATE
            RELM_MP_BUILD_TORCH_VERSION="${_relm_torch_version}"
            RELM_MP_BUILD_CUDA_TAG="${_relm_torch_cuda_tag}"
            RELM_MP_ABI_TARGET="${_relm_torch_abi_target}"
            RELM_MP_ABI_LANE="nonstable"
    )

    if (_relm_mp_cuda_enabled)
        set_source_files_properties("_ops/mp/mp_cuda_kernels.cu" PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(relm_mp_ops PRIVATE RELM_MP_HAS_CUDA=1 USE_CUDA)
    else ()
        target_compile_definitions(relm_mp_ops PRIVATE RELM_MP_HAS_CUDA=0)
    endif ()

    if (APPLE)
        set(_relm_rpath "@loader_path")
    elseif (UNIX)
        set(_relm_rpath "$ORIGIN")
    endif ()
    set_target_properties(relm_mp_ops PROPERTIES BUILD_RPATH "${_relm_rpath}" INSTALL_RPATH "${_relm_rpath}" INSTALL_RPATH_USE_LINK_PATH FALSE)

    if (RELM_RPATH_MODE STREQUAL "dev")
        file(REMOVE "${_relm_legacy_mp_artifact}")
        set_target_properties(relm_mp_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_relm_pkg_dir}")
        add_custom_command(
                TARGET relm_mp_ops POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E rm -f "${_relm_legacy_mp_artifact}"
        )
    endif ()

    install(TARGETS relm_mp_ops LIBRARY DESTINATION "relmo")
    install(CODE "file(REMOVE \"\${CMAKE_INSTALL_PREFIX}/relmo/relm_relmp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}\")")
else ()
    install(CODE "file(REMOVE \"\${CMAKE_INSTALL_PREFIX}/relmo/relm_mp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}\")")
    install(CODE "file(REMOVE \"\${CMAKE_INSTALL_PREFIX}/relmo/relm_relmp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}\")")
    if (RELM_RPATH_MODE STREQUAL "dev")
        file(REMOVE "${_relm_pkg_dir}/relm_mp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}")
        file(REMOVE "${_relm_legacy_mp_artifact}")
    endif ()
endif ()

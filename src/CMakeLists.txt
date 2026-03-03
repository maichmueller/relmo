cmake_minimum_required(VERSION 3.24)
project(relm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
endif ()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(RELM_RELMP_OPS "AUTO" CACHE STRING "Build stable-ABI relmp custom ops: AUTO, ON, OFF.")
set_property(CACHE RELM_RELMP_OPS PROPERTY STRINGS AUTO ON OFF)
set(RELM_RPATH_MODE "dev" CACHE STRING "Runtime path mode: dev or wheel")
set_property(CACHE RELM_RPATH_MODE PROPERTY STRINGS dev wheel)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

set(_relm_relmp_enabled FALSE)
set(_relm_torch_version "")
set(_relm_torch_cuda_raw "cpu")
set(_relm_torch_cuda_tag "cpu")
set(_relm_torch_stable_abi_target "")
set(_relm_torch_link_target "")
string(TOUPPER "${RELM_RELMP_OPS}" _relm_relmp_mode)

set(_relm_relmp_modes AUTO ON OFF)
list(FIND _relm_relmp_modes "${_relm_relmp_mode}" _relm_relmp_mode_idx)
if (_relm_relmp_mode_idx EQUAL -1)
    message(FATAL_ERROR "Invalid RELM_RELMP_OPS=${RELM_RELMP_OPS}. Use AUTO, ON, or OFF.")
endif ()

if (NOT _relm_relmp_mode STREQUAL "OFF")
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
                set(_relm_relmp_enabled TRUE)
            endif ()
        endif ()
    endif ()

    if (_relm_relmp_enabled)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+(\\.[0-9]+)?" _relm_torch_semver "${_relm_torch_version}")
        if (_relm_torch_semver STREQUAL "")
            set(_relm_relmp_enabled FALSE)
            if (_relm_relmp_mode STREQUAL "ON")
                message(FATAL_ERROR "Could not parse torch version '${_relm_torch_version}' for relmp stable-ABI checks.")
            endif ()
        elseif (_relm_torch_semver VERSION_LESS "2.10.0")
            set(_relm_relmp_enabled FALSE)
            if (_relm_relmp_mode STREQUAL "ON")
                message(FATAL_ERROR "RELM_RELMP_OPS requires torch>=2.10. Detected ${_relm_torch_version}.")
            else ()
                message(STATUS "RELM relmp ops disabled (AUTO): torch ${_relm_torch_version} is below 2.10.")
            endif ()
        endif ()
    endif ()

    if (_relm_relmp_enabled)
        if (_relm_torch_cuda_raw STREQUAL "" OR _relm_torch_cuda_raw STREQUAL "cpu" OR _relm_torch_cuda_raw STREQUAL "None")
            set(_relm_torch_cuda_tag "cpu")
        else ()
            string(REPLACE "." "" _relm_torch_cuda_digits "${_relm_torch_cuda_raw}")
            set(_relm_torch_cuda_tag "cu${_relm_torch_cuda_digits}")
        endif ()

        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _relm_torch_stable_abi_target "${_relm_torch_version}")
        if (_relm_torch_stable_abi_target STREQUAL "")
            set(_relm_torch_stable_abi_target "unknown")
        endif ()

        message(STATUS "RELM relmp ops enabled (torch=${_relm_torch_version}, cuda_tag=${_relm_torch_cuda_tag}).")
    elseif (_relm_relmp_mode STREQUAL "ON")
        message(FATAL_ERROR "RELM_RELMP_OPS=ON but pip torch could not be resolved for CMake.")
    else ()
        message(STATUS "RELM relmp ops disabled (AUTO): pip torch probe failed or Torch CMake package unavailable.")
        if (NOT _relm_torch_probe_result EQUAL 0 AND NOT _relm_torch_probe_error STREQUAL "")
            message(STATUS "Torch probe error: ${_relm_torch_probe_error}")
        endif ()
    endif ()
endif ()

if (_relm_relmp_enabled)
    set(_relm_relmp_sources "_ops/relmp/relmp_ops.cpp")
    set(_relm_relmp_cuda_enabled FALSE)

    if (NOT _relm_torch_cuda_tag STREQUAL "cpu")
        if (CMAKE_CUDA_COMPILER)
            list(APPEND _relm_relmp_sources "_ops/relmp/relmp_cuda_kernels.cu")
            set(_relm_relmp_cuda_enabled TRUE)
            message(STATUS "RELM relmp CUDA kernels enabled.")
        else ()
            message(WARNING "RELM relmp CUDA kernels disabled: torch CUDA detected but CMAKE_CUDA_COMPILER unavailable.")
        endif ()
    endif ()

    add_library(relm_relmp_ops SHARED ${_relm_relmp_sources})
    set_target_properties(relm_relmp_ops PROPERTIES OUTPUT_NAME "relm_relmp_ops" PREFIX "" POSITION_INDEPENDENT_CODE ON)

    if (TORCH_INCLUDE_DIRS)
        target_include_directories(relm_relmp_ops PRIVATE ${TORCH_INCLUDE_DIRS})
    endif ()

    if (NOT _relm_torch_link_target STREQUAL "")
        target_link_libraries(relm_relmp_ops PRIVATE "${_relm_torch_link_target}")
    elseif (TORCH_LIBRARIES)
        target_link_libraries(relm_relmp_ops PRIVATE ${TORCH_LIBRARIES})
    else ()
        message(FATAL_ERROR "Torch was found but no link libraries were exposed for relm_relmp_ops.")
    endif ()

    if (TORCH_CXX_FLAGS)
        separate_arguments(_relm_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(relm_relmp_ops PRIVATE ${_relm_torch_cxx_flags})
        unset(_relm_torch_cxx_flags)
    endif ()

    target_compile_definitions(
            relm_relmp_ops
            PRIVATE
            RELM_RELMP_BUILD_TORCH_VERSION="${_relm_torch_version}"
            RELM_RELMP_BUILD_CUDA_TAG="${_relm_torch_cuda_tag}"
            RELM_RELMP_STABLE_ABI_TARGET="${_relm_torch_stable_abi_target}"
    )

    if (_relm_relmp_cuda_enabled)
        set_source_files_properties("_ops/relmp/relmp_cuda_kernels.cu" PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(relm_relmp_ops PRIVATE RELM_RELMP_HAS_CUDA=1 USE_CUDA)
    else ()
        target_compile_definitions(relm_relmp_ops PRIVATE RELM_RELMP_HAS_CUDA=0)
    endif ()

    if (APPLE)
        set(_relm_rpath "@loader_path")
    elseif (UNIX)
        set(_relm_rpath "$ORIGIN")
    endif ()
    set_target_properties(relm_relmp_ops PROPERTIES BUILD_RPATH "${_relm_rpath}" INSTALL_RPATH "${_relm_rpath}" INSTALL_RPATH_USE_LINK_PATH FALSE)

    if (RELM_RPATH_MODE STREQUAL "dev")
        set_target_properties(relm_relmp_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/relm")
    endif ()

    install(TARGETS relm_relmp_ops LIBRARY DESTINATION "relm")
else ()
    install(CODE "file(REMOVE \"\${CMAKE_INSTALL_PREFIX}/relm/relm_relmp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}\")")
    if (RELM_RPATH_MODE STREQUAL "dev")
        file(REMOVE "${CMAKE_SOURCE_DIR}/src/relm/relm_relmp_ops${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif ()
endif ()

cmake_minimum_required(VERSION 3.16)

project(ldpc
    VERSION 1.0.0
    DESCRIPTION "SIMD-accelerated, multi-threaded LDPC decoding library"
    LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(LDPC_IS_TOP_LEVEL ON)
else()
    set(LDPC_IS_TOP_LEVEL OFF)
endif()

include(GNUInstallDirs)

# --- Options -----------------------------------------------------------------

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|i[3-6]86|x86)$")
    set(LDPC_ARCH_X86 ON)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(LDPC_ARCH_ARM64 ON)
endif()

option(LDPC_ENABLE_AVX2 "Build the AVX2 kernel (selected at run time)" ${LDPC_ARCH_X86})
option(LDPC_ENABLE_AVX512 "Build the AVX-512 kernel (selected at run time)" ${LDPC_ARCH_X86})
option(LDPC_ENABLE_NEON "Build the AArch64 NEON kernel" ${LDPC_ARCH_ARM64})
option(LDPC_BUILD_TESTS "Build the unit tests" ${LDPC_IS_TOP_LEVEL})
option(LDPC_BUILD_APPS "Build the command-line tools" ${LDPC_IS_TOP_LEVEL})
option(LDPC_BUILD_EXAMPLES "Build the examples" ${LDPC_IS_TOP_LEVEL})
option(LDPC_INSTALL "Generate the install target" ${LDPC_IS_TOP_LEVEL})

if(LDPC_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Library -----------------------------------------------------------------

find_package(Threads REQUIRED)

add_library(ldpc
    src/alist.cpp
    src/bch.cpp
    src/cpu_features.cpp
    src/decoder.cpp
    src/dvb.cpp
    src/dvb_tables.cpp
    src/encoder.cpp
    src/interleaver.cpp
    src/parallel_decoder.cpp
    src/sparse_matrix.cpp
    src/structure.cpp
    src/thread_pool.cpp
    src/kernel/dispatch.cpp
    src/kernel/kernel_scalar.cpp
)
add_library(ldpc::ldpc ALIAS ldpc)

target_compile_features(ldpc PUBLIC cxx_std_17)
set_target_properties(ldpc PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(ldpc
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ldpc PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(ldpc PRIVATE /W4)
else()
    target_compile_options(ldpc PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Each SIMD kernel lives in its own translation unit compiled for its ISA;
# the dispatcher picks one at run time based on the CPU.
function(ldpc_add_kernel source define)
    target_sources(ldpc PRIVATE ${source})
    target_compile_definitions(ldpc PRIVATE ${define})
    set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "${ARGN}")
endfunction()

if(LDPC_ENABLE_AVX2)
    if(MSVC)
        ldpc_add_kernel(src/kernel/kernel_avx2.cpp LDPC_HAVE_AVX2 /arch:AVX2)
    else()
        ldpc_add_kernel(src/kernel/kernel_avx2.cpp LDPC_HAVE_AVX2 -mavx2)
    endif()
endif()
if(LDPC_ENABLE_AVX512)
    if(MSVC)
        ldpc_add_kernel(src/kernel/kernel_avx512.cpp LDPC_HAVE_AVX512 /arch:AVX512)
    else()
        ldpc_add_kernel(src/kernel/kernel_avx512.cpp LDPC_HAVE_AVX512 -mavx512f -mavx512bw)
    endif()
endif()
if(LDPC_ENABLE_NEON)
    ldpc_add_kernel(src/kernel/kernel_neon.cpp LDPC_HAVE_NEON)
endif()

# --- Sub-projects ------------------------------------------------------------

if(LDPC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(LDPC_BUILD_APPS)
    add_subdirectory(apps)
endif()
if(LDPC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# --- Install / package -------------------------------------------------------

if(LDPC_INSTALL)
    include(CMakePackageConfigHelpers)

    install(TARGETS ldpc
        EXPORT ldpcTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ldpc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT ldpcTargets
        NAMESPACE ldpc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ldpc)

    configure_package_config_file(cmake/ldpcConfig.cmake.in
        ${PROJECT_BINARY_DIR}/ldpcConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ldpc)
    write_basic_package_version_file(${PROJECT_BINARY_DIR}/ldpcConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${PROJECT_BINARY_DIR}/ldpcConfig.cmake
        ${PROJECT_BINARY_DIR}/ldpcConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ldpc)
endif()

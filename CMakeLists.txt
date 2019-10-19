project(SmallRaytracer)
cmake_minimum_required(VERSION 3.0)

IF(NOT DEFINED ENV{ROCM_PATH})
    SET(ROCM_PATH /opt/rocm)
ELSE()
    SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH ${ROCM_PATH}/hip CACHE PATH "Path where HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path where HIP has been installed")
    endif()
endif()

IF(NOT DEFINED ENV{HIPRAND_PATH})
    SET(HIPRAND_PATH ${ROCM_PATH}/hiprand)
ELSE()
    SET(HIPRAND_PATH $ENV{HIPRAND_PATH})
ENDIF()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
set(hiprand_DIR ${HIPRAND_PATH}/lib/cmake/hiprand)

find_package(HIP REQUIRED)
find_package(hiprand REQUIRED)

include_directories(extern/headers_only)

set(Sources
        src/main.cpp)

set(Headers
        src/logger.h
        src/constexpr_string.h)

set(HIPCC_OPTIONS)
set(HCC_OPTIONS)
set(NVCC_OPTIONS -ccbin=gcc-8.3.0 -Xcompiler -O3,-Wall,-Wextra,-Werror=float-conversion,-Werror=double-promotion,-Wno-unused-function -Xlinker ${hiprand_LIBRARIES} -Xptxas -v -rdc=true -std=c++14 --expt-relaxed-constexpr)

set_source_files_properties(${Sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(raytracer ${Sources} ${Headers} HIPCC_OPTIONS ${HIPCC_OPTIONS} HCC_OPTIONS ${HCC_OPTIONS} NVCC_OPTIONS ${NVCC_OPTIONS})
target_include_directories(raytracer PRIVATE ${hiprand_INCLUDE_DIRS})
target_link_libraries(raytracer PRIVATE )

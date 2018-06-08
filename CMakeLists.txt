cmake_minimum_required(VERSION 2.6.0)

project(ICP)

find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIR})

file(GLOB srcs src/*.cpp)
file(GLOB cuda src/Cuda/*.cu)
file(GLOB containers src/Cuda/containers/*.cpp)

set(CUDA_ARCH_BIN "30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")

add_executable(ICP 
               ${srcs}
               ${cuda} 
               ${cuda_objs} 
               ${containers}
)

target_link_libraries(ICP
                      ${Pangolin_LIBRARIES}
                      ${Eigen_LIBRARIES}
                      ${CUDA_LIBRARIES}
)


cmake_minimum_required(VERSION 3.18)
project(CUDA_Inference_Project)

# Enable CUDA
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

# Set C++ standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(include)

# Add source files
set(SOURCES
    src/main.cpp
    src/inference.cu
    src/engine_loader.cpp
    src/utils.cpp
)

# Link CUDA and TensorRT libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})

# Specify CUDA architecture (for modern GPUs, adjust accordingly)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")


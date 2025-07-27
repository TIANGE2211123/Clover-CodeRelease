cmake_minimum_required(VERSION 3.18)
project(CLOVER LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)

# Find cuBLAS
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
if(NOT CUBLAS_LIBRARIES)
    message(FATAL_ERROR "cuBLAS library not found")
endif()

# Optional: Find FAISS if available
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(FAISS faiss)
    if(FAISS_FOUND)
        add_definitions(-DUSE_FAISS)
        message(STATUS "FAISS found: enabling FAISS support")
    else()
        message(STATUS "FAISS not found: compiling without FAISS support")
    endif()
else()
    message(STATUS "PkgConfig not found: compiling without FAISS support")
endif()

# Set CUDA architectures - Include Tensor Core capable architectures
# Volta (sm_70), Turing (sm_75), Ampere (sm_80, sm_86), Ada Lovelace (sm_89), Hopper (sm_90)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Alternatively, for older CMake versions:
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
#     -gencode arch=compute_70,code=sm_70
#     -gencode arch=compute_75,code=sm_75  
#     -gencode arch=compute_80,code=sm_80
#     -gencode arch=compute_86,code=sm_86
#     -gencode arch=compute_89,code=sm_89
#     -gencode arch=compute_90,code=sm_90
# )

# Compiler flags for optimization and Tensor Core support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")

# Enable Tensor Core operations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA_ENABLE_TENSOR_CORES")

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add Thrust configuration
thrust_create_target(Thrust)

# Create the main executable
add_executable(clover
    src/linear-scans.cu
    # Add other source files as needed
)

# Link libraries
target_link_libraries(clover 
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    Thrust
)

# Add FAISS support if available
if(FAISS_FOUND)
    target_link_libraries(clover ${FAISS_LIBRARIES})
    target_include_directories(clover PRIVATE ${FAISS_INCLUDE_DIRS})
    target_compile_definitions(clover PRIVATE ${FAISS_CFLAGS_OTHER})
endif()

# Set target properties
set_target_properties(clover PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optional: Enable profiling support
option(ENABLE_PROFILING "Enable NVIDIA profiling support" OFF)
if(ENABLE_PROFILING)
    target_compile_definitions(clover PRIVATE ENABLE_PROFILING)
    target_link_libraries(clover nvToolsExt)
endif()

# Print configuration summary
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "cuBLAS Library: ${CUBLAS_LIBRARIES}")
message(STATUS "Tensor Core Support: Enabled for sm_70+")

# Create a compile definition to check Tensor Core capability at runtime
target_compile_definitions(clover PRIVATE TENSOR_CORE_AVAILABLE)
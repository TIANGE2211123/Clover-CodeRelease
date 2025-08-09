cmake_minimum_required(VERSION 3.27)
set(CMAKE_CUDA_ARCHITECTURES "86")
project(gpu-knn LANGUAGES CUDA CXX)

file(GLOB CU_FILES
    src/*.cu
    include/treelogy/GPU/knearestneighbor/kdtree/gpu_non_lockstep/*.cu
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(linear-scans ${CU_FILES})

if(LINK_FAISS)
    add_compile_definitions(USE_FAISS)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/faiss)
    find_package(OpenMP REQUIRED)
    target_link_libraries(linear-scans PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(linear-scans PRIVATE -lcublas)
    target_link_libraries(linear-scans PRIVATE
    	    ${CMAKE_SOURCE_DIR}/external/faiss/build/faiss/libfaiss.a)
endif()

# 在 target_compile_options 中添加优化标志
target_compile_options(linear-scans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -arch=sm_86;
    -std=c++20;
    -Xptxas="-v";
    -O3;                    # 新增
    -use_fast_math;        # 新增
>)

# 总是链接 cuBLAS（不仅仅在 FAISS 时）
find_library(CUBLAS_LIBRARIES cublas REQUIRED)
target_link_libraries(linear-scans PRIVATE ${CUBLAS_LIBRARIES})

# !! 添加 Tensor Core 支持定义
target_compile_definitions(linear-scans PRIVATE TENSOR_CORE_AVAILABLE)
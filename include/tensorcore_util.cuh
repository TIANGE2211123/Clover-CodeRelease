#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "cuda_util.cuh"

namespace tensorcore {

/**
 * Convert float array to half precision (__half) array
 */
__global__ void convert_float_to_half(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

/**
 * Compute squared norms of matrix rows
 * Input: matrix of shape [M, D] 
 * Output: norms of shape [M]
 */
__global__ void compute_squared_norms_fp16(const __half* matrix, float* norms, int M, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < D; col++) {
            float val = __half2float(matrix[row * D + col]);
            sum += val * val;
        }
        norms[row] = sum;
    }
}

/**
 * Add squared norms to distance matrix
 * distances[i][j] += norms_A[i] + norms_B[j]
 */
__global__ void add_squared_norms(float* distances, const float* norms_A, const float* norms_B, 
                                  int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < M && j < N) {
        distances[i * N + j] += norms_A[i] + norms_B[j];
    }
}

/**
 * Batched L2 distance computation using cuBLAS GEMM with Tensor Cores
 * Uses the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
 * 
 * @param handle cuBLAS handle
 * @param A queries matrix, shape [M, D] in FP16
 * @param B hubs matrix, shape [N, D] in FP16  
 * @param distances output distance matrix, shape [M, N] in FP32
 * @param M number of query points
 * @param N number of hub points
 * @param D dimensionality
 */
void batched_l2_distance_gemm(cublasHandle_t handle,
                              const __half* A,     // queries [M, D]
                              const __half* B,     // hubs [N, D]
                              float* distances,    // output [M, N]
                              int M, int N, int D) {
    
    // 使用预分配的内存池避免频繁malloc/free
    static float *norms_A = nullptr, *norms_B = nullptr;
    static size_t max_M = 0, max_N = 0;
    
    // 动态分配或重用内存
    if (norms_A == nullptr || M > max_M) {
        if (norms_A) cudaFree(norms_A);
        CUDA_CALL(cudaMalloc(&norms_A, M * sizeof(float)));
        max_M = M;
    }
    if (norms_B == nullptr || N > max_N) {
        if (norms_B) cudaFree(norms_B);
        CUDA_CALL(cudaMalloc(&norms_B, N * sizeof(float)));
        max_N = N;
    }
    
    // Step 1: Compute dot products using GEMM with Tensor Cores
    // C = -2.0 * A * B^T + 0.0 * C
    const float alpha = -2.0f;
    const float beta = 0.0f;
    
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,        // B^T
        CUBLAS_OP_N,        // A
        N, M, D,            // dimensions: N x M result from (D x N)^T * (M x D)
        &alpha,
        B, CUDA_R_16F, D,   // B: D x N, leading dimension D
        A, CUDA_R_16F, D,   // A: M x D, leading dimension D  
        &beta,
        distances, CUDA_R_32F, N, // C: M x N, leading dimension N
        CUDA_R_32F,         // computation type
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GEMM failed");
    }
    
    // Step 2: Compute squared norms in parallel
    dim3 block_size(256);
    dim3 grid_size_A((M + block_size.x - 1) / block_size.x);
    dim3 grid_size_B((N + block_size.x - 1) / block_size.x);
    
    // 并行计算两个norm
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    compute_squared_norms_fp16<<<grid_size_A, block_size, 0, stream1>>>(A, norms_A, M, D);
    compute_squared_norms_fp16<<<grid_size_B, block_size, 0, stream2>>>(B, norms_B, N, D);
    
    // 等待两个stream完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    // Step 3: Add squared norms to complete L2 distance computation
    dim3 block_2d(16, 16);
    dim3 grid_2d((N + block_2d.x - 1) / block_2d.x, 
                 (M + block_2d.y - 1) / block_2d.y);
    
    add_squared_norms<<<grid_2d, block_2d>>>(distances, norms_A, norms_B, M, N);
}

/**
 * Helper class to manage cuBLAS handle lifecycle
 */
class CublasManager {
private:
    cublasHandle_t handle;
    
public:
    CublasManager() {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
        
        // Enable Tensor Core usage
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    }
    
    ~CublasManager() {
        cublasDestroy(handle);
    }
    
    cublasHandle_t get() const { return handle; }
    
    // Disable copy construction and assignment
    CublasManager(const CublasManager&) = delete;
    CublasManager& operator=(const CublasManager&) = delete;
};

} // namespace tensorcore
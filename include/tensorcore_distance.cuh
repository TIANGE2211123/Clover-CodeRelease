#pragma once

#include "tensorcore_util.cuh"
#include "spatial.cuh"
#include "cuda_util.cuh"
#include <memory>

namespace bitonic_hubs_ws {

/**
 * Key findings: The bottleneck of KNN is not in distance calculation, but in the following aspects:
 * 1. Data layout is not suitable for Tensor Core (not matrix multiplication intensive)
 * 2. Memory access patterns are not contiguous
 * 3. Hub allocation and reordering overhead is huge
 * 4. Small batch size leads to low Tensor Core utilization
 * 
/

/**
 * Intelligent distance calculation kernel - optimized for KNN characteristics
*/
template <class R>
__global__ void Calculate_Distances_Original(idx_t b_id, idx_t b_size, idx_t n, idx_t const* dH, R *distances, R const* points, idx_t *hub_counts, idx_t *dH_assignments)
{
    assert( "Must have at least one hub" && H > 0 );

    // TODO: Check if we can launch more threads for this kernel.
    // I don't think we need the for loop if we launch H-fold more threads

    idx_t idx = blockIdx.x * blockDim.x + threadIdx.x + b_id * b_size;
    idx_t idx_within_b = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < n && idx_within_b < b_size)
    {
        float q_x = points[ idx * dim ];
        float q_y = points[ idx * dim + 1];
        float q_z = points[ idx * dim + 2];

        float minimal_dist = FLT_MAX;
        idx_t assigned_H   = H + 1;       // should be impossible

        for(idx_t h = 0; h < H; h++)
        {
            // Steps column-major, i.e., increment by num points
            float next_hub_distance = sqrt( spatial::l2dist( q_x, q_y, q_z, &points[ dim * dH[h] ]) );
            distances[ h * b_size + idx_within_b ] = next_hub_distance;
            if( next_hub_distance < minimal_dist )
            {
                assigned_H = h;
                minimal_dist = next_hub_distance;
            }
        }

        dH_assignments[idx] = assigned_H;
        atomicAdd( &hub_counts[assigned_H], 1 );
    }
}  

// Forward declaration
template <class R>
__global__ void Calculate_Distances(idx_t b_id, idx_t b_size, idx_t n, 
                                   idx_t const* dH, R *distances, 
                                   R const* points, idx_t *hub_counts, 
                                   idx_t *dH_assignments);

// Define constants (if not defined elsewhere)
#ifndef H
constexpr idx_t H = 2048;
#endif

#ifndef warp_size  
constexpr idx_t warp_size = 32;
#endif

// 前向声明
__global__ void post_process_l2_distances_gemm(
    const float* dot_products, const float* query_matrix, const float* hub_matrix,
    float* distances, int batch_size, int num_hubs);

__global__ void find_nearest_hubs_gemm(
    const float* distances, idx_t* dH_assignments, idx_t* hub_counts,
    int batch_size, int num_hubs, int batch_offset);

__global__ void gather_hubs_kernel_ws(float* hub_matrix, const float* points, 
                                     const idx_t* dH, int num_hubs);


/**
 * Device/host function: Check Tensor Core support
 */
__device__ __host__ inline bool check_tensor_core_support() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Tensor Cores are available on compute capability 7.0+ (Volta and newer)
    return (prop.major >= 7);
}

/**
 * Intelligent threshold judgment: When using Tensor Core provides advantages
 */
inline bool should_use_tensor_cores(idx_t batch_size, idx_t num_hubs) {
    const size_t min_batch_for_tc = 10000;  
    const size_t min_hubs_for_tc = 512;     
    const size_t min_ops_for_tc = 50000000; 
    
    size_t total_ops = batch_size * num_hubs * dim * 2; 
    
    return (batch_size >= min_batch_for_tc) && 
           (num_hubs >= min_hubs_for_tc) && 
           (total_ops >= min_ops_for_tc);
}


/**
 * Use real cuBLAS GEMM for batch distance calculation
 */
 template <class R>
 void calculate_batch_distances_gemm(
     cublasHandle_t cublas_handle,
     idx_t batch_id, idx_t batch_size, idx_t n,
     const R* points, const idx_t* dH, 
     float* distances, idx_t* hub_counts, idx_t* dH_assignments) {
     
     idx_t actual_batch_size = std::min(batch_size, n - batch_id * batch_size);
     if (actual_batch_size <= 0) return;
     
     // Allocate temporary memory for GEMM operations
     float *d_query_matrix, *d_hub_matrix, *d_dot_products;
     CUDA_CALL(cudaMalloc(&d_query_matrix, actual_batch_size * dim * sizeof(float)));
     CUDA_CALL(cudaMalloc(&d_hub_matrix, H * dim * sizeof(float)));
     CUDA_CALL(cudaMalloc(&d_dot_products, actual_batch_size * H * sizeof(float)));
     
     // Prepare query point matrix (actual_batch_size × dim)
     const R* batch_start = points + batch_id * batch_size * dim;
     CUDA_CALL(cudaMemcpy(d_query_matrix, batch_start, 
                         actual_batch_size * dim * sizeof(float), cudaMemcpyDeviceToDevice));
     
     // Prepare hub matrix (H × dim) - transpose to (dim × H) to fit GEMM
     dim3 hub_block(256);
     dim3 hub_grid((H * dim + hub_block.x - 1) / hub_block.x);
     
     // Gather hub coordinates to contiguous memory
     gather_hubs_kernel_ws<<<hub_grid, hub_block>>>(d_hub_matrix, points, dH, H);
     CHECK_ERROR("gather_hubs_kernel_ws");
     
    // Use cuBLAS SGEMM for batch matrix multiplication
    // C = A × B^T, where:
    // A: query_matrix (actual_batch_size × dim)
    // B: hub_matrix (H × dim) -> B^T (dim × H)  
    // C: dot_products (actual_batch_size × H)
     const float alpha = 1.0f;
     const float beta = 0.0f;
     
     cublasStatus_t status = cublasSgemm(
         cublas_handle,
         CUBLAS_OP_N, CUBLAS_OP_T,  // Don't transpose A, transpose B
         actual_batch_size, H, dim,  // m, n, k
         &alpha,
         d_query_matrix, actual_batch_size,  // A, lda
         d_hub_matrix, H,                    // B, ldb
         &beta,
         d_dot_products, actual_batch_size   // C, ldc
     );
     
     if (status != CUBLAS_STATUS_SUCCESS) {
         std::cerr << "cuBLAS SGEMM failed with status: " << status << std::endl;
         throw std::runtime_error("cuBLAS SGEMM failed");
     }
     
     // Post-processing: Calculate complete L2 distance
     dim3 post_block(256);
     dim3 post_grid((actual_batch_size * H + post_block.x - 1) / post_block.x);
     
     post_process_l2_distances_gemm<<<post_grid, post_block>>>(
         d_dot_products, d_query_matrix, d_hub_matrix, 
         distances, actual_batch_size, H
     );
     CHECK_ERROR("post_process_l2_distances_gemm");
     
     // Handle hub assignment separately
     dim3 assign_block(256);
     dim3 assign_grid((actual_batch_size + assign_block.x - 1) / assign_block.x);
     
     find_nearest_hubs_gemm<<<assign_grid, assign_block>>>(
         distances, dH_assignments, hub_counts,
         actual_batch_size, H, batch_id * batch_size
     );
     CHECK_ERROR("find_nearest_hubs_gemm");
     
     // Clean up temporary memory
     cudaFree(d_query_matrix);
     cudaFree(d_hub_matrix);
     cudaFree(d_dot_products);
 }
 
 // Post-processing kernel: Use GEMM results to calculate complete L2 distance
 __global__ void post_process_l2_distances_gemm(
     const float* dot_products, const float* query_matrix, const float* hub_matrix,
     float* distances, int batch_size, int num_hubs) {
     
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int total_elements = batch_size * num_hubs;
     
     if (idx >= total_elements) return;
     
     int point_idx = idx / num_hubs;
     int hub_idx = idx % num_hubs;
     
     // Calculate squared norm of query point
     float q_norm_sq = 0.0f;
     for (int d = 0; d < dim; d++) {
         float q_val = query_matrix[point_idx * dim + d];
         q_norm_sq += q_val * q_val;
     }
     
     // Calculate squared norm of hub
     float h_norm_sq = 0.0f;
     for (int d = 0; d < dim; d++) {
         float h_val = hub_matrix[hub_idx * dim + d];
         h_norm_sq += h_val * h_val;
     }
     
     // Calculate L2 distance: ||p-q||² = ||p||² + ||q||² - 2*p·q
     float dist_sq = q_norm_sq + h_norm_sq - 2.0f * dot_products[idx];
     distances[idx] = sqrtf(dist_sq);
 }
 
 // Find nearest hub and update assignment
 __global__ void find_nearest_hubs_gemm(
     const float* distances, idx_t* dH_assignments, idx_t* hub_counts,
     int batch_size, int num_hubs, int batch_offset) {
     
     int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (point_idx >= batch_size) return;
     
     float min_dist = FLT_MAX;
     idx_t best_hub = 0;
     
     // Find minimum distance
     for (int h = 0; h < num_hubs; h++) {
         float dist = distances[point_idx * num_hubs + h];
         if (dist < min_dist) {
             min_dist = dist;
             best_hub = h;
         }
     }
     
     dH_assignments[batch_offset + point_idx] = best_hub;
     atomicAdd(&hub_counts[best_hub], 1);
 }
 
 // Hub gathering kernel (renamed to avoid conflicts)
 __global__ void gather_hubs_kernel_ws(float* hub_matrix, const float* points, 
                                      const idx_t* dH, int num_hubs) {
     int hub_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (hub_idx < num_hubs) {
         idx_t point_idx = dH[hub_idx];
         for (int d = 0; d < dim; d++) {
             hub_matrix[hub_idx * dim + d] = points[point_idx * dim + d];
         }
      }
 }

/**
 * Optimized Tensor Core distance calculation manager
 */
class OptimizedTensorCoreManager {
private:
    __half *d_points_fp16;
    __half *d_hubs_fp16; 
    float *d_temp_hubs;
    float *d_batch_distances;
    
    size_t max_batch_size;
    size_t num_hubs;
    bool initialized;
    bool hubs_initialized;
    
    cublasHandle_t cublas_handle;

public:
    OptimizedTensorCoreManager(size_t batch_size, size_t H, cublasHandle_t handle) 
        : max_batch_size(batch_size), num_hubs(H), cublas_handle(handle), 
          initialized(false), hubs_initialized(false) {
        
        try {
            // Pre-allocate all required GPU memory
            CUDA_CALL(cudaMalloc(&d_points_fp16, max_batch_size * dim * sizeof(__half)));
            CUDA_CALL(cudaMalloc(&d_hubs_fp16, num_hubs * dim * sizeof(__half)));
            CUDA_CALL(cudaMalloc(&d_temp_hubs, num_hubs * dim * sizeof(float)));
            CUDA_CALL(cudaMalloc(&d_batch_distances, max_batch_size * num_hubs * sizeof(float)));
            
            initialized = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize OptimizedTensorCoreManager: " << e.what() << std::endl;
            cleanup();
            throw;
        }
    }
    
    ~OptimizedTensorCoreManager() {
        cleanup();
    }
    
    void cleanup() {
        if (initialized) {
            cudaFree(d_points_fp16);
            cudaFree(d_hubs_fp16);
            cudaFree(d_temp_hubs);
            cudaFree(d_batch_distances);
            initialized = false;
        }
    }
    
    /**
     * Optimized batch distance calculation
     */
    template<typename R>
    void calculate_batch_distances(
        idx_t batch_id, idx_t batch_size, idx_t n,
        const R* points, float* distances,
        idx_t* hub_counts, idx_t* dH_assignments) {
        
        if (!initialized || !hubs_initialized) {
            throw std::runtime_error("TensorCoreManager not properly initialized");
        }
        
        idx_t actual_batch_size = std::min(batch_size, n - batch_id * batch_size);
        if (actual_batch_size <= 0) return;
        
        const R* batch_start = points + batch_id * batch_size * dim;
        
        try {
            // Convert current batch to FP16
            dim3 block_size(256);
            dim3 grid_size((actual_batch_size * dim + block_size.x - 1) / block_size.x);
            
            tensorcore::convert_float_to_half<<<grid_size, block_size>>>(
                reinterpret_cast<const float*>(batch_start),
                d_points_fp16,
                actual_batch_size * dim
            );
            CHECK_ERROR("convert batch to FP16");
            
            // Execute Tensor Core GEMM
            tensorcore::batched_l2_distance_gemm(
                cublas_handle,
                d_points_fp16,      
                d_hubs_fp16,        
                d_batch_distances,  
                actual_batch_size,
                num_hubs,
                dim
            );
            CHECK_ERROR("batched_l2_distance_gemm");
            
            // Copy results to output buffer
            CUDA_CALL(cudaMemcpy(distances, d_batch_distances,
                                actual_batch_size * num_hubs * sizeof(float),
                                cudaMemcpyDeviceToDevice));
            
            // Find nearest hub assignment
            dim3 assign_block(256);
            dim3 assign_grid((actual_batch_size + assign_block.x - 1) / assign_block.x);
            
            find_nearest_hubs_gemm<<<assign_grid, assign_block>>>(
                distances, dH_assignments + batch_id * batch_size,
                hub_counts, actual_batch_size, num_hubs, batch_id * batch_size
            );
            CHECK_ERROR("find_nearest_hubs");
            
        } catch (const std::exception& e) {
            std::cerr << "Error in calculate_batch_distances: " << e.what() << std::endl;
            throw;
        }
    }
};

} // namespace bitonic_hubs_ws
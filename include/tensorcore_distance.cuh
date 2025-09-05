/**
 * Complete implementation of Tensor Core-optimized distance calculation
 * This replaces the Calculate_Distances function in bitonic-hubs-ws.cuh
 */

 #pragma once

 #include "tensorcore_util.cuh"
 #include "spatial.cuh"
 #include "cuda_util.cuh"
 #include "bitonic-hubs-ws.cuh"
 
 namespace bitonic_hubs_ws {
 
 // Forward declaration of Calculate_Distances function
 template <class R>
 __global__ void Calculate_Distances(idx_t b_id, idx_t b_size, idx_t n, 
                                    idx_t const* dH, R *distances, 
                                    R const* points, idx_t *hub_counts, 
                                    idx_t *dH_assignments);
 
 // Define H and warp_size constants
 namespace {
     idx_t constexpr H = 2048;
     idx_t constexpr warp_size = 32;
 }
 
 /**
  * Device function to check if the current GPU supports Tensor Cores
  */
 __device__ __host__ bool check_tensor_core_support() {
     int device;
     cudaGetDevice(&device);
     
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, device);
     
     // Tensor Cores are available on compute capability 7.0+ (Volta and newer)
     return (prop.major >= 7);
 }
 
 /**
  * Kernel to gather hub coordinates into a contiguous array
  * MOVED BEFORE calculate_distances_batch_tensorcore to fix compilation error
  */
 __global__ void gather_hubs_kernel(float* temp_hubs, const float* points, 
                                   const idx_t* dH, int num_hubs) {
     int hub_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (hub_idx < num_hubs) {
         idx_t point_idx = dH[hub_idx];
         for (int d = 0; d < dim; d++) {
             temp_hubs[hub_idx * dim + d] = points[point_idx * dim + d];
         }
     }
 }
 
 /**
  * Kernel to find nearest hub for each point based on computed distances
  */
 __global__ void find_nearest_hubs(const float* distances, idx_t* dH_assignments,
                                  idx_t* hub_counts, int batch_size, int num_hubs,
                                  int batch_offset) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (idx < batch_size) {
         float minimal_dist = FLT_MAX;
         idx_t assigned_H = num_hubs + 1;  // Invalid hub initially
         
         // Find minimum distance among all hubs for this point
         for (int h = 0; h < num_hubs; h++) {
             float dist = sqrtf(distances[idx * num_hubs + h]);  // Take sqrt for actual L2 distance
             if (dist < minimal_dist) {
                 minimal_dist = dist;
                 assigned_H = h;
             }
         }
         
         // Assign point to nearest hub
         int global_point_idx = batch_offset + idx;
         dH_assignments[global_point_idx] = assigned_H;
         atomicAdd(&hub_counts[assigned_H], 1);
     }
 }
 
 /**
  * Host-side wrapper for batched distance calculation using Tensor Cores
  * This function manages memory and orchestrates the GEMM-based distance computation
  */
 template <class R>
 void calculate_distances_batch_tensorcore(
     idx_t batch_id, 
     idx_t batch_size, 
     idx_t n,
     idx_t const* dH,           // Hub indices
     R const* points,           // All points [n, dim]
     float* distances,          // Output distances [batch_size, H]
     idx_t* hub_counts,         // Hub assignment counts
     idx_t* dH_assignments,     // Point->hub assignments
     cublasHandle_t cublas_handle
 ) {
     // Calculate actual batch size (handle last batch being smaller)
     idx_t actual_batch_size = std::min(batch_size, n - batch_id * batch_size);
     if (actual_batch_size <= 0) return;
     
     // Allocate device memory for FP16 conversions
     __half *points_fp16, *hubs_fp16;
     float *temp_hubs;
     
     CUDA_CALL(cudaMalloc(&points_fp16, actual_batch_size * dim * sizeof(__half)));
     CUDA_CALL(cudaMalloc(&hubs_fp16, H * dim * sizeof(__half)));
     CUDA_CALL(cudaMalloc(&temp_hubs, H * dim * sizeof(float)));
     
     // Step 1: Extract and convert current batch points to FP16
     const R* batch_start = points + batch_id * batch_size * dim;
     
     dim3 block_size(256);
     dim3 grid_size((actual_batch_size * dim + block_size.x - 1) / block_size.x);
     
     tensorcore::convert_float_to_half<<<grid_size, block_size>>>(
         reinterpret_cast<const float*>(batch_start), 
         points_fp16, 
         actual_batch_size * dim
     );
     
     // Step 2: Extract hub coordinates and convert to FP16
     // Launch kernel to gather hub coordinates
     dim3 hub_block(256);
     dim3 hub_grid((H + hub_block.x - 1) / hub_block.x);
     
     // Launch hub gathering kernel (now properly defined above)
     gather_hubs_kernel<<<hub_grid, hub_block>>>(temp_hubs, points, dH, H);
     CHECK_ERROR("gather_hubs_kernel");
     
     // Convert hubs to FP16
     dim3 hub_conv_grid((H * dim + block_size.x - 1) / block_size.x);
     tensorcore::convert_float_to_half<<<hub_conv_grid, block_size>>>(
         temp_hubs, hubs_fp16, H * dim
     );
     CHECK_ERROR("convert hubs to FP16");
     
     // Step 3: Perform batched GEMM using Tensor Cores
     tensorcore::batched_l2_distance_gemm(
         cublas_handle,
         points_fp16,        // queries [actual_batch_size, dim]  
         hubs_fp16,          // hubs [H, dim]
         distances,          // output [actual_batch_size, H] (row-major)
         actual_batch_size, 
         H, 
         dim
     );
     CHECK_ERROR("batched_l2_distance_gemm");
     
     // Step 4: Find nearest hub assignments
     dim3 assign_block(256);
     dim3 assign_grid((actual_batch_size + assign_block.x - 1) / assign_block.x);
     
     find_nearest_hubs<<<assign_grid, assign_block>>>(
         distances, dH_assignments + batch_id * batch_size, 
         hub_counts, actual_batch_size, H, batch_id * batch_size
     );
     CHECK_ERROR("find_nearest_hubs");
     
     // Cleanup
     cudaFree(points_fp16);
     cudaFree(hubs_fp16);
     cudaFree(temp_hubs);
 }
 
 /**
  * Updated main distance calculation function with Tensor Core support
  */
 template <class R>
 __global__ void Calculate_Distances_TensorCore_Optimized(
     idx_t b_id, idx_t b_size, idx_t n, 
     idx_t const* dH, float *distances, 
     R const* points, idx_t *hub_counts, 
     idx_t *dH_assignments,
     cublasHandle_t cublas_handle,
     bool* use_tensor_cores  // Flag to enable/disable tensor cores
 ) {
     // This kernel now just calls the host function
     // In practice, you'd call the host function directly from C_and_Q
     
     // The actual work is done in calculate_distances_batch_tensorcore
     // which should be called from the host side
 }
 
 /**
  * Runtime check and fallback system
  */
 template <class R>  
 void smart_distance_calculation(
     idx_t batch_id, idx_t batch_size, idx_t n,
     idx_t const* dH, float* distances, R const* points,
     idx_t* hub_counts, idx_t* dH_assignments,
     cublasHandle_t cublas_handle
 ) {
     // Check if Tensor Cores are available and beneficial
     bool use_tensor_cores = check_tensor_core_support();
     
     // Additional heuristics: Tensor Cores are most beneficial for larger batches
     if (batch_size < 1000 || H < 64) {
         use_tensor_cores = false;  // Use traditional method for small batches
     }
     
     if (use_tensor_cores) {
         try {
             calculate_distances_batch_tensorcore(
                 batch_id, batch_size, n, dH, points, 
                 distances, hub_counts, dH_assignments, cublas_handle
             );
         } catch (const std::exception& e) {
             // Fallback to original method if Tensor Core path fails
             std::cerr << "Tensor Core path failed, falling back: " << e.what() << std::endl;
             use_tensor_cores = false;
         }
     }
     
     if (!use_tensor_cores) {
         // Launch original CUDA kernel
         dim3 block_size(1024);
         dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
         
         Calculate_Distances<<<grid_size, block_size>>>(
             batch_id, batch_size, n, dH, distances, points, hub_counts, dH_assignments
         );
         CHECK_ERROR("Calculate_Distances");
     }
 }
 
 } // namespace bitonic_hubs_ws
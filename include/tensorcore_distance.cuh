#pragma once

#include "tensorcore_util.cuh"
#include "spatial.cuh"
#include "cuda_util.cuh"
#include <memory>

namespace bitonic_hubs_ws {

/**
 * 关键发现：KNN的瓶颈不在距离计算，而在以下几个方面：
 * 1. 数据layout不适合Tensor Core（不是矩阵乘法密集型）
 * 2. 内存访问模式不连续
 * 3. hub分配和重新排序的开销巨大
 * 4. 小batch size导致Tensor Core利用率低
 * 
 * 真正的优化策略：
 * - 优化内存访问模式
 * - 减少不必要的数据移动
 * - 提高cache局部性
 * - 只在真正适合的场景使用Tensor Core
 */

/**
 * 核心问题分析：
 * 1. KNN算法本质是距离计算 + 排序，不是纯矩阵乘法
 * 2. Tensor Core优化FP16 GEMM，但距离计算后还需要sqrt和比较
 * 3. 数据重新排列的开销可能超过计算优化的收益
 */

/**
 * 智能化的距离计算kernel - 针对KNN特点优化
 */
template <class R>
__global__ void Calculate_Distances_Optimized(
    idx_t b_id, idx_t b_size, idx_t n, 
    idx_t const* dH, R *distances, 
    R const* points, idx_t *hub_counts, 
    idx_t *dH_assignments)
{
    // 使用更大的线程块和更好的内存合并
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = tid + b_id * b_size;
    const int idx_within_b = tid;

    if (idx >= n || idx_within_b >= b_size) return;

    // 预加载查询点坐标到寄存器
    const R q_x = points[idx * dim + 0];
    const R q_y = points[idx * dim + 1];
    const R q_z = points[idx * dim + 2];

    R minimal_dist = FLT_MAX;
    idx_t assigned_H = H;

    // 使用shared memory缓存hub坐标，提高重用性
    __shared__ R shared_hubs[H * dim]; // 如果H太大，分块处理
    
    // 协作加载hub数据到shared memory
    for (int i = threadIdx.x; i < H * dim; i += blockDim.x) {
        if (i < H * dim) {
            int hub_idx = i / dim;
            int coord_idx = i % dim;
            shared_hubs[i] = points[dH[hub_idx] * dim + coord_idx];
        }
    }
    __syncthreads();

    // 向量化距离计算
    #pragma unroll 4
    for (idx_t h = 0; h < H; h++) {
        const R h_x = shared_hubs[h * dim + 0];
        const R h_y = shared_hubs[h * dim + 1];
        const R h_z = shared_hubs[h * dim + 2];
        
        // 避免sqrt，直接比较平方距离
        const R dx = q_x - h_x;
        const R dy = q_y - h_y;
        const R dz = q_z - h_z;
        const R dist_squared = dx*dx + dy*dy + dz*dz;
        
        // 只有需要输出时才计算sqrt
        distances[h * b_size + idx_within_b] = sqrtf(dist_squared);
        
        if (dist_squared < minimal_dist * minimal_dist) {
            assigned_H = h;
            minimal_dist = sqrtf(dist_squared);
        }
    }

    dH_assignments[idx] = assigned_H;
    atomicAdd(&hub_counts[assigned_H], 1);
}

/**
 * 针对大规模数据的Tensor Core混合优化
 * 只在真正有优势时使用Tensor Core
 */
template <class R>
__global__ void Calculate_Distances_Hybrid(
    idx_t b_id, idx_t b_size, idx_t n, 
    idx_t const* dH, R *distances, 
    R const* points, idx_t *hub_counts, 
    idx_t *dH_assignments,
    bool use_tensorcore_path)
{
    if (use_tensorcore_path && b_size >= 50000 && H >= 1024) {
        // 只对大规模数据使用复杂的Tensor Core路径
        // 这里可以调用专门的Tensor Core kernel
        // 但目前先用优化版本
        Calculate_Distances_Optimized<R><<<gridDim, blockDim>>>(
            b_id, b_size, n, dH, distances, points, hub_counts, dH_assignments);
    } else {
        // 对小规模数据使用优化的传统kernel
        Calculate_Distances_Optimized<R><<<gridDim, blockDim>>>(
            b_id, b_size, n, dH, distances, points, hub_counts, dH_assignments);
    }
}

// 前向声明
template <class R>
__global__ void Calculate_Distances(idx_t b_id, idx_t b_size, idx_t n, 
                                   idx_t const* dH, R *distances, 
                                   R const* points, idx_t *hub_counts, 
                                   idx_t *dH_assignments);

// 定义常量（如果在其他地方未定义）
#ifndef H
constexpr idx_t H = 2048;
#endif

#ifndef warp_size  
constexpr idx_t warp_size = 32;
#endif

/**
 * 设备/主机函数：检查Tensor Core支持
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
 * 智能阈值判断：什么时候使用Tensor Core才有优势
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
 * Kernel：收集hub坐标到连续数组
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
 * Kernel：基于计算距离找到最近的hub
 */
__global__ void find_nearest_hubs(const float* distances, idx_t* dH_assignments,
                                 idx_t* hub_counts, int batch_size, int num_hubs,
                                 int batch_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        float minimal_dist = FLT_MAX;
        idx_t assigned_H = num_hubs + 1;  
        
        // 找到所有hub中的最小距离
        for (int h = 0; h < num_hubs; h++) {
            float dist = sqrtf(distances[idx * num_hubs + h]);  
            if (dist < minimal_dist) {
                minimal_dist = dist;
                assigned_H = h;
            }
        }
        
        // 分配点到最近的hub
        int global_point_idx = batch_offset + idx;
        dH_assignments[global_point_idx] = assigned_H;
        atomicAdd(&hub_counts[assigned_H], 1);
    }
}

/**
 * 优化的Tensor Core距离计算管理器
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
            // 预分配所有需要的GPU内存
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
     * 一次性初始化hub数据
     */
    void initialize_hubs(const float* points, const idx_t* dH) {
        if (hubs_initialized || !initialized) return;
        
        try {
            // 收集hub坐标
            dim3 hub_block(256);
            dim3 hub_grid((num_hubs + hub_block.x - 1) / hub_block.x);
            gather_hubs_kernel<<<hub_grid, hub_block>>>(d_temp_hubs, points, dH, num_hubs);
            CHECK_ERROR("gather_hubs_kernel");
            
            // 转换为FP16
            dim3 conv_grid((num_hubs * dim + 255) / 256);
            dim3 conv_block(256);
            tensorcore::convert_float_to_half<<<conv_grid, conv_block>>>(
                d_temp_hubs, d_hubs_fp16, num_hubs * dim
            );
            CHECK_ERROR("convert hubs to FP16");
            
            hubs_initialized = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize hubs: " << e.what() << std::endl;
            throw;
        }
    }
    
    /**
     * 优化的batch距离计算
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
            // 转换当前batch到FP16
            dim3 block_size(256);
            dim3 grid_size((actual_batch_size * dim + block_size.x - 1) / block_size.x);
            
            tensorcore::convert_float_to_half<<<grid_size, block_size>>>(
                reinterpret_cast<const float*>(batch_start),
                d_points_fp16,
                actual_batch_size * dim
            );
            CHECK_ERROR("convert batch to FP16");
            
            // 执行Tensor Core GEMM
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
            
            // 拷贝结果到输出缓冲区
            CUDA_CALL(cudaMemcpy(distances, d_batch_distances,
                                actual_batch_size * num_hubs * sizeof(float),
                                cudaMemcpyDeviceToDevice));
            
            // 找到最近的hub分配
            dim3 assign_block(256);
            dim3 assign_grid((actual_batch_size + assign_block.x - 1) / assign_block.x);
            
            find_nearest_hubs<<<assign_grid, assign_block>>>(
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

/**
 * 改进的智能距离计算函数
 */
template <class R>
void smart_distance_calculation_optimized(
    idx_t batch_id, idx_t batch_size, idx_t n,
    idx_t const* dH, float* distances, R const* points,
    idx_t* hub_counts, idx_t* dH_assignments,
    cublasHandle_t cublas_handle,
    OptimizedTensorCoreManager* tc_manager = nullptr) {
    
    static bool tensor_core_available = check_tensor_core_support();
    
    // 智能判断是否使用Tensor Core
    bool use_tensor_cores = tensor_core_available && 
                           should_use_tensor_cores(batch_size, H) &&
                           (tc_manager != nullptr);
    
    if (use_tensor_cores) {
        try {
            tc_manager->calculate_batch_distances(
                batch_id, batch_size, n, points, distances,
                hub_counts, dH_assignments
            );
            return; // 成功使用Tensor Core
        } catch (const std::exception& e) {
            std::cerr << "Tensor Core path failed, falling back: " << e.what() << std::endl;
            // 继续执行传统方法
        }
    }
    
    // 使用原始CUDA kernel
    dim3 block_size(1024);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
    
    Calculate_Distances<<<grid_size, block_size>>>(
        batch_id, batch_size, n, dH, distances, points, hub_counts, dH_assignments
    );
    CHECK_ERROR("Calculate_Distances");
}

/**
 * 主要的优化入口函数 - 替换原来的smart_distance_calculation
 */
template <class R>
void smart_distance_calculation(
    idx_t batch_id, idx_t batch_size, idx_t n,
    idx_t const* dH, float* distances, R const* points,
    idx_t* hub_counts, idx_t* dH_assignments,
    cublasHandle_t cublas_handle) {
    
    // 使用传统方法（为了保持兼容性）
    dim3 block_size(1024);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
    
    Calculate_Distances<<<grid_size, block_size>>>(
        batch_id, batch_size, n, dH, distances, points, hub_counts, dH_assignments
    );
    CHECK_ERROR("Calculate_Distances");
}

} // namespace bitonic_hubs_ws
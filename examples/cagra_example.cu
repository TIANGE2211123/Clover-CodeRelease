#include "../include/cagra.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

/**
 * CAGRA算法使用示例
 * 
 * 这个示例展示了如何使用CAGRA算法进行k近邻搜索
 */

int main() {
    std::cout << "CAGRA k-Nearest Neighbor Search Example" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // 配置参数
    const idx_t num_points = 10000;  // 数据点数量
    const idx_t num_queries = 100;   // 查询点数量
    const idx_t k = 10;              // 查找的最近邻数量
    const idx_t dim = 3;             // 数据维度
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Number of data points: " << num_points << std::endl;
    std::cout << "  - Number of queries: " << num_queries << std::endl;
    std::cout << "  - k value: " << k << std::endl;
    std::cout << "  - Dimension: " << dim << std::endl;
    std::cout << std::endl;
    
    // 生成随机数据点
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    std::vector<float> data_points(num_points * dim);
    for (idx_t i = 0; i < num_points * dim; ++i) {
        data_points[i] = dist(gen);
    }
    
    // 生成查询点
    std::vector<idx_t> query_indices(num_queries);
    for (idx_t i = 0; i < num_queries; ++i) {
        query_indices[i] = i;  // 使用前num_queries个点作为查询
    }
    
    // 分配GPU内存
    float* d_data_points;
    idx_t* d_query_indices;
    idx_t* d_results;
    float* d_distances;
    
    CUDA_CALL(cudaMalloc(&d_data_points, sizeof(float) * num_points * dim));
    CUDA_CALL(cudaMalloc(&d_query_indices, sizeof(idx_t) * num_queries));
    CUDA_CALL(cudaMalloc(&d_results, sizeof(idx_t) * num_queries * k));
    CUDA_CALL(cudaMalloc(&d_distances, sizeof(float) * num_queries * k));
    
    // 复制数据到GPU
    CUDA_CALL(cudaMemcpy(d_data_points, data_points.data(), 
                        sizeof(float) * num_points * dim, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_query_indices, query_indices.data(), 
                        sizeof(idx_t) * num_queries, cudaMemcpyHostToDevice));
    
    std::cout << "Running CAGRA kNN search..." << std::endl;
    
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 执行CAGRA kNN搜索
    cagra::knn_gpu(num_points, d_data_points, num_queries, d_query_indices, 
                   k, d_results, d_distances);
    
    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "CAGRA search completed in " << duration.count() << " microseconds" << std::endl;
    std::cout << std::endl;
    
    // 复制结果回CPU
    std::vector<idx_t> results(num_queries * k);
    std::vector<float> distances(num_queries * k);
    
    CUDA_CALL(cudaMemcpy(results.data(), d_results, 
                        sizeof(idx_t) * num_queries * k, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(distances.data(), d_distances, 
                        sizeof(float) * num_queries * k, cudaMemcpyDeviceToHost));
    
    // 显示前几个查询的结果
    std::cout << "Results for first 3 queries:" << std::endl;
    for (idx_t q = 0; q < 3 && q < num_queries; ++q) {
        std::cout << "Query " << q << " (point " << query_indices[q] << "):" << std::endl;
        std::cout << "  Coordinates: (" 
                  << data_points[query_indices[q] * dim] << ", "
                  << data_points[query_indices[q] * dim + 1] << ", "
                  << data_points[query_indices[q] * dim + 2] << ")" << std::endl;
        std::cout << "  k-nearest neighbors:" << std::endl;
        
        for (idx_t i = 0; i < k; ++i) {
            idx_t neighbor_idx = results[q * k + i];
            float distance = distances[q * k + i];
            
            std::cout << "    " << i + 1 << ". Point " << neighbor_idx 
                      << " (distance: " << distance << ")"
                      << " at (" 
                      << data_points[neighbor_idx * dim] << ", "
                      << data_points[neighbor_idx * dim + 1] << ", "
                      << data_points[neighbor_idx * dim + 2] << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    
    // 统计信息
    float total_distance = 0.0f;
    float min_distance = FLT_MAX;
    float max_distance = 0.0f;
    
    for (idx_t i = 0; i < num_queries * k; ++i) {
        total_distance += distances[i];
        min_distance = std::min(min_distance, distances[i]);
        max_distance = std::max(max_distance, distances[i]);
    }
    
    std::cout << "Statistics:" << std::endl;
    std::cout << "  - Average distance: " << total_distance / (num_queries * k) << std::endl;
    std::cout << "  - Minimum distance: " << min_distance << std::endl;
    std::cout << "  - Maximum distance: " << max_distance << std::endl;
    std::cout << "  - Total search time: " << duration.count() << " μs" << std::endl;
    std::cout << "  - Average time per query: " << duration.count() / num_queries << " μs" << std::endl;
    
    // 清理GPU内存
    CUDA_CALL(cudaFree(d_data_points));
    CUDA_CALL(cudaFree(d_query_indices));
    CUDA_CALL(cudaFree(d_results));
    CUDA_CALL(cudaFree(d_distances));
    
    std::cout << std::endl << "Example completed successfully!" << std::endl;
    
    return 0;
}


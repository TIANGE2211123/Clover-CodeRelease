#include "cagra.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <curand_kernel.h>

namespace cagra {

// Shared memory for graph construction
__shared__ float shared_distances[cagra_config::BLOCK_SIZE];
__shared__ idx_t shared_indices[cagra_config::BLOCK_SIZE];

/**
 * Initialize CAGRA graph structure
 */
__host__ void initialize_graph(CagraGraph& graph, idx_t num_vertices, idx_t max_degree) {
    graph.num_vertices = num_vertices;
    graph.max_degree = max_degree;
    
    // Allocate memory on device
    CUDA_CALL(cudaMalloc(&graph.neighbors, sizeof(idx_t) * num_vertices * max_degree));
    CUDA_CALL(cudaMalloc(&graph.degrees, sizeof(idx_t) * num_vertices));
    CUDA_CALL(cudaMalloc(&graph.distances, sizeof(float) * num_vertices * max_degree));
    
    // Initialize degrees to 0
    CUDA_CALL(cudaMemset(graph.degrees, 0, sizeof(idx_t) * num_vertices));
}

/**
 * Free CAGRA graph memory
 */
__host__ void free_graph(CagraGraph& graph) {
    if (graph.neighbors) {
        CUDA_CALL(cudaFree(graph.neighbors));
        graph.neighbors = nullptr;
    }
    if (graph.degrees) {
        CUDA_CALL(cudaFree(graph.degrees));
        graph.degrees = nullptr;
    }
    if (graph.distances) {
        CUDA_CALL(cudaFree(graph.distances));
        graph.distances = nullptr;
    }
}

/**
 * GPU kernel for graph construction using CAGRA approach
 */
template <class R>
__global__ void construct_graph_kernel(const R* points, idx_t num_points,
                                      idx_t* neighbors, idx_t* degrees,
                                      float* distances, idx_t max_degree) {
    idx_t vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vertex_id >= num_points) return;
    
    // Get current vertex coordinates
    float v_x = points[vertex_id * dim];
    float v_y = points[vertex_id * dim + 1];
    float v_z = points[vertex_id * dim + 2];
    
    // Temporary arrays for storing candidates
    float candidate_dists[cagra_config::MAX_DEGREE];
    idx_t candidate_indices[cagra_config::MAX_DEGREE];
    idx_t num_candidates = 0;
    
    // Find initial random neighbors
    curandState state;
    curand_init(vertex_id, 0, 0, &state);
    
    // Sample random neighbors for initial graph
    for (idx_t i = 0; i < max_degree && num_candidates < max_degree; ++i) {
        idx_t neighbor_id = curand(&state) % num_points;
        if (neighbor_id != vertex_id) {
            float dist = spatial::l2dist(v_x, v_y, v_z, 
                                       points[neighbor_id * dim],
                                       points[neighbor_id * dim + 1],
                                       points[neighbor_id * dim + 2]);
            
            // Insert into sorted candidates
            idx_t insert_pos = 0;
            while (insert_pos < num_candidates && candidate_dists[insert_pos] < dist) {
                insert_pos++;
            }
            
            if (insert_pos < max_degree) {
                // Shift elements to make room
                for (idx_t j = num_candidates; j > insert_pos; --j) {
                    candidate_dists[j] = candidate_dists[j-1];
                    candidate_indices[j] = candidate_indices[j-1];
                }
                candidate_dists[insert_pos] = dist;
                candidate_indices[insert_pos] = neighbor_id;
                if (num_candidates < max_degree) num_candidates++;
            }
        }
    }
    
    // Store results
    degrees[vertex_id] = num_candidates;
    for (idx_t i = 0; i < num_candidates; ++i) {
        neighbors[vertex_id * max_degree + i] = candidate_indices[i];
        distances[vertex_id * max_degree + i] = candidate_dists[i];
    }
}

/**
 * GPU kernel for kNN search using CAGRA graph
 */
template <class R>
__global__ void search_kernel(const R* query_points, idx_t num_queries,
                             const CagraGraph graph, const R* data_points,
                             idx_t k, idx_t* results, R* distances) {
    idx_t query_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_id >= num_queries) return;
    
    // Get query point coordinates
    float q_x = query_points[query_id * dim];
    float q_y = query_points[query_id * dim + 1];
    float q_z = query_points[query_id * dim + 2];
    
    // Priority queue for k nearest neighbors
    float knn_dists[cagra_config::SEARCH_WIDTH];
    idx_t knn_indices[cagra_config::SEARCH_WIDTH];
    idx_t knn_size = 0;
    
    // Visited vertices bitmap (using shared memory)
    __shared__ bool visited[cagra_config::BLOCK_SIZE];
    if (threadIdx.x < cagra_config::BLOCK_SIZE) {
        visited[threadIdx.x] = false;
    }
    __syncthreads();
    
    // Start with random vertex
    curandState state;
    curand_init(query_id, 0, 0, &state);
    idx_t current_vertex = curand(&state) % graph.num_vertices;
    
    // Search iterations
    for (idx_t iter = 0; iter < cagra_config::ITERATIONS; ++iter) {
        if (current_vertex >= graph.num_vertices) break;
        
        // Check if current vertex is better than our k-th best
        float current_dist = spatial::l2dist(q_x, q_y, q_z,
                                           data_points[current_vertex * dim],
                                           data_points[current_vertex * dim + 1],
                                           data_points[current_vertex * dim + 2]);
        
        // Insert into priority queue if better
        if (knn_size < k || current_dist < knn_dists[knn_size - 1]) {
            idx_t insert_pos = 0;
            while (insert_pos < knn_size && knn_dists[insert_pos] < current_dist) {
                insert_pos++;
            }
            
            if (insert_pos < k) {
                // Shift elements
                for (idx_t j = knn_size; j > insert_pos; --j) {
                    knn_dists[j] = knn_dists[j-1];
                    knn_indices[j] = knn_indices[j-1];
                }
                knn_dists[insert_pos] = current_dist;
                knn_indices[insert_pos] = current_vertex;
                if (knn_size < k) knn_size++;
            }
        }
        
        // Mark as visited
        visited[current_vertex % cagra_config::BLOCK_SIZE] = true;
        
        // Find best unvisited neighbor
        float best_neighbor_dist = FLT_MAX;
        idx_t best_neighbor = current_vertex;
        
        idx_t degree = graph.degrees[current_vertex];
        for (idx_t i = 0; i < degree; ++i) {
            idx_t neighbor = graph.neighbors[current_vertex * graph.max_degree + i];
            if (neighbor < graph.num_vertices && 
                !visited[neighbor % cagra_config::BLOCK_SIZE]) {
                
                float neighbor_dist = spatial::l2dist(q_x, q_y, q_z,
                                                    data_points[neighbor * dim],
                                                    data_points[neighbor * dim + 1],
                                                    data_points[neighbor * dim + 2]);
                
                if (neighbor_dist < best_neighbor_dist) {
                    best_neighbor_dist = neighbor_dist;
                    best_neighbor = neighbor;
                }
            }
        }
        
        // Move to best neighbor
        current_vertex = best_neighbor;
    }
    
    // Store results
    for (idx_t i = 0; i < k; ++i) {
        if (i < knn_size) {
            results[query_id * k + i] = knn_indices[i];
            distances[query_id * k + i] = knn_dists[i];
        } else {
            results[query_id * k + i] = 0;
            distances[query_id * k + i] = FLT_MAX;
        }
    }
}

/**
 * Build CAGRA graph from point data
 */
template <class R>
__host__ void build_graph(const R* points, idx_t num_points, CagraGraph& graph) {
    // Initialize graph
    initialize_graph(graph, num_points, cagra_config::MAX_DEGREE);
    
    // Launch graph construction kernel
    dim3 block_size(cagra_config::BLOCK_SIZE);
    dim3 grid_size((num_points + block_size.x - 1) / block_size.x);
    
    construct_graph_kernel<<<grid_size, block_size>>>(
        points, num_points, graph.neighbors, graph.degrees, 
        graph.distances, graph.max_degree);
    
    CUDA_CALL(cudaDeviceSynchronize());
}

/**
 * Search for k nearest neighbors using CAGRA
 */
template <class R>
__host__ void search_knn(const R* query_points, idx_t num_queries, 
                        const CagraGraph& graph, const R* data_points,
                        idx_t k, idx_t* results, R* distances) {
    // Launch search kernel
    dim3 block_size(cagra_config::BLOCK_SIZE);
    dim3 grid_size((num_queries + block_size.x - 1) / block_size.x);
    
    search_kernel<<<grid_size, block_size>>>(
        query_points, num_queries, graph, data_points, k, results, distances);
    
    CUDA_CALL(cudaDeviceSynchronize());
}

/**
 * Main CAGRA kNN function interface
 */
template <class R>
void knn_gpu(std::size_t n, R* data, std::size_t q, idx_t* queries, 
             std::size_t k, idx_t* results_knn, R* results_distances) {
    
    // Build CAGRA graph
    CagraGraph graph;
    build_graph(data, n, graph);
    
    // Prepare query points
    R* query_points;
    CUDA_CALL(cudaMalloc(&query_points, sizeof(R) * q * dim));
    
    // Copy query points to device
    for (idx_t i = 0; i < q; ++i) {
        idx_t query_idx = queries[i];
        CUDA_CALL(cudaMemcpy(query_points + i * dim, 
                            data + query_idx * dim, 
                            sizeof(R) * dim, 
                            cudaMemcpyDeviceToDevice));
    }
    
    // Perform search
    search_knn(query_points, q, graph, data, k, results_knn, results_distances);
    
    // Cleanup
    CUDA_CALL(cudaFree(query_points));
    free_graph(graph);
}

// Explicit template instantiations
template void knn_gpu<float>(std::size_t n, float* data, std::size_t q, idx_t* queries, 
                            std::size_t k, idx_t* results_knn, float* results_distances);

} // namespace cagra

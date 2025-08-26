#pragma once

#include <cuda_runtime.h>
#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cassert>
#include <vector>
#include <memory>

#include "cuda_util.cuh"
#include "spatial.cuh"

// CAGRA configuration parameters
namespace cagra_config {
    constexpr idx_t MAX_DEGREE = 64;           // Maximum degree of each vertex in the graph
    constexpr idx_t SEARCH_WIDTH = 64;         // Number of candidates to explore during search
    constexpr idx_t ITERATIONS = 10;           // Number of iterations for graph construction
    constexpr float EPSILON = 0.1f;           // Approximation factor
    constexpr idx_t BLOCK_SIZE = 256;         // CUDA block size for kernels
    constexpr idx_t WARP_SIZE = 32;           // CUDA warp size
}

// CAGRA graph structure
struct CagraGraph {
    idx_t num_vertices;
    idx_t max_degree;
    idx_t* neighbors;      // Neighbor indices for each vertex
    idx_t* degrees;        // Degree of each vertex
    float* distances;      // Distances to neighbors
};

// CAGRA search state
struct CagraSearchState {
    idx_t* candidates;     // Current search candidates
    float* candidate_dists; // Distances to candidates
    idx_t* visited;        // Visited vertices
    idx_t num_candidates;
    idx_t num_visited;
};

namespace cagra {

/**
 * Initialize CAGRA graph structure
 */
__host__ void initialize_graph(CagraGraph& graph, idx_t num_vertices, idx_t max_degree);

/**
 * Free CAGRA graph memory
 */
__host__ void free_graph(CagraGraph& graph);

/**
 * Build CAGRA graph from point data
 */
template <class R>
__host__ void build_graph(const R* points, idx_t num_points, CagraGraph& graph);

/**
 * Search for k nearest neighbors using CAGRA
 */
template <class R>
__host__ void search_knn(const R* query_points, idx_t num_queries, 
                        const CagraGraph& graph, const R* data_points,
                        idx_t k, idx_t* results, R* distances);

/**
 * GPU kernel for graph construction
 */
template <class R>
__global__ void construct_graph_kernel(const R* points, idx_t num_points,
                                      idx_t* neighbors, idx_t* degrees,
                                      float* distances, idx_t max_degree);

/**
 * GPU kernel for kNN search
 */
template <class R>
__global__ void search_kernel(const R* query_points, idx_t num_queries,
                             const CagraGraph graph, const R* data_points,
                             idx_t k, idx_t* results, R* distances);

/**
 * Main CAGRA kNN function interface
 */
template <class R>
void knn_gpu(std::size_t n, R* data, std::size_t q, idx_t* queries, 
             std::size_t k, idx_t* results_knn, R* results_distances);

} // namespace cagra

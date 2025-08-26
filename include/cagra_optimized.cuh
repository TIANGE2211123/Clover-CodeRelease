#pragma once

#include <cuda_runtime.h>
#include <cuda/std/limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <curand_kernel.h>
#include <cassert>
#include <vector>
#include <memory>

#include "cuda_util.cuh"
#include "spatial.cuh"

// Optimized CAGRA configuration parameters
namespace cagra_opt_config {
    constexpr idx_t MAX_DEGREE = 128;          // Increased max degree for better connectivity
    constexpr idx_t SEARCH_WIDTH = 128;        // Increased search width
    constexpr idx_t ITERATIONS = 20;           // More iterations for better accuracy
    constexpr float EPSILON = 0.05f;          // Tighter approximation factor
    constexpr idx_t BLOCK_SIZE = 512;         // Larger block size for better occupancy
    constexpr idx_t WARP_SIZE = 32;           // CUDA warp size
    constexpr idx_t GRAPH_BUILD_ITERATIONS = 5; // Multiple iterations for graph refinement
    constexpr idx_t CANDIDATE_POOL_SIZE = 256;  // Larger candidate pool
}

// Optimized CAGRA graph structure with additional metadata
struct CagraOptimizedGraph {
    idx_t num_vertices;
    idx_t max_degree;
    idx_t* neighbors;      // Neighbor indices for each vertex
    idx_t* degrees;        // Degree of each vertex
    float* distances;      // Distances to neighbors
    float* vertex_radii;   // Radius of each vertex's neighborhood
    idx_t* vertex_centers; // Center vertices for clustering
    bool* is_hub;          // Whether vertex is a hub
};

// Optimized search state with better memory layout
struct CagraOptimizedSearchState {
    idx_t* candidates;     // Current search candidates
    float* candidate_dists; // Distances to candidates
    bool* visited;         // Visited vertices bitmap
    idx_t* search_path;    // Search path for debugging/analysis
    idx_t num_candidates;
    idx_t num_visited;
    idx_t path_length;
};

namespace cagra_optimized {

/**
 * Initialize optimized CAGRA graph structure
 */
__host__ void initialize_graph(CagraOptimizedGraph& graph, idx_t num_vertices, idx_t max_degree);

/**
 * Free optimized CAGRA graph memory
 */
__host__ void free_graph(CagraOptimizedGraph& graph);

/**
 * Build optimized CAGRA graph with multiple refinement iterations
 */
template <class R>
__host__ void build_graph_optimized(const R* points, idx_t num_points, CagraOptimizedGraph& graph);

/**
 * Refine graph connections for better search performance
 */
template <class R>
__host__ void refine_graph(const R* points, idx_t num_points, CagraOptimizedGraph& graph);

/**
 * Search for k nearest neighbors using optimized CAGRA
 */
template <class R>
__host__ void search_knn_optimized(const R* query_points, idx_t num_queries, 
                                  const CagraOptimizedGraph& graph, const R* data_points,
                                  idx_t k, idx_t* results, R* distances);

/**
 * GPU kernel for optimized graph construction
 */
template <class R>
__global__ void construct_graph_optimized_kernel(const R* points, idx_t num_points,
                                                idx_t* neighbors, idx_t* degrees,
                                                float* distances, float* radii,
                                                idx_t max_degree, idx_t iteration);

/**
 * GPU kernel for graph refinement
 */
template <class R>
__global__ void refine_graph_kernel(const R* points, idx_t num_points,
                                   CagraOptimizedGraph graph, idx_t iteration);

/**
 * GPU kernel for optimized kNN search with early termination
 */
template <class R>
__global__ void search_optimized_kernel(const R* query_points, idx_t num_queries,
                                       const CagraOptimizedGraph graph, const R* data_points,
                                       idx_t k, idx_t* results, R* distances);

/**
 * GPU kernel for hub-based search initialization
 */
template <class R>
__global__ void initialize_search_kernel(const R* query_points, idx_t num_queries,
                                        const CagraOptimizedGraph graph, const R* data_points,
                                        idx_t* initial_candidates, float* initial_dists);

/**
 * Main optimized CAGRA kNN function interface
 */
template <class R>
void knn_gpu_optimized(std::size_t n, R* data, std::size_t q, idx_t* queries, 
                      std::size_t k, idx_t* results_knn, R* results_distances);

/**
 * Batch processing for multiple queries with shared graph
 */
template <class R>
void knn_batch_gpu(std::size_t n, R* data, std::size_t q, idx_t* queries, 
                  std::size_t k, idx_t* results_knn, R* results_distances);

} // namespace cagra_optimized

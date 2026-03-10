# CLOVER App Summary (One Page)

## What it is
CLOVER is a GPU-native exact k-nearest-neighbor (kNN) research codebase that accompanies the paper "CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN". It benchmarks multiple CUDA-based kNN strategies (linear-scan and spatio-graph variants) through one executable, `linear-scans`.

## Who it's for
Primary persona: GPU systems/ML researchers and performance engineers who need to evaluate exact kNN implementations and CUDA kernel optimizations.

## What it does
- Builds a CUDA/C++ executable (`linear-scans`) from `src/*.cu` and KNN-related CUDA headers.
- Supports multiple algorithm modes selected by CLI index (Bitonic, Warpwise, Hubs variants, FAISS variants, Treelogy KD-tree).
- Runs repeatable synthetic experiments by generating random 3D points with a fixed seed.
- Runs mesh-based experiments by scanning `../meshes`, loading point data from `.txt` files, and querying kNN.
- Moves points/queries to GPU memory, dispatches the selected kernel, and returns neighbor IDs + distances.
- Prints timing tuples in release mode and can print computed outputs for inspection.
- Optionally integrates FAISS-based paths behind a build-time `USE_FAISS` flag.

## How it works (repo-evidence architecture)
- Build layer: `CMakeLists.txt` compiles CUDA sources, links cuBLAS, and conditionally links FAISS/OpenMP when `LINK_FAISS` is enabled.
- Orchestration layer: `src/linear-scans.cu` parses the algorithm index, prepares datasets/queries, and loops through benchmark sizes.
- Data flow: host arrays/vectors -> `cudaMemcpy` to device buffers -> algorithm-specific GPU kernel call -> device vectors copied back to host -> stdout timing/results.
- Algorithm layer: CUDA headers in `include/` provide implementations for bitonic, warpwise, hubs, optional FAISS wrappers, and Treelogy integration.
- External services/components: Not found in repo (no web server, DB, queue, or RPC service definitions).

## How to run (minimal)
1. Ensure CUDA + GCC toolchain is installed (README lists CUDA 12.6 and GCC 13.3 for local Linux).
2. `mkdir build && cd build`
3. `cmake -DCMAKE_BUILD_TYPE=Release ..`
4. `make`
5. `./linear-scans 2` (or another algorithm index; running with no/invalid args prints usage/options).

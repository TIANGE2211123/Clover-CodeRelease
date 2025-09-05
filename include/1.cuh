/**
 * 完全优化的C_and_Q_TensorCore实现
 */
 template <class R>
 void C_and_Q_TensorCore_Optimized(std::size_t n, R *data, std::size_t q, 
                                   idx_t *queries, std::size_t k, 
                                   idx_t *results_knn, R *results_distances) {
     idx_t constexpr block_size = 1024;
     idx_t constexpr batch_size = 100000;
     idx_t batch_number = (n + batch_size - 1) / batch_size;
     
     // 静态变量控制输出
     static bool first_run = true;
     if (first_run) {
         std::cout << "C_and_Q_TensorCore_Optimized_START!!" << std::endl;
         first_run = false;
     }
     
     // 初始化cuBLAS
     tensorcore::CublasManager cublas_manager;
     cublasHandle_t cublas_handle = cublas_manager.get();
     
     // 检查Tensor Core支持
     bool tensor_cores_available = check_tensor_core_support();
     if (tensor_cores_available) {
         std::cout << "Tensor Cores detected, using optimized implementation." << std::endl;
     }
     
     // 内存分配（与原实现相同）
     idx_t *dH, *dH_psum, *dH_psum_copy, *dH_assignments, *d_psum_placeholder;
     float *distances, *arr_x, *arr_y, *arr_z, *D, *dD;
     idx_t *arr_idx, *iD;
     
     // ... 内存分配代码保持不变 ...
     CUDA_CALL(cudaMalloc((void **) &dH, sizeof(idx_t) * H));
     CUDA_CALL(cudaMalloc((void **) &dH_psum, sizeof(idx_t) * (H + 1)));
     CUDA_CALL(cudaMalloc((void **) &dH_psum_copy, sizeof(idx_t) * (H + 1)));
     CUDA_CALL(cudaMalloc((void **) &d_psum_placeholder, sizeof(idx_t) * (H + 1)));
     CUDA_CALL(cudaMalloc((void **) &dH_assignments, sizeof(idx_t) * n));
     CUDA_CALL(cudaMalloc((void **) &distances, sizeof(float) * H * batch_size));
     CUDA_CALL(cudaMalloc((void **) &arr_x, sizeof(float) * n));
     CUDA_CALL(cudaMalloc((void **) &arr_y, sizeof(float) * n));
     CUDA_CALL(cudaMalloc((void **) &arr_z, sizeof(float) * n));
     CUDA_CALL(cudaMalloc((void **) &arr_idx, sizeof(idx_t) * n));
     CUDA_CALL(cudaMalloc((void **) &D, sizeof(float) * H * H));
     CUDA_CALL(cudaMalloc((void **) &iD, sizeof(idx_t) * H * H));
     CUDA_CALL(cudaMalloc((void **) &dD, sizeof(float) * H * H));
     
     // 初始化
     cudaMemset(dH_psum, 0, sizeof(idx_t) * (H + 1));
     cudaMemset(dH_psum_copy, 0, sizeof(idx_t) * (H + 1));
     cudaMemset(d_psum_placeholder, 0, sizeof(idx_t) * (H + 1));
     cudaMemset(dH_assignments, 0, sizeof(idx_t) * n);
     
     // Hub选择
     std::size_t num_blocks = (H + block_size - 1) / block_size;
     Randomly_Select_Hubs<<<num_blocks, block_size>>>(n, dH);
     CHECK_ERROR("Randomly_Select_Hubs");
     
     set_max_float<<<(H * H + block_size - 1) / block_size, block_size>>>(D, H * H);
     CHECK_ERROR("set_max_float");
     
     // 创建优化的Tensor Core管理器
     std::unique_ptr<OptimizedTensorCoreManager> tc_manager;
     if (tensor_cores_available && should_use_tensor_cores(batch_size, H)) {
         tc_manager = std::make_unique<OptimizedTensorCoreManager>(batch_size, H, cublas_handle);
         tc_manager->initialize_hubs(data, dH); // 一次性初始化hub数据
     }
     
     // 批处理
     for (idx_t batch_id = 0; batch_id < batch_number; batch_id++) {
         smart_distance_calculation_optimized(
             batch_id, batch_size, n, dH, distances, data,
             dH_psum, dH_assignments, cublas_handle, tc_manager.get()
         );
         
         Construct_D<<<H, block_size>>>(distances, dH_assignments, batch_id, batch_size, n, D);
         CHECK_ERROR("Construct_D");
     }
     
     // 后续处理保持不变...
     cudaFree(distances);
     
     // 构建空间索引
     fused_prefix_sum_copy<<<1, dim3(warp_size, warp_size, 1)>>>(dH_psum, dH_psum_copy);
     cudaMemcpy(d_psum_placeholder, dH_psum_copy, (H + 1) * sizeof(idx_t), cudaMemcpyDeviceToDevice);
     cudaMemcpy(dH_psum_copy + 1, d_psum_placeholder, H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
     cudaMemcpy(dH_psum + 1, d_psum_placeholder, H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
     cudaMemset(dH_psum, 0, sizeof(idx_t));
     cudaMemset(dH_psum_copy, 0, sizeof(idx_t));
     cudaFree(d_psum_placeholder);
     
     num_blocks = (n + block_size - 1) / block_size;
     BucketSort<<<num_blocks, block_size>>>(n, arr_x, arr_y, arr_z, arr_idx, data, dH_assignments, dH_psum_copy);
     cudaFree(dH_psum_copy);
     
     fused_transform_sort_D<float, (H + block_size - 1) / block_size>
         <<<H, dim3{warp_size, block_size/warp_size, 1}>>>(D, iD, dD);
     cudaFree(D);
     
     // 查询执行
     int *d_hubsScanned, *d_pointsScanned;
     CUDA_CALL(cudaMalloc((void **) &d_hubsScanned, sizeof(int) * 1));
     CUDA_CALL(cudaMalloc((void **) &d_pointsScanned, sizeof(int) * 1));
     
     std::size_t constexpr queries_per_block = 128 / warp_size;
     num_blocks = util::CEIL_DIV(n, queries_per_block);
     
     // 查询kernel启动逻辑保持不变...
     switch (util::CEIL_DIV(k, warp_size))
     {
         case 1: {  // k <= 32: Use 32 registers per thread
             Query<32, 2, 128> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(
                 queries, results_knn, results_distances, k, n, data, 
                 dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, 
                 d_hubsScanned, d_pointsScanned
             ); 
         } break;
         case 2: {  // 33 <= k <= 64: Use 64 registers per thread
             Query<64, 3, 128> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(
                 queries, results_knn, results_distances, k, n, data, 
                 dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, 
                 d_hubsScanned, d_pointsScanned
             ); 
         } break;
         case 3: {  // 65 <= k <= 96: Use 128 registers per thread
             Query<128, 3, 128> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(
                 queries, results_knn, results_distances, k, n, data, 
                 dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, 
                 d_hubsScanned, d_pointsScanned
             ); 
         } break;
         case 4: {  // 97 <= k <= 128: Use 256 registers per thread
             Query<256, 4, 128> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(
                 queries, results_knn, results_distances, k, n, data, 
                 dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, 
                 d_hubsScanned, d_pointsScanned
             ); 
         } break;
         case 5: {  // 129 <= k <= 160: Use 512 registers per thread
             Query<512, 8, 128> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(
                 queries, results_knn, results_distances, k, n, data, 
                 dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, 
                 d_hubsScanned, d_pointsScanned
             ); 
         } break;
         default: 
             // Assertion failure: k value too large for available thread registers
             assert(false && "Rounds required to fulfill k value will exceed thread register allotment.");
     }
     
     CHECK_ERROR("Running scan kernel");
     
     // 清理
     cudaFree(iD); cudaFree(dD); cudaFree(dH_psum); cudaFree(dH_assignments);
     cudaFree(arr_idx); cudaFree(arr_x); cudaFree(arr_y); cudaFree(arr_z);
     cudaFree(dH); cudaFree(d_hubsScanned); cudaFree(d_pointsScanned);
     
     static bool completion_shown = false;
     if (tensor_cores_available && !completion_shown) {
         std::cout << "Optimized Tensor Core processing completed successfully." << std::endl;
         completion_shown = true;
     }
 }
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<int>(result) 
                  << " (" << cudaGetErrorString(result) << ") in " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS_ERROR(val) check_cublas_error((val), #val, __FILE__, __LINE__)
void check_cublas_error(cublasStatus_t status, const char* func, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error at " << file << ":" << line << " in " << func << ": ";
        switch(status) {
            case CUBLAS_STATUS_NOT_INITIALIZED: std::cerr << "Not initialized"; break;
            case CUBLAS_STATUS_ALLOC_FAILED: std::cerr << "Allocation failed"; break;
            case CUBLAS_STATUS_INVALID_VALUE: std::cerr << "Invalid value"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH: std::cerr << "Architecture mismatch"; break;
            case CUBLAS_STATUS_MAPPING_ERROR: std::cerr << "Memory mapping error"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: std::cerr << "Execution failed"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR: std::cerr << "Internal error"; break;
            case CUBLAS_STATUS_NOT_SUPPORTED: std::cerr << "Operation not supported"; break;
            default: std::cerr << "Unknown error";
        }
        std::cerr << " (code " << static_cast<int>(status) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // 显示设备信息
    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, 0));
    std::cout << "Using device: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Global memory: " << props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Max shared memory per block: " << props.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    
    // 设置矩阵尺寸 - 确保尺寸是64的倍数以优化Tensor Core性能
    const size_t M = 8192;  // 行数（结果矩阵）
    const size_t N = 8192;  // 列数（结果矩阵）
    const size_t K = 8192;  // 公共维度
    std::cout << "\nMatrix dimensions: A=" << M << "x" << K 
              << ", B=" << K << "x" << N 
              << ", C=" << M << "x" << N << std::endl;
    
    // 计算FLOP（避免整数溢出）
    const double flopsPerMatrixMul = 2.0 * static_cast<double>(M) * N * K;
    std::cout << "FP Operations per GEMM: " << flopsPerMatrixMul << " FLOP" << std::endl;

    // 分配主机内存（使用半精度）
    std::vector<__half> h_A(M * K);
    std::vector<__half> h_B(K * N);
    std::vector<__half> h_C(M * N);
    
    // 初始化矩阵为可重复的随机值
    for (size_t i = 0; i < h_A.size(); i++) 
        h_A[i] = __float2half(1.0f);
    for (size_t i = 0; i < h_B.size(); i++) 
        h_B[i] = __float2half(1.0f);
    for (size_t i = 0; i < h_C.size(); i++) 
        h_C[i] = __float2half(2.0f);
    
    // 分配设备内存
    __half *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, h_A.size() * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, h_B.size() * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, h_C.size() * sizeof(__half)));
    
    // 拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), cudaMemcpyHostToDevice));
    
    // 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 设置cuBLAS参数
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 使用Tensor Core配置
    cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    
    // 预热运行（确保所有初始化完成）
    std::cout << "\nRunning warm-up GEMM operation..." << std::endl;
    CHECK_CUBLAS_ERROR(cublasGemmEx(handle, 
                                  CUBLAS_OP_N, CUBLAS_OP_N, 
                                  M, N, K,  // 正确顺序：M, N, K
                                  &alpha,
                                  d_A, CUDA_R_16F, M,  // A的lda应为行数（M）
                                  d_B, CUDA_R_16F, K,  // B的ldb应为公共维度（K）
                                  &beta, 
                                  d_C, CUDA_R_16F, M, // C的ldc应为行数（M）
                                  computeType, 
                                  algo));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Warm-up completed.\n";
    
    // 执行多次GEMM操作并计时
    const int num_repeats = 50;
    
    // 记录总开始时间
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    for (int i = 0; i < num_repeats; i++) {
        CHECK_CUBLAS_ERROR(cublasGemmEx(handle, 
                                      CUBLAS_OP_N, CUBLAS_OP_N, 
                                      M, N, K,  // 正确顺序：M, N, K
                                      &alpha,
                                      d_A, CUDA_R_16F, M, 
                                      d_B, CUDA_R_16F, K, 
                                      &beta, 
                                      d_C, CUDA_R_16F, M,
                                      computeType, 
                                      algo));
    }
    
    // 记录总结束时间并等待所有操作完成
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    cudaDeviceSynchronize();
    
    // 计算总时间
    float total_time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_time_ms, start, stop));
    
    // 计算性能（TFLOPS = 10^12 浮点运算/秒）
    const double total_flops = flopsPerMatrixMul * num_repeats;
    const double tflops = total_flops / (total_time_ms / 1000.0) * 1e-12;
    const double avg_time_per_op = total_time_ms / num_repeats;
    
    // 显示结果
    std::cout << "\nPerformance Results (FP16 with FP16 accumulation):" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N 
              << " = " << M << "x" << N << std::endl;
    std::cout << "FP operations per GEMM: " << flopsPerMatrixMul << " FLOP" << std::endl;
    std::cout << "Number of repetitions: " << num_repeats << std::endl;
    std::cout << "Total execution time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average time per GEMM: " << avg_time_per_op << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;
    
    // RTX 4090的理论性能参考
    std::cout << "RTX 4090 specifications:" << std::endl;
    std::cout << " - Tensor Core peak (FP16): ~330 TFLOPS" << std::endl;
    std::cout << " - Memory bandwidth: 1008 GB/s" << std::endl;
    std::cout << "Percentage of theoretical peak: " 
              << std::round((tflops / 330.0) * 100.0) << "%" << std::endl;
    
    // 验证矩阵尺寸是否优化
    if (M % 64 != 0 || N % 64 != 0 || K % 64 != 0) {
        std::cout << "\n[WARNING] Matrix dimensions should be multiples of 64 for Tensor Core optimization" << std::endl;
    }

    // 在return前添加：拷贝并打印部分结果
    const int print_size = 10; // 打印10x10的子矩阵

    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // 打印矩阵左上角
    std::cout << "\nPartial output matrix (first 10x10 elements):\n";
    for(int i = 0; i < print_size; ++i) {
        for(int j = 0; j < print_size; ++j) {
            float val = __half2float(h_C[i * N + j]);
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // 验证计算结果（所有元素应为8192）
    std::cout << "\nElement [0][0] value: " 
            << __half2float(h_C[0]) 
            << " (expected: 8192)\n";
    
    // 清理资源
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return 0;
}
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

// nvcc -arch=compute_75 -code=sm_75 -o turing_wmma_matmul turing_wmma_matmul.cu && turing_wmma_matmul

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void naive_matmul_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
        }
        C[i * N + j] = sum;
    }
}

void naive_matmul(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    naive_matmul_kernel<<<gridDim, blockDim>>> (A, B, C, M, N, K);
    checkCudaError(cudaGetLastError(), "naive_matmul_kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "naive_matmul_kernel execution");
}

#define TILE_SIZE 16  // TILE_SIZE for Tensor Core usage (16x16x16)

using namespace nvcuda;
// A of size M x K
// B of size K x N
// C of size M x N
// Assumes dims are divisible by TILE_SIZE
__global__ void tensor_core_matmul_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;
    int global_warp_id = (blockIdx.x * gridDim.x + blockIdx.x) / warpSize;

    int tiled_row = global_warp_id % (M / TILE_SIZE);
    int tiled_col = global_warp_id / (M / TILE_SIZE);

    if (tiled_col >= N / TILE_SIZE || tiled_row >= M / TILE_SIZE) {
        return;
    }

    wmma::fill_fragment(c_frag, 0.0f);
    for (int tiled_k = 0; tiled_k < K / TILE_SIZE; tiled_k++) {
        wmma::load_matrix_sync(a_frag, A + tiled_row * TILE_SIZE * K + (TILE_SIZE * tiled_k), K);
        wmma::load_matrix_sync(b_frag, B + tiled_k * TILE_SIZE * N + TILE_SIZE * tiled_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + tiled_row * N * TILE_SIZE + tiled_col * TILE_SIZE, c_frag, N, wmma::mem_row_major);
}


void tensor_core_matmul(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 blockDim(32 * 4);
    // only schedule on x dim
    dim3 gridDim((M * N + blockDim.x - 1) / blockDim.x);

    tensor_core_matmul_kernel<<<gridDim, blockDim>>> (A, B, C, M, N, K);
    checkCudaError(cudaGetLastError(), "tensor_core_matmul_kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "tensor_core_matmul_kernel execution");
}


void init_matrix(half* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = __float2half(static_cast<float>(rand()) / RAND_MAX * 5.0f - 2.5f);
    }
}

bool check_result(const float* A, const float* B, int size) {
    bool corr = true;
    for (int i = 0; i < size; i++) {
        // the precision of half is not so good, so we some margin of error
        float error = abs(A[i] - B[i]);
        float magnitude = abs(B[i]) * 0.1f;
        if (error > magnitude) {
            std::cout << "Results do not match at index " << i << " A: " << A[i] << " B: " << B[i] << " Error: " << error << " Magnitude: " << magnitude << std::endl;
            corr = false;
        }
    }
    return corr;
}


int main() {
    int M = 32 * 100; 
    int N = 32 * 100;
    int K = 32 * 200;
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    half *h_A, *h_B;
    float *h_C, *h_C_ref;
    half *d_A, *d_B;
    float *d_C;

    h_A = (half*)malloc(size_A);
    h_B = (half*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    h_C_ref = (float*)malloc(size_C);

    checkCudaError(cudaMalloc(&d_A, size_A), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, size_B), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, size_C), "cudaMalloc d_C");

    srand(time(NULL));
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);

    checkCudaError(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), "cudaMemcpy d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "cudaMemcpy d_B");

    // Use naive CUDA matmul instead of reference matmul
    naive_matmul(d_A, d_B, d_C, M, N, K);
    checkCudaError(cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy h_C_ref");

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    naive_matmul(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Naive Matmul Time: " << elapsedTime / 1000.0f << " seconds" << std::endl;

    checkCudaError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy h_C");

    if (check_result(h_C, h_C_ref, M * N)) {
        std::cout << "Naive Success!" << std::endl;
    } else {
        std::cout << "Naive Failed!" << std::endl;
    }

    cudaEventRecord(start, 0);
    tensor_core_matmul(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tensor Core Matmul Time: " << elapsedTime / 1000.0f << " seconds" << std::endl;

    checkCudaError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy h_C");

    if (check_result(h_C, h_C_ref, M * N)) {
        std::cout << "Tensor Core Success!" << std::endl;
    } else {
        std::cout << "Tensor Core Failed!" << std::endl;
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");

    return 0;
}
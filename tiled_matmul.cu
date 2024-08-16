#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// nvcc -o tiled_matmul tiled_matmul.cu && tiled_matmul

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void reference_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    naive_matmul_kernel<<<gridDim, blockDim>>> (A, B, C, M, N, K);
    checkCudaError(cudaGetLastError(), "naive_matmul_kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "naive_matmul_kernel execution");
}

#define TILE_SIZE 32

// A of size M x K
// B of size K x N
// C of size M x N
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc = 0.0f;
    for (int tiled_k = 0; tiled_k < (K - 1) / TILE_SIZE + 1; tiled_k++) {
        // load shared mem
        if (row < M && (tiled_k * TILE_SIZE + tx) < K) {
            s_a[ty][tx] = A[(tiled_k * TILE_SIZE + tx) + row * K];
        } else {
            s_a[ty][tx] = 0.0f;
        }

        if (col < N && (tiled_k * TILE_SIZE + ty) < K) {
            s_b[ty][tx] = B[col + ((tiled_k * TILE_SIZE + ty) * N)];
        } else {
            s_b[ty][tx] = 0.0f;
        }
        __syncthreads();

        // reduce
        for (int i = 0; i < TILE_SIZE; i++){
            // tx is the row of the submatrix a
            // ty is the column of the submatrix b
            // i is the column of a/row of b
            acc += s_a[ty][i] * s_b[i][tx];
        }

        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    tiled_matmul_kernel<<<gridDim, blockDim>>> (A, B, C, M, N, K);
    checkCudaError(cudaGetLastError(), "tiled_matmul_kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "tiled_matmul_kernel execution");
}


void init_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool check_result(const float* A, const float* B, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(A[i] - B[i]) > 1e-4) {
            std::cout << "Results do not match at index " << i << " A: " << A[i] << " B: " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int M = 32 * 10 + 1; 
    int N = 32 * 15 + 1;
    int K = 32 * 20 + 1;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
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

    reference_matmul(h_A, h_B, h_C_ref, M, N, K);

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
    tiled_matmul(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Tiled Matmul Time: " << elapsedTime / 1000.0f << " seconds" << std::endl;

    checkCudaError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "cudaMemcpy h_C");

    if (check_result(h_C, h_C_ref, M * N)) {
        std::cout << "Tiled Success!" << std::endl;
    } else {
        std::cout << "Tiled Failed!" << std::endl;
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

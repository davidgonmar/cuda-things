#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// REFERENCE https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=shfl#warp-shuffle-functions

#define FULL_MASK 0xffffffff
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__device__ float warp_reduce_max(float val) {
    int lane_id = threadIdx.x % warpSize;
    for (uint16_t offset = 16; offset > 0; offset /= 2) {
        // gets the val of 'this_thread + offset'
        // then compares it with this val.
        // as we can see when this is printed, for (lane_id + offset) > 31, the retrieved value is the same as the current value.
        printf("val: %f, retrieved val: %f, offset: %d, lane_id: %d\n", val, __shfl_down_sync(FULL_MASK, val, offset), offset, lane_id);
        val = max(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (uint16_t offset = 16; offset > 0; offset /= 2) {
        // gets the val of 'this_thread + offset'
        // then adds it to this val.
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}


__global__ void kernel(float *d_out, float *d_in) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x % warpSize;
    float val = d_in[idx];
    float max_val = warp_reduce_max(val);
    float sum_val = warp_reduce_sum(val);
    if (lane_id == 0) {
        printf("block: %d, max: %f, sum: %f, global idx: %d, local idx: %d\n", blockIdx.x, max_val, sum_val, idx, threadIdx.x);
        d_out[idx] = max_val;
        d_out[idx + blockDim.x] = sum_val;
    }
    
}

int main() {
    constexpr size_t n = 128;
    int warpSize;
    cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
    // host memory
    float *h_in = new float[n];
    float *h_out = new float[n * 2];
    memset(h_out, 0, n * 2 * sizeof(float));
    // fill h_in with random integers
    for (size_t i = 0; i < n; i++) {
        int random_int_between_4_and_10 = rand() % 7 + 4;
        h_in[i] = random_int_between_4_and_10;
    }

    // device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * 2 * sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, n * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    kernel<<<1, n>>>(d_out, d_in);

    cudaDeviceSynchronize();

    // copy data from device to host
    cudaMemcpy(h_out, d_out, n * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // there should be n / warp_size / 2 blocks
    for (size_t i = 0; i < n; i+= warpSize) {
        printf("max: %f, sum: %f\n", h_out[i], h_out[i + n]);
    }

    // free memory
    delete[] h_in;
    delete[] h_out;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
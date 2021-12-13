#include <stdio.h>

// functions qualifers:
// __global__ launched by CPU on device (must return void)
// __device__ called from other GPU functions (never CPU)
// __host__ can be executed by CPU
// (can be used together with __device__)

// kernel launch:
// f_name<<<blocks,threads_per_block>>>(p1,... pN)

__global__ void print_from_gpu(void) {
    int tidx = blockIdx.x*blockDim.x+threadIdx.x;
    printf("Hello from device! My threadId = blockIdx.x *blockDim.x + threadIdx.x <=> %d = %d * %d + %d \n",
    tidx, blockIdx.x, blockDim.x, threadIdx.x);
}

int main(void) {
    printf("Hello World from host!\n");

    print_from_gpu<<<2,3>>>();  // <<<blocks, threads_per_block>>>
    cudaDeviceSynchronize();
    printf("-------------------------------\n");
    dim3 grid_dim(2,1,1);
    dim3 block_dim(3,1,1);
    print_from_gpu<<<grid_dim, block_dim>>>();  // <<<blocks, threads_per_block>>>
    cudaDeviceSynchronize();
    return 0;
}
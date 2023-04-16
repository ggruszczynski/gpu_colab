#include <stdio.h>
#include <assert.h>


//cudaMemcpy (void ∗dst, const void ∗src, size t count, enum cudaMemcpyKind kind)
#define MAX_THREADS_IN_BLOCK 1024

#define MAX_ERR 1e-6

using namespace std;

void cpu_vector_add(float *h_out, float *h_a, float *h_b, int n) {
    int tid = 0; // this is CPU zero, so we start at zero
    while (tid < n)
    {
        h_out[tid] = h_a[tid] + h_b[tid];
        tid += 1;  // we have one CPU, so we increment by one
    }

    // same, using the for loop
    // for(int i = 0; i < n; i++){
    //     h_out[i] = h_a[i] + h_b[i];
    // }
}

__global__ void gpu_vector_add(float *out, float *a, float *b, int n) {
    // built-in variable blockDim.x describes amount threads per block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // check if still inside array
    if (tid < n)
        out[tid] = a[tid] + b[tid];

    // more advanced version - handling arbitrary vector/kernel size
    // Consider case when gridDim*blockDim < vector size
    // int step = gridDim.x * blockDim.x;
    // while (tid < n)
    // {
    //     out[tid] = a[tid] + b[tid];
    //     tid += step;
    // }

    // same, using the for loop
    // for(; tid < n; tid += step){
    //     out[tid] = a[tid] + b[tid];
    // }
}

void CPU_version_wrapper(const int N)
{
    float *h_a, *h_b, *h_out; 

    // Allocate host memory (RAM for CPU)
    h_a   = (float*)malloc(sizeof(float) * N);
    h_b   = (float*)malloc(sizeof(float) * N);
    h_out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        h_a[i] = 1.0; 
        h_b[i] = 2.0;
    }

    // Main function
    cpu_vector_add(h_out, h_a, h_b, N);

    for(int i = 0; i < N; i++){
        assert(fabs(h_out[i] - h_a[i] - h_b[i]) < MAX_ERR);
    }
    printf("CPU assertion PASSED\n");
    printf("CPU Last element in the array: out[%d] = %.2f\n\n",N-1,  h_out[N-1]);

    // Cleanup host memory
    free(h_a); free(h_b); free(h_out);
}

void GPU_version_wrapper(const int N)
{

    // Allocate CPU memory
    float *h_a, *h_b, *h_out; 
    h_a   = (float*)malloc(sizeof(float) * N);
    h_b   = (float*)malloc(sizeof(float) * N);
    h_out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        h_a[i] = 1.0; 
        h_b[i] = 2.0;
    }
    
    // Allocate device memory for d_a
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device  (global) memory
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function: Call the kernel
    gpu_vector_add<<<1,MAX_THREADS_IN_BLOCK>>>(d_out, d_a, d_b, N);//  <<<blocks, threads_per_block>>>
   
    // implement a kernel for which gridDim*blockDim < vector size
    // gpu_vector_add<<<2,64>>>(d_out, d_a, d_b, N);//  <<<blocks, threads_per_block>>>

    // if N is a friendly multiplier of THREADS_PER_BLOCK
    // gpu_vector_add<<<N/MAX_THREADS_IN_BLOCK,MAX_THREADS_IN_BLOCK>>>(d_out, d_a, d_b, N);
    
    // if N is not a friendly multiplier of THREADS_PER_BLOCK
    // gpu_vector_add<<<(N + MAX_THREADS_IN_BLOCK-1) / MAX_THREADS_IN_BLOCK, MAX_THREADS_IN_BLOCK>>>(d_out, d_a, d_b, N);

    // Transfer data from device (global) memory to host
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy() Blocks the CPU until the copy is complete
    // Copy begins when all preceding CUDA calls have completed

    // Verification
    printf("GPU Last element in the array: out[%d] = %.2f\n",N-1,  h_out[N-1]);
    for(int i = 0; i < N; i++){
        assert(fabs(h_out[i] - h_a[i] - h_b[i]) < MAX_ERR);
    }

    printf("GPU assertion PASSED\n\n");
    
    // Cleanup memory after kernel execution
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_out);
    free(h_a);free(h_b);free(h_out);
}



int main(){
    const int N = 1024;
    CPU_version_wrapper(N);
    GPU_version_wrapper(N);

    return 0;
}


// experiment with inappriopate size of array/number of threads

// $ nvcc ex2_vector_add.cu -o ex2_vector_add
// $ cuda-memcheck ./ex2_vector_add 

// ========= CUDA-MEMCHECK
// CPU Result: 3
// ========= Invalid __global__ read of size 4
// =========     at 0x00000050 in gpu_vector_add(float*, float*, float*, int)
// =========     by thread (10,0,0) in block (9,0,0)
// =========     Address 0xb021c01b4 is out of bounds

//https://devtalk.nvidia.com/default/topic/978550/cuda-programming-and-performance/maximum-number-of-threads-on-thread-block/


// check the profiler log
// nvprof ./ex2_vector_add

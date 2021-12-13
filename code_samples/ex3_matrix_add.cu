#include <stdio.h>

#define MAX_THREADS_IN_BLOCK 1024

void cpu_add_matrix_elementwise (float *a, float *b, float  *c, int N)
{
    int index;
    for (int i=0; i <N; ++i)
    {
        for (int j=0; j <N; ++j)
        {
            index = i + j*N;
            c[index] = a[index] + b[index];
        }
    }   
}

void __global__ gpu_matrix_add_elementwise (float *a, float *b, float  *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = i + j*N;

    // check if still inside array
    if (index < N*N)
        c[index] = a[index] + b[index];
}

void  __global__ gpu_matrix_add_elementwise_naive(float *a, float *b, float  *c, int n) {
    // built-in variable blockDim.x describes amount threads per block

    // check if still inside array
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n*n)
        c[tid] = a[tid] + b[tid];
}

void print_matrix(float *Matrix, const int N)
{
    for (int i=0; i <N; ++i)
    {
        printf("\n");
        for (int j=0; j <N; ++j)
        {
            int index = i + j*N;
            printf(" %.2f ", Matrix[index]);
        }
    }  
}

void CPU_version_wrapper(const int N)
{
    const int mem_size = N*N*sizeof(float);

    float* A = (float*)malloc(mem_size);
    float* B = (float*)malloc(mem_size);
    float* C = (float*)malloc(mem_size);
    // initialize data
    for (int i=0; i <N; ++i)
    {
        for (int j=0; j <N; ++j)
        {
            int index = i + j*N;
            A[index] = 2.*index;
            B[index] = 3.*index;
        }
    }

    // run calculations
    cpu_add_matrix_elementwise(A,B,C,N);
    print_matrix(C, N);

    // Free memory
    free(A); free(B); free(C);
}

void GPU_version_wrapper(const int N)
{

    // Allocate CPU memory
    float *h_a, *h_b, *h_out; 
    const int mem_size = N*N*sizeof(float);
    h_a   = (float*)malloc(mem_size);
    h_b   = (float*)malloc(mem_size);
    h_out = (float*)malloc(mem_size);

    // Initialize data
    for (int i=0; i <N; ++i)
    {
        for (int j=0; j <N; ++j)
        {
            int index = i + j*N;
            h_a[index] = 2*index;
            h_b[index] = 3*index;
        }
    }   

    // Allocate device memory for d_a
    float *d_a, *d_b, *d_out;
    cudaMalloc((void**)&d_a, mem_size);
    cudaMalloc((void**)&d_b, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);

    int blockSize; 
    if (N < MAX_THREADS_IN_BLOCK)
        blockSize = N;
    else
        blockSize = MAX_THREADS_IN_BLOCK;

    dim3 dimBlock(blockSize, blockSize); // This variable describes number of threads in the block in each dimension.
    int gridSize = (N + blockSize-1) / blockSize;
    dim3 dimGrid(gridSize, gridSize);    // This variable describes number of blocks in the grid in each dimension.      
    
    // Run the kernel
    gpu_matrix_add_elementwise<<<dimGrid,dimBlock>>>(d_a, d_b, d_out, N);
    // my_kernel<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(args) when the N (size of problem) is a friendly multiplier of THREADS_PER_BLOCK
    // my_kernel<<<(N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(args);  when the N (size of problem) is not a friendly multiplier of THREADS_PER_BLOCK
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);
    
    print_matrix(h_out, N);

    // Free memory
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
}

int main(){
    const int N = 8;
    CPU_version_wrapper(N);
    printf("\n----------------------------------\n");
    GPU_version_wrapper(N);
    printf("\n");
    return 0;
}

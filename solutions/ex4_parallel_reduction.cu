#include <stdio.h>
#include <assert.h>
#define MAX_THREADS_IN_BLOCK 1024

/* function to generate and return random numbers */
int *GetRandomVector(int vectorSize) {

	int *r = (int*)malloc(vectorSize * sizeof(int));

	/* set the seed */
	// srand((unsigned)time(NULL)); // random seed
    // srand(123); // same seed
	for (int i = 0; i < vectorSize; ++i) {
		r[i] = 1; //rand();
		// printf("r[%d] = %d\n", i, r[i]);
	}

	return r;
}


void cpu_sum(int *x, int n)
{
    int result = 0;
    for(unsigned int i=0; i < n; ++i) { 
        result += x[i];
    }
    printf("CPU Sum is %d \t(there are %d elements in the array)\n",result, n);
}

//Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
// When declaring a variable in shared memory as an external array such as 
// extern __shared__ int sdata[];
// the size of the array is determined at launch time (see Execution Configuration). 
// All variables declared in this fashion, start at the same address in memory, 
// so that the layout of the variables in the array must be explicitly managed through offsets.

extern __shared__ int sdata[];

__global__ void gpu_shared_mem_sum(int *g_idata, int *g_odata)
{

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // Reduction #1: Interleaved Addressing - using for loop
    // highly divergent warps are very inefficient, and % operator is very slow
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            // printf("sdata[%d] = %d \n", tid,x[tid] );
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // // Reduction #1: Interleaved Addressing - using while loop
    // // highly divergent warps are very inefficient, and % operator is very slow
    // int step  = 1;
    // while (step < blockDim.x) {
    //     if (tid % (2*step) == 0){   
    //         //printf("tid = %d \n", tid);
    //         sdata[tid] += sdata[tid+step];
    //     }   
    //     __syncthreads(); 
    //     step *= 2;
    // }


    // // Reduction #2: Interleaved Addressing 
    // // strided index and non-divergent branch
    // for (unsigned int s=1; s < blockDim.x; s *= 2) {
    //     int index = 2*s*tid;
    //     if (index < blockDim.x) {
    //         sdata[index] += sdata[index + s];
    //     }
    //     __syncthreads();
    // }

    // Reduction #3: Sequential Addressing
    // for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    //     if (tid < s) {
    //         sdata[tid] += sdata[tid + s];
    //     }
    //     __syncthreads();
    // }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];   
}

__global__ void gpu_sum(int *x)
{   
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // without shared mem
    int N = gridDim.x * blockDim.x;

    // printf("Hello from device! My threadId = blockIdx.x *blockDim.x + threadIdx.x <=> %d = %d * %d + %d \n",
    // tid blockIdx.x, blockDim.x, threadIdx.x);

    // Reduction #1: Interleaved Addressing - using for loop
    // highly divergent warps are very inefficient, and % operator is very slow
    for(unsigned int s=1; s < N; s *= 2) {
        if (tid % (2*s) == 0) {
            // printf("sdata[%d] = %d \n", tid,x[tid] );
            x[tid] += x[tid + s];
        }
        __syncthreads();
    }
    
    // // Reduction #1: Interleaved Addressing - using while loop
    // // highly divergent warps are very inefficient, and % operator is very slow
    // int step  = 1;
    // while (step < N) {
    //     if (tid % (2*step) == 0){   
    //         //printf("tid = %d \n", tid);
    //         x[tid] += x[tid+step];
    //     }   
    //     __syncthreads(); 
    //     step *= 2;
    // }

    // // Reduction #2: Interleaved Addressing 
    // // strided index and non-divergent branch
    // for (unsigned int s=1; s < N; s *= 2) {
    //     int index = 2*s*tid;
    //     if (index < N) {
    //         x[index] += x[index + s];
    //     }
    //     __syncthreads();
    // }

    // Reduction #3: Sequential Addressing
    // for (unsigned int s=N/2; s>0; s>>=1) {
    //     if (tid < s) {
    //         x[tid] += x[tid + s];
    //     }
    //     __syncthreads();
    // }
}


void gpu_global_mem_wrapper(int gridSize, int blockSize, int N)
{
    // int h[] = {10, 1, 8, -1, 0, -2, 3, 5, -2, -3, 2, 7, 0, 11, 0, 2};
    // int size = sizeof(h);
    // int N = size/sizeof(int);

    int *h = GetRandomVector(N);
    int d_glob_mem_size = N*sizeof(int);

	int* d;
	cudaMalloc((void**)&d, d_glob_mem_size);
	cudaMemcpy(d, h, d_glob_mem_size, cudaMemcpyHostToDevice);

    gpu_sum <<<gridSize, blockSize>>>(d);
    
    int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("GPU Sum is %d \n",result);
    cpu_sum(h, N);

    // int* processed_d   = (int*)malloc(size);
    // cudaMemcpy(processed_d, d, size, cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < count; ++i)
    //     printf("processed_d[%d]=%d \n", i, processed_d[i]);

    cudaFree(d);
}

void gpu_shared_mem_sum_wrapper(int gridSize, int blockSize, int N)
{
    // int h[] = {10, 1, 8, -1, 0, -2, 3, 5, -2, -3, 2, 7, 0, 11, 0, 2};
    // int size = sizeof(h);
    // N = size/sizeof(int);

    int *h = GetRandomVector(N);
    int d_glob_mem_size = N*sizeof(int);

	int* d;
	cudaMalloc((void**)&d, d_glob_mem_size);
	cudaMemcpy(d, h, d_glob_mem_size, cudaMemcpyHostToDevice);

    int d_shared_mem_size = d_glob_mem_size/gridSize;
    int* h_subsum  = (int*)malloc(d_shared_mem_size);
    int* d_subsum;
    cudaMalloc((void**)&d_subsum, d_shared_mem_size);

    gpu_shared_mem_sum <<<gridSize, blockSize, d_shared_mem_size >>>(d, d_subsum);  // <<<blocks, threads_per_block, size_t sharedMem (bytes) >>>
    
    int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_subsum, d_subsum, d_shared_mem_size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < gridSize; ++i)
        printf("GPU block sub-sum: h_subsum[%d]=%d \n", i, h_subsum[i]);

    cudaFree(d);

    cpu_sum(h, N);
}

int main()
{   
    int N = 2*1024;         // N must be a power of 2
    int gridSize = 4;       // This variable describes number of blocks in the grid in each dimension.  
    int blockSize = 512;    // This variable describes number of threads in the block in each dimension.
    assert(gridSize*blockSize == N);

    gpu_global_mem_wrapper(gridSize, blockSize, N);
    printf("\n-------\n");
    gpu_shared_mem_sum_wrapper(gridSize, blockSize, N);
	return 0;
}

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
// Derived somewhat from code developed by Patrick Rogers, UNC-C
// http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#include <cassert>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 512
#define THREADS 1024
#define PI 3.14159265358979323846264338327  // known value of pi


typedef double real_t;
// typedef float real_t;

__global__ void gpu_monte_carlo(real_t *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	real_t x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(unsigned int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0 * points_in_circle / (real_t) TRIALS_PER_THREAD; // return estimate of pi
}

real_t host_monte_carlo(long trials) {
	real_t x, y;
	unsigned long points_in_circle;
	for(unsigned long i = 0; i < trials; i++) {
		x = rand() / (real_t) RAND_MAX;
		y = rand() / (real_t) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0);
	}
	return 4.0 * points_in_circle / (real_t) trials;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;
	double mem_size = (double)BLOCKS * THREADS * sizeof(real_t)/1000000.0 ;
	printf("Trying to allocate BLOCKS * THREADS * sizeof(real_t)/1E6 = %lf [MB].\n", mem_size);
	
	real_t host[BLOCKS * THREADS];
	real_t *dev;
	curandState *devStates;

	printf("No of trials per thread = %d, No of blocks = %d, No of threads/block = %d.\n", TRIALS_PER_THREAD, BLOCKS, THREADS);

	
	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(real_t)); // allocate device mem. for counts
	
	cudaMalloc((void **)&devStates, THREADS * BLOCKS * sizeof(curandState));

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(real_t), cudaMemcpyDeviceToHost); // return results 

	real_t pi_gpu = 0;
	for(unsigned int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);

	stop = clock();

    printf("GPU pi calculated in %lf s.\n", (stop-start)/(real_t)CLOCKS_PER_SEC);
    real_t pi_cuda_abs_error = pi_gpu - PI;
    printf("CUDA estimate of PI = %lf [error of %lf]\n", pi_gpu, pi_cuda_abs_error);
    assert(abs(pi_cuda_abs_error) < 1e-3);

	// start = clock();
	// real_t pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
	// stop = clock();
	// printf("CPU pi calculated in %f s.\n", (stop-start)/(real_t)CLOCKS_PER_SEC);
	// printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
 
	return 0;
}


// $ ulimit -a
// you will see sth like

// core file size          (blocks, -c) 0
// data seg size           (kbytes, -d) unlimited
// scheduling priority             (-e) 0
// file size               (blocks, -f) unlimited
// pending signals                 (-i) 127698
// max locked memory       (kbytes, -l) 65536
// max memory size         (kbytes, -m) unlimited
// open files                      (-n) 1024
// pipe size            (512 bytes, -p) 8
// POSIX message queues     (bytes, -q) 819200
// real-time priority              (-r) 0
// stack size              (kbytes, -s) 8192  <--- 8*1024*1000, so max array size is double host[1024*1000];
// cpu time               (seconds, -t) unlimited
// max user processes              (-u) 127698
// virtual memory          (kbytes, -v) unlimited
// file locks                      (-x) unlimited

// $ulimit -s STACK_SIZE to modify the stack size
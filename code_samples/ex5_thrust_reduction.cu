#include <stdio.h>
#include <thrust/device_vector.h>

void cpu_sum(int *x, int n)
{
    int result = 0;
    for(unsigned int i=0; i < n; ++i) { 
        result += x[i];
    }
    printf("CPU Sum is %d \n", result);
}

void gpu_sum(int *x, int n)
{
    thrust::device_vector<int> d_vec(n,0); // initialize all n integers of a device_vector to 0

    for(unsigned int i = 0; i < n; ++i){
        d_vec[i] = x[i];
    }

    int t_sum = thrust::reduce(d_vec.begin(), d_vec.end(), (int) 0, thrust::plus<int>());
    // int t_sum = thrust::reduce(d_vec.begin(), d_vec.begin() + 3, (int) 0, thrust::plus<int>()); // reduce from frist to third element of the array;
    printf("GPU (thrust) Sum is %d \n", t_sum);
}


int main()
{
    int h[] = {10, 1, 8, -1, 0, -2, 3, 5, -2, -3, 2, 7, 0, 11, 0, 2};
    
    int size = sizeof(h);
    int count = size/sizeof(int);

    cpu_sum(h, count);
    gpu_sum(h, count);

	return 0;
}

#include <stdio.h>
#include <thrust/device_vector.h>
#include <iostream>

struct saxpy_functor
{
    const float a;
    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(const float& x, const float& y) const 
    {
        return a * x + y;
    }

};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.end(), saxpy_functor(A));
}

int main(void)
{
    // allocate two device_vectors with 5 elements
    thrust::device_vector<float> X(5);
    thrust::device_vector<float> Y(5);

    // initialize the arrays to 0,1,2,3,4
    thrust::sequence(X.begin(), X.end());
    thrust::sequence(Y.begin(), Y.end());

    saxpy_fast(100, X, Y);

    // print contents of Y
    for(int i = 0; i < Y.size(); i++)
        std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
}

// output
// Y[0] = 0
// Y[1] = 101
// Y[2] = 202
// Y[3] = 303
// Y[4] = 404

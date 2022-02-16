#include <iostream>


template <typename scalar_t>
__device__ void warpReduce(volatile scalar_t *sdata, int tid, int blockSize) {
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (blockSize >= 64) {if (tid < 32) { sdata[tid] += sdata[tid + 32];} __syncthreads(); }
    if (blockSize >= 32) {if (tid < 16) { sdata[tid] += sdata[tid + 16];} __syncthreads(); }
    if (tid == 0)
    {
        for (int i = 1; i < 16; i += 2)
        {
            sdata[0] += sdata[i] + sdata[i+1];
        }
        sdata[0] += sdata[15];
    }
}

template<typename scalar_t>
__global__ void reduceSumKernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int n_elem)
{
    const int absolute_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int tid = threadIdx.x;
    extern __shared__ float sPool[];
    for (int i = absolute_pos; i < n_elem; i += stride)
    {
        sPool[tid] = x[i];
        warpReduce<scalar_t>(sPool, tid, blockDim.x);
        atomicAdd(&y[0], sPool[0]);
    }
}


int main()
{
    const unsigned int elems = 100000000;
    const unsigned int size = elems * sizeof(float);
    float *x, *y;
    cudaMalloc(&x, size);
    cudaMalloc(&y, 1);
    reduceSumKernel<float><<<1, 1024, 1024 * sizeof(float)>>>(x, y, elems);
    cudaFree(x);
    cudaFree(y);
}
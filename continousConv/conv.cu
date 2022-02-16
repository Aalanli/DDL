//#define IsPyExtension

#ifdef IsPyExtension
#include <torch/extension.h>
#else
#include <torch/torch.h>
#include <cuda_profiler_api.h>
#include <iostream>
#endif

#include <vector>


template <typename scalar_t>
__device__ __forceinline__ scalar_t weight_kernel(scalar_t x, scalar_t a, scalar_t b) {
    return 1 / (1 + pow(abs(x / a), 2 * b));
}

template <typename scalar_t>
__device__ __forceinline__ void d_weight_kernel(scalar_t x, scalar_t a, scalar_t b, scalar_t* w, scalar_t* dA, scalar_t* dB) {
    auto v = pow(abs(x / a), 2 * b);
    w[0] = 1 / (1 + v);
    auto vp = 2 / (1 / v + 2 + v);
    dA[0] = b * vp / a;
    dB[0] = -log(abs(x / a) + 1e-6) * vp;
} 

template <typename scalar_t>
__device__ __forceinline__ scalar_t cut_off_thres(scalar_t a, scalar_t b, float thres) {
    return a * pow(1/thres-1,1/(2*b));
}

template <typename scalar_t>  // expects length(sPool) == length(nThreads), max(tid) == nThreads - 1
__device__ __forceinline__ void reduceSum(volatile scalar_t* sPool, int tid, int nThreads) {

}

/*
simplified kernel, assumes that there is no stride in the channel dimension
*/
template <typename scalar_t>
__global__ void conv2d_kernel_v1( 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const scalar_t bS,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> y,
    const int* __restrict__ blockDims, // [batch, (y, x)] num blocks in y and x dimensions 
    const int* __restrict__ nKernels,
    const int* __restrict__ kernelW,
    const scalar_t* __restrict__ thresholds, // [batch, (thresY, thresX)]
    const float threadKPercent,
    const float cutThres)
{
    const int block = blockIdx.x;
    int b = 0;
    int tb = 0;
    while (block < blockDims[b * 2] * blockDims[b * 2 + 1] * x.size(1)) {tb += blockDims[b*2] * blockDims[b*2+1] * x.size(1); b += 1;}
    const int c = (block - tb) / (blockDims[b*2] * blockDims[b*2+1]);
    const int yBlock = (block - tb - c * blockDims[b*2] * blockDims[b*2+1]) / blockDims[b*2+1];
    const int xBlock = (block - tb - c * blockDims[b*2] * blockDims[b*2+1] - yBlock * blockDims[b*2+1]);
    
    const int threadsK = (threadKPercent * kernelW[b][0] * kernelW[b][1]); // number of threads each kernel requires
    const int spanBlock = blockDim.x / threadsK; // number of kernels each block computes
    const int localKernelId = threadIdx.x / threadsK; // the kernel within each block
    const int globalXKernel = xBlock * spanBlock + localKernelId;
    
    const scalar_t yStart = (scalar_t) yBlock * a[b][0] + a[b][0] / 2;
    const scalar_t xStart = (scalar_t) globalXKernel * a[b][1] + a[b][1] / 2;

    const int tid = threadIdx.x;
    const int localTid = tid % threadsK;
    const scalar_t thresY = thresholds[b][0];
    const scalar_t thresX = thresholds[b][1];
    const int yKernelW = kernelW[b][0];
    const int xKernelW = kernelW[b][1];

    const scalar_t aY = a[b][0]/2;
    const scalar_t aX = a[b][1]/2;

    const int indXS = max((int) (xStart - thresX + 0.5), 0);  // start of real index x, real index refers to the index of the pixel on the actual image
    const int indXE = min((int) (thresX + xStart) + 1, x.size(3)); // end of real index x

    const int indYS = max((int) (yStart - thresY + 0.5), 0);  // start of real index y
    const int indYE = min((int) (thresY + yStart) + 1, x.size(2)); // end of real index y
    const int kernelElems = (indXE - indXS) * (indYE - indYS);

    extern __shared__ float sData[]; // [yKernelW + spanBlock * xKernelW + threadsPerCol * xKernelW * spanBlock]
    scalar_t* yWeights = (scalar_t*) sData;
    scalar_t* xWeights = (scalar_t*) (sData + yKernelW + localKernelId * xKernelW);
    scalar_t* sPool = (scalar_t*) (sData + yKernelW + spanBlock * xKernelW + localKernelId * threadsK); // [0, threadsK]

    // calculate weights
    if (tid < yKernelW && tid < indYE) {
        yWeights[tid] = weight_kernel<scalar_t>((scalar_t) (tid + indYS), aY, b);
    }
    if (localTid < xKernelW && localTid < indXE) {
        xWeights[localTid] = weight_kernel<scalar_t>((scalar_t) (localTid + indXS), aX, b);
    }
    __syncthreads();

    sPool[localTid] = 0;
    for (int i=localTid; i < kernelElems; i++) {
        int iy = i / xKernelW;
        int ix = i - iy * xKernelW;
        sPool[localTid] += yWeights[iy] * xWeights[ix] * x[b][c][iy + indYS][ix + indXS];
    }
    __syncthreads();
    // reduce sum
    reduceSum(sPool, localTid, threadsK);
    if (localTid == 0) {
        y[b][c][yBlock][globalXKernel] = sPool[0];
    }
}





#ifndef IsPyExtension

int main() {
    std::cout << "hello";
}

#else

#endif
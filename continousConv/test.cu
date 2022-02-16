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

/*
kernel assumes that each kernel group gets maximum of 1 block
*/
template <typename scalar_t>
__global__ void conv2d_kernel_v1( 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
    const scalar_t bS,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> y,
    const int batch,
    const int ch,
    const int yDim,
    const int xDim,
    const int y1Dim,
    const int x1Dim,
    const int threadsPerCol,
    const float cutThres)
{
    const int b = blockIdx.y;
    const scalar_t thresY = cut_off_thres<scalar_t>(a[b][0], bS, cutThres);
    const scalar_t thresX = cut_off_thres<scalar_t>(a[b][1], bS, cutThres);
    const int yKernels = (yDim - a[b][0]) / a[b][0];  // number of kernel passes across y dimension
    const int xKernels = (xDim - a[b][1]) / a[b][1];  // number of kernel passes across x dimension
    const int yKernelW = 2 * thresY + 0.5;  // width of each y kernel
    const int xKernelW = 2 * thresX + 0.5;  // width of each x kernel

    const int threadsK = xKernelW * threadsPerCol; // num threads for each kernel
    const int spanBlock = blockDim.x / threadsK; // number of kernels each block handles
    const int blocksPerRow = xKernels / spanBlock; // number of blocks per row of 2D image
    const int tid = threadIdx.x;
    const int localBlockid = tid / threadsK; // kernel number within each block
    const int localTid = tid % threadsK; // local thread for each kernel (0 to xKernelW * threadsPerCol - 1)

    const int blocksPerCh = blocksPerRow * yKernels; // number of blocks needed for each 2D channel
    const int channel = blockIdx.x / blocksPerCh; // the current channel
    const int channelStride = gridDim.x / blocksPerCh; // number of channels processed in each step

    const int yKernelId = blockIdx.x % blocksPerRow;
    const int xKernelId = blockIdx.x - channel * blocksPerCh - yKernelId * blocksPerRow + localBlockid;

    const scalar_t yStart = (scalar_t) yKernelId * a[b][0] + a[b][0] / 2;
    const scalar_t xStart = (scalar_t) xKernelId * a[b][1] + a[b][1] / 2;

    const int localX = localTid % xKernelW;
    const int indXS = max((int) (xStart - thresX + 0.5), 0);  // start of real index x, real index refers to the index of the pixel on the actual image
    const int indXE = min((int) (thresX + xStart) + 1, xDim); // end of real index x

    const int localY = localTid / xKernelW;
    const int indYS = max((int) (yStart - thresY + 0.5), 0);  // start of real index y
    const int indYE = min((int) (thresY + yStart) + 1, yDim); // end of real index y

    extern __shared__ float sData[]; // [yKernelW + spanBlock * xKernelW + threadsPerCol * xKernelW * spanBlock]
    scalar_t* yWeights = (scalar_t*) sData;
    scalar_t* xWeights = (scalar_t*) (sData + yKernelW + localBlockid * xKernelW);
    scalar_t* sPool = (scalar_t*) (sData + yKernelW + spanBlock * xKernelW + localBlockid * threadsK); // [0, threadsPerCol * xKernelW - 1]

    // calculate weights
    if (tid < yKernelW && tid < indYE) {
        yWeights[tid] = weight_kernel<scalar_t>((scalar_t) (tid + indYS), a[b][0]/2, b);
    }
    if (localTid < xKernelW && localTid < indXE) {
        xWeights[localTid] = weight_kernel<scalar_t>((scalar_t) (localTid + indXS), a[b][1]/2, b);
    }
    __syncthreads();

    for (int c=channel; c < x.size(1); c+=channelStride) {
        sPool[localTid] = 0;
        for (int y=localY + indYS; y < indYE; y += threadsPerCol) {
            if (indXS + localX < indXE) {
                sPool[localTid] += yWeights[y - indYS] * x[b][channel][y][indXS + localX]; 
            }
        }
        sPool[localTid] *= xWeights[localX];
        __syncthreads();
        // reduce sum
        if (localY == 0) {
            
        }
    }

}





#ifndef IsPyExtension

int main() {

}

#else

#endif
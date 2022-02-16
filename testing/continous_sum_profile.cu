#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <iostream>


const float exp_overflow = 15;
const float exp_underflow = -14;

void errorchk()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Cuda error: " << cudaGetErrorString(err) << "\n";
        exit(-1);
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t inv_sigmoid(scalar_t x) {
    return 1.0 / (1.0 + exp(x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t weighted_sigmoid(scalar_t x, scalar_t theta, scalar_t stride, scalar_t beta) {
    return inv_sigmoid(beta * (x - theta - stride)) - inv_sigmoid(beta * (x - stride));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x) {
    const auto s = inv_sigmoid(x);
    return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ int overflow(scalar_t theta, scalar_t stride, scalar_t beta) {
    return exp_overflow / beta + stride + theta;
}

template <typename scalar_t>
__device__ __forceinline__ int underflow(scalar_t stride, scalar_t beta) {
    return exp_underflow / beta + stride;
}

template <typename scalar_t>
__device__ __forceinline__ void reduceOneLine(volatile scalar_t *sdata, int tid, int blockSize, int division) {
    if (blockSize >= division * 2) { 
        if (tid < division && tid + division < blockSize) {
            sdata[tid] += sdata[tid + division];
            sdata[tid + blockSize] += sdata[tid + blockSize + division];
            sdata[tid + blockSize * 2] += sdata[tid + blockSize * 2 + division];
        }
        __syncthreads();
    }
}


template <typename scalar_t>
__global__ void d_continous_sum_kernel(
    const scalar_t* __restrict__  y_grad,
    const scalar_t* __restrict__  x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__  dx,
    scalar_t* __restrict__ dTheta,
    scalar_t* __restrict__ dStride,
    scalar_t* __restrict__ dBeta,
    const int batch_stride,
    const int cols,
    const int rows,
    const int ybatch_stride)
{
    const int batch_id = blockIdx.z;
    const int yCol = blockIdx.y;
    scalar_t nconv = yCol * stride[0];

    const int block_stride = gridDim.x;
    const int block_size = blockDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;

    const int y_pos = batch_id * ybatch_stride + yCol * rows + tid;

    int uFlow = max(0, underflow<scalar_t>(nconv, beta[0]));
    int oFlow = min(cols, overflow<scalar_t>(theta[0], nconv, beta[0]));
    __shared__ scalar_t d[4];
    extern __shared__ scalar_t sPool[];
    sPool[tid] = 0; // dtheta
    sPool[tid + block_size] = 0; // dstride
    sPool[tid + block_size * 2] = 0; // dbeta
    for (int i = block + uFlow; i < oFlow; i += block_stride)
    {
        // i = col number
        int x_pos = batch_id * batch_stride + i * rows + tid;

        if (tid == 0) {
            scalar_t pos = (scalar_t) i;
            scalar_t s = d_sigmoid<scalar_t>(beta[0] * (pos - theta[0] - stride[0]));
            scalar_t s1 = d_sigmoid<scalar_t>(beta[0] * (pos - stride[0]));
            // d[dx, dt, ds, db]
            d[0] = weighted_sigmoid<scalar_t>(pos, theta[0], stride[0], beta[0]);
            d[1] = s * beta[0];
            d[2] = (s1 - s) * beta[0];
            d[3] = s * (pos - theta[0] - stride[0]) - s1 * (pos - stride[0]);
        }
        __syncthreads();
        atomicAdd(&dx[x_pos], y_grad[y_pos] * d[0]);
        sPool[tid] += d[1] * x[x_pos];
        sPool[tid + block_size] += d[2] * x[x_pos];
        sPool[tid + block_size * 2] += d[3] * x[x_pos];
    }
    sPool[tid] *= y_grad[y_pos];
    sPool[tid + block_size] *= y_grad[y_pos];
    sPool[tid + block_size * 2] *= y_grad[y_pos];

    // warp reduce
    reduceOneLine<scalar_t>(sPool, tid, block_size, 512);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 256);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 128);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 64);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 32);
    // removes last few __syncthreads(), faster
    if (tid == 0) {
        for (int i = 1; i < 15; i += 2) {
            sPool[0] += sPool[i] + sPool[i+1];
        }
        sPool[0] += sPool[15];
        atomicAdd(&dTheta[0], sPool[0]);
    }
    if (tid == 1) {
        for (int i = 1; i < 15; i += 2) {
            sPool[block_size] += sPool[i + block_size] + sPool[i + block_size + 1];
        }
        sPool[block_size] += sPool[block_size + 15];
        atomicAdd(&dStride[0], sPool[block_size]);
    }
    if (tid == 2) {
        for (int i = 1; i < 15; i += 2) {
            sPool[block_size * 2] += sPool[i + block_size * 2] + sPool[i + block_size * 2 + 1];
        }
        sPool[block_size * 2] += sPool[block_size * 2 + 15];
        atomicAdd(&dBeta[0], sPool[block_size * 2]);
    }

}

template <typename scalar_t>
scalar_t* allocTensor(int bytes)
{
    scalar_t *a_device;
    cudaMalloc(&a_device, sizeof(scalar_t) * bytes);
    return a_device;
}


void d_continous_sum_rowMajor(int batch, int cols, int rows, int blocks_per_batch, float theta, float stride, float beta)
{    
    int ycols = (cols - theta) / stride;
    float *input = allocTensor<float>(batch * cols * rows);
    float *dinput = allocTensor<float>(batch * cols * rows);
    float *t = allocTensor<float>(1);
    float *dt = allocTensor<float>(1);
    float *s = allocTensor<float>(1);
    float *ds = allocTensor<float>(1);
    float *b = allocTensor<float>(1);
    float *db = allocTensor<float>(1);
    float *ygrad = allocTensor<float>(batch * ycols * rows);

    const int threads = rows;
    const int y_seq_len = ycols;
    const dim3 blocks(blocks_per_batch, y_seq_len, batch);

    d_continous_sum_kernel<float><<<blocks, threads, sizeof(float) * threads * 3>>>(
        ygrad, input, t, s, b, dinput, dt, ds, db, cols * rows, cols, rows, rows * ycols 
    );
    errorchk();
    cudaFree(input);
    cudaFree(dinput);
    cudaFree(t);
    cudaFree(dt);
    cudaFree(s);
    cudaFree(ds);
    cudaFree(b);
    cudaFree(db);
    cudaFree(ygrad);
}


int main()
{
    d_continous_sum_rowMajor(1, 128, 128, 16, 5, 5, 2);
}
#define LinkingLibrary

//#define IsPyExtension

#ifdef IsPyExtension
#include <torch/extension.h>
#else
#include <torch/torch.h>
#include <cuda_profiler_api.h>
#endif

#include <vector>

const float e = 2.718281828459045;
//const float exp_overflow = 89;
//const float exp_underflow = -104;

const float exp_overflow = 20;
const float exp_underflow = -20;


void errorchk()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Cuda error: " << cudaGetErrorString(err) << "\n";
        exit(-1);
    }
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
__device__ __forceinline__ scalar_t inv_sigmoid(scalar_t x) {
    return 1.0 / (1.0 + __expf(x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t weighted_sigmoid(scalar_t x, scalar_t theta, scalar_t stride, scalar_t beta) {
    return inv_sigmoid(beta * (x - theta - stride)) - inv_sigmoid(beta * (x - stride));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lifted_softplus(scalar_t x) {
    return log(e + __expf32(x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x) {
    const auto s = inv_sigmoid(x);
    return (1.0 - s) * s;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t d_softplus(scalar_t x) {
    return 1 / (exp(e - x) + 1);
}

template <typename scalar_t>
__device__ void warpReduce(volatile scalar_t *sdata, int tid, int blockSize) {
    if (blockSize >= 1024) { if (tid < 512 && tid + 512 < blockSize) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512)  { if (tid < 256 && tid + 256 < blockSize) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256)  { if (tid < 128 && tid + 128 < blockSize) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128)  { if (tid < 64 && tid + 64 < blockSize)   { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (blockSize >= 64)   { if (tid < 32 && tid + 32 < blockSize)   { sdata[tid] += sdata[tid + 32];} __syncthreads(); }
    if (blockSize >= 32)   { if (tid < 16 && tid + 16 < blockSize)   { sdata[tid] += sdata[tid + 16];} __syncthreads(); }
    // removes last few __syncthreads(), faster
    if (tid == 0)
    {
        for (int i = 1; i < 15; i += 2)
        {
            sdata[0] += sdata[i] + sdata[i+1];
        }
        sdata[0] += sdata[15];
    }
    __syncthreads();
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
__global__ void continous_sum_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ y,
    const int batch, const int col, const int row, const int yCol)
{
    const int batch_id = blockIdx.z;
    scalar_t nconv = (scalar_t) blockIdx.y * stride[0];

    const int block_stride = gridDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;

    const int batch_start = batch_id * col * row;

    int uFlow = max(0, underflow<scalar_t>(nconv, beta[0]));
    int oFlow = min(col, overflow<scalar_t>(theta[0], nconv, beta[0]));
    scalar_t accum = 0;
    __shared__ scalar_t w[1];

    for (int i = block + uFlow; i < oFlow; i += block_stride)
    {
        // i = col number
        if (tid == 0) {
            w[0] = weighted_sigmoid<scalar_t>((scalar_t) i, theta[0], nconv, beta[0]);
        }
        __syncthreads();
        accum += w[0] * x[batch_start + i * row + tid];
    }
    atomicAdd(&y[batch_id * yCol * row + blockIdx.y * row + tid], accum);
}


torch::Tensor continous_sum_cuda(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    int blocks_per_batch)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = theta.cpu().item<float>();
    float s_ = stride.cpu().item<float>();
    int y_seq_len = (seq_len - t_) / s_ + 1;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, y_seq_len, dim}, opt);

    const int threads = dim;
    const dim3 blocks(blocks_per_batch, y_seq_len, batch);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_forward_cuda", ([&] {
        continous_sum_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            theta.data<scalar_t>(),
            stride.data<scalar_t>(),
            beta.data<scalar_t>(),
            y.data<scalar_t>(),
            batch, seq_len, dim, y_seq_len
        );
    }));
    return y;
}


template <typename scalar_t>
__global__ void d_continous_sum_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> y_grad,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx,
    scalar_t* __restrict__ dTheta,
    scalar_t* __restrict__ dStride,
    scalar_t* __restrict__ dBeta)
{
    const int batch_id = blockIdx.z;
    const int yCol = blockIdx.y;
    scalar_t nconv = (scalar_t) yCol * stride[0];

    const int block_stride = gridDim.x;
    const int block_size = blockDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;

    int uFlow = max(0, underflow<scalar_t>(nconv, beta[0]));
    int oFlow = min(x.size(1), overflow<scalar_t>(theta[0], nconv, beta[0]));
    extern __shared__ float sdata[];
    scalar_t* sPool = (scalar_t*) sdata;

    __shared__ scalar_t w;
    __shared__ scalar_t dt;
    __shared__ scalar_t ds;
    __shared__ scalar_t db;

    sPool[tid] = 0; // dtheta
    sPool[tid + block_size] = 0; // dstride
    sPool[tid + block_size * 2] = 0; // dbeta
    for (int r = block + uFlow; r < oFlow; r += block_stride)
    {
        // i = col number
        if (tid == 0)
        {
            w = weighted_sigmoid<scalar_t>((scalar_t) r, theta[0], nconv, beta[0]);
        }
        if (tid == 1) {
            scalar_t pos = (scalar_t) r;
            scalar_t s = d_sigmoid<scalar_t>(beta[0] * (pos - theta[0] - nconv));
            scalar_t s1 = d_sigmoid<scalar_t>(beta[0] * (pos - nconv));
            // d[dx, dt, ds, db]
            dt = s * beta[0];
            ds = (s - s1) * beta[0] * yCol;
            db = s1 * (pos - nconv) - s * (pos - theta[0] - nconv);
        }
        __syncthreads();
        atomicAdd(&dx[batch_id][r][tid], y_grad[batch_id][yCol][tid] * w);
        sPool[tid] += x[batch_id][r][tid] * dt;
        sPool[tid + block_size] += x[batch_id][r][tid] * ds;
        sPool[tid + block_size * 2] += x[batch_id][r][tid] * db;
    }
    sPool[tid] *= y_grad[batch_id][yCol][tid];
    sPool[tid + block_size] *= y_grad[batch_id][yCol][tid];
    sPool[tid + block_size * 2] *= y_grad[batch_id][yCol][tid];

    // warp reduce
    reduceOneLine<scalar_t>(sPool, tid, block_size, 512);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 256);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 128);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 64);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 32);
    reduceOneLine<scalar_t>(sPool, tid, block_size, 16);
    // removes last few __syncthreads(), faster
    if (tid == 0) {
        for (int i = 1; i < 14; i += 2) {
            sPool[0] += sPool[i] + sPool[i+1];
        }
        sPool[0] += sPool[15];
        atomicAdd(&dTheta[0], sPool[0]);
    }
    if (tid == 1) {
        for (int i = 1; i < 14; i += 2) {
            sPool[block_size] += sPool[i + block_size] + sPool[i + block_size + 1];
        }
        sPool[block_size] += sPool[block_size + 15];
        atomicAdd(&dStride[0], sPool[block_size]);
    }
    if (tid == 2) {
        for (int i = 1; i < 14; i += 2) {
            sPool[block_size * 2] += sPool[i + block_size * 2] + sPool[i + block_size * 2 + 1];
        }
        sPool[block_size * 2] += sPool[block_size * 2 + 15];
        atomicAdd(&dBeta[0], sPool[block_size * 2]);
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

std::vector<torch::Tensor> d_continous_sum_cuda(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    int blocks_per_batch)
{
    torch::Tensor d_input = torch::zeros_like(input);
    torch::Tensor d_theta = torch::zeros_like(theta);
    torch::Tensor d_stride = torch::zeros_like(stride);
    torch::Tensor d_beta = torch::zeros_like(beta);

    const int threads = input.size(2);
    const int y_seq_len = grad.size(1);
    const int batch = input.size(0);
    const dim3 blocks(blocks_per_batch, y_seq_len, batch);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_forward_cpu", ([&] {
        d_continous_sum_kernel<scalar_t><<<blocks, threads, sizeof(scalar_t) * threads * 3>>>(
            grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            theta.data<scalar_t>(),
            stride.data<scalar_t>(),
            beta.data<scalar_t>(),
            d_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            d_theta.data<scalar_t>(),
            d_stride.data<scalar_t>(),
            d_beta.data<scalar_t>()
        );
    }));
    return {d_input, d_theta, d_stride, d_beta};
}

#ifndef LinkingLibrary
#ifndef IsPyExtension

int main()
{
    auto i = torch::rand({1, 256, 256}).cuda();
    auto t = torch::tensor({5.0}).cuda();
    auto s = torch::tensor({3.0}).cuda();
    auto b = torch::tensor({2.0}).cuda();
    auto y = continous_sum_cuda(i, t, s, b, 1);
    auto z = d_continous_sum_cuda(i, t, s, b, y, 1);
    errorchk();
    std::cout << (y == 0).all() << (z[0] == 0).all() << "\n";
}


#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("continous_sum_cuda", &continous_sum_cuda, "(CUDA)");
    m.def("d_continous_sum_cuda", &d_continous_sum_cuda, "(CUDA)");
}
#endif
#endif
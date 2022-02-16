#define LinkingLibrary

//#define IsPyExtension

#ifdef IsPyExtension
#include <torch/extension.h>
#else
#include <torch/torch.h>
#include <cuda_profiler_api.h>
#include <iostream>
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
    scalar_t h = beta * (x - stride);
    return 1.0 / (1.0 + __expf(-h) + __expf(h - beta * theta));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t lifted_softplus(scalar_t x, scalar_t a, scalar_t b) {
    return log(a + __expf(x)) + b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x) {
    const auto s = inv_sigmoid(x);
    return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_softplus(scalar_t x, scalar_t a) {
    return 1 / (1 + a * __expf(-x));
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
    scalar_t lifted_a, scalar_t lifted_b,
    const int batch, const int col, const int row, const int yCol)
{
    const int batch_id = blockIdx.z;
    const int nconv = blockIdx.y;

    const int block_stride = gridDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    const int batch_start = batch_id * col * row;
    
    const int pos_r = batch_id * row + tid;
    scalar_t lT = lifted_softplus<scalar_t>(theta[pos_r], lifted_a, lifted_b);
    scalar_t lS = lifted_softplus<scalar_t>(stride[pos_r], lifted_a, lifted_b);
    scalar_t lB = lifted_softplus<scalar_t>(beta[pos_r], lifted_a, lifted_b);
    const int max_conv = (col - lT) / lS + 1;

    if (nconv < max_conv)
    {
        lS *= nconv;
        int uFlow = max(0, underflow<scalar_t>(lS, lB));
        int oFlow = min(col, overflow<scalar_t>(lT, lS, lB));
        scalar_t accum = 0;
        for (int i = block + uFlow; i < oFlow; i += block_stride)
        {
            // i = col number            
            accum += weighted_sigmoid<scalar_t>((scalar_t) i, lT, lS, lB) * x[batch_start + i * row + tid];
        }
        atomicAdd(&y[batch_id * yCol * row + blockIdx.y * row + tid], accum);
    }
}

template <typename scalar_t>
__global__ void continous_sum_kernelv2(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> theta,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stride,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> y,
    scalar_t lifted_a, scalar_t lifted_b)
{
    const int batch_id = blockIdx.z;
    const int dimx = blockIdx.y;
    const int conv_block = blockIdx.x;

    const int conv = threadIdx.y;
    const int tid = threadIdx.x;
    const int conv_stride = blockDim.y;
    const int stride_x = blockDim.x;

    __shared__ scalar_t lT;
    __shared__ scalar_t lS;
    __shared__ scalar_t lB;
    __shared__ int max_conv;
    extern __shared__ float sData[];
    scalar_t *sumPool = (scalar_t*) sData;
    if (tid == 0)
    {
        if (conv == 0)
            lT = lifted_softplus<scalar_t>(theta[batch_id][dimx], lifted_a, lifted_b);
        if (conv == 1)
            lS = lifted_softplus<scalar_t>(stride[batch_id][dimx], lifted_a, lifted_b);
        if (conv == 2)
            lB = lifted_softplus<scalar_t>(beta[batch_id][dimx], lifted_a, lifted_b);
        if (conv == 3)
            max_conv = (x.size(1) - lT) / lS + 1;
    }
    __syncthreads();
    const int conv_start = conv_block * conv_stride + conv;
    const int pool_start = conv * stride_x;
    if (conv_start < max_conv)
    {
        scalar_t lS_local = lS * conv_start;
        int uFlow = max(0, underflow<scalar_t>(lS_local, lB));
        int oFlow = min(x.size(1), overflow<scalar_t>(lT, lS_local, lB));
        sumPool[conv * stride_x + tid] = 0;
        for (int r = tid + uFlow; r < oFlow; r += stride_x)
        {
            sumPool[pool_start + tid] += weighted_sigmoid<scalar_t>((scalar_t) r, lT, lS_local, lB) * x[batch_id][r][dimx];
        }
        // reduce_sum
        if (tid == 0)
        {
            for (int i = 1; i < stride_x; i++)
            {
                sumPool[pool_start] += sumPool[pool_start + i];
            }
            y[batch_id][conv_start][dimx] = sumPool[pool_start];
        }
    }
}

torch::Tensor continous_sum_cuda(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    float lifted_a,
    float lifted_b,
    int blocks_per_batch)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = theta.min().cpu().item<float>();
    float s_ = stride.min().cpu().item<float>();
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
            lifted_a,
            lifted_b,
            batch, seq_len, dim, y_seq_len
        );
    }));
    return y;
}

torch::Tensor continous_sum_cuda_v2(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    float lifted_a,
    float lifted_b,
    int threads_per_conv)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = theta.min().cpu().item<float>();
    float s_ = stride.min().cpu().item<float>();
    int y_seq_len = (seq_len - t_) / s_ + 1;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, y_seq_len, dim}, opt);

    const int conv_blocks = (y_seq_len * threads_per_conv + 1023) / 1024;
    const int conv = (y_seq_len + conv_blocks - 1) / conv_blocks;

    const dim3 threads(threads_per_conv, conv, 1);
    const dim3 blocks(conv_blocks, dim, batch);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_forward_cuda", ([&] {
        continous_sum_kernelv2<scalar_t><<<blocks, threads, sizeof(scalar_t) * threads_per_conv * conv>>>(
            input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            theta.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            stride.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            y.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            lifted_a,
            lifted_b
        );
    }));
    return y;
}

template <typename scalar_t>
__global__ void d_continous_sum_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> y_grad,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> theta,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> stride,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> beta,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dx,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dTheta,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dStride,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dBeta,
    scalar_t lifted_a,
    scalar_t lifted_b)
{
    const int batch_id = blockIdx.z;
    const int yCol = blockIdx.y;

    const int block_stride = gridDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;

    scalar_t lT = lifted_softplus<scalar_t>(theta[batch_id][tid], lifted_a, lifted_b);
    scalar_t lS = lifted_softplus<scalar_t>(stride[batch_id][tid], lifted_a, lifted_b);
    scalar_t lB = lifted_softplus<scalar_t>(beta[batch_id][tid], lifted_a, lifted_b);
    const int max_conv = (x.size(1) - lT) / lS + 1;

    if (yCol < max_conv)
    {
        scalar_t nconv = (scalar_t) yCol * lS;
        int uFlow = max(0, underflow<scalar_t>(nconv, lB));
        int oFlow = min(x.size(1), overflow<scalar_t>(lT, nconv, lB));

        scalar_t dt = 0; scalar_t ds = 0; scalar_t db = 0;

        for (int r = block + uFlow; r < oFlow; r += block_stride)
        {
            scalar_t pos = (scalar_t) r;
            scalar_t ea = __expf(-lB * (pos - nconv));
            scalar_t eb = __expf(lB * (pos - nconv - lT));
            scalar_t e2 = __powf(1 + ea + eb, 2);
            // d[dx, dt, ds, db]
            dt += x[batch_id][r][tid] * lB / (e2 / eb);
            ds += x[batch_id][r][tid] * lB * yCol / (e2 / (eb - ea));
            db += x[batch_id][r][tid] / (e2 / (ea * (pos - nconv) - eb * (pos - nconv - lT)));

            atomicAdd(&dx[batch_id][r][tid], y_grad[batch_id][yCol][tid] * weighted_sigmoid<scalar_t>(pos, lT, nconv, lB));
        }
        atomicAdd(&dTheta[batch_id][tid], dt * d_softplus<scalar_t>(theta[batch_id][tid], lifted_a));
        atomicAdd(&dStride[batch_id][tid], ds * d_softplus<scalar_t>(stride[batch_id][tid], lifted_a));
        atomicAdd(&dBeta[batch_id][tid], db * d_softplus<scalar_t>(beta[batch_id][tid], lifted_a));
    }
}

std::vector<torch::Tensor> d_continous_sum_cuda(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    float lifted_a,
    float lifted_b,
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
            theta.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            stride.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            d_theta.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_stride.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_beta.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lifted_a,
            lifted_b
        );
    }));
    return {d_input, d_theta, d_stride, d_beta};
}

#ifndef LinkingLibrary
#ifndef IsPyExtension

int main()
{
    auto i = torch::rand({1, 32, 32}).cuda();
    auto t = torch::full({1, 32}, 5.0).cuda();
    auto s = torch::full({1, 32}, 3.0).cuda();
    auto b = torch::full({1, 32}, 2.0).cuda();
    auto y = continous_sum_cuda_v2(i, t, s, b, 1, 0, 2);
    //auto z = d_continous_sum_cuda(i, t, s, b, y, 1, 0, 16);
    errorchk();
    std::cout << (y == 0).all() << "\n";
    //std::cout << (y == 0).all() << (z[0] == 0).all() << "\n";
}


#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("continous_sum_cuda", &continous_sum_cuda, "(CUDA)");
    m.def("d_continous_sum_cuda", &d_continous_sum_cuda, "(CUDA)");
}
#endif
#endif
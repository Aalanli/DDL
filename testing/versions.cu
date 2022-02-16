#include <torch/torch.h>
#include <vector>

const float e = 2.718281828459045;
const float exp_overflow = 89;
const float exp_underflow = -104;

int lowest_tessellation(int seq_len, int min, int max)
{
    int remainder = seq_len % min;
    int tess = min;
    for (int i = min * 2; i <= max; i *= 2)
    {
        if (seq_len % i < remainder)
        {
            remainder = seq_len % i;
            tess = i;
        }
    }
    return tess;
}

std::vector<int> calculate_over_under_flow(float theta, float stride, float beta) {
    int under = exp_overflow / beta + theta + stride;
    int over = exp_underflow / beta + stride;
    return {under, over};
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
__device__ __forceinline__ scalar_t lifted_softplus(scalar_t x) {
    return log(e + exp(x));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x) {
    const auto s = inv_sigmoid(x);
    return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t* d_weighted_sigmoid(scalar_t x, scalar_t theta, scalar_t stride, scalar_t beta) {
    scalar_t dy[4];  // [dx, dtheta, dstride, dbeta]
    const auto s = d_sigmoid(beta * (x - theta - stride));
    const auto s1 = d_sigmoid(beta * (x - stride));
    dy[0] = (s - s1) * beta;
    dy[1] = s * beta;
    dy[2] = dy[0];
    dy[3] = s * (x - theta - stride) - s1 * (x - stride);
    return dy;
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

/*
// case1 kernel, if non_zero portion is smaller than num threads, 1024
template <typename scalar_t, typename standard = float>
__global__ void continous_sum_kernelc1(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ y,
    const unsigned int batch,
    const unsigned int cols,      // col dim
    const unsigned int rows,      // row dim
    const unsigned int non_zero,  // threads per row
    const unsigned int block_per_conv,    // total number of convolutions
    const unsigned int n_sum_pool)
{
    unsigned int nconv = blockIdx.y;

    const unsigned int block = blockIdx.x;
    const unsigned int threads = blockDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int rows_per_block = threads / non_zero;
    const unsigned int block_start = rows_per_block * block * rows;
    const unsigned int row_start = tid / non_zero * rows + block_start;
    extern __shared__ standard weights[];
    standard* sum_pool = &weights[non_zero];
    // calculate weights
    if (tid < non_zero)
    {
        weights[tid] = d_weighted_sigmoid<scalar_t>(tid, theta[0], nconv * stride[0], beta[0])
    }
    __syncthreads();
}
*/

// baseline kernel, computes weights across the full range of sequence
template <typename scalar_t, typename standard = float>
__global__ void continous_sum_kernelb(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ y,
    const int x_rows,
    const int x_cols,
    const int y_cols)
{
    const int nconv = blockIdx.y;

    const int block_size = blockDim.x;
    const int blocks_per_row = (x_cols + block_size - 1) / block_size;
    const int rows_per_cycle = gridDim.x / blocks_per_row;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    // each group uses blocks_per_row blocks to calculate each row
    const int group_id = block / blocks_per_row;
    // row_idx refers to the local index within a row/block group
    int row_idx = (block + blocks_per_row) % blocks_per_row * blockDim.x + tid;
    extern __shared__ standard sumPool[];
    sumPool[tid] = 0;
    // calculate weights
    scalar_t w = weighted_sigmoid<scalar_t>((scalar_t) row_idx, theta[0], ((scalar_t) nconv) * stride[0], beta[0]);
    for (int i = group_id; i < x_rows; i += rows_per_cycle)
    {
        if (row_idx < x_cols)
            sumPool[tid] = w * x[i * x_cols + row_idx];
        __syncthreads();
        warpReduce(sumPool, tid, block_size);
        if (tid == 0)
        {
            atomicAdd(&y[y_cols * i + nconv], sumPool[0]);
        }
    }
}

template <typename scalar_t, typename standard = float, int tensorDim=3>
__global__ void continous_sum_kernelc1(
    const torch::PackedTensorAccessor32<scalar_t, tensorDim, torch::RestrictPtrTraits> x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    torch::PackedTensorAccessor32<scalar_t, tensorDim, torch::RestrictPtrTraits> y)
{
    const int nconv = blockIdx.y;
    const int b = x.size(0), r = x.size(2);
    const int elems = b * r;
    const int block_size = blockDim.x;
    const int blocks_per_col = (x.size(1) + block_size - 1) / block_size;
    const int cols_per_cycle = gridDim.x / blocks_per_col;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    // each group uses blocks_per_row blocks to calculate each row
    const int group_id = block / blocks_per_col;
    // row_idx refers to the local index within a row/block group
    int col_idx = (block + blocks_per_col) % blocks_per_col * blockDim.x + tid;
    extern __shared__ standard sumPool[];
    sumPool[tid] = 0;
    // calculate weights
    int b_, r_;
    scalar_t w = weighted_sigmoid<scalar_t>((scalar_t) col_idx, theta[0], ((scalar_t) nconv) * stride[0], beta[0]);
    for (int i = group_id; i < elems; i += cols_per_cycle)
    {
        b_ = i / r;
        r_ = i - b_ * r;
        if (col_idx < x.size(1))
            sumPool[tid] = w * x[b_][col_idx][r_];
        __syncthreads();
        warpReduce(sumPool, tid, block_size);
        if (tid == 0)
        {
            atomicAdd(&y[b_][nconv][r_], sumPool[0]);
        }
    }
}


template <typename scalar_t, typename standard = float, int tensorDim=3>
__global__ void continous_sum_kernelc2(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__  y,
    const int bCol, const int cCyl,
    const int c, const int r, const int c_y, const int elems)
{
    const int nconv = blockIdx.y;
    const int block_size = blockDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    // each group uses blocks_per_row blocks to calculate each row
    const int group_id = block / bCol;
    // row_idx refers to the local index within a row/block group
    int col_idx = (block + bCol) % bCol * blockDim.x + tid;
    extern __shared__ standard sumPool[];
    // calculate weights
    scalar_t w = weighted_sigmoid<scalar_t>((scalar_t) col_idx, theta[0], ((scalar_t) nconv) * stride[0], beta[0]);
    sumPool[tid] = 0;
    /*
    int b_, r_;
    for (int i = group_id; i < elems; i += cCyl)
    {
        b_ = i / r;
        r_ = i - b_ * r;
        if (col_idx < c)
            sumPool[tid] = w * x[b_ * c * r + col_idx * r + r_];
        __syncthreads();
        //warpReduce(sumPool, tid, block_size);
        if (tid == 0)
        {
            atomicAdd(&y[b_ * c_y * r + nconv * r + r_], sumPool[0]);
        }
    }
    */
}


torch::Tensor continous_sum_colmajor1(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    int div)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = *theta.cpu().data<float>();
    float s_ = *stride.cpu().data<float>();
    int y_seq_len = (seq_len - t_) / s_;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, y_seq_len, dim}, opt);

    int threads;
    int dimx;
    if (seq_len <= 1024) {
        threads = seq_len;
        dimx = batch * dim / div;
    } 
    else {
        threads = lowest_tessellation(seq_len, 256, 1024);
        dimx = (int) ((seq_len + threads - 1) / threads) * batch * dim / div;
    }
    const dim3 blocks(dimx, y_seq_len);

    const int blocks_per_col = (seq_len + threads - 1) / threads;
    const int cols_per_cycle = dimx / blocks_per_col;
    
    continous_sum_kernelc2<float, float, 3><<<blocks, threads, threads*sizeof(float)>>>(
        input.data<float>(),
        theta.data<float>(),
        stride.data<float>(),
        beta.data<float>(),
        y.data<float>(),
        blocks_per_col, cols_per_cycle, seq_len, dim, y_seq_len, batch * dim);
    
    return y;
}

torch::Tensor continous_sum(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta)
{
    // input.transpose_(-1, -2);
    int batch = input.size(0);
    int dim = input.size(2);
    int seq_len = input.size(1);
    float t_ = *theta.cpu().data<float>();
    float s_ = *stride.cpu().data<float>();
    int y_seq_len = (seq_len - t_) / s_;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, dim, y_seq_len}, opt);

    int threads;
    int dimx;
    if (seq_len <= 1024) {
        threads = seq_len;
        dimx = batch * dim / 4;
    } 
    else {
        threads = lowest_tessellation(seq_len, 256, 1024);
        dimx = (int) ((seq_len + threads - 1) / threads) * batch * dim / 4;
    }
    const dim3 blocks(dimx, y_seq_len);
    
    continous_sum_kernelb<float, float><<<blocks, threads, threads*sizeof(float)>>>(
        input.data<float>(),
        theta.data<float>(),
        stride.data<float>(),
        beta.data<float>(),
        y.data<float>(),
        batch * dim,
        seq_len,
        y_seq_len
    );
    
    // input.transpose_(-1, -2);
    // y.transpose_(-1, -2);
    return y;
}

template <typename scalar_t, typename standard = float, int tensorDim=3>
__global__ void continous_sum_kernelc(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ stride,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__  y,
    const int bCol, const int cCyl,
    const int c, const int r, const int c_y, const int elems)
{
    const int nconv = blockIdx.y;
    const int block_size = blockDim.x;
    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    // each group uses blocks_per_row blocks to calculate each row
    const int group_id = block / bCol;
    // row_idx refers to the local index within a row/block group
    int col_idx = (block + bCol) % bCol * blockDim.x + tid;
    extern __shared__ standard sumPool[];
    // calculate weights
    scalar_t w = weighted_sigmoid<scalar_t>((scalar_t) col_idx, theta[0], ((scalar_t) nconv) * stride[0], beta[0]);
    sumPool[tid] = 0;
    
    int b_, r_;
    for (int i = group_id; i < elems; i += cCyl)
    {
        b_ = i / r;
        r_ = i - b_ * r;
        if (col_idx < c)
            sumPool[tid] = w * x[b_ * c * r + col_idx * r + r_];
        __syncthreads();
        warpReduce(sumPool, tid, block_size);
        if (tid == 0)
        {
            atomicAdd(&y[b_ * c_y * r + nconv * r + r_], sumPool[0]);
        }
    }
}


torch::Tensor continous_sum_colmajor(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    int div)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = *theta.cpu().data<float>();
    float s_ = *stride.cpu().data<float>();
    int y_seq_len = (seq_len - t_) / s_;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, y_seq_len, dim}, opt);

    int threads;
    int dimx;
    if (seq_len <= 1024) {
        threads = seq_len;
        dimx = batch * dim / div;
    } 
    else {
        threads = lowest_tessellation(seq_len, 256, 1024);
        dimx = (int) ((seq_len + threads - 1) / threads) * batch * dim / div;
    }
    std::cout << "rowMajor blocks:" << dimx << " yseq: " << y_seq_len << "\n"; 

    const dim3 blocks(dimx, y_seq_len);

    const int blocks_per_col = (seq_len + threads - 1) / threads;
    const int cols_per_cycle = dimx / blocks_per_col;
    
    //cudaProfilerStart();
    continous_sum_kernelc<float, float, 3><<<blocks, threads, threads*sizeof(float)>>>(
        input.data<float>(),
        theta.data<float>(),
        stride.data<float>(),
        beta.data<float>(),
        y.data<float>(),
        blocks_per_col, cols_per_cycle, seq_len, dim, y_seq_len, batch * dim);
    //cudaProfilerStop();
    return y;
}
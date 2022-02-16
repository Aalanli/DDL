#define IsPyExtension

#ifdef IsPyExtension
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif

#include <iostream>
#include <vector>
#include <math.h>
#include <string>

const float exp_overflow = 20;
const float exp_underflow = -20;


template <typename scalar_t>
inline int overflow(scalar_t theta, scalar_t stride, scalar_t beta) {
    return exp_overflow / beta + stride + theta;
}

template <typename scalar_t>
inline int underflow(scalar_t stride, scalar_t beta) {
    return exp_underflow / beta + stride;
}

template <typename scalar_t>
inline scalar_t inv_sigmoid(scalar_t x) {
    return 1.0 / (1.0 + std::exp(x));
}

template <typename scalar_t>
inline scalar_t weighted_sigmoid(scalar_t x, scalar_t theta, scalar_t stride, scalar_t beta) {
    return 1.0 / (1.0 + std::exp(-beta * (x - stride)) + std::exp(beta * (x - stride - theta)));
}

template <typename scalar_t>
inline scalar_t d_sigmoid(scalar_t x) {
    const auto s = inv_sigmoid(x);
    return (1.0 - s) * s;
}

template <typename scalar_t>
inline scalar_t lifted_softplus(scalar_t x, scalar_t alpha, scalar_t beta) {
    return std::log(alpha + std::exp(x)) + beta;
}

template <typename scalar_t>
inline scalar_t d_lifted_softplus(scalar_t x, scalar_t alpha) {
    return 1 / (1 + alpha * std::exp(-x));
}

#ifdef IsPyExtension
torch::Tensor continous_sum_cuda(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    float lifted_a,
    float lifted_b,
    int blocks_per_batch);

std::vector<torch::Tensor> d_continous_sum_cuda(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    float lifted_a,
    float lifted_b,
    int blocks_per_batch);

torch::Tensor continous_sum_cuda_v2(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    float lifted_a,
    float lifted_b,
    int threads_per_conv);
#endif

template <typename scalar_t>
void continous_sum_op_cpu(
    const scalar_t* __restrict_arr x, // [batch, seq_len, dim]
    const scalar_t* __restrict_arr theta, // [batch, dim]
    const scalar_t* __restrict_arr stride, // [batch, dim]
    const scalar_t* __restrict_arr beta, // [batch, dim]
    scalar_t* __restrict_arr y, // [batch, seq_len, dim]
    const scalar_t lifted_a,
    const scalar_t lifted_b,
    const int batch, const int col, const int row, const int yCol)
{
    for (int b = 0; b < batch; b++)
    {
        for (int r = 0; r < row; r++)
        {
            int pos = b * row + r;
            scalar_t lT = lifted_softplus<scalar_t>(theta[pos], lifted_a, lifted_b);
            scalar_t lS = lifted_softplus<scalar_t>(stride[pos], lifted_a, lifted_b);
            scalar_t lB = lifted_softplus<scalar_t>(beta[pos], lifted_a, lifted_b);
            int max_conv = (col - lT) / lS + 1;
            for (int nconv = 0; nconv < max_conv; nconv++)
            {
                int uFlow = std::max(0, underflow<scalar_t>(nconv * lS, lB));
                int oFlow = std::min(col, overflow<scalar_t>(lT, nconv * lS, lB));
                int yPos = b * col * row + nconv * row + r;
                for (int c = uFlow; c < oFlow; c++)
                {
                    y[yPos] += x[b * col * row + c * row + r] * weighted_sigmoid((scalar_t) c, lT, lS * nconv, lB);
                }
            }   
        }
    }
}

template <typename scalar_t>
void d_continous_sum_op_cpu(
    const scalar_t* __restrict_arr  y_grad,
    const scalar_t* __restrict_arr  x,
    const scalar_t* __restrict_arr theta,
    const scalar_t* __restrict_arr stride,
    const scalar_t* __restrict_arr beta,
    scalar_t* __restrict_arr  dx,
    scalar_t* __restrict_arr dTheta,
    scalar_t* __restrict_arr dStride,
    scalar_t* __restrict_arr dBeta,
    scalar_t lifted_a,
    scalar_t lifted_b,
    const int batch,
    const int col,
    const int row,
    const int ycol)
{
    for (int b = 0; b < batch; b++)
    {
        for (int r = 0; r < row; r++)
        {
            int pos_t = b * row + r;
            scalar_t lT = lifted_softplus<scalar_t>(theta[pos_t], lifted_a, lifted_b);
            scalar_t lS = lifted_softplus<scalar_t>(stride[pos_t], lifted_a, lifted_b);
            scalar_t lB = lifted_softplus<scalar_t>(beta[pos_t], lifted_a, lifted_b);

            int max_conv = (col - lT) / lS + 1;
            for (int nconv = 0; nconv < max_conv; nconv++)
            {
                scalar_t stride_r = lS * nconv;
                int uFlow = std::max(0, underflow<scalar_t>(stride_r, lB));
                int oFlow = std::min(col, overflow<scalar_t>(lT, stride_r, lB));
                int yPos = b * ycol * row + nconv * row + r;
                scalar_t dt = 0; scalar_t ds = 0; scalar_t db = 0;
                for (int c = uFlow; c < oFlow; c++)
                {
                    scalar_t pos_c = (scalar_t) c;
                    scalar_t ea = exp(-lB * (pos_c - stride_r));
                    scalar_t eb = exp(lB * (pos_c - stride_r - lT));
                    scalar_t e2 = powf(1 + ea + eb, 2);
                    int xPos = b * col * row + c * row + r;
                    dx[xPos] += y_grad[yPos] * weighted_sigmoid<scalar_t>(pos_c, lT, stride_r, lB);
                    dt += x[xPos] * lB / (e2 / eb);
                    ds += x[xPos] * lB * nconv / (e2 / (eb - ea));
                    db += x[xPos] / (e2 / (-eb * (pos_c - stride_r - lT) + ea * (pos_c - stride_r)));
                }
                dTheta[pos_t] += dt * y_grad[yPos];
                dStride[pos_t] += ds * y_grad[yPos];
                dBeta[pos_t] += db * y_grad[yPos];
            }
            dTheta[pos_t] *= d_lifted_softplus<scalar_t>(theta[pos_t], lifted_a);
            dStride[pos_t] *= d_lifted_softplus<scalar_t>(stride[pos_t], lifted_a);
            dBeta[pos_t] *= d_lifted_softplus<scalar_t>(beta[pos_t], lifted_a);
        }
    }
}

std::vector<torch::Tensor> d_continous_sum_cpu(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    float lifted_a,
    float lifted_b)
{
    torch::Tensor d_input = torch::zeros_like(input);
    torch::Tensor d_theta = torch::zeros_like(theta);
    torch::Tensor d_stride = torch::zeros_like(stride);
    torch::Tensor d_beta = torch::zeros_like(beta);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_forward_cpu", ([&] {
        d_continous_sum_op_cpu<scalar_t>(
            grad.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            theta.data_ptr<scalar_t>(),
            stride.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            d_input.data_ptr<scalar_t>(),
            d_theta.data_ptr<scalar_t>(),
            d_stride.data_ptr<scalar_t>(),
            d_beta.data_ptr<scalar_t>(),
            (scalar_t) lifted_a,
            (scalar_t) lifted_b,
            input.size(0),
            input.size(1),
            input.size(2),
            grad.size(1)
        );
    }));

    return {d_input, d_theta, d_stride, d_beta};
}


torch::Tensor continous_sum_cpu(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    float lifted_a, float lifted_b)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = theta.min().cpu().item<float>();
    float s_ = stride.min().cpu().item<float>();
    int y_seq_len = (seq_len - t_) / s_ + 1;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, y_seq_len, dim}, opt);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_backward_cpu", ([&] {
        continous_sum_op_cpu<scalar_t>(
            input.data_ptr<scalar_t>(),
            theta.data_ptr<scalar_t>(),
            stride.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            (scalar_t) lifted_a,
            (scalar_t) lifted_b,
            batch, seq_len, dim, y_seq_len
        );
    }));
    return y;
}

#ifdef IsPyExtension
torch::Tensor continous_sum(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    float lifted_a,
    float lifted_b,
    int block_size,
    std::string exec_type="v1")
{
    if (input.type().is_cuda())
    {
        if (exec_type == "v1")
            return continous_sum_cuda(input, theta, stride, beta, lifted_a, lifted_b, block_size);
        if (exec_type == "v2")
            return continous_sum_cuda_v2(input, theta, stride, beta, lifted_a, lifted_b, block_size);
    }
    else {
        return continous_sum_cpu(input, theta, stride, beta, lifted_a, lifted_b);
    }
}

std::vector<torch::Tensor> d_continous_sum(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    float lifted_a,
    float lifted_b,
    int block_size)
{
    if (input.type().is_cuda())
    {
        return d_continous_sum_cuda(input, theta, stride, beta, grad, lifted_a, lifted_b, block_size);
    }
    else {
        return d_continous_sum_cpu(input, theta, stride, beta, grad, lifted_a, lifted_b);
    }
}
#endif

#ifndef IsPyExtension
int main()
{
    float t = 5.0, s = 5.0, b = 2.0;
    int batch = 1, c = 256, r = 256;
    int yC = (c - t) / s;
    auto x = torch::randn({batch, c, r});
    auto theta = torch::full({batch, r}, t);
    auto stride = torch::full({batch, r}, s);
    auto beta = torch::full({batch, r}, b);
    auto out = continous_sum_cpu(x, theta, stride, beta, 0, 1);
    auto d = d_continous_sum_cpu(x, theta, stride, beta, torch::ones_like(out), 0, 1);
    
    std::cout << "done\n";
}

#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d_continous_sum", &d_continous_sum, "");
    m.def("continous_sum", &continous_sum, "");
}

#endif
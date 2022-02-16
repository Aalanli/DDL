#define IsPyExtension

#ifdef IsPyExtension
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif

#include <iostream>
#include <vector>
#include <math.h>


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
    return inv_sigmoid(beta * (x - theta - stride)) - inv_sigmoid(beta * (x - stride));
}

template <typename scalar_t>
inline scalar_t d_sigmoid(scalar_t x) {
    const auto s = inv_sigmoid(x);
    return (1.0 - s) * s;
}

torch::Tensor continous_sum_cuda(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    int blocks_per_batch);

std::vector<torch::Tensor> d_continous_sum_cuda(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    int blocks_per_batch);

template <typename scalar_t>
void continous_sum_op_cpu(
    const scalar_t* __restrict_arr x,
    const scalar_t* __restrict_arr theta,
    const scalar_t* __restrict_arr stride,
    const scalar_t* __restrict_arr beta,
    scalar_t* __restrict_arr y,
    const int batch, const int col, const int row, const int yCol)
{
    for (int nconv = 0; nconv < yCol; nconv++)
    {
        int uFlow = std::max(0, underflow<scalar_t>(nconv * stride[0], beta[0]));
        int oFlow = std::min(col, overflow<scalar_t>(theta[0], nconv * stride[0], beta[0]));

        for (int c = uFlow; c < oFlow; c++)
        {
            scalar_t w = weighted_sigmoid<scalar_t>((scalar_t) c, theta[0], stride[0] * nconv, beta[0]);

            for (int b = 0; b < batch; b++)
            {
                for (int r = 0; r < row; r++)
                {
                    y[b * yCol * row + nconv * row + r] += w * x[b * col * row + c * row + r];
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
    const int batch,
    const int col,
    const int row,
    const int ycol)
{
    for (int nconv = 0; nconv < ycol; nconv++)
    {
        int uFlow = std::max(0, underflow<scalar_t>(nconv * stride[0], beta[0]));
        int oFlow = std::min(col, overflow<scalar_t>(theta[0], nconv * stride[0], beta[0]));

        for (int c = uFlow; c < oFlow; c++)
        {
            scalar_t pos = (scalar_t) c;
            scalar_t pos_s = stride[0] * nconv;
            scalar_t s = d_sigmoid<scalar_t>(beta[0] * (pos - theta[0] - pos_s));
            scalar_t s1 = d_sigmoid<scalar_t>(beta[0] * (pos - pos_s));
            scalar_t dw = weighted_sigmoid<scalar_t>(pos, theta[0], pos_s, beta[0]);
            scalar_t dt = s * beta[0];
            scalar_t ds = (s - s1) * beta[0] * nconv;
            scalar_t db = -s * (pos - theta[0] - pos_s) + s1 * (pos - pos_s);
            for (int b = 0; b < batch; b++)
            {
                for (int r = 0; r < row; r++)
                {
                    int xPos = b * col * row + c * row + r;
                    int yPos = b * ycol * row + nconv * row + r;
                    dx[xPos] += y_grad[yPos] * dw;
                    dTheta[0] += y_grad[yPos] * x[xPos] * dt;
                    dStride[0] += y_grad[yPos] * x[xPos] * ds;
                    dBeta[0] += y_grad[yPos] * x[xPos] * db;
                }
            }
        }
    }
}

std::vector<torch::Tensor> d_continous_sum_cpu(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad)
{
    torch::Tensor d_input = torch::zeros_like(input);
    torch::Tensor d_theta = torch::zeros_like(theta);
    torch::Tensor d_stride = torch::zeros_like(stride);
    torch::Tensor d_beta = torch::zeros_like(beta);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_forward_cpu", ([&] {
        d_continous_sum_op_cpu<scalar_t>(
            grad.data<scalar_t>(),
            input.data<scalar_t>(),
            theta.data<scalar_t>(),
            stride.data<scalar_t>(),
            beta.data<scalar_t>(),
            d_input.data<scalar_t>(),
            d_theta.data<scalar_t>(),
            d_stride.data<scalar_t>(),
            d_beta.data<scalar_t>(),
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
    torch::Tensor beta)
{
    int batch = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    float t_ = theta.cpu().item<float>();
    float s_ = stride.cpu().item<float>();
    int y_seq_len = (seq_len - t_) / s_ + 1;

    auto opt = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    auto y = torch::zeros({batch, y_seq_len, dim}, opt);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "continous_sum_backward_cpu", ([&] {
        continous_sum_op_cpu<scalar_t>(
            input.data<scalar_t>(),
            theta.data_ptr<scalar_t>(),
            stride.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch, seq_len, dim, y_seq_len
        );
    }));
    return y;
}


torch::Tensor continous_sum(
    torch::Tensor input, 
    torch::Tensor theta, 
    torch::Tensor stride, 
    torch::Tensor beta,
    int block_size)
{
    if (input.type().is_cuda())
    {
        return continous_sum_cuda(input, theta, stride, beta, block_size);
    }
    else {
        return continous_sum_cpu(input, theta, stride, beta);
    }
}

std::vector<torch::Tensor> d_continous_sum(
    torch::Tensor input,
    torch::Tensor theta,
    torch::Tensor stride,
    torch::Tensor beta,
    torch::Tensor grad,
    int block_size)
{
    if (input.type().is_cuda())
    {
        return d_continous_sum_cuda(input, theta, stride, beta, grad, block_size);
    }
    else {
        return d_continous_sum_cpu(input, theta, stride, beta, grad);
    }
}


#ifndef IsPyExtension
int main()
{
    float t = 5.0, s = 5.0, b = 2.0;
    int batch = 1, c = 256, r = 256;
    int yC = (c - t) / s;
    auto x = torch::randn({batch, c, r});
    auto theta = torch::tensor({t});
    auto stride = torch::tensor({s});
    auto beta = torch::tensor({b});
    auto yGrad = torch::randn({batch, yC, r});
    auto grads = d_continous_sum_cpu(x, theta, stride, beta, yGrad);
    std::cout << "done\n";
}

#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d_continous_sum", &d_continous_sum, "");
    m.def("continous_sum", &continous_sum, "");
}

#endif
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

template <typename T>
void print(std::vector<T> s) {
    std::cout << s << "\n";
}

template <typename scalar_t>
inline scalar_t weight_kernel(scalar_t x, scalar_t a, scalar_t b) {
    return 1 / (1 + pow(abs(x / a), 2 * b));
}

template <typename scalar_t>
inline void d_weight_kernel(scalar_t x, scalar_t a, scalar_t b, scalar_t* w, scalar_t* dA, scalar_t* dB) {
    auto v = pow(abs(x / a), 2 * b);
    w[0] = 1 / (1 + v);
    auto vp = 2 / (1 / v + 2 + v);
    dA[0] = b * vp / a;
    dB[0] = -log(abs(x / a) + 1e-6) * vp;
} 

template <typename scalar_t>
scalar_t cut_off_thres(scalar_t a, scalar_t b, float thres) {
    return a * pow(1/thres-1,1/(2*b));
}

template <typename scalar_t>
void conv1d_kernel_cpu(
    const scalar_t* __restrict_arr x, // [batch, channels, xDim]
    const scalar_t* __restrict_arr aS, // [batch] (kernel_sizes for each batch)
    const scalar_t* __restrict_arr bS, // [batch] (hardness for each batch)
    const scalar_t* __restrict_arr sT, // [batch] (strides for each batch)
    scalar_t* __restrict_arr y,
    const int batch,
    const int ch,
    const int xDim,
    const int yDim,
    const float cutThres) 
{
    for (int b=0; b < batch; b++) {
        scalar_t thres = cut_off_thres(aS[b], bS[b], cutThres) - aS[b];
        scalar_t lowB = -thres;
        scalar_t hiB = 2 * aS[b] + thres;
        scalar_t funcMid = aS[b];

        for (int k=0; k < (int) ((xDim - aS[b])/sT[b]) + 1; k++) {
            int indL = std::max(0, (int) lowB);
            int indH = std::min(xDim, (int) hiB);
            //print<int>({k, indL, indH});
            scalar_t weights[indH - indL];
            for (int i=indL; i < indH; i++) {
                weights[i-indL] = weight_kernel<scalar_t>((scalar_t) i - funcMid, aS[b], bS[b]);
            }
            
            for (int c=0; c < ch; c++) {
                scalar_t s = 0;
                for (int i=indL; i < indH; i++) {
                    s += x[b * ch * xDim + c * xDim + i] * weights[i-indL];
                }
                y[b * ch * yDim + c * yDim + k] = s;
            }
            lowB += sT[b];
            hiB += sT[b];
            funcMid += sT[b];
        }
    }
}

template <typename scalar_t>
void conv2d_kernel_cpu(
    const scalar_t* __restrict_arr x,  // [batch, channels, yDim, xDim]
    const scalar_t* __restrict_arr aS, // [batch, 2] (kernel_sizes for each batch)
    const scalar_t* __restrict_arr bS, // [batch, 2] (hardness for each batch)
    const scalar_t* __restrict_arr sT, // [batch, 2] (strides for each batch)
    scalar_t* __restrict_arr y,
    const int batch,
    const int ch,
    const int yDim,
    const int xDim,
    const int y1Dim,
    const int x1Dim,
    const float cutThres) 
{
    for (int b=0; b < batch; b++) {
        const int yc = b * 2;
        const int xc = b * 2 + 1;
        const scalar_t thresY = cut_off_thres(aS[yc] / 2, bS[yc], cutThres);
        const scalar_t thresX = cut_off_thres(aS[xc] / 2, bS[xc], cutThres);
        const int yKernels = (yDim - aS[yc]) / sT[yc];  // number of kernel passes across y dimension
        const int xKernels = (xDim - aS[xc]) / sT[xc];  // number of kernel passes across x dimension
        const int xKernelW = 2 * thresX + 0.5;  // width of each x kernel

        scalar_t xWeights[xKernels][xKernelW];
        for (int i=0; i < xKernels; i++) {
            scalar_t start = i * sT[xc] + aS[xc] / 2;
            int indS = std::max((int) (start - thresX + 0.5), 0);
            for (int j=indS; j < indS + xKernelW; j++) {
                xWeights[i][j - indS] = weight_kernel<scalar_t>((scalar_t) j - start, aS[xc] / 2, bS[xc]);
            }
        }

        for (int ky=0; ky < y1Dim; ky++) {
            scalar_t yStart = ky * sT[yc] + aS[yc] / 2;
            int yIndS = std::max((int) (yStart - thresY + 0.5), 0);
            int yIndE = std::min((int) (thresY + yStart) + 1, yDim);

            for (int iy=yIndS; iy < yIndE; iy++) {
                scalar_t yWeight = weight_kernel<scalar_t>((scalar_t) iy - yStart, aS[yc] / 2, bS[yc]);
                
                for (int kx=0; kx < xKernels; kx++) {
                    scalar_t xStart = kx * sT[xc] + aS[xc] / 2;
                    int xIndS = std::max((int) (xStart - thresX + 0.5), 0);
                    int xIndE = std::min((int) (thresX + xStart) + 1, xDim);
                    for (int c=0; c < ch; c++) {
                        scalar_t s = 0;
                        for (int ix=xIndS; ix < xIndE; ix++) {
                            s += x[b * ch * yDim * xDim + c * yDim * xDim + iy * xDim + ix] * xWeights[kx][ix-xIndS];
                        }
                        y[b * ch * y1Dim * x1Dim + c * y1Dim * x1Dim + ky * x1Dim + kx] += s * yWeight;
                    }
                }
                
            }
        }
    }
}

template <typename scalar_t>
void d_conv1d_kernel_cpu(
    const scalar_t* __restrict_arr x,
    const scalar_t* __restrict_arr aS,
    const scalar_t* __restrict_arr bS,
    const scalar_t* __restrict_arr sT,
    const scalar_t* __restrict_arr grad,
    scalar_t* __restrict_arr dX,
    scalar_t* __restrict_arr dA,
    scalar_t* __restrict_arr dB,
    const int batch,
    const int ch,
    const int xDim,
    const int yDim,
    const float cutThres)
{
    for (int b=0; b < batch; b++) {
        scalar_t thres = cut_off_thres(aS[b], bS[b], cutThres) - aS[b];
        scalar_t lowB = -thres;
        scalar_t hiB = 2 * aS[b] + thres;
        scalar_t funcMid = aS[b];

        for (int k=0; k < (int) ((xDim - aS[b])/sT[b]) + 1; k++) {
            int indL = std::max(0, (int) lowB);
            int indH = std::min(xDim, (int) hiB);
            //print<int>({k, indL, indH});
            scalar_t weights[indH - indL];
            scalar_t dwA[indH - indL];
            scalar_t dwB[indH - indL];
            for (int i=0; i < (indH - indL); i++) {
                d_weight_kernel<scalar_t>((scalar_t) (i + indL) - funcMid, aS[i], bS[i], weights + i, dwA + i, dwB + i);
            }
            
            for (int c=0; c < ch; c++) {
                auto k0 = b * ch * yDim + c * yDim + k;
                scalar_t tA = 0;
                scalar_t tB = 0;
                for (int i=indL; i < indH; i++) {
                    dX[b * ch * xDim + c * xDim + i] += grad[k0] * weights[i - indL];
                    tA += x[b * ch * xDim + c * xDim + i] * dwA[i - indL];
                    tB += x[b * ch * xDim + c * xDim + i] * dwB[i - indL];
                }
                dA[b] += tA * grad[k0];
                dB[b] += tB * grad[k0];
            }
            lowB += sT[b];
            hiB += sT[b];
            funcMid += sT[b];
        }
    }
}


template <typename scalar_t>
void d_conv2d_kernel_cpu(
    const scalar_t* __restrict_arr x,
    const scalar_t* __restrict_arr aS,
    const scalar_t* __restrict_arr bS,
    const scalar_t* __restrict_arr sT,
    const scalar_t* __restrict_arr grad,
    scalar_t* __restrict_arr dX,
    scalar_t* __restrict_arr dA,
    scalar_t* __restrict_arr dB,
    const int batch,
    const int ch,
    const int yDim,
    const int xDim,
    const int y1Dim,
    const int x1Dim,
    const float cutThres) 
{
    for (int b=0; b < batch; b++) {
        const int yc = b * 2;
        const int xc = b * 2 + 1;
        const scalar_t thresY = cut_off_thres(aS[yc] / 2, bS[yc], cutThres);
        const scalar_t thresX = cut_off_thres(aS[xc] / 2, bS[xc], cutThres);
        const int yKernels = (yDim - aS[yc]) / sT[yc];  // number of kernel passes across y dimension
        const int xKernels = (xDim - aS[xc]) / sT[xc];  // number of kernel passes across x dimension
        const int xKernelW = 2 * thresX + 0.5;  // width of each x kernel

        scalar_t xWeights[xKernels][xKernelW];
        scalar_t aXGrad[xKernels][xKernelW];
        scalar_t bXGrad[xKernels][xKernelW];
        for (int i=0; i < xKernels; i++) {
            scalar_t start = i * sT[xc] + aS[xc] / 2;
            int indS = std::max((int) (start - thresX + 0.5), 0);
            for (int j=0; j < xKernelW; j++) {
                d_weight_kernel<scalar_t>((scalar_t) (j + indS) - start, aS[xc] / 2, bS[xc], (xWeights[i] + j), (aXGrad[i] + j), (bXGrad[i] + j));
            }
        }

        for (int ky=0; ky < y1Dim; ky++) {
            scalar_t yStart = ky * sT[yc] + aS[yc] / 2;
            int yIndS = std::max((int) (yStart - thresY + 0.5), 0);
            int yIndE = std::min((int) (thresY + yStart) + 1, yDim);

            for (int iy=yIndS; iy < yIndE; iy++) {
                scalar_t yWeight, aYGrad, bYGrad;
                d_weight_kernel<scalar_t>((scalar_t) iy - yStart, aS[yc] / 2, bS[yc], &yWeight, &aYGrad, &bYGrad);
                
                for (int kx=0; kx < xKernels; kx++) {
                    scalar_t xStart = kx * sT[xc] + aS[xc] / 2;
                    int xIndS = std::max((int) (xStart - thresX + 0.5), 0);
                    int xIndE = std::min((int) (thresX + xStart) + 1, xDim);
                    for (int c=0; c < ch; c++) {
                        int k0 = b * ch * y1Dim * x1Dim + c * y1Dim * x1Dim + ky * x1Dim + kx;
                        scalar_t tDY = 0;
                        scalar_t tDAX = 0, tDBX = 0;
                        for (int ix=xIndS; ix < xIndE; ix++) {
                            int i0 = b * ch * yDim * xDim + c * yDim * xDim + iy * xDim + ix;
                            dX[i0] += yWeight * xWeights[kx][ix-xIndS] * grad[k0];
                            tDY += x[i0] * xWeights[kx][ix-xIndS];
                            tDAX += x[i0] * aXGrad[kx][ix-xIndS];
                            tDBX += x[i0] * bXGrad[kx][ix-xIndS];
                        }
                        dA[yc] += tDY * aYGrad * grad[k0];
                        dB[yc] += tDY * bYGrad * grad[k0];
                        dA[xc] += tDAX * yWeight * grad[k0];
                        dB[xc] += tDBX * yWeight * grad[k0];
                    }
                }
            }
        }
        dA[yc] /= 2;
        dA[xc] /= 2;
    }
}


torch::Tensor conv1d_cpu(
    torch::Tensor x,
    torch::Tensor aS,
    torch::Tensor hS,
    torch::Tensor sT,
    float cutThres)
{
    float minA = torch::min(aS).data_ptr<float>()[0];
    float minS = torch::min(sT).data_ptr<float>()[0];
    int batch = x.size(0);
    int ch = x.size(1);
    int xDim = x.size(2);
    int yDim = (int) ((xDim - minA) / minS) + 1;
    std::cout << "filter size " << 2 * cut_off_thres(minA, hS.min().data_ptr<float>()[0], cutThres) << "\n";
    auto tensorOpt = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor y = torch::empty({batch, ch, yDim}, tensorOpt);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1D forward cpu", ([&] {
        conv1d_kernel_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            aS.data_ptr<scalar_t>(),
            hS.data_ptr<scalar_t>(),
            sT.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch, ch, xDim, yDim,
            cutThres
        );
    }));
    return y;
}

std::vector<torch::Tensor> d_conv1d_cpu(
    torch::Tensor x,
    torch::Tensor aS,
    torch::Tensor bS,
    torch::Tensor sT,
    torch::Tensor grad,
    float cutThres)
{
    int batch = x.size(0);
    int ch = x.size(1);
    int xDim = x.size(2);
    int yDim = grad.size(2);

    auto dX = torch::zeros_like(x);
    auto dA = torch::zeros_like(aS);
    auto dB = torch::zeros_like(bS);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1D forward cpu", ([&] {
        d_conv1d_kernel_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            aS.data_ptr<scalar_t>(),
            bS.data_ptr<scalar_t>(),
            sT.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            dX.data_ptr<scalar_t>(),
            dA.data_ptr<scalar_t>(),
            dB.data_ptr<scalar_t>(),
            batch, ch, xDim, yDim,
            cutThres
        );
    }));
    return {dX, dA, dB};
}

torch::Tensor conv2d_cpu(
    torch::Tensor x,
    torch::Tensor aS,
    torch::Tensor bS,
    torch::Tensor sT,
    double cutThres)
{
    float minAY = torch::min(aS.index({torch::indexing::Slice(), 0})).item<float>();
    float minSY = torch::min(sT.index({torch::indexing::Slice(), 0})).item<float>();
    float minAX = torch::min(aS.index({torch::indexing::Slice(), 1})).item<float>();
    float minSX = torch::min(sT.index({torch::indexing::Slice(), 1})).item<float>();
    int batch = x.size(0);
    int ch = x.size(1);
    int yDim = x.size(2);
    int xDim = x.size(3);
    int y1Dim = (int) ((yDim - minAY) / minSY) + 1;
    int x1Dim = (int) ((xDim - minAX) / minSX) + 1;

    auto tensorOpt = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor y = torch::zeros({batch, ch, y1Dim, x1Dim}, tensorOpt);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv2D forward cpu", ([&] {
        conv2d_kernel_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            aS.data_ptr<scalar_t>(),
            bS.data_ptr<scalar_t>(),
            sT.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch, ch, yDim, xDim,
            y1Dim, x1Dim,
            (float) cutThres
        );
    }));
    
    return y;
}

std::vector<torch::Tensor> d_conv2d_cpu(
    torch::Tensor x,
    torch::Tensor aS,
    torch::Tensor bS,
    torch::Tensor sT,
    torch::Tensor grad,
    double cutThres)
{
    int batch = x.size(0);
    int ch = x.size(1);
    int yDim = x.size(2);
    int xDim = x.size(3);
    int y1Dim = grad.size(2);
    int x1Dim = grad.size(3);

    torch::Tensor dX = torch::zeros_like(x);
    torch::Tensor dA = torch::zeros_like(aS);
    torch::Tensor dB = torch::zeros_like(bS);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv2D forward cpu", ([&] {
        d_conv2d_kernel_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            aS.data_ptr<scalar_t>(),
            bS.data_ptr<scalar_t>(),
            sT.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            dX.data_ptr<scalar_t>(),
            dA.data_ptr<scalar_t>(),
            dB.data_ptr<scalar_t>(),
            batch, ch, yDim, xDim,
            y1Dim, x1Dim,
            (float) cutThres
        );
    }));
    return {dX, dA, dB};
}


#ifndef IsPyExtension
int main() {
    auto opt = torch::TensorOptions().dtype(torch::kFloat64);
    auto x = torch::rand({1, 3, 64, 64}, opt);
    auto aS = torch::full({1, 2}, 4.2, opt);
    auto sT = torch::clone(aS);
    auto hS = torch::full_like(aS, 4.6);

    auto y = conv2d_cpu(x, aS, hS, sT, 0.01);
    auto g = d_conv2d_cpu(x, aS, hS, sT, y, 0.01);
    std::cout << y.sizes();
    //std::cout << torch::allclose(x, g[0]);
}

#else
int test() {
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_cpu", &conv1d_cpu, "");
    m.def("d_conv1d_cpu", &d_conv1d_cpu, "");
    m.def("conv2d_cpu", &conv2d_cpu, "");
    m.def("d_conv2d_cpu", &d_conv2d_cpu, "");
    m.def("test", &test, "");
}

#endif
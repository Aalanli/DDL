# %%
import os

import torch
from torch.utils.cpp_extension import load
from torchvision.datasets import Caltech101
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

def build_extension(name, *sources):
    build_dir = 'extension_cache/'
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    if not os.path.exists(build_dir + name):
        os.mkdir(build_dir + name)
    
    return load(name, sources=sources, verbose=True, build_directory=build_dir+name)

ext = build_extension('conv', 'continousConv/conv.cpp')
print(ext.test())
def conv2d(x, a, b, stride=None, thres_cut=0.01):
    batch = x.shape[0]
    if type(a) != torch.Tensor:
        a = torch.full([batch, 2], a).to(x)
    if type(b) != torch.Tensor:
        b = torch.full([batch, 2], b).to(x)
    if stride is None:
        stride = torch.clone(a)
    elif type(stride) != torch.Tensor:
        stride = torch.full([batch, 2], stride).to(x)
    return ext.conv2d_cpu(x, a, b, stride, thres_cut)

def d_conv2d(grad, x, a, b, stride=None, thres_cut=0.01):
    batch = x.shape[0]
    if type(a) != torch.Tensor:
        a = torch.full([batch, 2], a).to(x)
    if type(b) != torch.Tensor:
        b = torch.full([batch, 2], b).to(x)
    if stride is None:
        stride = torch.clone(a)
    elif type(stride) != torch.Tensor:
        stride = torch.full([batch, 2], stride).to(x)
    return ext.d_conv2d_cpu(x, a, b, stride, grad, thres_cut)

def show(x: torch.Tensor):
    plt.imshow(x.permute(1, 2, 0))
    plt.show()

def plot(x: torch.Tensor):
    plt.plot(torch.arange(0, x.shape[-1]), x)
    plt.show()

def equal_all(a, b, rtol=0.00001, atol=1e-8):
    for i, j in zip(a, b):
        print(torch.allclose(i, j.to(i), rtol, atol))
"""
data = Caltech101('datasets', transform=ToTensor())
im = data[0][0]
show(im)
im1 = conv2d(im.unsqueeze(0), 2, 12, 5).squeeze()
print(im1.shape)
show(im1 / im1.max())
print(im1.max(), im1.min(), im1.mean())"""



class Conv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, thres):
        ctx.thres = thres
        ctx.save_for_backward(x, a, b)
        return conv2d(x, a, b, thres_cut=thres)
    
    @staticmethod
    def backward(ctx, grad):
        x, a, b = ctx.saved_tensors
        return tuple(d_conv2d(grad, x, a, b, thres_cut=ctx.thres) + [None, None])

from torch.autograd import gradcheck

dim=4
y=10
x=10
a=4.5
b=7
input = torch.randn((1, dim, y, x), dtype=torch.double, requires_grad=True)
a = torch.full((1, 2), a, dtype=torch.double, requires_grad=True)
b = torch.full((1, 2), b, dtype=torch.double, requires_grad=True)

gradcheck(Conv2D.apply, (input, a, b, 0.01))

# %%
y = conv2d(input, a, b)
grad = torch.rand_like(y)
dx = d_conv2d(grad, input, a, b)

print(dx[1])

# %%
import torch.nn.functional as F
import triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(6, 30)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'rowMajor',
            #'torch-native',
            'colMajor',
        ],  # possible values for `line_arg``
        line_names=[
            "row major",
            #"Torch (native)",
            "col major",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 1024, 'K': 1},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, K, provider):
    x = torch.randn(K, N, M, device='cuda', dtype=torch.float32)
    t = torch.full((K, M), 5, device='cuda', dtype=torch.float32)
    s = torch.full((K, M), 5, device='cuda', dtype=torch.float32)
    b = torch.full((K, M), 2, device='cuda', dtype=torch.float32)
    if provider == 'rowMajor':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: continous_sum(x, t, s, b, 0., 1., blocks=2))
    if provider == 'colMajor':
        y = torch.ones(K, int((N - 5) / 2) + 1, M, device='cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: d_continous_sum(x, t, s, b, y, 0, 1, blocks=1))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)
# %%
def f1(x: torch.Tensor, t, s, b):
    return 1 / (1 + (b * (x - s - t)).exp()) - 1 / (1 + (b * (x - s)).exp())

def f2(x: torch.Tensor, t, s, b):
    return 1 / (1 + (-b * (x - s)).exp() + ((b * (x - t - s)).exp()))

a = torch.randn(512, 512, dtype=torch.float) * 100
b = 5.0
c = 5.0
d = 2.0
print((f1(a, b, c, d) - f2(a, b, c, d)).max())

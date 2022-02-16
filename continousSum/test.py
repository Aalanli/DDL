# %%
import os

import torch
from torch.utils.cpp_extension import load

from matplotlib import pyplot as plt

def build_extension(name, *sources):
    build_dir = 'extension_cache/'
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    if not os.path.exists(build_dir + name):
        os.mkdir(build_dir + name)
    
    return load(name, sources=sources, verbose=True, build_directory=build_dir+name)

ext = build_extension('csum', 'continous_sum.cpp', 'continous_sum.cu')

def continous_sum(x, t, s, b, alpha, beta, blocks=2, exec_type="v1"):
    if type(t) != torch.Tensor:
        t = torch.tensor(t, dtype=torch.float32).to(x)
    if type(s) != torch.Tensor:
        s = torch.tensor(s, dtype=torch.float32).to(x)
    if type(b) != torch.Tensor:
        b = torch.tensor(b, dtype=torch.float32).to(x)
    return ext.continous_sum(x, t, s, b, alpha, beta, blocks, exec_type)

def d_continous_sum(x, t, s, b, y, alpha, beta, blocks=16):
    if type(t) != torch.Tensor:
        t = torch.tensor(t, dtype=torch.float32).to(x)
    if type(s) != torch.Tensor:
        s = torch.tensor(s, dtype=torch.float32).to(x)
    if type(b) != torch.Tensor:
        b = torch.tensor(b, dtype=torch.float32).to(x)
    return ext.d_continous_sum(x, t, s, b, y, alpha, beta, blocks)


def inv_sigmoid(x):
    return 1 / (1 + x.exp())

def weighted_sum(x: torch.Tensor, t, s, b):
    return inv_sigmoid(b * (x - t - s)) - inv_sigmoid(b * (x - s))

def continous_sum_torch(x: torch.Tensor, t, s, b):
    t1 = float(t); s1 = float(s)
    seq_len = int((x.shape[1] - t1) / s1) + 1
    y = torch.empty(x.shape[0], seq_len, x.shape[2]).to(x)
    stride = 0
    for i in range(seq_len):
        w = torch.arange(0, x.shape[1], 1).to(x)
        w_ = weighted_sum(w, t, stride, b)
        w = (w_[None, :, None] * x).sum(1)
        y[:, i, :] = w
        stride += s
    return y

def show(x: torch.Tensor):
    plt.imshow(x)
    plt.show()

def plot(x: torch.Tensor):
    plt.plot(torch.arange(0, x.shape[-1]), x)
    plt.show()

def equal_all(a, b, rtol=0.00001, atol=1e-8):
    for i, j in zip(a, b):
        print(torch.allclose(i, j.to(i), rtol, atol))

class ContinousSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t, s, b, alpha, beta):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.save_for_backward(x, t, s, b)
        return continous_sum(x, t, s, b, alpha, beta)
    
    @staticmethod
    def backward(ctx, grad):
        x, t, s, b = ctx.saved_tensors
        return tuple(d_continous_sum(x, t, s, b, grad, ctx.alpha, ctx.beta) + [None, None])

from torch.autograd import gradcheck

dim = 32
t1 = 5; s1 = 5; b1 = 2
input = torch.randn((1, 32, dim), dtype=torch.double, requires_grad=True)
t = torch.full((1, dim), t1, dtype=torch.double, requires_grad=True)
s = torch.full((1, dim), s1, dtype=torch.double, requires_grad=True)
b = torch.full((1, dim), b1, dtype=torch.double, requires_grad=True)


# %%
x = continous_sum(input, t, s, b, 1, 0)
y = d_continous_sum(x, t, s, b, torch.ones_like(x), 1, 0)
input = input.cuda(); t = t.cuda(); s = s.cuda(); b = b.cuda()
x1 = continous_sum(input, t, s, b, 1, 0)
y1 = d_continous_sum(x1, t, s, b, torch.ones_like(x1), 1, 0)

equal_all([x], [x1])
equal_all(y, y1)

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

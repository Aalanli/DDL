# %%
import os

import torch
from torch.utils.cpp_extension import load

def build_extension(name, *sources):
    build_dir = 'extension_cache/'
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)
    if not os.path.exists(build_dir + name):
        os.mkdir(build_dir + name)
    
    return load(name, sources=sources, verbose=True, build_directory=build_dir+name)

ext = build_extension('csum', 'continous_sum.cpp', 'continous_sum.cu')


class ContinousSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t, s, b, alpha, beta):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.save_for_backward(x, t, s, b)
        return ext.continous_sum(x, t, s, b, alpha, beta, 2, exec_type="v1")
    
    @staticmethod
    def backward(ctx, grad):
        x, t, s, b = ctx.saved_tensors
        return tuple(ext.d_continous_sum(x, t, s, b, grad, ctx.alpha, ctx.beta, 1) + [None, None])


class ContinousPool1D(torch.nn.Module):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, b: torch.Tensor):
        if type(t) == float:
            t = torch.full(x.shape, t, device=x.device)
        if type(s) == float:
            s = torch.full(x.shape, s, device=x.device)
        if type(b) == float:
            b = torch.full(x.shape, b, device=x.device)
        return ContinousSum.apply(x, t, s, b, self.alpha, self.beta)


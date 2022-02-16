
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

values = np.random.normal(0, 3, 100)
stop_val = 10
beta_val = 20.0
theta_val = 4.0
arr = values[:stop_val+1]

x = np.arange(0.0, stop_val, 0.01)


def flat_sigmoid(arr: np.ndarray, beta, theta):
    return 1 / (1 + np.exp(arr * beta)) - 1 / (1 + np.exp(beta * (arr + theta)))

def bucket_sum(arr: np.ndarray, alpha: int, beta, theta):
    dist = np.arange(0, arr.shape[0]) - alpha
    w = flat_sigmoid(dist, beta, theta)

    return (arr * w).sum()

def real(arr: np.ndarray, alpha: int, theta):
    stop = math.ceil(alpha)
    return arr[max(int(stop-theta), 0):stop].sum()

def f_estimate(arr, x, beta, theta):
    y = [bucket_sum(arr, i, beta, theta) for i in x]
    return np.array(y)

def f_real(arr, x, theta):
    y_real = [real(arr, i, theta) for i in x]
    return np.array(y_real)


est, = plt.plot(x, f_estimate(arr, x, beta_val, theta_val), label='approximate')
rel, = plt.plot(x, f_real(arr, x, theta_val), label='real')

axcolor = 'lightgoldenrodyellow'
beta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
theta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

beta = Slider(beta, 'Beta', 0.1, 40.0, valinit=beta_val, valstep=0.01)
theta = Slider(theta, 'Theta', 0.1, stop_val, valinit=theta_val, valstep=0.01)


def update(val):
    b = beta.val
    t = theta.val

    est.set_ydata(f_estimate(arr, x, b, t))
    rel.set_ydata(f_real(arr, x, t))
    fig.canvas.draw_idle()

beta.on_changed(update)
theta.on_changed(update)



plt.show()
import torch

def d_flat_sigmoid(x: torch.Tensor, b: int, c: int):
    e_x = (x * b).exp()
    ce_x = (b * x + b * c).exp()
    return b * e_x / (1 + e_x) ** 2 - b * ce_x / (1 + ce_x) ** 2

def safe_d_flat_sigmoid(x: torch.Tensor, b: int, c: int):
    e_x = (x * b).exp()
    ce_x = (b * x + b * c).exp()
    return b / (1 / e_x + e_x + 2) - b / (1 / ce_x + ce_x + 2)

a = torch.rand(1)
b = 20
c = 5

print(d_flat_sigmoid(a, b, c), safe_d_flat_sigmoid(a, b, c))

# %%
import torch
from torch.autograd import Function, gradcheck

class Split(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        return (2 * x).sin(), (3 * x).exp()
    
    @staticmethod
    def backward(ctx, d1, d2):
        x, = ctx.saved_tensors
        return d1 * (2 * x).cos() * 2 + d2 * (3 * x).exp() * 3

i = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),)
test = gradcheck(Split.apply, i)
print(test)

python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationFunctions:
    class SiLU(nn.Module):
        @staticmethod
        def forward(x):
            return x * torch.sigmoid(x)

    class Hardswish(nn.Module):
        @staticmethod
        def forward(x):
            return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0

    class Mish(nn.Module):
        @staticmethod
        def forward(x):
            return x * F.softplus(x).tanh()

    class MemoryEfficientMish(nn.Module):
        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.mul(torch.tanh(F.softplus(x)))

            @staticmethod
            def backward(ctx, grad_output):
                x = ctx.saved_tensors[0]
                sx = torch.sigmoid(x)
                fx = F.softplus(x).tanh()
                return grad_output * (fx + x * sx * (1 - fx * fx))

        def forward(self, x):
            return self.F.apply(x)

    class FReLU(nn.Module):
        def __init__(self, c1, k=3):
            super().__init__()
            self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
            self.bn = nn.BatchNorm2d(c1)

        def forward(self, x):
            return torch.max(x, self.bn(self.conv(x)))

    class AconC(nn.Module):
        def __init__(self, c1):
            super().__init__()
            self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
            self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
            self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

        def forward(self, x):
            dpx = (self.p1 - self.p2) * x
            return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x

    class MetaAconC(nn.Module):
        def __init__(self, c1, k=1, s=1, r=16):
            super().__init__()
            c2 = max(r, c1 // r)
            self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
            self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
            self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
            self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)

        def forward(self, x):
            y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
            beta = torch.sigmoid(self.fc2(self.fc1(y)))
            dpx = (self.p1 - self.p2) * x
            return dpx * torch.sigmoid(beta * dpx) + self.p2 * x

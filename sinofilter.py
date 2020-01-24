import time

import opt_einsum
import torch
import torch_radon_cuda


def bench(f, w, x):
    with torch.no_grad():
        for i in range(50):
            f(x, w)

        torch.cuda.synchronize()
        s = time.time()
        for i in range(1000):
            f(x, w)
        torch.cuda.synchronize()
        e = time.time()

    return e - s


def transpose(x, w):
    y = x.transpose(0, 1).reshape(8, 16 * 256, 256).contiguous()
    return torch.bmm(y, w).reshape(8, 16, 256, 256).transpose(0, 1).contiguous()


device = torch.device("cuda")

X = torch.Tensor(16, 8, 256, 256).normal_().to(device)
W = torch.Tensor(8, 256, 256).normal_().to(device)

# print(opt_einsum.contract_expression("bcwh,bcwh,chj->bcwj", X.size(), X.size(), W.size()))
# print(opt_einsum.contract_expression("bcwh,cjh->bcwj", X.size(), W.size()))
T = False
y = torch.einsum("bchw,chj->bcjw" if T else "bcwh,cjh->bcwj", X, W)  #
y_ = torch_radon_cuda.sinofilter(X, W)
# print(y_.size())
#
print(y[0, 0])
print(y_[0, 0])
print(torch.sum(y ** 2), torch.sum(y_ ** 2), torch.sum((y - y_) ** 2))
# print(torch.allclose(y, y_))
print("Einsum:", bench(lambda x, w: torch.einsum("bcwh,cjh->bcwj", x, w), W, X))
# print("Einsum:", bench(lambda x, w: torch.einsum("bcwh,chj->bcwj", x, w), W, X))
# # opt = opt_einsum.contract_expression("bcwh,chj->bcwj", X.size(), W.size())
# # print("Opt Einsum:", bench(lambda x, w: opt(x, w, backend="torch"), W, X))
# # # # # print("Einsum Transpose W:", bench(lambda x, w: torch.einsum("bcwh,chj->bcjw", x, w), W, X))
print("Manual:", bench(torch_radon_cuda.sinofilter, W, X))

try:
    import torch_radon_cuda
except Exception as e:
    print("Importing exception")

import torch
from torch.autograd import Function


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, angles, tex_cache, vol_cfg, proj_cfg, exec_cfg_generator, exec_cfg=None):
        exec_cfg = exec_cfg_generator(vol_cfg, proj_cfg,  x.dtype == torch.half) if exec_cfg is None else exec_cfg
        sinogram = torch_radon_cuda.forward(x, angles, tex_cache, vol_cfg, proj_cfg, exec_cfg)
        ctx.tex_cache = tex_cache
        ctx.vol_cfg = vol_cfg
        ctx.proj_cfg = proj_cfg.copy()
        ctx.exec_cfg_generator = exec_cfg_generator
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        exec_cfg = ctx.exec_cfg_generator(ctx.vol_cfg, ctx.proj_cfg, grad_x.dtype == torch.half)
        grad = torch_radon_cuda.backward(grad_x, angles, ctx.tex_cache, ctx.vol_cfg, ctx.proj_cfg, exec_cfg)
        return grad, None, None, None, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, angles, tex_cache, vol_cfg, proj_cfg, exec_cfg_generator, exec_cfg=None):
        exec_cfg = exec_cfg_generator(vol_cfg, proj_cfg,  x.dtype == torch.half) if exec_cfg is None else exec_cfg
        image = torch_radon_cuda.backward(x, angles, tex_cache,  vol_cfg, proj_cfg, exec_cfg)
        ctx.tex_cache = tex_cache
        ctx.vol_cfg = vol_cfg
        ctx.proj_cfg = proj_cfg.copy()
        ctx.exec_cfg_generator = exec_cfg_generator
        ctx.save_for_backward(angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        exec_cfg = ctx.exec_cfg_generator(ctx.vol_cfg, ctx.proj_cfg, grad_x.dtype == torch.half)
        grad = torch_radon_cuda.forward(grad_x, angles, ctx.tex_cache, ctx.vol_cfg, ctx.proj_cfg, exec_cfg)
        return grad, None, None, None, None, None, None

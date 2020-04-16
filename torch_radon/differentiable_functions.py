import torch_radon_cuda
from torch.autograd import Function


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, rays, angles, tex_cache):
        print(x.dtype, rays.dtype, angles.dtype)
        sinogram = torch_radon_cuda.forward(x, rays, angles, tex_cache)
        ctx.tex_cache = tex_cache
        ctx.save_for_backward(rays, angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.backward(grad_x, rays, angles, ctx.tex_cache, False), None, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, rays, angles, tex_cache, extend):
        image = torch_radon_cuda.backward(x, rays, angles, tex_cache, extend)
        ctx.tex_cache = tex_cache
        ctx.save_for_backward(rays, angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.forward(grad_x, rays, angles, ctx.tex_cache), None, None, None, None

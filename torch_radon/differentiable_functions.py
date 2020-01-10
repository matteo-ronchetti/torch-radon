import torch_radon_cuda
from torch.autograd import Function


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, rays, angles, tex_cache, back_tex_cache):
        sinogram = torch_radon_cuda.forward(x, rays, angles, tex_cache)
        ctx.back_tex_cache = back_tex_cache
        ctx.save_for_backward(rays, angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.backward(grad_x, rays, angles, ctx.back_tex_cache), None, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, rays, angles, tex_cache, back_tex_cache):
        image = torch_radon_cuda.backward(x, rays, angles, back_tex_cache)
        ctx.tex_cache = tex_cache
        ctx.save_for_backward(rays, angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        rays, angles = ctx.saved_variables
        return torch_radon_cuda.forward(grad_x, rays, angles, ctx.tex_cache), None, None, None, None

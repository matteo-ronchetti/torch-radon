import torch_radon_cuda
from torch.autograd import Function


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, det_count, det_spacing, angles, tex_cache):
        sinogram = torch_radon_cuda.forward(x, det_count, det_spacing, angles, tex_cache)
        ctx.tex_cache = tex_cache
        ctx.det_count = det_count
        ctx.det_spacing = det_spacing
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        return torch_radon_cuda.backward(grad_x, ctx.det_count, ctx.det_spacing, angles, ctx.tex_cache), None, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, det_count, det_spacing, angles, tex_cache):
        image = torch_radon_cuda.backward(x, det_count, det_spacing, angles, tex_cache)
        ctx.tex_cache = tex_cache
        ctx.det_count = det_count
        ctx.det_spacing = det_spacing
        ctx.save_for_backward(angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        return torch_radon_cuda.forward(grad_x, ctx.det_count, ctx.det_spacing, angles, ctx.tex_cache), None, None, None, None

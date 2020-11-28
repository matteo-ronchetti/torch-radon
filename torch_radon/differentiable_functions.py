try:
    import torch_radon_cuda
except Exception as e:
    print("Importing exception")

from torch.autograd import Function


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, angles, tex_cache, rays_cfg):
        sinogram = torch_radon_cuda.forward(x, angles, tex_cache, rays_cfg)
        ctx.tex_cache = tex_cache
        ctx.rays_cfg = rays_cfg
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        angles, = ctx.saved_variables
        grad = torch_radon_cuda.backward(grad_x, angles, ctx.tex_cache, ctx.rays_cfg)
        return grad, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, angles, tex_cache, rays_cfg):
        image = torch_radon_cuda.backward(x, angles, tex_cache, rays_cfg)
        ctx.tex_cache = tex_cache
        ctx.rays_cfg = rays_cfg
        ctx.save_for_backward(angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        if not grad_x.is_contiguous():
            grad_x = grad_x.contiguous()

        angles, = ctx.saved_variables
        grad = torch_radon_cuda.forward(grad_x, angles, ctx.tex_cache, ctx.rays_cfg)
        return grad, None, None, None

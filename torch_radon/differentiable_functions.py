import torch_radon_cuda
from torch.autograd import Function


class RadonForward(Function):
    @staticmethod
    def forward(ctx, x, det_count, det_spacing, angles, tex_cache, clip_to_circle):
        sinogram = torch_radon_cuda.forward(x, det_count, det_spacing, angles, tex_cache, clip_to_circle)
        ctx.tex_cache = tex_cache
        ctx.det_count = det_count
        ctx.det_spacing = det_spacing
        ctx.clip_to_circle = clip_to_circle
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        grad = torch_radon_cuda.backward(grad_x, ctx.det_count, ctx.det_spacing, angles, ctx.tex_cache,
                                         ctx.clip_to_circle)
        return grad, None, None, None, None, None


class RadonForwardFanbeam(Function):
    @staticmethod
    def forward(ctx, x, s_dist, d_dist, det_count, det_spacing, angles, tex_cache, clip_to_circle):
        sinogram = torch_radon_cuda.forward_fanbeam(x, s_dist, d_dist, det_count, det_spacing, angles, tex_cache,
                                                    clip_to_circle)
        ctx.tex_cache = tex_cache
        ctx.s_dist = s_dist
        ctx.d_dist = d_dist
        ctx.det_count = det_count
        ctx.det_spacing = det_spacing
        ctx.clip_to_circle = clip_to_circle
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        grad = torch_radon_cuda.backward_fanbeam(grad_x, ctx.s_dist, ctx.d_dist, ctx.det_count, ctx.det_spacing, angles,
                                                 ctx.tex_cache, ctx.clip_to_circle)
        return grad, None, None, None, None, None, None, None


class RadonBackprojectionFanbeam(Function):
    @staticmethod
    def forward(ctx, x, s_dist, d_dist, det_count, det_spacing, angles, tex_cache, clip_to_circle):
        sinogram = torch_radon_cuda.backward_fanbeam(x, s_dist, d_dist, det_count, det_spacing, angles, tex_cache,
                                                     clip_to_circle)
        ctx.tex_cache = tex_cache
        ctx.s_dist = s_dist
        ctx.d_dist = d_dist
        ctx.det_count = det_count
        ctx.det_spacing = det_spacing
        ctx.clip_to_circle = clip_to_circle
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        grad = torch_radon_cuda.forward_fanbeam(grad_x, ctx.s_dist, ctx.d_dist, ctx.det_count, ctx.det_spacing, angles, ctx.tex_cache,
                                         ctx.clip_to_circle)
        return grad, None, None, None, None, None, None, None


class RadonBackprojection(Function):
    @staticmethod
    def forward(ctx, x, det_count, det_spacing, angles, tex_cache, clip_to_circle):
        image = torch_radon_cuda.backward(x, det_count, det_spacing, angles, tex_cache, clip_to_circle)
        ctx.tex_cache = tex_cache
        ctx.det_count = det_count
        ctx.det_spacing = det_spacing
        ctx.clip_to_circle = clip_to_circle
        ctx.save_for_backward(angles)

        return image

    @staticmethod
    def backward(ctx, grad_x):
        angles, = ctx.saved_variables
        grad = torch_radon_cuda.forward(grad_x, ctx.det_count, ctx.det_spacing, angles, ctx.tex_cache,
                                        ctx.clip_to_circle)
        return grad, None, None, None, None, None

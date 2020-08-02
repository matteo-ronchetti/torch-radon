import torch


def _normalize_shape(x, d):
    old_shape = x.size()[:-d]
    x = x.view(-1, *(x.size()[-d:]))
    return x, old_shape


def _unnormalize_shape(y, old_shape):
    if isinstance(y, torch.Tensor):
        y = y.view(*old_shape, *(y.size()[1:]))
    elif isinstance(y, tuple):
        y = [yy.view(*old_shape, *(yy.size()[1:])) for yy in y]

    return y


def normalize_shape(d):
    """
    Input with shape (batch_1, ..., batch_n, s_1, ..., s_d) is reshaped to (batch, s_1, s_2, ...., s_d)
    fed to f and output is reshaped to (batch_1, ..., batch_n, s_1, ..., s_o).
    :param d: Number of non-batch dimensions
    """

    def wrap(f):
        def wrapped(self, x, *args, **kwargs):
            x, old_shape = _normalize_shape(x, d)

            y = f(self, x, *args, **kwargs)

            return _unnormalize_shape(y, old_shape)

        wrapped.__doc__ = f.__doc__
        return wrapped

    return wrap

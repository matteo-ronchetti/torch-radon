import torch


def normalize_shape(in_dim):
    def wrap(f):
        def wrapped(self, x, *args, **kwargs):
            old_shape = x.size()[:-in_dim]
            x = x.view(-1, *(x.size()[-in_dim:]))

            y = f(self, x, *args, **kwargs)

            if isinstance(y, torch.Tensor):
                y = y.view(*old_shape, *(y.size()[1:]))
            elif isinstance(y, tuple):
                y = [yy.view(*old_shape, *(yy.size()[1:])) for yy in y]

            return y

        return wrapped

    return wrap


class Sample:
    @normalize_shape(3)
    def sample(self, x):
        print("    ", x.size())
        y = x[:, 0]
        print("    ", y.size())
        return y


x = torch.empty(2, 3, 4, 5, 6)
print(x.size())
s = Sample()
y = s.sample(x)
print(y.size())


x = torch.empty(4, 5, 6)
print(x.size())
s = Sample()
y = s.sample(x)
print(y.size())


x = torch.empty(5, 6)
print(x.size())
s = Sample()
y = s.sample(x)
print(y.size())
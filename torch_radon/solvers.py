import torch


class Landweber:
    def __init__(self, operator, circle_mask=True):
        self.operator = operator
        self.circle_mask = circle_mask

    def project(self, x):
        with torch.no_grad():
            x = torch.relu(x)
            # TODO circle mask
            return x

    def normalize(self, x):
        size = x.size()
        x = x.view(size[0], -1)
        norm = torch.norm(x, dim=1)
        x /= norm.view(-1, 1)
        return x.view(*size), torch.max(norm).item()

    def estimate_alpha(self, img_size, device, n_iter=50, batch_size=8):
        x = torch.randn((batch_size, img_size, img_size), device=device)
        x, _ = self.normalize(x)
        for i in range(n_iter):
            next_x = self.operator.backward(self.operator.forward(x))
            x, v = self.normalize(next_x)

        return 2.0/v

    def run(self, guess, y, alpha, iterations=1, callback=None):
        res = []
        with torch.no_grad():
            x = guess
            for i in range(iterations):
                x = self.project(x - alpha * self.operator.backward(self.operator.forward(x) - y))
                if callback is not None:
                    res.append(callback(x))

        if callback is not None:
            return x, res
        else:
            return x

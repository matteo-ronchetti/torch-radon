import torch
from torch_radon import Radon
import torch.nn
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.radon = Radon(256, np.linspace(0, np.pi, 256))

    def forward(self, x):
        print("Angles", self.radon.angles.device)
        return self.radon.forward(x)


# def run(gpu, args):
#     dist.init_process_group(backend='nccl')
#
#     radon = Radon(256, torch.device("cuda", gpu))


x = torch.FloatTensor(256, 1, 256, 256).normal_()  # .to(torch.device("cuda"))

model = Model().to(torch.device("cuda", 0))
y = model(x.to(torch.device("cuda", 0)))

# model = model.to(torch.device("cuda", 0))
# y = model(x.to(torch.device("cuda", 0)))

# model = model.to(torch.device("cuda", 1))
# y = model(x.to(torch.device("cuda", 1)))

model = torch.nn.DataParallel(model, [0, 1])
for i in range(100):
    y_ = model(x)
print("Done", flush=True)
print(y.device, y_.device)

print(torch.allclose(y, y_))


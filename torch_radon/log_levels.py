import torch_radon_cuda

DEBUG = 0
INFO = 1
WARN = 2
WARNING = 2
ERROR = 3

def set_log_level(level):
    level = max(0, min(int(level), ERROR))
    torch_radon_cuda.set_log_level(level)
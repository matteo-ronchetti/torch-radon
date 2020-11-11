import numpy as np
import torch
import time
import json

from torch_radon import Radon, RadonFanbeam


def benchmark(f, x, warmup, repeats):
    for _ in range(warmup):
        y = f(x)

    torch.cuda.synchronize()
    s = time.time()
    for _ in range(repeats):
        y = f(x)
        torch.cuda.synchronize()
    execution_time = time.time() - s

    return execution_time, y


def bench_parallel_forward(phantom, det_count, num_angles, warmup, repeats):
    radon = Radon(phantom.size(1), np.linspace(0, np.pi, num_angles, endpoint=False), det_count)

    f = lambda x: radon.forward(x)

    return benchmark(f, phantom, warmup, repeats)


def bench_parallel_backward(phantom, det_count, num_angles, warmup, repeats):
    radon = Radon(phantom.size(1), np.linspace(0, np.pi, num_angles, endpoint=False), det_count)

    sino = radon.forward(phantom)
    f = lambda x: radon.forward(x)

    return benchmark(f, sino, warmup, repeats)


def bench_fanbeam_forward(phantom, det_count, num_angles, source_dist, det_dist, warmup, repeats):
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    radon = RadonFanbeam(phantom.size(1), angles, source_dist, det_dist, det_count)

    f = lambda x: radon.forward(x)

    return benchmark(f, phantom, warmup, repeats)


def bench_fanbeam_backward(phantom, det_count, num_angles, source_dist, det_dist, warmup, repeats):
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    radon = RadonFanbeam(phantom.size(1), angles, source_dist, det_dist, det_count)

    sino = radon.forward(phantom)
    f = lambda x: radon.backward(x)

    return benchmark(f, sino, warmup, repeats)


def do_benchmarks(config, phantom):
    output_prefix = "hp_" if phantom.dtype == torch.float16 else ""

    batch_size = config["batch size"]
    warmup = config["warmup"]
    repeats = config["repeats"]

    results = []
    for task in config["tasks"]:
        print(f"Benchmarking task '{task['task']}'")

        if task["task"] == "parallel forward":
            exec_time, y = bench_parallel_forward(phantom, task["num angles"], task["det count"], warmup, repeats)
        elif task["task"] == "parallel backward":
            exec_time, y = bench_parallel_backward(phantom, task["num angles"], task["det count"], warmup, repeats)
        elif task["task"] == "fanbeam forward":
            exec_time, y = bench_fanbeam_forward(phantom,
                                                 task["num angles"], task["det count"],
                                                 task["source distance"], task["detector distance"],
                                                 warmup, repeats)
        elif task["task"] == "fanbeam backward":
            exec_time, y = bench_fanbeam_backward(phantom,
                                                  task["num angles"], task["det count"],
                                                  task["source distance"], task["detector distance"],
                                                  warmup, repeats)
        else:
            print(f"Unknown task '{task['task']}'")
            continue

        print("Execution time:", exec_time)
        fps = (batch_size * repeats) / exec_time
        print("FPS:", fps)
        np.save(output_prefix + task["output"], y[0].cpu().float().numpy())

        res = dict()
        for k in task:
            if k != "output":
                res[k] = task[k]

        res["time"] = exec_time
        res["fps"] = fps
        results.append(res)
        print("")

    return results


def main():
    with open("../config.json") as f:
        config = json.load(f)

    batch_size = config["batch size"]

    # load phantom
    phantom = np.load("../" + config["input"])

    # add bach dimension and repeat to match batch size
    phantom = np.expand_dims(phantom, axis=0)
    phantom = np.vstack([phantom] * batch_size)
    print(phantom.shape)

    # Place phantom on the GPU
    device = torch.device("cuda")
    phantom = torch.FloatTensor(phantom).to(device)

    gpu_name = torch.cuda.get_device_name(device)
    print(f"Running benchmarks on {gpu_name}")

    print("\nBenchmarking Single Precision")
    results = do_benchmarks(config, phantom)
    print("\n\nBenchmarking Half Precision")
    results_hp = do_benchmarks(config, phantom.half())

    with open("../torch_radon_results.json", "w") as f:
        json.dump({
            "library": "TorchRadon",
            "batch_size": config["batch size"],
            "warmup": config["warmup"],
            "repeats": config["repeats"],
            "gpu": gpu_name,

            "results": results
        }, f, indent=4)

    with open("../torch_radon_hp_results.json", "w") as f:
        json.dump({
            "library": "TorchRadon half",
            "batch_size": config["batch size"],
            "warmup": config["warmup"],
            "repeats": config["repeats"],
            "gpu": gpu_name,

            "results": results_hp
        }, f, indent=4)


main()

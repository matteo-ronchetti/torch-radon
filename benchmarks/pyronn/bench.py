import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import time
import json

from pyronn.ct_reconstruction.layers.projection_2d import parallel_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import parallel_backprojection2d
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D

from pyronn.ct_reconstruction.layers.projection_2d import fan_projection2d
from pyronn.ct_reconstruction.layers.backprojection_2d import fan_backprojection2d
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D

from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory

import sys
import ctypes


def get_gpu_name():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    name = b' ' * 100
    device = ctypes.c_int()

    cuda.cuDeviceGet(ctypes.byref(device), 0)
    cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
    return name.split(b'\0', 1)[0].decode()


def benchmark(f, x, warmup, repeats):
    for _ in range(warmup):
        y = f(x)

    s = time.time()
    for _ in range(repeats):
        y = f(x)
    execution_time = time.time() - s

    return execution_time, y


def create_parallel_geometry(det_count, num_angles):
    # Detector Parameters:
    detector_shape = det_count
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = num_angles
    angular_range = np.pi

    # create Geometry class
    geometry = GeometryParallel2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections,
                                  angular_range)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    return geometry


def create_fanbeam_geometry(det_count, num_angles, source_dist, det_dist):
    # Detector Parameters:
    detector_shape = det_count
    detector_spacing = 1

    # Trajectory Parameters:
    number_of_projections = num_angles
    angular_range = np.pi

    source_detector_distance = source_dist + det_dist

    # create Geometry class
    geometry = GeometryFan2D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections,
                             angular_range, source_detector_distance, source_dist)
    geometry.set_trajectory(circular_trajectory.circular_trajectory_2d(geometry))

    return geometry


def bench_parallel_forward(phantom, det_count, num_angles, warmup, repeats):
    geometry = create_parallel_geometry(det_count, num_angles)

    f = lambda x: parallel_projection2d(x, geometry)

    return benchmark(f, phantom, warmup, repeats)


def bench_parallel_backward(phantom, det_count, num_angles, warmup, repeats):
    geometry = create_parallel_geometry(det_count, num_angles)

    sino = parallel_projection2d(phantom, geometry)
    f = lambda x: parallel_backprojection2d(x, geometry)

    return benchmark(f, sino, warmup, repeats)


def bench_fanbeam_forward(phantom, det_count, num_angles, source_dist, det_dist, warmup, repeats):
    geometry = create_fanbeam_geometry(det_count, num_angles, source_dist, det_dist)

    f = lambda x: fan_projection2d(x, geometry)

    return benchmark(f, phantom, warmup, repeats)


def bench_fanbeam_backward(phantom, det_count, num_angles, source_dist, det_dist, warmup, repeats):
    geometry = create_fanbeam_geometry(det_count, num_angles, source_dist, det_dist)

    sino = fan_projection2d(phantom, geometry)
    f = lambda x: fan_backprojection2d(x, geometry)

    return benchmark(f, sino, warmup, repeats)


with open("../config.json") as f:
    config = json.load(f)

batch_size = config["batch size"]
warmup = config["warmup"]
repeats = config["repeats"]

# load phantom
phantom = np.load("../" + config["input"])

# define volume
volume_size = phantom.shape[0]
volume_shape = [volume_size, volume_size]
volume_spacing = [1, 1]

# add bach dimension and repeat to match batch size
phantom = np.expand_dims(phantom, axis=0)
phantom = np.vstack([phantom] * batch_size)
print(phantom.shape)

# Place phantom on the GPU
with tf.device('/GPU:0'):
    phantom = tf.convert_to_tensor(phantom, dtype=tf.float32)

gpu_name = get_gpu_name()
print(f"Running benchmarks on {gpu_name}")

print("\n")
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
        print(f"ERROR Unknown task '{task['task']}'")

    print("Execution time:", exec_time)
    fps = (batch_size * repeats) / exec_time
    print("FPS:", fps)
    np.save(task["output"], y[0])

    res = dict()
    for k in task:
        if k != "output":
            res[k] = task[k]

    res["time"] = exec_time
    res["fps"] = fps
    results.append(res)
    print("")

with open("../pyronn_results.json", "w") as f:
    config = json.dump({
        "library": "pyronn",
        "batch_size": config["batch size"],
        "warmup": config["warmup"],
        "repeats": config["repeats"],
        "gpu": gpu_name,

        "results": results
    }, f, indent=4)

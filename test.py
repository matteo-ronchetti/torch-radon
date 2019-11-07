import torch
import cv2
import numpy as np
import radon
import time
import astra


def astra_single_fp(x, angles):
    vol_geom = astra.create_vol_geom(128, 128)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 128, -angles.cpu().numpy())
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    x_ = x.cpu().numpy()

    return astra.create_sino(x_, proj_id)


def astra_batch_fp(x, angles):
    x_ = x.cpu().numpy()

    vol_geom = astra.create_vol_geom(x.size(1), x.size(2), x.size(0))
    phantom_id = astra.data3d.create('-vol', vol_geom, data=x_)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, x_.shape[0], x_.shape[1], angles.cpu().numpy())

    return astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)

def astra_batch_bp(proj_id, angles, s, bs):
    vol_geom = astra.create_vol_geom(s, s, bs)
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    return astra.data3d.get(rec_id)

def save(name, xx):
    x = xx.copy()
    x -= np.min(x)
    x /= np.max(x) / 255
    x = x.astype(np.uint8)
    cv2.imwrite(name, x)


def main():
    print(f"Found GPU {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')

    # read test image
    img = cv2.imread("phantom.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    angles = np.linspace(0, 2 * np.pi, 180).astype(np.float32)

    # compute rays
    s = img.shape[0] // 2
    locations = np.arange(2 * s) - s + 0.5
    ys = np.sqrt(s ** 2 - locations ** 2)
    locations = locations.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    rays = np.hstack((locations, -ys, locations, ys))
    print("Rays shape", rays.shape)

    batch_size = 16
    # move to gpu
    x = torch.FloatTensor(img).to(device).view(1, 128, 128).repeat(batch_size, 1, 1)
    rays = torch.FloatTensor(rays).to(device)
    angles = torch.FloatTensor(angles).to(device)

    s = time.time()
    y = radon.forward(x, rays, angles)
    e = time.time()
    my_fp_time = e - s
    print("My FP Time", my_fp_time)

    f =  2 * np.abs(np.fft.fftfreq(256))
    f = torch.FloatTensor(f).to(device)
    yf = radon.filter_sinogram(y, f)
    print(yf.shape)
    save("filtered_sino.png", yf[0].cpu().numpy())
    
    s = time.time()
    x_ = radon.backward(yf, rays, angles)
    e = time.time()
    my_bp_time = e - s
    print("My BP Time", my_bp_time)
    save("fbp.png", x_[0].cpu().numpy())
    
    s = time.time()
    proj_id, y_ = astra_batch_fp(x, angles)
    e = time.time()
    astra_fp_time = e - s
    print("Astra FP Time", astra_fp_time)

    s = time.time()
    ax_ = astra_batch_bp(proj_id, angles, 128, batch_size)
    e = time.time()
    astra_bp_time = e - s
    print("Astra BP Time", astra_bp_time)
    
    print("Speedup, fp:", astra_fp_time/my_fp_time, " bp:", astra_bp_time/my_bp_time, " total:", (astra_bp_time + astra_fp_time)/(my_bp_time + my_fp_time))
    
    save("astra_bp.png", ax_[0])
    
    print("Batch error", np.linalg.norm(y_ - y.cpu().numpy()) / np.linalg.norm(y_))
    print("Batch BP error", np.linalg.norm(ax_ - x_.cpu().numpy()) / np.linalg.norm(ax_))

    
    _, ref = astra_single_fp(x[0], angles)
    error = np.linalg.norm(ref - y[0].cpu().numpy()) / np.linalg.norm(ref)
    print("My-ref Error", error)
    error = np.linalg.norm(ref - y_[0]) / np.linalg.norm(ref)
    print("Astra-ref Error", error)


main()
# y_[0] -= np.min(y_[0])
# y_[0] /= np.max(y_[0]) / 255
# y_[0] = y_[0].astype(np.uint8)
# cv2.imwrite("res.png", y_[0])
#
# ref -= np.min(ref)
# ref /= np.max(ref) / 255
# ref = ref.astype(np.uint8)
# cv2.imwrite("ref.png", ref)

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
from torch_radon import Radon
from astra_wrapper import AstraWrapper


def save(name, xx):
    x = xx.copy()
    x -= np.min(x)
    x /= np.max(x) / 255
    x = x.astype(np.uint8)
    cv2.imwrite(name, x)


def relative_error(ref, x):
    return np.linalg.norm(ref - x) / np.linalg.norm(ref)


def test_backward_projection(radon, astra, my_fp, astra_fp_id, angles, img_size, batch_size):
    s = time.time()
    my_bp = radon.backprojection(my_fp, angles)
    e = time.time()
    my_bp_time = e - s
    save("backprojection.png", my_bp[0].cpu().numpy())

    s = time.time()
    astra_bp = astra.backproject(astra_fp_id, img_size, batch_size)
    e = time.time()
    astra_bp_time = e - s

    error = relative_error(astra_bp, my_bp.cpu().numpy())

    return error, my_bp_time, astra_bp_time


def test_forward_projection(radon, astra, x, x_cpu, angles):
    s = time.time()
    for i in range(100):
        my_fp = radon.forward(x, angles)
    e = time.time()
    my_fp_time = e - s
    save("sinogram.png", my_fp[0].cpu().numpy())

    s = time.time()
    for i in range(100):
        astra_fp_id, astra_fp = astra.forward(x_cpu)
    e = time.time()
    astra_fp_time = e - s

    error = relative_error(astra_fp, my_fp.cpu().numpy())

    return error, my_fp_time, astra_fp_time, my_fp, astra_fp_id


def main():
    n_angles = 180
    img_size = 128
    batch_size = 8

    print(f"Found GPU {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')

    # read test image
    img = cv2.imread("phantom.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    angles = np.linspace(0, 2 * np.pi, n_angles).astype(np.float32)

    radon = Radon(img_size).to(device)
    astra = AstraWrapper(angles)

    # move to gpu
    x = torch.FloatTensor(img).to(device).view(1, img_size, img_size).repeat(batch_size, 1, 1)
    x_cpu = x.cpu().numpy()
    angles = torch.FloatTensor(angles).to(device)

    # Check if results are close to what astra produces
    with torch.no_grad():
        print("Testing FP...")
        error, my_fp_time, astra_fp_time, my_fp, astra_fp_id = test_forward_projection(radon, astra, x, x_cpu, angles)
        print("FP error", error)

        print("\n\nTesting BP...")
        error, my_bp_time, astra_bp_time = test_backward_projection(radon, astra, my_fp, astra_fp_id, angles, img_size,
                                                                    batch_size)
        print("BP error", error)

    # print timings
    print("\n\nSPEED RESULTS")
    print("My FP Time", my_fp_time)
    print("Astra FP Time", astra_fp_time)
    print("My BP Time", my_bp_time)
    print("Astra BP Time", astra_bp_time)
    print("Speedup, fp:", astra_fp_time / my_fp_time, " bp:", astra_bp_time / my_bp_time, " total:",
          (astra_bp_time + astra_fp_time) / (my_bp_time + my_fp_time))

    print("\n\nTesting convolutional filter...")
    # load ramp filter
    f = np.load("ramp-filter.npy") # / 0.012281854 * np.pi/(2.88*2*n_angles)
    print("filter sum", np.sum(f))
    pad = f.shape[0] // 2
    conv_filter = torch.FloatTensor(f.reshape(1, 1, 1, -1)).to(device)
    conv_filter.requires_grad = True
    
    s = time.time()
    for _ in range(1):
        sinogram = radon.forward(x, angles)

        # apply filter to sinogram
        filtered_sinogram = F.conv2d(sinogram.view(batch_size, 1, n_angles, img_size), conv_filter, padding=(0, pad)).view(batch_size, n_angles, img_size)
        #print(filtered_sinogram.size())
#         fs = filtered_sinogram[0].detach().cpu().numpy()
#         x_ = x[0].cpu().numpy()
#         print(np.min(fs), np.max(fs), np.min(x_), np.max(x_))
#         print(fs.shape, x_.shape)
#         print(np.linalg.norm(fs)/ np.linalg.norm(x_))

#         save("filtered_sinogram.png", fs)

        # backproject
        fbp = radon.backprojection(filtered_sinogram, angles)
        fbp += -8235.569
        fbp /= (45317.203 -8235.569)/255.0
        fs = fbp[0].detach().cpu().numpy()
        print(np.min(fs), np.max(fs))
#         save("fbp.png", fbp[0].detach().cpu().numpy())

#         print(fbp[0].size(), x[0].size())
#         print(np.linalg.norm(fbp[0].detach().cpu().numpy()) / np.linalg.norm(x[0].cpu().numpy()))

        loss = F.mse_loss(fbp/255, x/255)
        print(loss.item())
        loss.backward()
    e = time.time()
    print(e-s)
        
    # s = time.time()
    # yf = radon.filter_sinogram(y)
    # my_fbp = radon.backward(yf, rays, angles)
    # e = time.time()
    # my_fbp_time = e - s
    # my_fbp /= 256 * 128
    # my_fbp /= (2 * angles.size(0))
    # my_fbp *= np.pi
    #
    # print("My FBP Time", my_fbp_time)
    # my_fbp = my_fbp[0].cpu().numpy()
    # save("fbp.png", my_fbp)
    #
    # save("filtered_sino.png", yf[0].cpu().numpy())
    #
    # s = time.time()
    # x_ = radon.backward(y, rays, angles)
    # e = time.time()
    # my_bp_time = e - s
    # print("My BP Time", my_bp_time)
    # save("bp.png", x_[0].cpu().numpy())
    #
    # s = time.time()
    # proj_id, y_ = astra_batch_fp(x, angles)
    # e = time.time()
    # astra_fp_time = e - s
    # print("Astra FP Time", astra_fp_time)
    #
    # s = time.time()
    # proj_id, _ = astra_single_fp(x[0], angles)
    # a_fbp = astra_fbp(proj_id, 128)
    # e = time.time()
    # astra_fbp_time = e - s
    # print("Astra FBP Time", astra_fbp_time)
    # save("astra_fbp.png", a_fbp)
    #
    # save("astra_bp.png", ax_[0])
    #
    # print("Batch error", np.linalg.norm(y_ - y.cpu().numpy()) / np.linalg.norm(y_))
    # print("Batch BP error", np.linalg.norm(ax_ - x_.cpu().numpy()) / np.linalg.norm(ax_))
    # print(np.max(a_fbp), np.max(my_fbp))
    # print("FBP error", np.linalg.norm(a_fbp - my_fbp) / np.linalg.norm(a_fbp))
    #
    # _, ref = astra_single_fp(x[0], angles)
    # error = np.linalg.norm(ref - y[0].cpu().numpy()) / np.linalg.norm(ref)
    # print("My-ref Error", error)
    # error = np.linalg.norm(ref - y_[0]) / np.linalg.norm(ref)
    # print("Astra-ref Error", error)


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

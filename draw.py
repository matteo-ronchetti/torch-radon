import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 256
pad = 256
det_spacing = 1.0
angle = 0.0  # np.pi / 4
clip_to_circle = True

img = np.zeros((2 * pad + size, 2 * pad + size), dtype=np.uint8)

cv2.circle(img, (pad + size // 2, pad + size // 2), size // 2, 255)

for ray_id in range(0, size, 1):
    v = size / 2.0
    sx = int((ray_id - v) * det_spacing + 0.5)
    sy = int(0.71 * size)
    ex = sx
    ey = -sy

    cs = np.cos(angle)
    sn = np.sin(angle)

    rsx = sx * cs - sy * sn
    rsy = sx * sn + sy * cs
    rdx = ex * cs - ey * sn - rsx
    rdy = ex * sn + ey * cs - rsy

    if clip_to_circle:
        a = rdx * rdx + rdy * rdy;
        b = rsx * rdx + rsy * rdy;
        c = rsx * rsx + rsy * rsy - v * v;

        delta_sqrt = np.sqrt(max(b * b - a * c, 1.0));
        alpha_s = (-b - delta_sqrt) / a;
        alpha_e = (-b + delta_sqrt) / a;

        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        cv2.line(img, (int(rsx + pad), int(rsy) + pad), (int(rsx + rdx) + pad, int(rsy + rdy) + pad), 255)
    else:
        alpha_x_m = (-v - rsx) / rdx
        alpha_x_p = (v - rsx) / rdx
        alpha_y_m = (-v - rsy) / rdy
        alpha_y_p = (v - rsy) / rdy
        print(alpha_x_m, alpha_x_p, alpha_y_m, alpha_y_p)

        alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m))
        alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m))

        print(alpha_s, alpha_e)
        if alpha_s < alpha_e:
            rsx += rdx * alpha_s + v
            rsy += rdy * alpha_s + v
            rdx *= (alpha_e - alpha_s)
            rdy *= (alpha_e - alpha_s)

            cv2.line(img, (int(rsx + pad), int(rsy) + pad), (int(rsx + rdx) + pad, int(rsy + rdy) + pad), 255)

print(img.shape)

plt.imshow(img, cmap="gray")
plt.show()

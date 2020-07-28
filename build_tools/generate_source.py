import re

DEFINE_ACCUMULATOR = """
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }
"""

PARALLEL_BEAM_RAY = """
        const float v = img_size / 2.0;
        const float sx = (ray_id - v + 0.5f)*det_spacing;
        const float sy = 0.71f * img_size;
        const float ex = sx;
        const float ey = -sy;
"""

FANBEAM_RAY = """
        const float v = img_size / 2.0;
        const float sy = s_dist;
        const float sx = 0.0f;
        const float ey = -d_dist;
        const float ex = (ray_id - v + 0.5f)*det_spacing;
"""

ROTATE_RAY = """
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;
"""

CLIP_TO_SQUARE = """
        // clip to square (to reduce memory reads)
        const float alpha_x_m = (-v - rsx)/rdx;
        const float alpha_x_p = (v - rsx)/rdx;
        const float alpha_y_m = (-v -rsy)/rdy;
        const float alpha_y_p = (v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        if(alpha_s > alpha_e){
            #pragma unroll
            for (int b = 0; b < channels; b++) {
                output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] = 0.0f;
            }
            return;
        }

        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);
"""

CLIP_TO_CIRCLE = """
        // clip rays to circle (to reduce memory reads)
        const float a = rdx * rdx + rdy * rdy;
        const float b = rsx * rdx + rsy * rdy;
        const float c = rsx * rsx + rsy * rsy - v * v;

        // min_clip to 1 to avoid getting empty rays
        const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
        const float alpha_s = (-b - delta_sqrt) / a;
        const float alpha_e = (-b + delta_sqrt) / a;
        
        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);
"""

ACCUMULATE_LOOP = """
        const uint n_steps = __float2uint_ru(hypot(rdx, rdy));
        const float vx = rdx / n_steps;
        const float vy = rdy / n_steps;
        const float n = hypot(vx, vy);

        for (uint j = 0; j <= n_steps; j++) { //changing j and n_steps to int makes everything way slower (WHY???)
            if (channels == 1) {
                accumulator[0] += tex2DLayered<float>(texture, rsx, rsy, blockIdx.z);
            } else {
                float4 read = tex2DLayered<float4>(texture, rsx, rsy, blockIdx.z);
                accumulator[0] += read.x;
                accumulator[1] += read.y;
                accumulator[2] += read.z;
                accumulator[3] += read.w;
            }
            rsx += vx;
            rsy += vy;
        }
"""

OUTPUT_LOOP = """
        #pragma unroll
        for (int b = 0; b < channels; b++) {
            output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] =
                    accumulator[b] * n;
        }
"""

COMPUTE_SIN_COS = """
    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();"""

COMPUTE_IMAGE_COORDINATES = """
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const float center = img_size / 2.0f;
    const float dx = float(x) - center + 0.5f;
    const float dy = float(y) - center + 0.5f;
"""

BACK_FP1_LOOP = """
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;
            tmp += tex2DLayered<float>(texture, j, i + 0.5f, batch_id);
        }
"""

BACK_FPCH_LOOP = """
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;

            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x;
            tmp[1] += read.y;
            tmp[2] += read.z;
            tmp[3] += read.w;
        }"""

BACK_HP_LOOP = """
    const float ids = __fdividef(1.0f, det_spacing);
    for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;
#pragma unroll
        for (int h = 0; h < wpt; h++) {
            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z * wpt + h);
            tmp[h * 4 + 0] += read.x;
            tmp[h * 4 + 1] += read.y;
            tmp[h * 4 + 2] += read.z;
            tmp[h * 4 + 3] += read.w;
        }
    }"""

BACK_FB_FP1_LOOP = """
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;

            tmp += tex2DLayered<float>(texture, j, i + 0.5f, batch_id) / den;
        }
"""

BACK_FB_FPCH_LOOP = """
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;

            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x * iden;
            tmp[1] += read.y * iden;
            tmp[2] += read.z * iden;
            tmp[3] += read.w * iden;
        }"""

BACK_FB_HP_LOOP = """
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;
#pragma unroll
        for (int h = 0; h < wpt; h++) {
            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z * wpt + h);
            tmp[h * 4 + 0] += read.x * iden;
            tmp[h * 4 + 1] += read.y * iden;
            tmp[h * 4 + 2] += read.z * iden;
            tmp[h * 4 + 3] += read.w * iden;
        }
    }"""

variables = {
    "DEFINE_ACCUMULATOR": DEFINE_ACCUMULATOR,

    "PARALLEL_BEAM_RAY": PARALLEL_BEAM_RAY,
    "FANBEAM_RAY": FANBEAM_RAY,
    "ROTATE_RAY": ROTATE_RAY,

    "CLIP_TO_SQUARE": CLIP_TO_SQUARE,
    "CLIP_TO_CIRCLE": CLIP_TO_CIRCLE,

    "ACCUMULATE_LOOP": ACCUMULATE_LOOP,
    "OUTPUT_LOOP": OUTPUT_LOOP,

    "COMPUTE_SIN_COS": COMPUTE_SIN_COS,
    "COMPUTE_IMAGE_COORDINATES": COMPUTE_IMAGE_COORDINATES,

    "BACK_FP1_LOOP": BACK_FP1_LOOP,
    "BACK_FPCH_LOOP": BACK_FPCH_LOOP,
    "BACK_HP_LOOP": BACK_HP_LOOP,

    "BACK_FB_FP1_LOOP": BACK_FB_FP1_LOOP,
    "BACK_FB_FPCH_LOOP": BACK_FB_FPCH_LOOP,
    "BACK_FB_HP_LOOP": BACK_FB_HP_LOOP,
}


def replace(m):
    var_name = m.group(1)
    return variables[var_name]


def generate_source(template, dst):
    with open(template, "r") as f:
        template = f.read()
        code = re.sub(r"{{\s*([A-Z_0-9]+)\s*}}", replace, template)

        with open(dst, "w") as of:
            of.write(code)

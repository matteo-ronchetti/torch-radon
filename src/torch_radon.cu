#include "torch_radon.h"
#include "forward.h"
#include "utils.h"
#include "rmath.h"
#include "vectors.hpp"



std::vector<float *> radon_forward(const float *x, const std::vector<float> &angles, TextureCache &tex_cache,
                                   const VolumeCfg &vol_cfg, Projection3D &proj_cfg, const ExecCfg &exec_cfg) {
    std::vector<float *> result;
    const size_t volumeSize = vol_cfg.width * vol_cfg.height * vol_cfg.slices * sizeof(float);
    const size_t projSize = proj_cfg.det_count_u * proj_cfg.det_count_v * sizeof(float);
    const size_t projectionsSize = angles.size() * projSize;

    float *x_gpu;
    float *angles_gpu;
    float *y_gpu;

    proj_cfg.n_angles = angles.size();
    cudaMalloc((void **) &x_gpu, volumeSize);
    cudaMalloc((void **) &angles_gpu, angles.size() * sizeof(float));
    cudaMalloc((void **) &y_gpu, projectionsSize);
    for (int i = 0; i < angles.size(); i++)
        result.push_back((float *) malloc(projSize));

    cudaMemcpy(x_gpu, x, volumeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(angles_gpu, &angles[0], angles.size() * sizeof(float), cudaMemcpyHostToDevice);

    radon_forward_cuda_3d(x_gpu, angles_gpu, y_gpu, tex_cache, vol_cfg, proj_cfg, exec_cfg, 1, 0);

    for (int i = 0; i < angles.size(); i++)
        cudaMemcpy(result[i], &y_gpu[i * proj_cfg.det_count_u * proj_cfg.det_count_v], projSize, cudaMemcpyDeviceToHost);

    cudaFree(x_gpu);
    cudaFree(angles_gpu);
    cudaFree(y_gpu);

    return result;
}

//std::vector<float> project_points(const std::vector<float> &points, float angle, const VolumeCfg &vol_cfg, Projection3D &proj_cfg) {
//    // ASSERT(points.size() % 3 == 0);
//    std::vector<float> res;
//
//    proj_cfg.updateMatrices(vol_cfg);
//
//
//    const float cs = rosh::cos(angle);
//    const float sn = rosh::sin(angle);
//    const float k = proj_cfg.s_dist + proj_cfg.d_dist;
//
//    vec3 source = {0.0, -proj_cfg.s_dist, 0.0};
//    // rotate start/end positions and add pitch * angle / (2*pi) to z
//    source = rotxy_transz(source, sn, cs, proj_cfg.pitch * angle * 0.1591549f);
//    vec3 u = rotxy(vec3{k, 0, 0}, sn, cs);
//    vec3 v = rotxy(vec3{0, 0, k}, sn, cs);
//    vec3 d = rotxy(vec3{0, 1, 0}, sn, cs);
//
//    for (int i = 0; i < points.size(); i += 3) {
//        const vec3 p = *(reinterpret_cast<const vec3 *>(&points[i])) - source;
//        float x = dot(u, p);
//        float y = dot(v, p);
//        float a = dot(d, p);
//        res.push_back(x / a);
//        res.push_back(y / a);
//    }
//
//    return res;
//}

void projection_matrices(float* res_data, const float* angles_data, const VolumeCfg &vol_cfg, std::vector<Projection3D> &proj_cfgs){
    for(int i = 0; i < proj_cfgs.size(); i++){
        float angle = angles_data[i];
        Projection3D& proj_cfg = proj_cfgs[i];
        proj_cfg.updateMatrices(vol_cfg);

        const float cs = rosh::cos(angle);
        const float sn = rosh::sin(angle);
        const float k = proj_cfg.s_dist + proj_cfg.d_dist;

        vec3 source = vec3{0.0, -proj_cfg.s_dist, 0.0};
        // rotate start/end positions and add pitch * angle / (2*pi) to z
        source = rotxy_transz(source, sn, cs, proj_cfg.pitch * angle * 0.1591549f);
        vec3 u = rotxy(vec3{k, 0, 0}, sn, cs) / proj_cfg.det_spacing_u;
        vec3 v = rotxy(vec3{0, 0, k}, sn, cs) / proj_cfg.det_spacing_v;
        vec3 d = rotxy(vec3{0, 1, 0}, sn, cs);

        source = proj_cfg.worldToVoxel * source;
        u = rotate_scale(proj_cfg.worldToVoxel, u);
        v = rotate_scale(proj_cfg.worldToVoxel, v);
        d = rotate_scale(proj_cfg.worldToVoxel, d);

        res_data[i*12 + 0] = u.x;
        res_data[i*12 + 1] = u.y;
        res_data[i*12 + 2] = u.z;
        res_data[i*12 + 3] = -dot(u, source);
        res_data[i*12 + 4] = v.x;
        res_data[i*12 + 5] = v.y;
        res_data[i*12 + 6] = v.z;
        res_data[i*12 + 7] = -dot(v, source);
        res_data[i*12 + 8] = d.x;
        res_data[i*12 + 9] = d.y;
        res_data[i*12 + 10] = d.z;
        res_data[i*12 + 11] = -dot(d, source);
    }
}

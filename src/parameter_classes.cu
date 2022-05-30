#include "parameter_classes.h"
#include "vectors.hpp"
#include "utils.h"
#include "rmath.h"
#include <cuda.h>


VolumeCfg::VolumeCfg(int s, int h, int w, float sz, float sy, float sx)
        : slices(s), height(h), width(w),
          spacing({sx, sy, sz}),
          inv_spacing({1 / sx, 1 / sy, 1 / sz}) {
            voxelToImage.x = {sx, 0, 0};
            voxelToImage.y = {0, -sy, 0};
            voxelToImage.z = {0, 0, -sz};
            voxelToImage.d = {-sx * (w / 2.0f + 0.5f), sy * (h / 2.0f + 0.5f), sz * (s / 2.0f + 0.5f)};
            imageToVoxel = inverse(voxelToImage);
          }

bool VolumeCfg::is_3d() const {
    return slices > 1;
}


Projection2D::Projection2D(ProjectionType t, int dc, float ds) : type(t), det_count(dc), det_spacing(ds) {}

Projection2D Projection2D::ParallelBeam(int det_count, float det_spacing) {
    return {ProjectionType::ParallelBeam, det_count, det_spacing};
}

Projection2D Projection2D::FanBeam(int det_count, float src_dist, float det_dist, float det_spacing) {
    Projection2D res(ProjectionType::FanBeam, det_count, det_spacing);
    res.s_dist = src_dist;
    res.d_dist = det_dist;
    return res;
}

Projection3D::Projection3D(ProjectionType t) : type(t) {}


Projection3D Projection3D::ConeBeam(int det_count_u, int det_count_v, float src_dist, float det_dist, float det_spacing_u, float det_spacing_v, float pitch) {
    Projection3D res(ProjectionType::ConeBeam);
    res.det_count_u = det_count_u;
    res.det_count_v = det_count_v;
    res.det_spacing_u = det_spacing_u;
    res.det_spacing_v = det_spacing_v;
    res.s_dist = src_dist;
    res.d_dist = det_dist;
    res.pitch = pitch;
    return res;
}

pose getPose(vec3 rot, vec3 d)
{
    float cx = cos(rot.x), sx = sin(rot.x), cy = cos(rot.y), sy = sin(rot.y), cz = cos(rot.z), sz = sin(rot.z);

    pose P;
    P.x = {cy*cz, cz*sx*sy - cx*sz, sx*sz + cx*cz*sy};
    P.y = {cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - cz*sx};
    P.z = {-sy, cy*sx, cx*cy};
    P.d = d;
    return P;
}

void Projection3D::setPose(float rx, float ry, float rz, float dx, float dy, float dz){
    imageToWorld = getPose({rx, ry, rz}, {dx, dy, dz});
}

void Projection3D::updateMatrices(const VolumeCfg& vol){
    worldToImage = inverse(imageToWorld);

    worldToVoxel = vol.imageToVoxel * worldToImage;

    // vec3 center = {vol.width / 2.0f + 0.5f, vol.height / 2.0f + 0.5f, vol.slices / 2.0f + 0.5f};
    // worldToVoxel.x = worldToImage.x / vol.spacing.x;
    // worldToVoxel.y = worldToImage.y / vol.spacing.y;
    // worldToVoxel.z = worldToImage.z / vol.spacing.z;
    // worldToVoxel.d = (worldToImage.d / vol.spacing) + center;
}


ExecCfg::ExecCfg(int x, int y, int z, int ch)
        : bx(x), by(y), bz(z), channels(ch) {}

dim3 ExecCfg::get_block_dim() const {
    return dim3(bx, by, bz);
}

dim3 ExecCfg::get_grid_size(int x, int y, int z) const {
    return dim3(roundup_div(x, bx), roundup_div(y, by), roundup_div(z, bz));
}

int ExecCfg::get_channels(int batch_size) const {
    return (batch_size % 4 == 0) ? this->channels : 1;
}
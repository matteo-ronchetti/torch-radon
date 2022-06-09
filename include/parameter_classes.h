#pragma once

#include <string>

#include "defines.h"
#include "vectors.h"

struct VolumeCfg
{
    // dimensions of the measured volume
    int slices;
    int height;
    int width;

    vec3 spacing;
    vec3 inv_spacing;

    pose voxelToImage;
    pose imageToVoxel;

    VolumeCfg(int s, int h, int w, float sz, float sy, float sx);
    bool is_3d() const;
};

struct Projection2D
{
    Projection2D(ProjectionType t, int dc, float ds = 1.0);

    static Projection2D ParallelBeam(int det_count, float det_spacing = 1.0);
    static Projection2D FanBeam(int det_count, float src_dist, float det_dist, float det_spacing = 1.0);

    // number of pixels of the detector
    int det_count = 0;

    // detector spacing
    float det_spacing = 1.0;

    // source and detector distances (for fan-beam)
    float s_dist = 0.0;
    float d_dist = 0.0;

    int n_angles = 0;

    ProjectionType type;

    // Image pose
    float dx = 0.0;
    float dy = 0.0;
};

class Projection3D
{
public:
    Projection3D(ProjectionType t);

    static Projection3D ConeBeam(int det_count_u, int det_count_v, float src_dist, float det_dist, float det_spacing_u = 1.0, float det_spacing_v = 1.0, float pitch = 0.0);

    // number of pixels of the detector
    int det_count_u;
    int det_count_v = 0;

    // detector spacing
    float det_spacing_u = 1.0;
    float det_spacing_v = 1.0;

    // source and detector distances (for fan-beam and cone-beam)
    float s_dist = 0.0;
    float d_dist = 0.0;

    // pitch = variation in z after a full rotation (for cone-beam)
    float pitch = 0.0;

    int n_angles;

    ProjectionType type;

    // Volume Pose
    pose imageToWorld;
    pose worldToImage;

    // Computed with the volume
    // mutable pose voxelToWorld;
    mutable pose worldToVoxel;

    void setPose(float rx, float ry, float rz, float dx, float dy, float dz);
    void updateMatrices(const VolumeCfg &vol);
};

class ExecCfg
{
public:
    int bx, by, bz;

    int channels;

    ExecCfg(int x, int y, int z, int ch);

    dim3 get_block_dim() const;

    dim3 get_grid_size(int x, int y = 1, int z = 1) const;

    int get_channels(int batch_size) const;
};
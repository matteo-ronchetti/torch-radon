#include "parameter_classes.h"
#include "texture.h"

#include <vector>

std::vector<float *> radon_forward(const float *x, const std::vector<float> &angles, TextureCache &tex_cache,
                                   const VolumeCfg &vol_cfg, Projection3D &proj_cfg, const ExecCfg &exec_cfg);

//std::vector<float> project_points(const std::vector<float> &points, float angle, const VolumeCfg &vol_cfg, Projection3D &proj_cfg);

void projection_matrices(float* res_data, const float* angles, const VolumeCfg &vol_cfg, std::vector<Projection3D> &proj_cfgs);
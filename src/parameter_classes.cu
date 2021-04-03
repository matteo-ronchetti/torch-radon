#include "parameter_classes.h"
#include "utils.h"

VolumeCfg::VolumeCfg(int d, int h, int w, float _dz, float _dy, float _dx, float _sz, float _sy, float _sx, bool ddd)
        : depth(d), height(h), width(w),
        dz(_dz), dy(_dy), dx(_dx), 
        sz(_sz), sy(_sy), sx(_sx),
        inv_scale_z(1.0f / sz), inv_scale_y(1.0f / sy), inv_scale_x(1.0f / sx), 
        is_3d(ddd) {}

ProjectionCfg::ProjectionCfg(int dc_u, float ds_u, int dc_v, float ds_v, float sd, float dd,
                  float pi, float iz, int pt)
            : det_count_u(dc_u), det_spacing_u(ds_u), det_count_v(dc_v),
              det_spacing_v(ds_v), s_dist(sd), d_dist(dd), pitch(pi), initial_z(iz),
              projection_type(pt) {}

ProjectionCfg::ProjectionCfg(const ProjectionCfg& src)
    :det_count_u(src.det_count_u), det_spacing_u(src.det_spacing_u),
    det_count_v(src.det_count_v), det_spacing_v(src.det_spacing_v),
    s_dist(src.s_dist), d_dist(src.d_dist), pitch(src.pitch), initial_z(src.initial_z),
              projection_type(src.projection_type), n_angles(src.n_angles) {}

bool ProjectionCfg::is_2d() const{
    return projection_type == PARALLEL || projection_type == FANBEAM;
}

ProjectionCfg ProjectionCfg::copy() const{
    return ProjectionCfg(*this);
}

//std::string ProjectionCfg::to_string() const{
//    switch(projection_type){
//        case PARALLEL:
//            return string_format("Projection(type=ParallelBeam, det_count=%d, det_spacing=%f)", det_count_u, det_spacing_u);
//        default:
//            return "TODO";
//    }
//}


ExecCfg::ExecCfg(int x, int y, int z, int ch, float sr)
        :block_dim(x, y, z), channels(ch), sampling_rate(sr) {}

dim3 ExecCfg::get_grid_size(int x, int y, int z) const{
    return dim3(roundup_div(x, block_dim.x), roundup_div(y, block_dim.y), roundup_div(z, block_dim.z));
}

int ExecCfg::get_channels(int batch_size) const{
    return (batch_size % 4 == 0) ? this->channels : 1;
}
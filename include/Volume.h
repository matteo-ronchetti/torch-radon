#pragma once

#include "vectors.h"
#include "Tensor.h"

class Volume
{
public:
    Volume(Tensor &&tensor, vec3 spacing);

    bool isCuda() const { return m_tensor.isCuda(); }
    int device() const { return m_tensor.device(); }
    Volume cuda(int device) const;
    Volume cpu() const;

    Tensor::Type type() const { return m_tensor.type(); }

    int width() const { return m_tensor.shape(2); }
    int height() const { return m_tensor.shape(1); }
    int slices() const { return m_tensor.shape(0); }

    bool is3D() const { return m_tensor.shape(0) > 1; }

    Tensor &tensor() { return m_tensor; }
    const Tensor &tensor() const { return m_tensor; }

private:
    Tensor m_tensor;

    vec3 m_spacing;

    pose m_voxelToImage;
    pose m_imageToVoxel;
};
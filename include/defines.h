#pragma once
typedef unsigned int uint;
typedef unsigned short ushort;

#define PRECISION_FLOAT 1
#define PRECISION_HALF 0

enum class ProjectionType{
    ParallelBeam = 0,
    FanBeam = 1,
    ConeBeam = 2
};

#define TEX_1D_LAYERED 0
#define TEX_2D_LAYERED 1
#define TEX_3D 2
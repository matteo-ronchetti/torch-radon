#include <iostream>
#include <vector>
#include <math.h>

#include "parameter_classes.h"

struct Gaussian
{
    float k;
    float cx;
    float cy;
    float a, b;

    Gaussian(float _k, float _cx, float _cy, float _a, float _b);

    float line_integral(float s_x, float s_y, float e_x, float e_y) const;
    float evaluate(float x, float y) const;
};

struct Ellipse
{
    float k;
    float cx;
    float cy;

    float radius;
    float aspect;

    Ellipse(float _k, float _cx, float _cy, float _r, float a);
    float line_integral(float s_x, float s_y, float e_x, float e_y) const;
    float evaluate(float x, float y) const;
};

class SymbolicFunction
{
    std::vector<Gaussian> gaussians;
    std::vector<Ellipse> ellipses;

public:
    SymbolicFunction();

    void add_gaussian(float k, float cx, float cy, float a, float b);
    void add_ellipse(float k, float cx, float cy, float r, float a);

    void move(float dx, float dy);

    void discretize(float *data, int w, int h) const;
    float line_integral(float s_x, float s_y, float e_x, float e_y) const;
};

void symbolic_forward(const SymbolicFunction &f, const ProjectionCfg &proj, const float *angles, const int n_angles, const int width, const int height, float *sinogram);
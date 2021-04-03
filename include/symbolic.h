#include <iostream>
#include <vector>
#include <math.h>

struct Gaussian{
    float k;
    float cx;
    float cy;
    float a, b;

    Gaussian(float _k, float _cx, float _cy, float _a, float _b):k(_k), cx(_cx), cy(_cy), a(_a), b(_b){}
};

class SymbolicFunction{
    std::vector<Gaussian> gaussians;

    public:
    SymbolicFunction();

    void add_gaussian(float k, float cx, float cy, float a, float b);

    void discretize(float* data, int w, int h);

    void move(float dx, float dy);

};
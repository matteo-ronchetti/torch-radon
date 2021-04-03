#include "symbolic.h"

template<typename T>
T line_integral(T s_x, T s_y, T e_x, T e_y, T c_x, T c_y, T a, T b){
    T x0 = a*pow(s_x, 2);
    T x1 = b*pow(s_y, 2);
    T x2 = 2*a*e_x;
    T x3 = 2*b*e_y;
    T x4 = s_x*x2;
    T x5 = s_y*x3;
    T x6 = 2*a*c_x*s_x + 2*b*c_y*s_y;
    T x7 = -c_x*x2 - c_y*x3 - 2*x0 - 2*x1 + x4 + x5 + x6;
    T x8 = a*pow(e_x, 2) + b*pow(e_y, 2) + x0 + x1 - x4 - x5;
    T x9 = sqrt(x8);
    T x10 = (1.0/2.0)/x9;
    T x11 = x10*x7;
    T x12 = sqrt(M_PI)*x10*exp(-a*pow(c_x, 2) - b*pow(c_y, 2) - x0 - x1 + x6 + (1.0/4.0)*pow(x7, 2)/x8);

    return x12*(-erf(x11) + erf(x11 + x9));
}
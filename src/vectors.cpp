#include "vectors.h"

float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 rotate_scale(const mat &m, const vec3 &v) {
    return {dot(m.x, v), dot(m.y, v), dot(m.z, v)};
}

vec3 operator-(const vec3 &v) {
    return {-v.x, -v.y, -v.z};
}

mat inverse(const mat &m) {
    mat r;
    r.x = {m.x.x, m.y.x, m.z.x};
    r.y = {m.x.y, m.y.y, m.z.y};
    r.z = {m.x.z, m.y.z, m.z.z};
    r.d = -rotate_scale(r, m.d);
    return r;
}

vec3 operator/(const vec3 &a, const float x){
    return {a.x / x, a.y / x, a.z / x};
}

vec3 operator/(const vec3 &a, const vec3 &b);

vec3 operator+(const vec3 &a, const vec3 &b){
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 operator/(const vec3 &a, const vec3 &b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

#pragma once

#include "rmath.h"


vec3 operator+(const vec3 &a, const vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

void operator+=(vec3 &a, const vec3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

vec3 operator-(const vec3 &a, const vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 operator*(const vec3 &a, const vec3 &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

vec3 operator*(const vec3 &a, const float x) {
    return {a.x * x, a.y * x, a.z * x};
}

vec3 operator*(const float x, const vec3 &a) {
    return {a.x * x, a.y * x, a.z * x};
}

float norm(const vec3 &a) {
    return rosh::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

vec3 rotxy(const vec3 &v, float sn, float cs) {
    return {v.x * cs - v.y * sn, v.x * sn + v.y * cs, v.z};
}

vec3 rotxy_transz(const vec3 &v, float sn, float cs, float dz) {
    return {v.x * cs - v.y * sn, v.x * sn + v.y * cs, v.z + dz};
}

float dot(const vec3 &a, const vec3 &b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

vec3 rotate_scale(const mat &m, const vec3 &v) {
    return {dot(m.x, v), dot(m.y, v), dot(m.z, v)};
}

vec3 operator*(const mat &m, const vec3 &v) {
    return rotate_scale(m, v) + m.d;
}

vec3 operator-(const vec3 &v) {
    return {-v.x, -v.y, -v.z};
}

vec3 operator/(const vec3 &a, const float x){
    return {a.x / x, a.y / x, a.z / x};
}

vec3 operator/(const vec3 &a, const vec3 &b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}


mat inverse(const mat &m) {
    mat r;
    r.x = {m.x.x, m.y.x, m.z.x};
    r.y = {m.x.y, m.y.y, m.z.y};
    r.z = {m.x.z, m.y.z, m.z.z};
    r.d = -rotate_scale(r, m.d);
    return r;
}
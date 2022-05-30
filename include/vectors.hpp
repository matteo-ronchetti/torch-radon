#pragma once

#include "rmath.h"

inline vec3 operator+(const vec3 &a, const vec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline void operator+=(vec3 &a, const vec3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline vec3 operator-(const vec3 &a, const vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline vec3 operator*(const vec3 &a, const vec3 &b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline vec3 operator*(const vec3 &a, const float x)
{
    return {a.x * x, a.y * x, a.z * x};
}

inline vec3 operator*(const float x, const vec3 &a)
{
    return {a.x * x, a.y * x, a.z * x};
}

inline float norm(const vec3 &a)
{
    return rosh::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline vec3 rotxy(const vec3 &v, float sn, float cs)
{
    return {v.x * cs - v.y * sn, v.x * sn + v.y * cs, v.z};
}

inline vec3 rotxy_transz(const vec3 &v, float sn, float cs, float dz)
{
    return {v.x * cs - v.y * sn, v.x * sn + v.y * cs, v.z + dz};
}

inline float dot(const vec3 &a, const vec3 &b)
{
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

inline vec3 rotate_scale(const pose &m, const vec3 &v)
{
    return {dot(m.x, v), dot(m.y, v), dot(m.z, v)};
}

inline vec3 operator*(const pose &m, const vec3 &v)
{
    return rotate_scale(m, v) + m.d;
}

inline pose operator*(const pose &a, const pose &b)
{
    vec3 tx = {b.x.x, b.y.x, b.z.x};
    vec3 ty = {b.x.y, b.y.y, b.z.y};
    vec3 tz = {b.x.z, b.y.z, b.z.z};
    pose c;
    c.x = {dot(c.x, tx), dot(c.x, ty), dot(c.x, tz)};
    c.y = {dot(c.y, tx), dot(c.y, ty), dot(c.y, tz)};
    c.z = {dot(c.z, tx), dot(c.z, ty), dot(c.z, tz)};
    c.d = a * b.d;
    return c;
}

inline vec3 operator-(const vec3 &v)
{
    return {-v.x, -v.y, -v.z};
}

inline vec3 operator/(const vec3 &a, const float x)
{
    return {a.x / x, a.y / x, a.z / x};
}

inline vec3 operator/(const vec3 &a, const vec3 &b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

inline pose inverse(const pose &m)
{
    float nx = dot(m.x, m.x);
    float ny = dot(m.y, m.y);
    float nz = dot(m.z, m.z);
    pose r;
    r.x = vec3{m.x.x, m.y.x, m.z.x} / nx;
    r.y = vec3{m.x.y, m.y.y, m.z.y} / ny;
    r.z = vec3{m.x.z, m.y.z, m.z.z} / nz;
    r.d = -rotate_scale(r, m.d);
    return r;
}
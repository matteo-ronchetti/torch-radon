#pragma once

#include "vectors.h"

__inline__ __device__ vec3 operator+(const vec3 &a, const vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__inline__ __device__ void operator+=(vec3 &a, const vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__inline__ __device__ vec3 operator-(const vec3 &a, const vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__inline__ __device__ vec3 operator*(const vec3 &a, const vec3 &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__inline__ __device__ vec3 operator*(const vec3 &a, const float x) {
    return {a.x * x, a.y * x, a.z * x};
}

__inline__ __device__ vec3 operator*(const float x, const vec3 &a) {
    return {a.x * x, a.y * x, a.z * x};
}

__inline__ __device__ vec3 operator/(const vec3 &a, const float x) {
    return {a.x / x, a.y / x, a.z / x};
}

__inline__ __device__ float norm(const vec3 &a) {
    return norm3df(a.x, a.y, a.z);
}

__inline__ __device__ vec3 rotxy(const vec3 &v, float sn, float cs) {
    return {v.x * cs - v.y * sn, v.x * sn + v.y * cs, v.z};
}

__inline__ __device__ vec3 rotxy_transz(const vec3 &v, float sn, float cs, float dz) {
    return {v.x * cs - v.y * sn, v.x * sn + v.y * cs, v.z + dz};
}

__inline__ __device__ float dot(const vec3 &a, const vec3 &b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__inline__ __device__ vec3 rotate_scale(const pose &m, const vec3 &v) {
    return {dot(m.x, v), dot(m.y, v), dot(m.z, v)};
}

__inline__ __device__ vec3 operator*(const pose &m, const vec3 &v) {
    return rotate_scale(m, v) + m.d;
}
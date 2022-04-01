#pragma once

struct vec3{
    float x;
    float y;
    float z;
};

struct mat{
    vec3 x = {1, 0, 0};
    vec3 y = {0, 1, 0};
    vec3 z = {0, 0, 1};
    vec3 d = {0, 0, 0};
};

//vec3 operator+(const vec3 &a, const vec3 &b);
//vec3 operator-(const vec3 &a, const vec3 &b);
//
//vec3 operator*(const vec3 &a, const float x);
//vec3 operator*(const float x, const vec3 &a);
//
//vec3 normalized(const vec3 &a);
//
//float dot(const vec3 &a, const vec3 &b);
//float norm(const vec3 &a);
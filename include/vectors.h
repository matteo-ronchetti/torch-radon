#pragma once

struct vec3
{
    float x;
    float y;
    float z;
};

struct vec3i
{
    int x;
    int y;
    int z;
};

struct pose
{
    vec3 x = {1, 0, 0};
    vec3 y = {0, 1, 0};
    vec3 z = {0, 0, 1};
    vec3 d = {0, 0, 0};
};
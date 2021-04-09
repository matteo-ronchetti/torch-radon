#include "symbolic.h"

Gaussian::Gaussian(float _k, float _cx, float _cy, float _a, float _b) : k(_k), cx(_cx), cy(_cy), a(_a), b(_b) {}

float Gaussian::line_integral(float s_x, float s_y, float e_x, float e_y) const
{
    float x0 = a * pow(s_x, 2);
    float x1 = b * pow(s_y, 2);
    float x2 = 2 * a * e_x;
    float x3 = 2 * b * e_y;
    float x4 = s_x * x2;
    float x5 = s_y * x3;
    float x6 = 2 * a * cx * s_x + 2 * b * cy * s_y;
    float x7 = -cx * x2 - cy * x3 - 2 * x0 - 2 * x1 + x4 + x5 + x6;
    float x8 = a * pow(e_x, 2) + b * pow(e_y, 2) + x0 + x1 - x4 - x5;
    float x9 = sqrt(x8);
    float x10 = (1.0 / 2.0) / x9;
    float x11 = x10 * x7;
    float x12 = sqrt(M_PI) * x10 * exp(-a * pow(cx, 2) - b * pow(cy, 2) - x0 - x1 + x6 + (1.0 / 4.0) * pow(x7, 2) / x8);
    float len = std::hypot(e_x - s_x, e_y - s_y);

    return k * len * x12 * (-erf(x11) + erf(x11 + x9));
}

float Gaussian::evaluate(float x, float y) const
{
    float dx = x - cx;
    float dy = y - cy;

    return k * exp(-a * dx * dx - b * dy * dy);
}

Ellipse::Ellipse(float _k, float _cx, float _cy, float _r, float a) : k(_k), cx(_cx), cy(_cy), radius(_r), aspect(a) {}

float Ellipse::line_integral(float s_x, float s_y, float e_x, float e_y) const
{
    const float dx = e_x - s_x;
    const float dy = aspect * (e_y - s_y);
    const float cdx = s_x - cx;
    const float cdy = aspect * (s_y - cy);

    const float a = dx * dx + dy * dy;
    const float b = cdx * dx + cdy * dy;
    const float c = cdx * cdx + cdy * cdy - radius * radius;

    const float delta = b * b - a * c;
    if (delta <= 0)
        return 0.0f;

    // min_clip to 1 to avoid getting empty rays
    const float delta_sqrt = std::sqrt(delta);
    const float alpha_s = std::min(std::max((-b - delta_sqrt) / a, 0.0f), 1.0f);
    const float alpha_e = std::min(std::max((-b + delta_sqrt) / a, 0.0f), 1.0f);

    return std::hypot(dx, e_y - s_y) * (alpha_e - alpha_s);
}

float Ellipse::evaluate(float x, float y) const
{
    float tmp = 0;
    float dx = cx - x;
    float dy = aspect * (cy - y);
    constexpr float r = 1.0f / 3;

    tmp += std::hypot(dx - r, dy - r) <= radius;
    tmp += std::hypot(dx - r, dy) <= radius;
    tmp += std::hypot(dx - r, dy + r) <= radius;
    tmp += std::hypot(dx, dy - r) <= radius;
    tmp += std::hypot(dx, dy) <= radius;
    tmp += std::hypot(dx, dy + r) <= radius;
    tmp += std::hypot(dx + r, dy - r) <= radius;
    tmp += std::hypot(dx + r, dy) <= radius;
    tmp += std::hypot(dx + r, dy + r) <= radius;

    return tmp / 9.0f;
}

SymbolicFunction::SymbolicFunction() {}

void SymbolicFunction::add_gaussian(float k, float cx, float cy, float a, float b)
{
    gaussians.push_back(Gaussian(k, cx, cy, a, b));
}

void SymbolicFunction::add_ellipse(float k, float cx, float cy, float r, float a)
{
    ellipses.push_back(Ellipse(k, cx, cy, r, a));
}

void SymbolicFunction::move(float dx, float dy)
{
    for (auto &gaussian : gaussians)
    {
        gaussian.cx += dx;
        gaussian.cy += dy;
    }
    for (auto &ellipse : ellipses)
    {
        ellipse.cx += dx;
        ellipse.cy += dy;
    }
}

void SymbolicFunction::discretize(float *data, int w, int h) const
{
    for (int i = 0; i < h; i++)
    {
        float y = i + 0.5f;
        for (int j = 0; j < w; j++)
        {
            float x = j + 0.5f;

            float tmp = 0.0f;
            for (auto &gaussian : gaussians)
                tmp += gaussian.evaluate(x, y);
            for (auto &ellipse : ellipses)
                tmp += ellipse.evaluate(x, y);

            data[i * w + j] = tmp;
        }
    }
}

float SymbolicFunction::line_integral(float s_x, float s_y, float e_x, float e_y) const
{
    float tmp = 0.0f;
    for (auto &gaussian : gaussians)
        tmp += gaussian.line_integral(s_x, s_y, e_x, e_y);
    for (auto &ellipse : ellipses)
        tmp += ellipse.line_integral(s_x, s_y, e_x, e_y);

    return tmp;
}

void symbolic_forward(const SymbolicFunction &f, const ProjectionCfg &proj, const float *angles, const int n_angles, const int width, const int height, float *sinogram)
{
    for (int angle_id = 0; angle_id < n_angles; angle_id++)
    {
        for (int ray_id = 0; ray_id < proj.det_count_u; ray_id++)
        {
            // compute ray
            float sx, sy, ex, ey;
            if (proj.projection_type == PARALLEL)
            {
                sx = (ray_id - proj.det_count_u * 0.5f + 0.5f) * proj.det_spacing_u;
                sy = hypot(width * 0.5f, height * 0.5f);
                ex = sx;
                ey = -sy;
            }
            else
            {
                sy = proj.s_dist;
                sx = 0.0f;
                ey = -proj.d_dist;
                ex = (ray_id - proj.det_count_u * 0.5f + 0.5f) * proj.det_spacing_u;
            }

            // rotate ray
            const float angle = angles[angle_id];
            const float cs = cos(angle);
            const float sn = sin(angle);

            // start position rs and direction rd (in detector coordinate system)
            float rsx = sx * cs - sy * sn;
            float rsy = sx * sn + sy * cs;
            float rdx = ex * cs - ey * sn - rsx;
            float rdy = ex * sn + ey * cs - rsy;

            rsx += 0.5f * width;
            rsy += 0.5f * height;

            // clip to volume (to reduce memory reads)
            float dx = rdx >= 0 ? std::max(rdx, 1e-6f) : std::min(rdx, -1e-6f);
            float dy = rdy >= 0 ? std::max(rdy, 1e-6f) : std::min(rdy, -1e-6f);

            const float alpha_x_m = (-rsx) / dx;
            const float alpha_x_p = (width - rsx) / dx;
            const float alpha_y_m = (-rsy) / dy;
            const float alpha_y_p = (height - rsy) / dy;
            const float alpha_s = std::max(std::min(alpha_x_p, alpha_x_m), std::min(alpha_y_p, alpha_y_m));
            const float alpha_e = std::min(std::max(alpha_x_p, alpha_x_m), std::max(alpha_y_p, alpha_y_m));

            // if ray volume intersection is empty exit
            if (alpha_s < alpha_e)
            {
                rsx += rdx * alpha_s;
                rsy += rdy * alpha_s;
                float rex = rsx + rdx * (alpha_e - alpha_s);
                float rey = rsy + rdy * (alpha_e - alpha_s);

                sinogram[angle_id * proj.det_count_u + ray_id] = f.line_integral(rsx, rsy, rex, rey);
            }
            else
            {
                sinogram[angle_id * proj.det_count_u + ray_id] = 0.0f;
            }
        }
    }
}

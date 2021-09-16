#include "symbolic.h"
#include "log.h"
#include "rmath.h"

Gaussian::Gaussian(float _k, float _cx, float _cy, float _a, float _b) : k(_k), cx(_cx), cy(_cy), a(_a), b(_b) {}

float Gaussian::line_integral(float s_x, float s_y, float e_x, float e_y) const
{
    float x0 = a * rosh::sq(s_x);
    float x1 = b * rosh::sq(s_y);
    float x2 = 2 * a * e_x;
    float x3 = 2 * b * e_y;
    float x4 = s_x * x2;
    float x5 = s_y * x3;
    float x6 = 2 * a * cx * s_x + 2 * b * cy * s_y;
    float x7 = -cx * x2 - cy * x3 - 2 * x0 - 2 * x1 + x4 + x5 + x6;
    float x8 = rosh::max(a * rosh::sq(e_x) + b * rosh::sq(e_y) + x0 + x1 - x4 - x5, 1e-6f);
    float x9 = rosh::sqrt(x8);
    float x10 = (1.0 / 2.0) / x9;
    float x11 = x10 * x7;
    float lg_x12 = rosh::log(rosh::sqrt(rosh::pi) * x10) - a * rosh::sq(cx) - b * rosh::sq(cy) - x0 - x1 + x6 + (1.0 / 4.0) * rosh::sq(x7) / x8;

    // this is not precise
    if (lg_x12 >= 5)
    {
        return 0.0f;
    }

    float len = rosh::hypot(e_x - s_x, e_y - s_y);

    float y = k * len * rosh::exp(lg_x12) * (-rosh::erf(x11) + rosh::erf(x11 + x9));

    // if(y != y){
    //     LOG_ERROR("len: " << len << " x11: " << x11 << " erf(x11): " << erf(x11) << " x8: " << x8);
    //     LOG_ERROR("x9: " << x9 << " erf(x11 + x9): " << erf(x11 + x9) << " log(x12): " << lg_x12);
    // }

    return y;
}

float Gaussian::evaluate(float x, float y) const
{
    float dx = x - cx;
    float dy = y - cy;

    return k * rosh::exp(-a * dx * dx - b * dy * dy);
}

void Gaussian::move(float dx, float dy)
{
    cx += dx;
    cy += dy;
}

void Gaussian::scale(float sx, float sy)
{
    cx *= sx;
    cy *= sy;
    a /= sx * sx;
    b /= sy * sy;
}

Ellipse::Ellipse(float _k, float _cx, float _cy, float rx, float ry) : k(_k), cx(_cx), cy(_cy), radius_x(rx), radius_y(ry), aspect(rx / ry) {}

float Ellipse::line_integral(float s_x, float s_y, float e_x, float e_y) const
{
    const float dx = e_x - s_x;
    const float dy = aspect * (e_y - s_y);
    const float cdx = s_x - cx;
    const float cdy = aspect * (s_y - cy);

    const float a = dx * dx + dy * dy;
    const float b = cdx * dx + cdy * dy;
    const float c = cdx * cdx + cdy * cdy - radius_x * radius_x;

    const float delta = b * b - a * c;
    if (delta <= 0)
        return 0.0f;

    // min_clip to 1 to avoid getting empty rays
    const float delta_sqrt = rosh::sqrt(delta);
    const float alpha_s = rosh::min(rosh::max((-b - delta_sqrt) / a, 0.0f), 1.0f);
    const float alpha_e = rosh::min(rosh::max((-b + delta_sqrt) / a, 0.0f), 1.0f);

    return rosh::hypot(dx, e_y - s_y) * (alpha_e - alpha_s);
}

float Ellipse::evaluate(float x, float y) const
{
    float tmp = 0;
    float dx = cx - x;
    float dy = aspect * (cy - y);
    constexpr float r = 1.0f / 3;

    tmp += rosh::hypot(dx - r, dy - r) <= radius_x;
    tmp += rosh::hypot(dx - r, dy) <= radius_x;
    tmp += rosh::hypot(dx - r, dy + r) <= radius_x;
    tmp += rosh::hypot(dx, dy - r) <= radius_x;
    tmp += rosh::hypot(dx, dy) <= radius_x;
    tmp += rosh::hypot(dx, dy + r) <= radius_x;
    tmp += rosh::hypot(dx + r, dy - r) <= radius_x;
    tmp += rosh::hypot(dx + r, dy) <= radius_x;
    tmp += rosh::hypot(dx + r, dy + r) <= radius_x;

    return tmp / 9.0f;
}

void Ellipse::move(float dx, float dy)
{
    cx += dx;
    cy += dy;
}

void Ellipse::scale(float sx, float sy)
{
    cx *= sx;
    cy *= sy;
    radius_x *= sx;
    radius_y *= sy;
    aspect = radius_x / radius_y;
}

SymbolicFunction::SymbolicFunction(float h, float w) : min_x(-w / 2), min_y(-h / 2), max_x(w / 2), max_y(h / 2) {}

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
    min_x += dx;
    min_y += dy;
    max_x += dx;
    max_y += dy;

    for (auto &gaussian : gaussians)
    {
        gaussian.move(dx, dy);
    }
    for (auto &ellipse : ellipses)
    {
        ellipse.move(dx, dy);
    }
}

void SymbolicFunction::scale(float sx, float sy)
{
    min_x *= sx;
    min_y *= sy;
    max_x *= sx;
    max_y *= sy;

    for (auto &gaussian : gaussians)
    {
        gaussian.scale(sx, sy);
    }
    for (auto &ellipse : ellipses)
    {
        ellipse.scale(sx, sy);
    }
}

float SymbolicFunction::max_distance_from_origin() const
{
    float x = rosh::max(rosh::abs(min_x), rosh::abs(max_x));
    float y = rosh::max(rosh::abs(min_y), rosh::abs(max_y));

    return rosh::hypot(x, y);
}

void SymbolicFunction::discretize(float *data, int h, int w) const
{
    for (int i = 0; i < h; i++)
    {
        float y = i + 0.5f - float(h) / 2;
        for (int j = 0; j < w; j++)
        {
            float x = j + 0.5f - float(w) / 2;

            if (min_x <= x && x <= max_x && min_y <= y && y <= max_y)
            {
                float tmp = 0.0f;
                for (auto &gaussian : gaussians)
                    tmp += gaussian.evaluate(x, y);
                for (auto &ellipse : ellipses)
                    tmp += ellipse.evaluate(x, y);

                data[i * w + j] = tmp;
            }
            else
            {
                data[i * w + j] = 0.0f;
            }
        }
    }
}

float SymbolicFunction::line_integral(float s_x, float s_y, float e_x, float e_y) const
{
    // clip segment to function domain
    float dx = e_x - s_x;
    float dy = e_y - s_y;
    dx = dx >= 0 ? rosh::max(dx, 1e-6f) : rosh::min(dx, -1e-6f);
    dy = dy >= 0 ? rosh::max(dy, 1e-6f) : rosh::min(dy, -1e-6f);

    const float alpha_x_m = (min_x - s_x) / dx;
    const float alpha_x_p = (max_x - s_x) / dx;
    const float alpha_y_m = (min_y - s_y) / dy;
    const float alpha_y_p = (max_y - s_y) / dy;
    const float alpha_s = rosh::max(rosh::min(alpha_x_p, alpha_x_m), rosh::min(alpha_y_p, alpha_y_m));
    const float alpha_e = rosh::min(rosh::max(alpha_x_p, alpha_x_m), rosh::max(alpha_y_p, alpha_y_m));

    if (alpha_s >= alpha_e)
    {
        return 0.0f;
    }

    s_x += dx * alpha_s;
    s_y += dy * alpha_s;
    e_x = s_x + dx * (alpha_e - alpha_s);
    e_y = s_y + dy * (alpha_e - alpha_s);

    float tmp = 0.0f;
    for (auto &gaussian : gaussians)
        tmp += gaussian.line_integral(s_x, s_y, e_x, e_y);
    for (auto &ellipse : ellipses)
        tmp += ellipse.line_integral(s_x, s_y, e_x, e_y);

    return tmp;
}

void symbolic_forward(const SymbolicFunction &f, const ProjectionCfg &proj, const float *angles, const int n_angles, float *sinogram)
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
                sy = f.max_distance_from_origin();
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
            const float cs = rosh::cos(angle);
            const float sn = rosh::sin(angle);

            float rsx = sx * cs + sy * sn;
            float rsy = -sx * sn + sy * cs;
            float rex = ex * cs + ey * sn;
            float rey = -ex * sn + ey * cs;

            sinogram[angle_id * proj.det_count_u + ray_id] = f.line_integral(rsx, rsy, rex, rey);
        }
    }
}

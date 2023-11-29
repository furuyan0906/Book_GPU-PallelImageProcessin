#include  <cmath>
#include  "CpuKernel.hpp"


static constexpr double pi = 3.141592653589793;

static inline float Host_U(float x, float y, float t)
{
    return (-2.0f * (float)std::cos(pi * t / 8.0f) * (float)std::sin(pi * x) * (float)std::sin(pi * x) * (float)std::cos(pi * y) * (float)std::sin(pi * y));
}

static inline float Host_V(float x, float y, float t)
{
    return (2.0f * (float)std::cos(pi * t / 8.0f) * (float)std::cos(pi * x) * (float)std::sin(pi * x) * (float)std::sin(pi * y) * (float)std::sin(pi * y));
}


void Host_RungeKutta(int nthPartice, float (*particles)[2], float time, float dt)
{
    float xn = particles[nthPartice][0];
    float yn = particles[nthPartice][1];

    float x, y, t;

    // 1段目
    x = xn;
    y = yn;
    t = time;
    float p1 = Host_U(x, y, t);
    float q1 = Host_V(x, y, t);

    // 2段目
    x = xn + 0.5f * p1 * dt;
    y = yn + 0.5f * q1 * dt;
    t = time + 0.5f * dt;
    float p2 = Host_U(x, y, t);
    float q2 = Host_V(x, y, t);

    // 3段目
    x = xn + 0.5f * p2 * dt;
    y = yn + 0.5f * q2 * dt;
    t = time + 0.5f * dt;
    float p3 = Host_U(x, y, t);
    float q3 = Host_V(x, y, t);

    // 4段目
    x = xn + p3 * dt;
    y = yn + q3 * dt;
    t = time + dt;
    float p4 = Host_U(x, y, t);
    float q4 = Host_V(x, y, t);

    particles[nthPartice][0] = xn + (p1 + p2 + p3 + p4) / 6.0f * dt;
    particles[nthPartice][1] = yn + (q1 + q2 + q3 + q4) / 6.0f * dt;
}


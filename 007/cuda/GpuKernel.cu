#include  <cuda_runtime_api.h>
#include  "GpuKernel.cuh"


static constexpr double pi = 3.141592653589793;

static inline __device__ float Device_U(float x, float y, float t)
{
    return -2.0f * __cosf(pi * t / 8.0f) * __sinf(pi * x) * __sinf(pi * x) * __cosf(pi * y) * __sinf(pi * y);
}

static inline __device__ float Device_V(float x, float y, float t)
{
    return 2.0f * __cosf(pi * t / 8.0f) * __cosf(pi * x) * __sinf(pi * x) * __sinf(pi * y) * __sinf(pi * y);
}

// GPUカーネル用 ルンゲクッタ法
__global__ void Device_RungeKutta(int nMaxParticle, float (*particles)[2], float time, float dt)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nMaxParticle)
    {
        return ;
    }

    float xn = particles[index][0];
    float yn = particles[index][1];

    float x = xn;
    float y = yn;
    float t = time;

    float p1 = Device_U(x, y, t);
    float q1 = Device_V(x, y, t);

    x = xn + 0.5f * p1 * dt;
    y = yn + 0.5f * q1 * dt;
    t = time + 0.5f * dt;
    float p2 = Device_U(x, y, t);
    float q2 = Device_V(x, y, t);

    x = xn + 0.5f * p2 * dt;
    y = yn + 0.5f * q2 * dt;
    t = time + 0.5f * dt;
    float p3 = Device_U(x, y, t);
    float q3 = Device_V(x, y, t);

    x = xn + p3 * dt;
    y = yn + q3 * dt;
    t = time + dt;
    float p4 = Device_U(x, y, t);
    float q4 = Device_V(x, y, t);

    particles[index][0] = xn + (p1 + p2 + p3 + p4) / 6.0f * dt;
    particles[index][1] = yn + (q1 + q2 + q3 + q4) / 6.0f * dt;
}


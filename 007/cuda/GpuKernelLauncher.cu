#include  "GpuParticle2DContainer.hpp"
#include  "GpuKernel.cuh"
#include  "GpuKernelLauncher.hpp"


static constexpr int CudaThreadSize = 512;

void launchGpuKernel(int nParticles, std::unique_ptr<GpuParticle2DContainer>& particles, double time, double dt)
{
    GpuParticle2DContainer* particle2DContainer = particles.get();
    float (*raw_particles)[2] = reinterpret_cast<float(*)[2]>(particle2DContainer->getDeviceMemoryPtr());

    dim3 grid(nParticles / CudaThreadSize + 1, 1);
    dim3 block(CudaThreadSize, 1, 1);
    Device_RungeKutta <<< grid, block >>> (nParticles, raw_particles, time, dt);
}


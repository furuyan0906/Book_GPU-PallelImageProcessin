#include  "CpuKernel.hpp"
#include  "CpuKernelLauncher.hpp"


void launchCpuKernel(int nMaxParticles, std::unique_ptr<Particle2DContainer<float>>& particles, double time, double dt)
{
    Particle2DContainer<float>* particle2DContainer = particles.get();
    float (*raw_particles)[2] = reinterpret_cast<float(*)[2]>(particle2DContainer->getRawParticles());

    for (int i = 0; i < nMaxParticles; ++i)
    {
        Host_RungeKutta(i, raw_particles, time, dt);
    }
}


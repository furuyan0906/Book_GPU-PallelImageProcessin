#ifndef  H__CPU_KERNEL_LAUNCHER__H
#define  H__CPU_KERNEL_LAUNCHER__H


#include  <memory>
#include  "Particle2DContainer.hpp"

void launchCpuKernel(int nMaxParticles, std::unique_ptr<Particle2DContainer<float>>& particles, double time, double dt);


#endif  // H__CPU_KERNEL_LAUNCHER__H


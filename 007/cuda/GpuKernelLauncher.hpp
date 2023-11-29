#ifndef  H__GPU_KERNEL_LAUNCHER__H
#define  H__GPU_KERNEL_LAUNCHER__H


#include  <memory>
#include  "cuda_typedef.hpp"
#include  "GpuParticle2DContainer.hpp"


void launchGpuKernel(int nMaxParticles, std::unique_ptr<GpuParticle2DContainer>& particles, double time, double dt);


#endif  // H__GPU_KERNEL_LAUNCHER__H


#ifndef H__CUDA_DIFFUSION_2D__H
#define H__CUDA_DIFFUSION_2D__H

#include <cstdint>
#include <utility>

namespace cuda
{
    void Initialize(
            std::uint64_t width,
            std::uint64_t height,
            float kappa,
            float max_density,
            const std::pair<float, float>& xy_max,
            const std::pair<float, float>& xy_min);
    
    std::uint32_t* Launch(float dt, float dx, float dy);
};

#endif  // H__CUDA_DIFFUSION_2D__H


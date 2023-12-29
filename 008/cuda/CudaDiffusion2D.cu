#include "CudaDiffusion2D.hpp"
#include <memory>
#include <stdexcept>
#include "CudaDiffusion2DService.cuh"

namespace cuda
{
    static std::unique_ptr<CudaDiffusion2DService> cudaDiffusion2DService;

    void Initialize(
            std::uint64_t width,
            std::uint64_t height,
            float kappa,
            float max_density,
            const std::pair<float, float>& xy_max,
            const std::pair<float, float>& xy_min)
    {
        if (!cudaDiffusion2DService)
        {
            cudaDiffusion2DService = std::make_unique<CudaDiffusion2DService>(width, height, kappa, max_density);
        }

        cudaDiffusion2DService->Initialize(xy_max, xy_min);
    }
    
    std::uint32_t* Launch(float dt, float dx, float dy)
    {
        if (!cudaDiffusion2DService)
        {
            throw std::runtime_error("Not initialized.");
        }

        return cudaDiffusion2DService->Launch(dt, dx, dy);
    }
};


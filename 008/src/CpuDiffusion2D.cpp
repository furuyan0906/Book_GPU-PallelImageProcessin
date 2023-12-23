#include "CpuDiffusion2D.hpp"
#include <memory>
#include <stdexcept>
#include "CpuDiffusion2DService.hpp"

namespace cpu
{
    static std::unique_ptr<CpuDiffusion2DService> cpuDiffusion2DService;

    void Initialize(
            std::uint64_t width,
            std::uint64_t height,
            float kappa,
            float max_density,
            const std::pair<float, float>& xy_max,
            const std::pair<float, float>& xy_min)
    {
        if (!cpuDiffusion2DService)
        {
            cpuDiffusion2DService = std::make_unique<CpuDiffusion2DService>(width, height, kappa, max_density);
        }

        cpuDiffusion2DService->Initialize(xy_max, xy_min);
    }
    
    std::uint32_t* Launch(float dt, float dx, float dy)
    {
        if (!cpuDiffusion2DService)
        {
            throw std::runtime_error("Not initialized.");
        }

        return cpuDiffusion2DService->Launch(dt, dx, dy);
    }
}


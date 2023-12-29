#ifndef  CUH__GPU_DIFFUSION_2D_SERVICE__CUH
#define  CUH__GPU_DIFFUSION_2D_SERVICE__CUH

#include <cstdint>
#include <array>
#include <memory>
#include <utility>
#include <cstdint>
#include "cuda_util.cuh"
#include "Diffusion2DServiceBase.hpp"

namespace cuda
{
    class CudaDiffusion2DService final : public Diffusion2DServiceBase
    {
        private:
            static constexpr int NFields = 2;
    
            std::array<cuda::shared_ptr<float[]>, NFields> fields;
            int srcFieldIndex;
            int dstFieldIndex;
    
            cuda::shared_ptr<std::uint32_t[]> deviceGraphicalResource;
            std::shared_ptr<std::uint32_t[]> hostGraphicalResource;

            void UpdateIndecies() noexcept;
    
        public: 
            CudaDiffusion2DService(
                    std::uint64_t width,
                    std::uint64_t height, 
                    float kappa,
                    float max_density) noexcept;
    
            ~CudaDiffusion2DService() noexcept;
    
            void Initialize(
                    const std::pair<float, float>& xy_max,
                    const std::pair<float, float>& xy_min) noexcept override;
    
            std::uint32_t* Launch(float dt, float dx, float dy) noexcept override;
    };
}

#endif  // CUH__GPU_DIFFUSION_2D_SERVICE__CUH


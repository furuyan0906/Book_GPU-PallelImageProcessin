#ifndef  H__DIFFUSION_2D_SERVICE__H
#define  H__DIFFUSION_2D_SERVICE__H

#include <array>
#include <memory>
#include <utility>
#include <cstdint>
#include "Diffusion2DServiceBase.hpp"
#include "GraphicalResourceConverter.hpp"

namespace cpu
{
    class CpuDiffusion2DService final : public Diffusion2DServiceBase
    {
        private:
            static constexpr int NFields = 2;
            std::array<std::shared_ptr<float[]>, NFields> fields;
            int srcFieldIndex;
            int dstFieldIndex;
    
            std::shared_ptr<std::uint32_t[]> graphicalResource;
            GraphicalResourceConverter graphicalResourceConverter;
    
            void UpdateIndecies() noexcept;
    
            std::uint64_t LaunchKernel(
                    const float* const srcField,
                    float* const dstField,
                    std::uint64_t kx,
                    std::uint64_t ky,
                    float c0,
                    float c1,
                    float c2) noexcept;
    
            void ConvertToRGB(std::uint32_t* dst, float src, float src_max);
    
        public:
            CpuDiffusion2DService(
                    std::uint64_t width,
                    std::uint64_t height, 
                    float kappa,
                    float max_density) noexcept;
    
            ~CpuDiffusion2DService() noexcept;
    
            void Initialize(
                    const std::pair<float, float>& xy_max,
                    const std::pair<float, float>& xy_min) noexcept override;
    
            std::uint32_t* Launch(float dt, float dx, float dy) noexcept override;
    };
}

#endif  // H__DIFFUSION_2D_SERVICE__H


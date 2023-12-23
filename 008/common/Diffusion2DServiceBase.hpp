#ifndef  H__DIFFUSION_2D_SERVICE_BASE_H
#define  H__DIFFUSION_2D_SERVICE_BASE_H

#include <memory>
#include <tuple>
#include <cstdint>

class Diffusion2DServiceBase
{
    protected:
        std::uint64_t width;
        std::uint64_t height;
        float kappa;
        float max_density;

        Diffusion2DServiceBase(
                std::uint64_t width,
                std::uint64_t height, 
                float kappa,
                float max_density) noexcept
            : width(width), height(height), kappa(kappa), max_density(max_density)
        {
        }

        std::tuple<float, float, float> GetDiffusion2DParameters(float dt, float dx, float dy) noexcept
        {
            auto c0 = kappa * dt / (dx * dx);
            auto c1 = kappa * dt / (dy * dy);
            auto c2 = 1.0f - 2.0f * (c0 + c1);
        
            return std::tuple<float, float, float>(c0, c1, c2);
        }
        
        std::uint64_t GetIndex(std::uint64_t kx, std::uint64_t ky) noexcept
        {
            return this->width * ky + kx;
        }
    
    public:
        ~Diffusion2DServiceBase(){}

        virtual void Initialize(
                const std::pair<float, float>& xy_max,
                const std::pair<float, float>& xy_min) = 0;

        virtual std::uint32_t* Launch(float dt, float dx, float dy) = 0;
};

#endif // H__DIFFUSION_2D_SERVICE_BASE_H


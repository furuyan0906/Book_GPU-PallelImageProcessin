#include "CpuDiffusion2DService.hpp"
#include "GraphicalResourceConverter.hpp"
#include "BinaryLog.hpp"

namespace cpu
{
    // ----- Public ------
    
    CpuDiffusion2DService::CpuDiffusion2DService(
            std::uint64_t width,
            std::uint64_t height,
            float kappa,
            float max_density) noexcept
        : Diffusion2DServiceBase(width, height, kappa, max_density),
          graphicalResourceConverter(GraphicalResourceConverter())
    {
        this->fields[0] = std::shared_ptr<float[]>(new float [width * height]);
        this->fields[1] = std::shared_ptr<float[]>(new float [width * height]);
        this->srcFieldIndex = 0;
        this->dstFieldIndex = 1;
    
        this->graphicalResource = std::shared_ptr<std::uint32_t[]>(new std::uint32_t [width * height]);
    }
    
    CpuDiffusion2DService::~CpuDiffusion2DService() noexcept
    {
    }
    
    void CpuDiffusion2DService::Initialize(
            const std::pair<float, float>& xy_max,
            const std::pair<float, float>& xy_min) noexcept
    {
        auto x_max = xy_max.first; auto x_min = xy_min.first;
        auto y_max = xy_max.second; auto y_min = xy_min.second;
    
        auto dx = (x_max - x_min) / this->width;
        auto dy = (y_max - y_min) / this->height;
    
        auto field = this->fields[this->srcFieldIndex];
    
        for (std::uint64_t ky = 0; ky < this->height; ++ky)
        {
            for (std::uint64_t kx = 0; kx < this->width; ++kx)
            {
                auto index = this->GetIndex(kx, ky);
    
                auto x = dx * (kx + 0.5f) + x_min;
                auto y = dy * (ky + 0.5f) + y_min;
                auto condition = (x > (0.75f * x_min + 0.25f * x_max)) &&
                                 (x < (0.25f * x_min + 0.75f * x_max)) &&
                                 (y > (0.75f * y_min + 0.25f * y_max)) &&
                                 (y < (0.25f * y_min + 0.75f * y_max));
                
                field[index] = condition ? this->max_density : 0.0f;
            }
        }
    
        //WriteIntoBinaryFile(this->fields[this->srcFieldIndex], this->height * this->width);
    }
    
    std::uint32_t* CpuDiffusion2DService::Launch(float dt, float dx, float dy) noexcept
    {
        auto parameters = this->GetDiffusion2DParameters(dt, dx, dy);
        auto c0 = std::get<0>(parameters);
        auto c1 = std::get<1>(parameters);
        auto c2 = std::get<2>(parameters);
    
        auto srcField = this->fields[this->srcFieldIndex].get();
        auto dstField = this->fields[this->dstFieldIndex].get();
        auto resource = this->graphicalResource.get();
    
        for (std::uint64_t ky = 0; ky < this->height; ++ky)
        {
            for (std::uint64_t kx = 0; kx < this->width; ++kx)
            {
                auto index = this->LaunchKernel(srcField, dstField, kx, ky, c0, c1, c2);
                this->ConvertToRGB(&resource[index], dstField[index], this->max_density);
            }
        }
    
        // WriteIntoBinaryFile("Field", dstField, this->height * this->width);
        // WriteIntoBinaryFile("Resource", resource, this->height * this->width);
    
        this->UpdateIndecies();
    
        return resource;
    }
    
    // ----- Private ------
    
    void CpuDiffusion2DService::UpdateIndecies() noexcept
    {
        this->srcFieldIndex = this->dstFieldIndex;
        this->dstFieldIndex = (this->srcFieldIndex + 1) % NFields;
    }
    
    std::uint64_t CpuDiffusion2DService::LaunchKernel(
            const float* const srcField,
            float* const dstField,
            std::uint64_t kx,
            std::uint64_t ky,
            float c0,
            float c1,
            float c2) noexcept
    {
        auto index = this->GetIndex(kx, ky);
    
        auto fc = srcField[index];
        auto fl = (kx == 0) ? fc : srcField[index - 1];
        auto fr = (kx >= (this->width - 1)) ? fc : srcField[index + 1];
        auto fd = (ky == 0) ? fc : srcField[index - this->width];
        auto ft = (ky >= (this->height - 1)) ? fc : srcField[index + this->width];
    
        auto density = c0 * (fl + fr) + c1 * (fd + ft) + c2 * fc; 
        dstField[index] = density;
    
        return index;
    }
    
    void CpuDiffusion2DService::ConvertToRGB(std::uint32_t* dst, float src, float src_max)
    {
        this->graphicalResourceConverter.ConvertToRGB(dst, src, src_max);
    }
}


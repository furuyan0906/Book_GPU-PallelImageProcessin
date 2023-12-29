#include "CudaDiffusion2DProvider.cuh"
#include <cuda_device_runtime_api.h>

namespace cuda
{
    __device__ static void ConvertToRGB(std::uint32_t* dst, float src, float src_max);

    __global__ void LaunchKernel(
            const float* const srcField,
            float* const dstField,
            std::uint32_t* resource,
            std::uint64_t width,
            std::uint64_t height,
            float c0,
            float c1,
            float c2,
            float max_density)
    {
        auto ky = blockDim.y * blockIdx.y + threadIdx.y;
        auto kx = blockDim.x * blockIdx.x + threadIdx.x;
    
        if ((kx >= width) || (ky >= height))
        {
            return ;
        }
    
        auto index = width * ky + kx;
    
        auto fc = srcField[index];
        auto fl = kx == 0 ? fc : srcField[index - 1];
        auto fr = kx >= (width - 1) ? fc : srcField[index + 1];
        auto fd = ky == 0 ? fc : srcField[index - width];
        auto ft = ky >= (height - 1) ? fc : srcField[index + width];
    
        dstField[index] = c0 * (fr + fl) + c1 * (ft + fd) + c2 * fc;

        ConvertToRGB(&resource[index], dstField[index], max_density);
    }    
    
    __device__ void ConvertToRGB(std::uint32_t* dst, float src, float src_max)
    {
        auto value = 255.0f - (src / src_max) * 255.0f;
        value = ::fmaxf(value, 0.0f);
        value = ::fminf(value, 255.0f);
    
        auto r = std::uint32_t(value);
        auto g = std::uint32_t(value);
        auto b = std::uint32_t(value);
        auto bgr = static_cast<std::uint32_t>((b << 16) | (g << 8) | (r << 0));
    
        *dst = bgr;
    }
};


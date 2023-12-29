#ifndef CUH__CUDA_DIFFUSION_2D_PROVIDER__CUH
#define CUH__CUDA_DIFFUSION_2D_PROVIDER__CUH

#include <cstdint>

namespace cuda
{
    __global__ void LaunchKernel(
            const float* const srcField,
            float* const dstField,
            std::uint32_t* resource,
            std::uint64_t width,
            std::uint64_t height,
            float c0,
            float c1,
            float c2,
            float max_density);
};

#endif  // CUH__CUDA_DIFFUSION_2D_PROVIDER__CUH


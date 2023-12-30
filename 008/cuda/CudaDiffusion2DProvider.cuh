#ifndef CUH__CUDA_DIFFUSION_2D_PROVIDER__CUH
#define CUH__CUDA_DIFFUSION_2D_PROVIDER__CUH

#include <cstdint>
#include <texture_types.h>

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

    __global__ void LaunchKernelWithSharedMemory(
            const float* const srcField,
            float* const dstField,
            std::uint32_t* resource,
            std::uint64_t width,
            std::uint64_t height,
            float c0,
            float c1,
            float c2,
            float max_density);

    __global__ void LaunchKernelWithTextureMemory(
            const cudaTextureObject_t& srcField,
            float* dstField,
            std::uint32_t* resource,
            std::uint64_t width,
            std::uint64_t height,
            float c0,
            float c1,
            float c2,
            float max_density);
};

#endif  // CUH__CUDA_DIFFUSION_2D_PROVIDER__CUH


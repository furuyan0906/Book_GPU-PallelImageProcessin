#include "CudaDiffusion2DService.cuh"
#include <vector>
#include <iostream>
#include <stdexcept>
#include "CudaDiffusion2DParameters.hpp"
#include "CudaDiffusion2DProvider.cuh"

#define USE_AS_NORMAL           (1 << 0)
#define USE_SHARED_MEMORY       (1 << 1)
#define USE_TEXTURE_MEMORY      (1 << 2)
#define USE_PIXEL_BUFFER_OBJECT (1 << 3)

#define USE_CUDA_MODE USE_TEXTURE_MEMORY

namespace cuda
{
    // ----- Public ------

    CudaDiffusion2DService::CudaDiffusion2DService(
            std::uint64_t width,
            std::uint64_t height, 
            float kappa,
            float max_density)
        : Diffusion2DServiceBase(width, height, kappa, max_density)
    {
        this->fields[0] = cuda::make_shared<float[]>(width * height);
        this->fields[1] = cuda::make_shared<float[]>(width * height);
        this->srcFieldIndex = 0;
        this->dstFieldIndex = 1;

        this->deviceGraphicalResource = cuda::make_shared<std::uint32_t[]>(width * height);
        this->hostGraphicalResource = std::shared_ptr<std::uint32_t[]>(new std::uint32_t[width * height]);

#if USE_CUDA_MODE == USE_TEXTURE_MEMORY
        if (!this->TryCreateCudaTextureObjects())
        {
            throw std::runtime_error("Failed to create texture objects");
        }
#endif
    }
    
    CudaDiffusion2DService::~CudaDiffusion2DService()
    {
#if USE_CUDA_MODE == USE_TEXTURE_MEMORY
        this->DisposeCudaTextureObjects();
#endif
    }
    
    void CudaDiffusion2DService::Initialize(
            const std::pair<float, float>& xy_max,
            const std::pair<float, float>& xy_min) noexcept
    {
        auto x_max = xy_max.first; auto x_min = xy_min.first;
        auto y_max = xy_max.second; auto y_min = xy_min.second;
    
        auto dx = (x_max - x_min) / this->width;
        auto dy = (y_max - y_min) / this->height;

        auto deviceResource = this->fields[this->srcFieldIndex];
        auto hostResource = std::vector<float>(this->width * this->height);

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
                
                hostResource[index] = condition ? this->max_density : 0.0f;
            }
        }

        ::cudaMemcpy(deviceResource.get(), hostResource.data(), this->width * this->height * sizeof(float), ::cudaMemcpyHostToDevice);
    }
    
    std::uint32_t* CudaDiffusion2DService::Launch(float dt, float dx, float dy) noexcept
    {
        auto parameters = this->GetDiffusion2DParameters(dt, dx, dy);
        auto c0 = std::get<0>(parameters);
        auto c1 = std::get<1>(parameters);
        auto c2 = std::get<2>(parameters);

        auto srcField = this->fields[this->srcFieldIndex].get();
        auto dstField = this->fields[this->dstFieldIndex].get();
        auto deviceResource = this->deviceGraphicalResource.get();

        auto grid = dim3(this->width / BlockDimX + 1, height / BlockDimY + 1);
        auto block = dim3(BlockDimX, BlockDimY);

#if USE_CUDA_MODE == USE_SHARED_MEMORY
        LaunchKernelWithSharedMemory <<< grid, block >>> (srcField, dstField, deviceResource, this->width, this->height, c0, c1, c2, this->max_density);
#elif USE_CUDA_MODE == USE_TEXTURE_MEMORY
        const auto textureSource = this->textureSources[this->srcFieldIndex];
        LaunchKernelWithTextureMemory <<< grid, block >>> (textureSource, dstField, deviceResource, this->width, this->height, c0, c1, c2, this->max_density);
#elif USE_CUDA_MODE == USE_PIXEL_BUFFER_OBJECT
        // TODO
#else
        // シェアードメモリ, テクスチャメモリ, PBO(Pixel Buffer Object)のいずれも使用しない
        LaunchKernel <<< grid, block >>> (srcField, dstField, deviceResource, this->width, this->height, c0, c1, c2, this->max_density);
#endif

        auto hostResource = this->hostGraphicalResource.get();

        ::cudaMemcpy(hostResource, deviceResource, this->width * this->height * sizeof(std::uint32_t), ::cudaMemcpyDeviceToHost);

        this->UpdateIndecies();

        return hostResource;
    }

    // ----- Private ------

    void CudaDiffusion2DService::UpdateIndecies() noexcept
    {
        this->srcFieldIndex = this->dstFieldIndex;
        this->dstFieldIndex = (this->srcFieldIndex + 1) % NFields;
    }

    bool CudaDiffusion2DService::TryCreateCudaTextureObjects()
    {
        try
        {
            auto channelDesc = ::cudaCreateChannelDesc(32, 0, 0, 0, ::cudaChannelFormatKindFloat);

            for (int i = 0; i < NFields; ++i)
            {
                ::cudaResourceDesc resourceDesc;
                {
                    memset(&resourceDesc, 0, sizeof(::cudaResourceDesc));
                    resourceDesc.resType = ::cudaResourceTypeLinear;
                    resourceDesc.res.linear.devPtr = this->fields[i].get();
                    resourceDesc.res.linear.desc = channelDesc;
                    resourceDesc.res.linear.sizeInBytes = this->width * this->height * sizeof(float);
                }

                ::cudaTextureDesc textureDesc;
                {
                    memset(&textureDesc, 0, sizeof(::cudaTextureDesc));
                    textureDesc.filterMode = ::cudaFilterModeLinear;
                    textureDesc.readMode = ::cudaReadModeElementType;
                }

                CHECK_CUDA_ERROR(::cudaCreateTextureObject(&(this->textureSources[i]), &resourceDesc, &textureDesc, nullptr));
            }

            return true;
        }
        catch (std::exception& e)
        {
            std::cout << "[ErrorReason] " << e.what() << std::endl;
            return false;
        }
    }

    void CudaDiffusion2DService::DisposeCudaTextureObjects()
    {
        for (int i = 0; i < NFields; ++i)
        {
            CHECK_CUDA_ERROR(::cudaDestroyTextureObject(this->textureSources[i]));
        }
    }
};


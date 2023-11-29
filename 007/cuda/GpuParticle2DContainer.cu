#include  "cuda_typedef.hpp"
#include  "cuda_error_check.cuh"
#include  "cuda_util.cuh"
#include  "GpuParticle2DContainer.hpp"
#include  <cuda_gl_interop.h>

GpuParticle2DContainer::GpuParticle2DContainer(int nParticle)
    : nParticle_(nParticle),
      nbytes_(nParticle * 2 * sizeof(float)),
      particles_(cuda::make_unique<float[]>(this->nbytes_))
{
}

GpuParticle2DContainer::~GpuParticle2DContainer()
{
    if (this->graphicsResource_ != nullptr)
    {
        this->unmapGraphicsResource();
    }

    ::cudaDeviceReset();
}

void GpuParticle2DContainer::copyToDevice(float* hostMemoryPtr, size_t nbytes)
{
    ThrowInvalidArgumentExceptionIfTooLargeMemorySize(nbytes);

    float* deviceMemoryPtr = this->particles_.get();

    CHECK_CUDA_ERROR(::cudaMemcpy(deviceMemoryPtr, hostMemoryPtr, nbytes, ::cudaMemcpyHostToDevice));
}

void GpuParticle2DContainer::copyToHost(float* hostMemoryPtr, size_t nbytes)
{
    ThrowInvalidArgumentExceptionIfTooLargeMemorySize(nbytes);

    float* deviceMemoryPtr = this->particles_.get();

    CHECK_CUDA_ERROR(::cudaMemcpy(hostMemoryPtr, deviceMemoryPtr, nbytes, ::cudaMemcpyDeviceToHost));
}

void GpuParticle2DContainer::registerGraphicsResource(const GLuint& vbo)
{
    this->ThrowRuntimeErrorIfGraphicsResourceAlreadyRegistered();

    this->vbo_ = vbo;

    auto rawGraphicsResource = &this->graphicsResource_;
    CHECK_CUDA_ERROR(::cudaGraphicsGLRegisterBuffer(rawGraphicsResource, vbo, cudaGraphicsRegisterFlagsNone));
}

void GpuParticle2DContainer::unregisterGraphicsResource(void)
{
    this->ThrowRuntimeErrorIfGraphicsResourceNotRegistered();

    CHECK_CUDA_ERROR(::cudaGraphicsUnregisterResource(this->graphicsResource_));

    this->graphicsResource_ = nullptr;
}

void GpuParticle2DContainer::mapGraphicsResource(void)
{
    this->ThrowRuntimeErrorIfGraphicsResourceNotRegistered();

    CHECK_CUDA_ERROR(::cudaGraphicsMapResources(1, &this->graphicsResource_, 0));

    auto rawDeviceMemoryPtr = this->particles_.get();
    CHECK_CUDA_ERROR(::cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&rawDeviceMemoryPtr), nullptr, this->graphicsResource_));
}

void GpuParticle2DContainer::unmapGraphicsResource(void)
{
    this->ThrowRuntimeErrorIfGraphicsResourceNotRegistered();

    CHECK_CUDA_ERROR(::cudaGraphicsUnmapResources(1, &this->graphicsResource_, 0));
}


#ifndef  H__GPU_PARTICLE_2D_CONTAINER__H
#define  H__GPU_PARTICLE_2D_CONTAINER__H


#include  <memory>
#include  <stdexcept>
#include  <string>
#include  <sstream>
#include  <unordered_map>
#include  <GL/glew.h>
#include  "cuda_typedef.hpp"


class GpuParticle2DContainer
{
    private:
        size_t nParticle_ = 0;

        // CUDAデバイスメモリのバイト数
        size_t nbytes_ = 0;

        // CUDAデバイスメモリをスマートポインタで管理する
        cuda::unique_ptr<float[]> particles_;

        // デバイスメモリ上の頂点バッファオブジェクトを管理する
        struct cudaGraphicsResource *graphicsResource_ = nullptr;

        GLuint vbo_;

        GpuParticle2DContainer(int nParticle);

        inline bool isIndexOutOfRange(size_t nthParticle) noexcept
        {
            return (this->nParticle_ <= nthParticle);
        }

        inline void ThrowRangeErrorExceptionIfIndexIsOutOfRange(int nthParticle)
        {
            if (this->isIndexOutOfRange(nthParticle))
            {
                std::stringstream ss;
                ss << "Argument Exception: `nthParticle` must be positive and less than " << this->nParticle_ << " (nthParticle = " << nthParticle << ")";
                throw std::range_error(ss.str());
            }
        }

        static inline bool isNumOfParticleNegativeOfZero(size_t nParticle) noexcept
        {
            return nParticle <= 0;
        }

        static inline void ThrowInvalidArgumentExceptionIfNumOfParticleIsNegativeOrZero(size_t nParticle)
        {
            if (isNumOfParticleNegativeOfZero(nParticle))
            {
                throw std::invalid_argument("Argument Exception: `nParticle` must be positive");
            }
        }

        inline bool isTooLargeMemorySize(size_t nbytes) noexcept
        {
            return this->nbytes_ < nbytes;
        }

        inline void ThrowInvalidArgumentExceptionIfTooLargeMemorySize(size_t nbytes)
        {
            if (isTooLargeMemorySize(nbytes))
            {
                std::stringstream ss;
                ss << "Argument Exception: 'nbytes' must be less than " << this->nbytes_ << " (nbytes = " << nbytes << ")";
                throw std::invalid_argument(ss.str());
            }
        }

        inline bool IsGraphicsResourceRegistered() noexcept
        {
            return this->graphicsResource_ != nullptr;
        }

        inline void ThrowRuntimeErrorIfGraphicsResourceAlreadyRegistered()
        {
            if (this->IsGraphicsResourceRegistered())
            {
                std::stringstream ss;
                ss << "Exception: GraphicsResource has been already registered";
                throw std::runtime_error(ss.str());
            }
        }

        inline void ThrowRuntimeErrorIfGraphicsResourceNotRegistered()
        {
            if (!this->IsGraphicsResourceRegistered())
            {
                std::stringstream ss;
                ss << "Exception: GraphicsResource does not been registered yet";
                throw std::runtime_error(ss.str());
            }
        }
 
    public:
        ~GpuParticle2DContainer() noexcept;

        // 疑似Factoryパターン
        static std::unique_ptr<GpuParticle2DContainer> CreateParticles(size_t nParticle)
        {
            ThrowInvalidArgumentExceptionIfNumOfParticleIsNegativeOrZero(nParticle);

            return std::unique_ptr<GpuParticle2DContainer>(new GpuParticle2DContainer(nParticle));
        }

        inline float* getDeviceMemoryPtr() noexcept
        {
            return static_cast<float*>(this->particles_.get());
        }

        inline uint64_t getDeviceMemoryBytes() noexcept
        {
            return this->nbytes_;
        }

        void copyToDevice(float* hostMemoryPtr, size_t nbytes);

        void copyToHost(float* hostMemoryPtr, size_t nbytes);

        // --- Vertex Buffer Object ---

        void registerGraphicsResource(const GLuint& vbo);

        void unregisterGraphicsResource(void);

        void mapGraphicsResource(void);

        void unmapGraphicsResource(void);
};


#endif  // H__GPU_PARTICLE_2D_CONTAINER__H


#ifndef  H__PARTICLE_2D_CONTAINER__H
#define  H__PARTICLE_2D_CONTAINER__H


#include  <memory>
#include  <stdexcept>
#include  <string>
#include  <sstream>
#include  <cmath>

template <typename T>
class Particle2DContainer final
{
    private:
        static constexpr int X = 0;
        static constexpr int Y = 1;

        size_t nParticle_ = 0;

        size_t nbytes_ = 0;

        // 連続したメモリ領域を確保するため, 配列を使用する
        std::unique_ptr<T[]> particles_;

        Particle2DContainer(size_t nParticle)
            : nParticle_(nParticle),
              nbytes_(2 * nParticle * sizeof(T)),
              particles_(std::make_unique<T[]>(2 * nParticle))
        {
        }

        inline bool isIndexOutOfRange(size_t nthParticle)
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

        static inline bool isNumOfParticleNegativeOfZero(int nParticle)
        {
            return nParticle <= 0;
        }

        static inline void ThrowInvalidArgumentExceptionIfNumOfParticleIsNegativeOrZero(int nParticle)
        {
            if (isNumOfParticleNegativeOfZero(nParticle))
            {
                throw std::invalid_argument("Argument Exception: `nParticle` must be positive");
            }
        }
 
    public:
        // 疑似Factoryパターン
        static std::unique_ptr<Particle2DContainer<T>> CreateParticles(int nParticle)
        {
            ThrowInvalidArgumentExceptionIfNumOfParticleIsNegativeOrZero(nParticle);

            return std::unique_ptr<Particle2DContainer<T>>(new Particle2DContainer<T>(nParticle));
        }

        inline void setPosition(int nthParticle, T x, T y)
        {
            this->ThrowRangeErrorExceptionIfIndexIsOutOfRange(nthParticle);

            this->particles_[2 * nthParticle + X] = x;
            this->particles_[2 * nthParticle + Y] = y;
        }

        inline T getX(int nthParticle)
        {
            this->ThrowRangeErrorExceptionIfIndexIsOutOfRange(nthParticle);

            return this->particles_[2 * nthParticle + X];
        }

        inline T getY(int nthParticle)
        {
            this->ThrowRangeErrorExceptionIfIndexIsOutOfRange(nthParticle);

            return this->particles_[2 * nthParticle + Y];
        }

        inline T* getParticleAsRawPointer(size_t nthParticle)
        {
            this->ThrowRangeErrorExceptionIfIndexIsOutOfRange(nthParticle);

            T* ptr = this->particles_.get();
            return &ptr[2 * nthParticle];
        }

        inline T* getRawParticles()
        {
            return this->particles_.get();
        }

        inline size_t getMemorySize()
        {
            return this->nbytes_;
        }
};


#endif  // H__PARTICLE_2D_CONTAINER__H


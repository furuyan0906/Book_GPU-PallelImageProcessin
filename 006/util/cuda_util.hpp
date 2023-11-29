#ifndef  H__CUDA_UTIL__H
#define  H__CUDA_UTIL__H


#include  <stdexcept>
#include  <sstream>
#include  <memory>
#include  <type_traits>
#include  <cuda_runtime_api.h>


#define  CHECK_CUDA_ERROR(e)  (cuda::check_error(e, __FILE__, __LINE__))

namespace cuda
{
    template<typename F, typename N>
    void check_error(const ::cudaError_t e, F&& f, N&& n)
    {
        if (e != ::cudaSuccess)
        {
            std::stringstream s;
            s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": " << ::cudaGetErrorString(e);
            throw std::runtime_error{s.str()};
        }
    }

    struct Deleter
    {
        void operator()(void* p) const
        {
            CHECK_CUDA_ERROR(::cudaFree(p));
            p = nullptr;
        }
    };

    template<typename T>
    using unique_ptr = std::unique_ptr<T, Deleter>;

    template<typename T>
    typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(const std::size_t n)
    {
        using U = typename std::remove_extent<T>::type;

        U* p;
        CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
        return cuda::unique_ptr<T>{p};
    }

    template<typename T>
    cuda::unique_ptr<T> make_unique()
    {
        T* p;
        CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T)));
        return cuda::unique_ptr<T>{p};
    }

}


#endif  /* H__CUDA_UTIL__H */


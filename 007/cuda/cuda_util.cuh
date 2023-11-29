#ifndef  CUH__CUDA_UTIL__CUH
#define  CUH__CUDA_UTIL__CUH


#include  <type_traits>
#include  <cuda_runtime_api.h>
#include  "cuda_typedef.hpp"
#include  "cuda_error_check.cuh"

namespace cuda
{
    template<typename T>
    typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(size_t n)
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
};


#endif  // CUH__CUDA_UTIL__CUH


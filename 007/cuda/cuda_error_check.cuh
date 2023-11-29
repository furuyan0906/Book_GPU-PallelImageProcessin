#ifndef  CUH__CUDA_ERROR_CHECK__CUH
#define  CUH__CUDA_ERROR_CHECK__CUH


#include  <string>
#include  <sstream>
#include  <stdexcept>
#include  <cuda_runtime_api.h>

namespace cuda
{
    template<typename F, typename N>
    void check_error(const ::cudaError_t e, F&& f, N&& n)
    {
        if (e != ::cudaSuccess)
        {
            std::stringstream ss;
            ss << ::cudaGetErrorName(e) << " (cudaError=" << e << ") @" << f << "#L" << n << ": " << ::cudaGetErrorString(e);
            throw std::runtime_error(ss.str());
        }
    }

    #define CHECK_CUDA_ERROR(e) (cuda::check_error(e, __FILE__, __LINE__))
};


#endif  // CUH__CUDA_ERROR_CHECK__CUH


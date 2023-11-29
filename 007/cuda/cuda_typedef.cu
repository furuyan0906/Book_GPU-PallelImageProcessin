#include  <cuda_runtime_api.h>
#include  "cuda_error_check.cuh"
#include  "cuda_typedef.hpp"


namespace cuda {
    void Deleter::operator() (void* p) const
    {
        CHECK_CUDA_ERROR(::cudaFree(p));
    }
};


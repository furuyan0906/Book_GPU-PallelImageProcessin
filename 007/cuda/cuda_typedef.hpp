#ifndef  H__CUDA_TYPEDEF__H
#define  H__CUDA_TYPEDEF__H


#include  <memory>

namespace cuda
{
    struct Deleter
    {
        void operator() (void* p) const;
    };

    template <typename T>
    using unique_ptr = std::unique_ptr<T, Deleter>;
};


#endif  // H__CUDA_TYPEDEF__H


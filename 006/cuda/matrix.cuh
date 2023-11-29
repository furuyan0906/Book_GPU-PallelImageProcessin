#ifndef  CUH__MATRIX__CUH
#define  CUH__MATRIX__CUH


// CUDAによる行列積
double Device_Multiply(const std::shared_ptr<float[]>& hA, const std::shared_ptr<float[]>& hB, std::shared_ptr<float[]>& hC, const int N);


#endif  /* CUH__MATRIX__CUH */


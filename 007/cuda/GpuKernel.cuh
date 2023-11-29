#ifndef  CUH__GPU_KERNEL__CUH
#define  CUH__GPU_KERNEL__CUH


__global__ void Device_RungeKutta(int nMaxParticle, float (*particles)[2], float time, float dt);


#endif  // CUH__GPU_KERNEL__CUH


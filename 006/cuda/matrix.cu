#include  <iostream>
#include  <sstream>
#include  <chrono>
#include  <stdexcept>
#include  <cassert>
#include  <cuda_runtime_api.h>
#include  "cuda_util.hpp"
#include  "matrix.cuh"


#define do_nothing()

static constexpr int Block = 16;

__global__ void device_Multiply(const float* dA, const float* dB, float* dC, const int N)
{
    unsigned int r = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int c = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float smA[Block][Block];
    __shared__ float smB[Block][Block];

    float product = 0.0f;

    for (int i = 0; i < N; i += Block)
    {
        // Block x Block個のスレッドが一斉に起動し, グローバルメモリからシェアードメモリへコピー
        // threadIdx.x = 0 ~ Block-1
        // threadIdx.y = 0 ~ Block-1
        smA[threadIdx.y][threadIdx.x] = dA[N * r + i + threadIdx.x];
        smB[threadIdx.y][threadIdx.x] = dB[N * (i + threadIdx.y) + c];
        __syncthreads();  // 全てのスレッドの処理が到達するまで, 速く終わったスレッドはサスペンド

        // シェアードメモリで積を計算する
        for (int j = 0; j < Block; ++j)
        {
            product += smA[threadIdx.y][j] * smB[j][threadIdx.x];
        }
        __syncthreads();
    }

    dC[N * r + c] = product;
}

double Device_Multiply(const std::shared_ptr<float[]>& hA, const std::shared_ptr<float[]>& hB, std::shared_ptr<float[]>& hC, const int N)
{
    double time = -1.0;

    try
    {
        int cudaDriverVersion, cudaRuntimeVersion;
        {
            ::cudaDriverGetVersion(&cudaDriverVersion);
            ::cudaRuntimeGetVersion(&cudaRuntimeVersion);
            if ((cudaDriverVersion / 100) != (cudaRuntimeVersion / 100))
            {
                std::stringstream s;
                s << "CUDA Driver = " << cudaDriverVersion << ", CUDA Runtime = " << cudaRuntimeVersion;

                throw std::runtime_error{s.str()};
            }
        }

        // デバイス(GPU)側の行列へのポインタ
        // デバイス側に行列演算用のメモリ領域を確保する
        auto dA = cuda::make_unique<float[]>(N * N);
        auto dB = cuda::make_unique<float[]>(N * N);
        auto dC = cuda::make_unique<float[]>(N * N);

        std::chrono::system_clock::time_point start, end;
        {
            start = std::chrono::system_clock::now();

            // ホスト側の行列のデータをデバイス側の行列用メモリ領域へ転送
            CHECK_CUDA_ERROR(::cudaMemcpy(dA.get(), hA.get(), sizeof(float) * N * N, ::cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(::cudaMemcpy(dB.get(), hB.get(), sizeof(float) * N * N, ::cudaMemcpyHostToDevice));

            // GPUのブロックとグリッドの定義
            dim3 block(Block, Block);
            dim3 grid(N / Block, N / Block);

            // GPU処理の起動
            device_Multiply<<< grid, block >>>(dA.get(), dB.get(), dC.get(), N);

            // 計算結果が格納されているデバイス側メモリ領域(dC)からホスト側メモリ領域(hc)へ転送
            CHECK_CUDA_ERROR(::cudaMemcpy(hC.get(), dC.get(), sizeof(float) * N * N, ::cudaMemcpyDeviceToHost));

            end = std::chrono::system_clock::now();
        }
        time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
    }
    catch (std::exception& e)
    {
        std::cout << "Exception : " << e.what() << std::endl;
    }

    return time;
}


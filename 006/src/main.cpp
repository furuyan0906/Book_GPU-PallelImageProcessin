#include  <iostream>
#include  <vector>
#include  <chrono>
#include  <memory>
#include  <cstdlib>
#include  <cstdio>
#include  <cmath>
#include  <cfloat>
#include  "matrix.hpp"
#include  "matrix.cuh"

using Func = void (*)(const std::shared_ptr<float[]>&, const std::shared_ptr<float[]>&, std::shared_ptr<float[]>&, int);

constexpr int N = 1024;
constexpr int Nsample = 10;

int main(void)
{
    auto A = std::shared_ptr<float[]>(new float[N * N]);
    auto B = std::shared_ptr<float[]>(new float[N * N]);
    auto C = std::shared_ptr<float[]>(new float[N * N]);
    auto Ans = std::shared_ptr<float[]>(new float[N * N]);

    {
        for (int i = 0; i < N * N; ++i)
        {
            A[i] = i;
            B[i] = i;
        }

        Host_Multiply_Default(A, B, Ans, N);
    }

    std::cout << "[--- Host Process ---]" << std::endl;
    {
        auto funcV = std::vector<Func>(0);
        {
            funcV.emplace_back(Host_Multiply_Default);
            funcV.emplace_back(Host_Multiply_LoopOrderExchange);
            funcV.emplace_back(Host_Multiply_LoopUnrolling);
            funcV.emplace_back(Host_Multiply_CacheBlocking);
        }
        int Nfunction = funcV.size();

        auto avgtimes = std::vector<double>(Nfunction, 0.0);
        auto isOKs = std::vector<bool>(Nfunction, true);

        for (int order = 0; order < (int)Nfunction; ++order)
        {
            for (int n = 0; n < Nsample; ++n)
            {
                for (int i = 0; i < N * N; i += 4)
                {
                    C[i + 0] = 0.0;
                    C[i + 1] = 0.0;
                    C[i + 2] = 0.0;
                    C[i + 3] = 0.0;
                }

                std::chrono::system_clock::time_point start, end;
                {
                    auto f = funcV[order];

                    start = std::chrono::system_clock::now();
                    f(A, B, C, N);
                    end = std::chrono::system_clock::now();
                }
                double time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);

                avgtimes[order] = (avgtimes[order] * n + time) / (double)(n + 1);
            }

            // 最後の一回だけベリファイする
            bool isOK = true;
            {
                for (int i = 0; i < N * N; ++i)
                {
                    isOK &= (::fabs(C[i] - Ans[i]) < FLT_EPSILON);
                }

                isOKs[order] = isOK;
            }
        }

        std::cout << "Nsample = " << Nsample << std::endl;
        for (int order = 0; order < Nfunction; ++order)
        {
            std::printf("[%03d] time = %12.3lfus (Verify: %s)\n", order + 1, avgtimes[order], isOKs[order] ? "OK" : "NG");
        }
    }

    // CUDA
    std::cout << "[--- Device(CUDA) Process ---]" << std::endl;
    {
        for (int i = 0; i < N * N; i += 4)
        {
            C[i + 0] = 0.0;
            C[i + 1] = 0.0;
            C[i + 2] = 0.0;
            C[i + 3] = 0.0;
        }

        double avg_actual_time = 0;
        double avg_total_time = 0;
        for (int n = 0; n < Nsample; ++n)
        {
            std::chrono::system_clock::time_point start, end;

            start = std::chrono::system_clock::now();
            double actual_time = Device_Multiply(A, B, C, N);
            end = std::chrono::system_clock::now();

            double total_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);

            avg_actual_time = (avg_actual_time * n + actual_time) / (double)(n + 1);
            avg_total_time = (avg_total_time * n + total_time) / (double)(n + 1);
        }

        bool isOK = true;
        {
            for (int i = 0; i < N * N; ++i) 
            {
                isOK &= (::fabs(C[i] - Ans[i]) < FLT_EPSILON);
            }
        }

        std::cout << "Nsample = " << Nsample << std::endl;
        std::printf("[GPU] actual time = %9.3lfus (Verify: %s)\n", avg_actual_time, isOK ? "OK" : "NG");
        std::printf("[GPU]  total time = %9.3lfus\n", avg_total_time);
    }

	return EXIT_SUCCESS;
}


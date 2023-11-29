#include  "matrix.hpp"


// 最適化なし
void Host_Multiply_Default(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < N; ++k)
            {
                c[i*N + j] += a[i*N + k] * b[k*N + j];
            }
        }
    }
}

// ループ順序交換法 (キャッシュメモリの空間局在性を利用)
void Host_Multiply_LoopOrderExchange(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < N; ++k)
        {
            for (int j = 0; j < N; ++j)
            {
                c[i*N + j] += a[i*N + k] * b[k*N + j];
            }
        }
    }
}

// (4段)ループアンローリング (キャッシュメモリの空間局在性の利用 + 分岐命令の削減)
void Host_Multiply_LoopUnrolling(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N)
{
    int i0, i1, i2, i3;
    int k0, k1, k2, k3;

    for (int i = 0; i < N; i += 4)
    {
        i0 = i + 0;
        i1 = i + 1;
        i2 = i + 2;
        i3 = i + 3;

        for (int k = 0; k < N; k += 4)
        {
            k0 = k + 0;
            k1 = k + 1;
            k2 = k + 2;
            k3 = k + 3;

            for (int j = 0; j < N; ++j)
            {
                c[i0 * N + j] += a[i0 * N + k0] * b[k0 * N + j];
                c[i0 * N + j] += a[i0 * N + k1] * b[k1 * N + j];
                c[i0 * N + j] += a[i0 * N + k2] * b[k2 * N + j];
                c[i0 * N + j] += a[i0 * N + k3] * b[k3 * N + j];

                c[i1 * N + j] += a[i1 * N + k0] * b[k0 * N + j];
                c[i1 * N + j] += a[i1 * N + k1] * b[k1 * N + j];
                c[i1 * N + j] += a[i1 * N + k2] * b[k2 * N + j];
                c[i1 * N + j] += a[i1 * N + k3] * b[k3 * N + j];

                c[i2 * N + j] += a[i2 * N + k0] * b[k0 * N + j];
                c[i2 * N + j] += a[i2 * N + k1] * b[k1 * N + j];
                c[i2 * N + j] += a[i2 * N + k2] * b[k2 * N + j];
                c[i2 * N + j] += a[i2 * N + k3] * b[k3 * N + j];

                c[i3 * N + j] += a[i3 * N + k0] * b[k0 * N + j];
                c[i3 * N + j] += a[i3 * N + k1] * b[k1 * N + j];
                c[i3 * N + j] += a[i3 * N + k2] * b[k2 * N + j];
                c[i3 * N + j] += a[i3 * N + k3] * b[k3 * N + j];
            }
        }
    }
}


// キャッシュブロッキング (キャッシュメモリの時間局在性の利用)
constexpr int Block = 64;
void Host_Multiply_CacheBlocking(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N)
{
    int ii1, ii2, ii3;
    int kk1, kk2, kk3;

    for (int i = 0; i < N; i += Block)
    {
        for (int k = 0; k < N; k += Block)
        {
            for (int j = 0; j < N; j += Block)
            {
                for (int ii0 = i; ii0 < (i + Block); ii0 += 4)
                {
                    ii1 = ii0 + 1;
                    ii2 = ii0 + 2;
                    ii3 = ii0 + 3;
                    for (int kk0 = k; kk0 < (k + Block); kk0 += 4)
                    {
                        kk1 = kk0 + 1;
                        kk2 = kk0 + 2;
                        kk3 = kk0 + 3;

                        for (int jj0 = j; jj0 < (j + Block); jj0++)
                        {
                            c[ii0 * N + jj0] += a[ii0 * N + kk0] * b[kk0 * N + jj0];
                            c[ii0 * N + jj0] += a[ii0 * N + kk1] * b[kk1 * N + jj0];
                            c[ii0 * N + jj0] += a[ii0 * N + kk2] * b[kk2 * N + jj0];
                            c[ii0 * N + jj0] += a[ii0 * N + kk3] * b[kk3 * N + jj0];

                            c[ii1 * N + jj0] += a[ii1 * N + kk0] * b[kk0 * N + jj0];
                            c[ii1 * N + jj0] += a[ii1 * N + kk1] * b[kk1 * N + jj0];
                            c[ii1 * N + jj0] += a[ii1 * N + kk2] * b[kk2 * N + jj0];
                            c[ii1 * N + jj0] += a[ii1 * N + kk3] * b[kk3 * N + jj0];

                            c[ii2 * N + jj0] += a[ii2 * N + kk0] * b[kk0 * N + jj0];
                            c[ii2 * N + jj0] += a[ii2 * N + kk1] * b[kk1 * N + jj0];
                            c[ii2 * N + jj0] += a[ii2 * N + kk2] * b[kk2 * N + jj0];
                            c[ii2 * N + jj0] += a[ii2 * N + kk3] * b[kk3 * N + jj0];

                            c[ii3 * N + jj0] += a[ii3 * N + kk0] * b[kk0 * N + jj0];
                            c[ii3 * N + jj0] += a[ii3 * N + kk1] * b[kk1 * N + jj0];
                            c[ii3 * N + jj0] += a[ii3 * N + kk2] * b[kk2 * N + jj0];
                            c[ii3 * N + jj0] += a[ii3 * N + kk3] * b[kk3 * N + jj0];
                        }
                    }
                }
            }
        }
    }
}


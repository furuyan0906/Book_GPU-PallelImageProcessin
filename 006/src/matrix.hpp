#ifndef  H__MATRIX__H
#define  H__MATRIX__H


#include  <memory>


// 最適化なし
void Host_Multiply_Default(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N);

// ループ順序交換法 (キャッシュメモリの空間局在性を利用)
void Host_Multiply_LoopOrderExchange(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N);

// (4段)ループアンローリング (キャッシュメモリの空間局在性の利用 + 分岐命令の削減)
void Host_Multiply_LoopUnrolling(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N);

// キャッシュブロッキング (キャッシュメモリの時間局在性の利用)
void Host_Multiply_CacheBlocking(const std::shared_ptr<float[]>& a, const std::shared_ptr<float[]>& b, std::shared_ptr<float[]>& c, const int N);


#endif  /* H__MATRIX__H */


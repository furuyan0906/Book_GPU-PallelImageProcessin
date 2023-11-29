#ifndef  H__BASIC_DEF__H
#define  H__BASIC_DEF__H


namespace glcuda {

// 点の構造体
typedef struct
{
    double coord[3];  // 座標
    int index;        // 点に付随するユニークなインデックス
} point_t;

// 辺の構造体
typedef struct
{
    int start;
    int end;
    edge_t* next;
} edge_t;

}

#endif  // H__BASIC_DEF__H

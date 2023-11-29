namespace glcuda {

// 座標 インデックス
constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;

constexpr int A = 0;
constexpr int B = 1;
constexpr int C = 2;
constexpr int D = 3;

// 閾値
constexpr double Eps = 0.00001;
constexpr double Large = 10000.0;

// 円周率
constexpr double Pi = 3.141592653589793;


// 図形要素の最大個数
constexpr int max_num_points    = 2000000;
constexpr int max_num_edges     = 5000000;
constexpr int max_num_triangles = 2000000;


// 点
double point[max_num_points][3];
int num_points = 0;

// 辺
int edge[max_num_edges][2];
int num_edges = 0;

// 三角形ポリゴンの頂点と周囲の辺
int triangle[max_num_triangles][3];
int edge_triangle[max_num_triangles][3];
int num_triangles = 0;

}

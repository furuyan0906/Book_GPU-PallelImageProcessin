#include  <iostream>
#include  <string>
#include  <fstream>
#include  <cstdlib>
#include  <cstring>
#include  <cmath>
#include  <basicdef.hpp>


namespace glcuda {

static std::vector<point_t> point_vec(max_num_points);
static int num_point_vec = 0;

static std::vector<edge_t> connecting_edge;


template<typename T>
void swap(T& a, T& b)
{
    auto tmp = a;
    a = b;
    b = tmp;
}

int compare(const point_t& p0, const point_t& p1)
{
    // X座標値で比較
    {
        if (p1.coord[X] - p0.coord[X] > Eps) { return -1; }
        if (p0.coord[X] - p1.coord[X] > Eps) { return 1; }
    }

    // Y座標値で比較
    {
        if (p1.coord[Y] - p0.coord[Y] > Eps) { return -1; }
        if (p0.coord[Y] - p1.coord[Y] > Eps) { return 1; }
    }

    // Z座標値で比較
    {
        if (p1.coord[Z] - p0.coord[Z] > Eps) { return -1; }
        if (p0.coord[Z] - p1.coord[Z] > Eps) { return 1; }
    }

    // 2点を同一と見なす
    return 0;
}

void quicksort(int start, int end, const std::vector<point_t>& a)
{
    if (end <= start) { return ; }

    auto left = start;
    auto right = end;
    auto mid = left + (right - left) / 2;
    auto pivot = a[mid];  // ソート前の中央のインデックスの値をpivotに設定する

    while (true)
    {
        // leftより小さいインデックスの要素は全てpivotより小さく
        // rightより大きいインデックスの要素は全てpivotより大きく
        while (compare(a[left], pivot) == -1) { left++; }
        while (compare(a[right], pivot) == 1) { right--; }

        if (right <= left) { break; }

        swap<point_t>(a[left], a[right]);
    }

    // 期待としては left = right = pivotのインデックス
    quicksort(start, left - 1, a);
    quicksort(right + 1, end, a);
}

bool readASCIISTLFile(const std::string&& STL_file)
{
    auto ifs = std::ifstream(STL_file, std::ios::in);
    {
        if (ifs.fail()) { return false; }
    }

    std::cout << "Trying text STL file ... ";

    auto nPoint = 0, nTriangle = 0;
    auto line = std::string();
    while (!ifs.eof())
    {
        std::getline(ifs, line)

        // "vertex"が見つかるまで読み飛ばす
        if (std::strstr(line.c_str, "vertex") == nullptr) { continue; }

        // 連続する3頂点を読み込んでポリゴンを登録する

        // 1点目
        {
            // 上で既に読み込み済み
            auto iss = istringstream(line);
            iss.ignore(std::numeric_limits<std::streamsize>::max(), " ");

            double x, y, z;
            {
                iss >> x > y >> z;
            }
            point_vec[num_point_vec].coord[X] = x;
            point_vec[num_point_vec].coord[Y] = y;
            point_vec[num_point_vec].coord[Z] = z;
            point_vec[num_point_vec].index = num_point_vec;

            num_point_vec++;
        }

        // 2点目
        {
            std::getline(ifs, line);
            auto iss = istringstream(line);
            iss.ignore(std::numeric_limits<std::streamsize>::max(), " ");

            double x, y, z;
            {
                iss >> x > y >> z;
            }
            point_vec[num_point_vec].coord[X] = x;
            point_vec[num_point_vec].coord[Y] = y;
            point_vec[num_point_vec].coord[Z] = z;
            point_vec[num_point_vec].index = num_point_vec;

            num_point_vec++;
        }

        // 3点目
        {
            std::getline(ifs, line);
            auto iss = istringstream(line);
            iss.ignore(std::numeric_limits<std::streamsize>::max(), " ");

            double x, y, z;
            {
                iss >> x > y >> z;
            }
            point_vec[num_point_vec].coord[X] = x;
            point_vec[num_point_vec].coord[Y] = y;
            point_vec[num_point_vec].coord[Z] = z;
            point_vec[num_point_vec].index = num_point_vec;

            num_point_vec++;
        }

        num_triangles++;
    }

    if (num_triangles > 0) std::cout << "Done" << std::endl;
    else std::cout << "Failed" << std::endl;

    return num_triangles > 0;
}

bool readBinarySTLFile(const std::string& STL_file)
{
    auto ifs = std::ifstream(STL_file, std::ios::in);
    {
        if (ifs.fail()) { return false; }
    }

    // ヘッダーは読み捨てる
    char header[85];  // 84文字 + '\0'
    ifs.read(header, 84);

    std::cout << "Trying binary STL file ... ";

    num_point_vec = 0;
    num_triangles = 0;

    char buf[51];  // 50文字 + '\0'
    while (!ifs.eof())
    {
        // 連続する3頂点を読み込んでポリゴンを登録する
        ifs.read(buf, 50);
        {
            if (std::ios::bad()) { return false; }
        }

        // 1点目
        {
            auto coord = static_cast<float*>(buf);

            point_vec[num_point_vec].coord[X] = coord[3];
            point_vec[num_point_vec].coord[Y] = coord[4];
            point_vec[num_point_vec].coord[Z] = coord[5];
            point_vec[num_point_vec].index = num_point_vec;

            num_point_vec++;
        }

        // 2点目
        {
            auto coord = static_cast<float*>(buf);

            point_vec[num_point_vec].coord[X] = coord[6];
            point_vec[num_point_vec].coord[Y] = coord[7];
            point_vec[num_point_vec].coord[Z] = coord[8];
            point_vec[num_point_vec].index = num_point_vec;

            num_point_vec++;
        }

        // 3点目
        {
            auto coord = static_cast<float*>(buf);

            point_vec[num_point_vec].coord[X] = coord[9];
            point_vec[num_point_vec].coord[Y] = coord[10];
            point_vec[num_point_vec].coord[Z] = coord[11];
            point_vec[num_point_vec].index = num_point_vec;

            num_point_vec++;
        }

        num_triangles++;
    }

    if (num_triangles) std::cout << "Done" << std::endl;
    else std::cout << "Failed" << std::endl;

    return num_triangles > 0;
}

bool loadSTLFile(const std::string& STL_file)
{
    if (readASCIISTLFile(STL_file))
    {
        std::cout << "Triangles: " << num_triangles << std::endl;
    }
    else if (readBinarySTLFile(STL_file))
    {
        std::cout << "Triangles: " << num_triangles << std::endl;
    }
    else
    {
        std::cout << "Cannot open " << STL_file << std::endl;
        return false;
    }

    quicksort(0, num_point_vec, point_vec);

    point_t ref_point;
    {
        ref_point.coord[X] = Large;
        ref_point.coord[Y] = Large;
        ref_point.coord[Z] = Large;
        ref_point.index = -1;
    }
    num_points = 0;

    // ソート済みの点列を先頭からスキャン
    int start, end;
    for (int i = 0; i <= num_point_vec; ++i)
    {
        // 最後の点か参照点とは異なる点が見つかった
        if ((i == num_point_vec) || (compare(ref_point, point_vec[i]) != 0))
        {
            // 同一座標の点が複数見つかっているはずなので
            // それらを同じ点として登録し直す
            // ただし, 最初だけ無視する
            end = i;
            if (i > 0)
            {
                point[num_points][X] = ref_point[X];
                point[num_points][Y] = ref_point[Y];
                point[num_points][Z] = ref_point[Z];
                for (int j = start; j < end; ++j)
                {
                    auto tri_index = (int)(point_vec[j].index / 3);
                    auto ver_index = point_vec[j].index % 3;
                    triangle[tri_index][ver_index] = num_points;
                }
                num_points++;
            }

            // 参照点を更新する
            if (end < num_point_vec)
            {
                ref_point[X] = point_vec[end].coord[X];
                ref_point[Y] = point_vec[end].coord[Y];
                ref_point[Z] = point_vec[end].coord[Z];
                start = end;
            }
        }
    }

    // 辺を登録する
    num_edges = 0;

    // 各点に接続する辺の記録場所の初期化
    connecting_edge = std::vector<edge_t>(num_points);

    // 各三角形の周囲の辺と登録済みの辺を比較
    for (int i = 0; i < num_triangles; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            auto ed = connecting_edge
        }
    }
}

}

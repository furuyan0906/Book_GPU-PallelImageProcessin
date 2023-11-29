#include  <iostream>
#include  <algorithm>
#include  <cstdio>
#include  <cstdlib>
#include  <cmath>
#include  <GL/freeglut.h>


static constexpr double PI = 3.141592653589793;

static constexpr int X = 0;
static constexpr int Y = 1;
static constexpr int Z = 2;

static constexpr int num_points = 8;
static double point[num_points][3] = {
    {1.0, 1.0, -1.0},
    {-1.0, 1.0, -1.0},
    {-1.0, -1.0, -1.0},
    {1.0, -1.0, -1.0},
    {1.0, 1.0, 1.0},
    {-1.0, 1.0, 1.0},
    {-1.0, -1.0, 1.0},
    {1.0, -1.0, 1.0},
};

static constexpr int num_quads = 6;
static constexpr int quad[num_quads][4] = {
    {3, 2, 1, 0},
    {0, 1, 5, 4},
    {1, 2, 6, 5},
    {2, 3, 7, 6},
    {3, 0, 4, 7},
    {4, 5, 6, 7},
};

static constexpr int num_triangles = 12;
static constexpr int triangle[num_triangles][3] = {
    {3, 2, 1},
    {1, 0, 3},
    {0, 1, 5},
    {5, 4, 0},
    {1, 2, 6},
    {6, 5, 1},
    {2, 3, 7},
    {7, 6, 2},
    {3, 0, 4},
    {4, 7, 3},
    {4, 5, 6},
    {6, 7, 4},
};

static double eye[3];
static double center[3] = {0.0, 0.0, 0.0};
static double up[3];

static double phi = 30.0;
static double theta = 30.0;

static int window_width;
static int window_height;

static int mouse_old_x;
static int mouse_old_y;
static bool motion_p;


double dot(const double vec0[], const double vec1[])
{
    auto ret = 0.0;
    for (int k = X; k <= Z; ++k)
    {
        ret += vec0[k] * vec1[k];
    }

    return ret;
}

void cross(const double vec0[], const double vec1[], double vec2[])
{
    vec2[X] = vec0[Y] * vec1[Z] - vec0[Z] * vec1[Y];
    vec2[Y] = vec0[Z] * vec1[X] - vec0[X] * vec1[Z];
    vec2[Z] = vec0[X] * vec1[Y] - vec0[Y] * vec1[X];
}

void normVec(double vec[])
{
    double norm = std::sqrt(dot(vec, vec));
    vec[X] /= norm;
    vec[Y] /= norm;
    vec[Z] /= norm;
}

void normal(const double p0[], const double p1[], const double p2[], double nrml[])
{
    double v0[3], v1[3];
    {
        for (int i = 0; i < 3; ++i)
        {
            v0[i] = p2[i] - p1[i];
            v1[i] = p0[i] - p1[i];
        }
    }

    cross(v0, v1, nrml);
    normVec(nrml);
}

void defineViewMatrix(double phi, double theta)
{
    // 視点の設定 (視点は単位球上に存在する)
    {
        auto c = cos(phi * PI / 180.0);
        auto s = sin(phi * PI / 180.0);

        eye[X] = c * cos(theta * PI / 180.0);
        eye[Y] = s * cos(theta * PI / 180.0);
        eye[Z] = sin(theta * PI / 180.0);

        double view_x_axis[3] { -s, c, 0.0 };

        cross(eye, view_x_axis, up);
    }

    // 視点を原点とする座標系の定義
    // OpenGLの仕様でZ軸方向は視点の向きとは逆方向が正なのに注意
    double z_axis[3];
    {
        for (int k = X; k <= Z; ++k) { z_axis[k] = eye[k] - center[k]; }
        normVec(z_axis);
    }

    double x_axis[3];
    {
        cross(up, z_axis, x_axis);
        normVec(x_axis);
    }

    double y_axis[3];
    {
        cross(z_axis, x_axis, y_axis);
    }

    // 必ず表示する範囲を算出する
    double left, right, bottom, top, farVal, nearVal;
    {
        left = bottom = farVal = 10000.0;
        right = top = nearVal = -10000.0;

        // 座標範囲の読み取り
        for (int i = 0; i < num_points; ++i)
        {
            double vec[3];
            {
                for (int k = X; k <= Z; ++k) { vec[k] = point[i][k] - eye[k]; }
            }

            // vecのx成分と比較 -> 視点座標系のX成分を取得
            left = std::min(left, dot(x_axis, vec));
            right = std::max(right, dot(x_axis, vec));

            // vecのy成分と比較 -> 視点座標系のY成分を取得
            bottom = std::min(bottom, dot(y_axis, vec));
            top = std::max(top, dot(y_axis, vec));

            // vecのz成分と比較 -> 視点座標系のZ成分を取得
            farVal = std::min(farVal, dot(z_axis, vec));
            nearVal = std::max(nearVal, dot(z_axis, vec));
        }
    }

    // アスペクト比補正後の表示範囲, 必ず表示する範囲にマージンを5%分加える
    {
        auto marginX = (right - left) * 0.05;
        left -= marginX;
        right += marginX;

        auto marginY = (top - bottom) * 0.05;
        bottom -= marginY;
        top += marginY;

        auto marginZ = (nearVal - farVal) * 0.05;
        farVal -= marginZ;
        nearVal += marginZ;
    }

    double dx, dy, d_aspect, w_aspect;
    {
        // 表示範囲のアスペクト比
        dx = right - left;
        dy = top - bottom;
        d_aspect = dy / dx;

        // ウィンドウのアスペクト比
        w_aspect = (double)window_height / (double)window_width;
    }

    if (w_aspect > d_aspect)
    {
        // ウィンドウが表示範囲よりも縦長なので, 表示範囲を縦に広げる
        auto d = (dy * (w_aspect / d_aspect - 1.0)) * 0.5;

        // 必ず表示する範囲をベースに広げる
        bottom -= d;
        top += d;
    }
    else
    {
        // ウィンドウが表示範囲よりも横長なので, 表示範囲を横に広げる
        auto d = (dx * (d_aspect / w_aspect - 1.0)) * 0.5;

        // 必ず表示する範囲をベースに広げる
        left -= d;
        right += d;
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, -nearVal, -farVal);  // 視点座標系からの視点になるように正投影の行列を適用する. OpenGLの仕様により視点の後方が負になるよう設定する
    glViewport(0, 0, window_width, window_height);  // ウィンドウ上での描画範囲をウィンドウサイズで指定

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye[X], eye[Y], eye[Z],
            center[X], center[Y], center[Z],
            up[X], up[Y], up[Z]
        );
}

void initGL(void)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);  // デプスバッファを有効化
    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glEnable(GL_LIGHT0);  // 光源GL_LIGHT0を点灯状態にする
}

void pass(void)
{
}

void display(void)
{
    // 光源の設定
    float light_pos[4];
    {
        light_pos[0] = (float)eye[X];
        light_pos[1] = (float)eye[Y];
        light_pos[2] = (float)eye[Z];
        light_pos[3] = 0.0f;
    }
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);

    // 照明の点灯
    glEnable(GL_LIGHTING);

    // 正射影の定義
    defineViewMatrix(phi, theta);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // 陰面消去
    glBegin(GL_TRIANGLES);

    for (int i = 0; i < num_triangles; ++i)
    {
        double nrml_vec[3];
        {
            normal(point[triangle[i][0]], point[triangle[i][1]], point[triangle[i][2]], nrml_vec);
        }

        // triangle[i][0], triangle[i][1], triangle[i][2]の3頂点の法線情報としてnrml_vecを登録
        glNormal3dv(nrml_vec);
        glVertex3dv(point[triangle[i][0]]);
        glVertex3dv(point[triangle[i][1]]);
        glVertex3dv(point[triangle[i][2]]);
    }

    glEnd();
    glFlush();
}

void resize(int width, int height)
{
    std::printf("Size: %dx%d\n", width, height);

    // ウィンドウサイズの取得
    window_width = width;
    window_height = height;
}

void mouse_button(int buton, int state, int x, int y)
{
    if ((state == GLUT_DOWN) && (buton == GLUT_LEFT_BUTTON)) { motion_p = true; }
    else if (state == GLUT_UP) { motion_p = false; }
    else { pass(); }

    mouse_old_x = x;
    mouse_old_y = y;
}

void mouse_motion(int x, int y)
{
    int dx, dy;
    {
        dx = x - mouse_old_x;
        dy = y - mouse_old_y;
    }

    if (motion_p)
    {
        phi -= dx * 0.2;
        theta += dy * 0.2;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    glutInitWindowPosition(128, 128);
    glutInitWindowSize(768, 768);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);  // 陰面消去
    glutCreateWindow(argv[0]);
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    initGL();
    glutMainLoop();

    return EXIT_SUCCESS;
}


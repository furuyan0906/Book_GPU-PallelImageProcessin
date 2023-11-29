#include  <iostream>
#include  <algorithm>
#include  <cstdio>
#include  <cstdlib>
#include  <cmath>
#include  <GL/freeglut.h>


// 円周率
static constexpr double PI = 3.141592653589793;

// ウィンドウの初期位置と初期サイズ
static constexpr int init_x_pos = 128;
static constexpr int init_y_pos = 128;
static constexpr int init_width = 512;
static constexpr int init_height = 512;

// 座標値参照の際のインデックス
static constexpr int X = 0;
static constexpr int Y = 1;
static constexpr int D = 2;

// 点
static constexpr int max_num_points = 10000;
static double point[max_num_points][D];
static int num_points;

// 表示モード
static constexpr int mode_display_init = 0;
static constexpr int mode_display_points = 1;
static constexpr int mode_dipslay_cones = 2;
static int display_mode = mode_display_init;

// ウィンドウサイズ
static int window_width, window_height;

// 画像の表示範囲
static double left, right, bottom, top;

// 保存可能な画像の最大サイズ
static constexpr int max_image_width = 2048;
static constexpr int max_image_height = 1024;

// 画像データ (RGBAフォーマット)
static GLubyte image[max_image_width * max_image_height][4];

// 検出した母点
static int detected_point_index = -1;


void initGL(void)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // 背景色は白
    glEnable(GL_DEPTH_TEST);  // デプスバッファ機能を有効化
    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glDisable(GL_LIGHTING);  // 照明は使用しない
}

void genPoints(int num)
{
    num = std::min(num, max_num_points);

    num_points = 0;
    while (true)
    {
        if (num <= num_points)
        {
            left = 0; right = window_width;
            bottom = (double)0; top = (double)window_height;

            break;
        }

        auto x = window_width * ((double)rand() / (double)RAND_MAX);
        auto y = window_height * ((double)rand() / (double)RAND_MAX);

        point[num_points][X] = x;
        point[num_points][Y] = y;
        num_points++;
    }
}

void IDToColor(uint32_t id)
{
    GLubyte r, g, b;
    {
        r = (id & 0x00FF0000) >> 16;
        g = (id & 0x0000FF00) >> 8;
        b = (id & 0x000000FF) >> 0;
    }
    glColor3ub(r, g, b);
}

uint32_t ColorToID(GLubyte r, GLubyte g, GLubyte b)
{
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

void displayPoints(void)
{
    glPointSize(4.0f);

    glBegin(GL_POINTS);

    for (int i = 0; i < num_points; ++i)
    {
        if (i == detected_point_index)
        {
            glColor3d(1.0, 1.0, 0.0);
        }
        else
        {
            glColor3d(1.0, 0.0, 0.0);
        }
        glVertex3d(point[i][X], point[i][Y], 0.0);
    }

    glEnd();
}

void displayCone(double peak_point[])
{
    // ウィンドウ内に中心を持つ円がウィンドウを覆いつくために必要な半径
    double radius;
    {
        radius = std::sqrt((double)(window_width * window_width + window_height * window_height)) * 1.1;
    }

    // peak_pointを頂点とする円錐を描く
    glBegin(GL_TRIANGLE_FAN);

    glVertex3d(peak_point[X], peak_point[Y], 0.0);  // 円錐の頂点
    for (int i = 0; i <= 360; ++i)
    {
        int x = radius * cos((i % 360) / 180.0 * PI);
        int y = radius * sin((i % 360) / 180.0 * PI);
        glVertex3d(peak_point[X] + x, peak_point[Y] + y, -radius);  // 円錐の高さをradiusに設定
    }

    glEnd();
}

void displayCones(void)
{
    for (int i = 0; i < num_points; ++i)
    {
        IDToColor(i);
        displayCone(point[i]);
    }
}

void display()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, -1.0, 2000.0);  // z軸の範囲は[-2000.0, 1.0]
    glViewport(0, 0, window_width, window_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // 陰面消去

    switch (display_mode)
    {
        case mode_display_points:
            displayPoints();
            break;
        case mode_dipslay_cones:
            displayCones();
            glReadPixels(0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, image);
            displayPoints();
            break;
        default:
            break;
    }

    glFlush();
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'q':
        case 'Q':
            exit(0);
        // ランダムな点群の生成・描画
        case 'g':
        case 'G':
            genPoints(1000);
            display_mode = mode_display_points;
            glutPostRedisplay();
            break;
        // ボロノイ図の生成
        case 'v':
        case 'V':
            display_mode = mode_dipslay_cones;
            glutPostRedisplay();
            break;
        default:
            break;
    }
}

void resize(int width, int height)
{
    auto old_width = window_width;
    auto old_height = window_height;

    window_width = width;
    window_height = height;

    // ウィンドウサイズの変更分を実サイズに置き換え, 描画範囲を補正
    auto dx = (double)(window_width - old_width) * 0.5;
    auto dy = (double)(window_height - old_height) * 0.5;
    {
        right += dx;
        left -= dx;

        top += dy;
        bottom -= dy;
    }
}

void mouse_button(int button, int state, int x, int y)
{
    switch (button)
    {
        case GLUT_LEFT_BUTTON:
            {
                if (state == GLUT_DOWN)
                {
                    auto offset = window_width * (window_height - y) + x;
                    auto idx = ColorToID(image[offset][0], image[offset][1], image[offset][2]);

                    if (idx < num_points)
                    {
                        detected_point_index = idx;
                        std::cout << "Nearest point : " << idx << std::endl;
                    }
                    glutPostRedisplay();
                }
            }
            break;
        default:
            break;
    }
}


int main(int argc, char** argv)
{
    glutInitWindowPosition(init_x_pos, init_y_pos);
    glutInitWindowSize(init_width, init_height);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);  // 陰面消去
    glutCreateWindow("Voronoi Diagram");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(resize);
    glutMouseFunc(mouse_button);
    initGL();
    glutMainLoop();

	return EXIT_SUCCESS;
}


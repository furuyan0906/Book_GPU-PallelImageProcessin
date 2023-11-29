#include  <iostream>
#include  <algorithm>
#include  <cstdio>
#include  <cstdlib>
#include  <GL/freeglut.h>


static constexpr int X = 0;
static constexpr int Y = 1;
static constexpr int Z = 2;

static constexpr int num_points = 5;
static double point[num_points][3] = {
    {1.3, 1.3, 0.0},
    {0.3, 1.3, 0.0},
    {0.3, 0.3, 0.0},
    {1.3, 0.3, 0.0},
    {0.8, 0.8, 0.0},
};

static int window_width;
static int window_height;

// 必ず表示する範囲
static double init_left = -2.0;
static double init_right = 2.0;
static double init_bottom = -2.0;
static double init_top = 2.0;

// アスペクト比補正後の表示範囲
static double left;
static double right;
static double bottom;
static double top;

void initGL(void)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void pass(void)
{
}

void display(void)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, -100.0, 100.0);
    glViewport(0, 0, window_width, window_height);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3dv(point[0]);
    glVertex3dv(point[1]);
    glVertex3dv(point[4]);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3dv(point[1]);
    glVertex3dv(point[2]);
    glVertex3dv(point[4]);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3dv(point[2]);
    glVertex3dv(point[3]);
    glVertex3dv(point[4]);

    glEnd();
    glFlush();
}

void resize(int width, int height)
{
    std::printf("Size: %dx%d\n", width, height);

    // ウィンドウサイズの取得
    window_width = width;
    window_height = height;

    // 座標範囲の読み取り
    init_left = init_bottom = 10000.0;
    init_right = init_top = -10000.0;
    for (int i = 0; i < num_points; ++i)
    {
        init_left = std::min(init_left, point[i][X]);
        init_right = std::max(init_right, point[i][X]);
        init_bottom = std::min(init_bottom, point[i][Y]);
        init_top = std::max(init_top, point[i][Y]);
    }
    
    // マージンを5%分加える
    auto marginX = (init_right - init_left) * 0.05;
    init_left -= marginX;
    init_right += marginX;

    auto marginY = (init_top - init_bottom) * 0.05;
    init_bottom -= marginY;
    init_top += marginY;

    // 表示範囲のアスペクト比
    auto dx = init_right - init_left;
    auto dy = init_top - init_bottom;
    auto d_aspect = dy / dx;

    // ウィンドウのアスペクト比
    auto w_aspect = (double)height / (double)width;

    if (w_aspect > d_aspect)
    {
        // ウィンドウが表示範囲よりも縦長なので, 表示範囲を縦に広げる
        auto d = (dy * (w_aspect / d_aspect - 1.0)) * 0.5;

        // 必ず表示する範囲をベースに広げる
        left = init_left;
        right = init_right;
        bottom = init_bottom - d;
        top = init_top + d;
    }
    else
    {
        // ウィンドウが表示範囲よりも横長なので, 表示範囲を横に広げる
        auto d = (dx * (d_aspect / w_aspect - 1.0)) * 0.5;

        // 必ず表示する範囲をベースに広げる
        left = init_left - d;
        right = init_right + d;
        bottom = init_bottom;
        top = init_top;
    }
}

int main(int argc, char** argv)
{
    glutInitWindowPosition(128, 128);
    glutInitWindowSize(768, 768);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutCreateWindow(argv[0]);
    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    initGL();
    glutMainLoop();

    return EXIT_SUCCESS;
}


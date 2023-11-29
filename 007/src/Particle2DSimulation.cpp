#include  <iostream>
#include  <algorithm>
#include  <random>
#include  <stdexcept>
#include  <cstdlib>
#include  <cmath>
#include  <GL/glew.h>
#include  "Particle2DContainer.hpp"
#include  "GpuParticle2DContainer.hpp"
#include  "GpuKernelLauncher.hpp"
#include  "CpuKernel.hpp"
#include  "CpuKernelLauncher.hpp"
#include  "Particle2DSimulation.hpp"


static constexpr int init_x_pos_ = 128;
static constexpr int init_y_pos_ = 128;
static constexpr int init_width_ = 512;
static constexpr int init_height_ = 512;

static double init_left_ = -0.25;
static double init_right_ = 1.25;
static double init_bottom_ = -0.25;
static double init_top_ = 1.25;

static int window_width_;
static int window_height_;

static double left_;
static double right_;
static double bottom_;
static double top_;


// 粒子数
static constexpr size_t nMaxParticles_ = 1024 * 1024;

// CPUカーネル用の粒子の位置情報
static std::unique_ptr<Particle2DContainer<float>> hostParticles_ = Particle2DContainer<float>::CreateParticles(nMaxParticles_);

// GPUカーネル用の粒子の位置情報
static std::unique_ptr<GpuParticle2DContainer> deviceParticles_ = GpuParticle2DContainer::CreateParticles(nMaxParticles_);

// 頂点バッファオブジェクト
GLuint vbo_;

// 処理時間と時間刻み
static float animation_time_ = 0.0f;
static float animation_dt_ = 0.01f;


// GLUT関連の初期化
static void initGLUT(int argc, const char** argv);

// OpenGL関連の初期化
static bool initGL(void);

// 頂点バッファオブジェクトを初期化する
static void initVBO(GLuint& vbo, GLsizeiptr size, float* data);

// 頂点バッファオブジェクトを削除する
static void deleteVBO(GLuint& vbo);

// 粒子を初期位置に配置する
static void setInitialPosition(void);

// 解析を実行する (CPU)
static void runCpuKernel(void);

// 解析を実行する (GPU)
void runGpuKernel(void);

// 描画処理
static void display(void);

// リサイズ処理
static void resize(int width, int height);

// キーボード入力処理
static void keyboard(unsigned char key, int x, int y);

// 後処理
static void cleanup(void);


/** Public Methods **/

// 二次元粒子法シミュレーションの初期化処理
bool initParticle2DSimulation(int argc, const char** argv)
{
    initGLUT(argc, argv);
    if (!initGL())
    {
        return false;
    }

    setInitialPosition();

    auto size = hostParticles_->getMemorySize();
    auto data = hostParticles_->getRawParticles();
    initVBO(vbo_, size, data);

    return true;
}

void startParticle2DSimulation(void) noexcept
{
    glutMainLoop();
}


/** Private Methods **/

// GLUT関連の初期化
void initGLUT(int argc, const char** argv)
{
    left_ = init_left_;
    right_ = init_right_;
    bottom_ = init_bottom_;
    top_ = init_top_;

    glutInit(&argc, const_cast<char**>(argv));
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition(init_x_pos_, init_y_pos_);
    glutInitWindowSize(init_width_, init_height_);
    glutCreateWindow("2D Particle Simulation");

    glutDisplayFunc(display);
    glutReshapeFunc(resize);
    glutKeyboardFunc(keyboard);

    atexit(cleanup);
}

// OpenGL関連の初期化
bool initGL(void)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // glewの初期化
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0"))
    {
        throw std::runtime_error("Support for necessary OpenGL extensions missing.");
    }

    return true;
}

// 頂点バッファオブジェクトを初期化する
void initVBO(GLuint& vbo, GLsizeiptr size, float* data)
{
    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);  // 設定修了

    deviceParticles_->registerGraphicsResource(vbo);
}

// 頂点バッファオブジェクトを削除する
void deleteVBO(GLuint& vbo)
{
    deviceParticles_->unregisterGraphicsResource();
    glDeleteBuffers(1, &vbo);
    vbo = 0;
}

void updateVBO(GLuint& vbo, GLsizeiptr size, GLvoid* data)
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);  // 設定修了
}

// 粒子を初期位置に配置する
void setInitialPosition(void)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < nMaxParticles_; ++i)
    {
        auto x = dist(engine) * 0.5f + 0.25f; 
        auto y = dist(engine) * 0.5f; 
        hostParticles_->setPosition(i, x, y);
    }

    //deviceParticles_->copyToDevice(hostParticles_->getRawParticles(), hostParticles_->getMemorySize());
}

// 解析を実行する (CPU)
void runCpuKernel(void)
{
    launchCpuKernel(nMaxParticles_, hostParticles_, animation_time_, animation_dt_);
    animation_time_ += animation_dt_;
}

// 解析を実行する (GPU)
void runGpuKernel(void)
{
    deviceParticles_->mapGraphicsResource();

    launchGpuKernel(nMaxParticles_, deviceParticles_, animation_time_, animation_dt_);

    //float* hostMemoryPtr = hostParticles_->getRawParticles();
    //size_t nbytes = hostParticles_->getMemorySize();
    //deviceParticles_->copyToHost(hostMemoryPtr, nbytes);

    //updateVBO(vbo_, hostParticles_->getMemorySize(), hostMemoryPtr);

    deviceParticles_->unmapGraphicsResource();

    animation_time_ += animation_dt_;
}

// 描画処理
void display(void)
{
    // 射影設定
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left_, right_, bottom_, top_, -100.0f, 100.0f);
    glViewport(0, 0, window_width_, window_height_);

    // 粒子の位置を更新する
    //runCpuKernel();
    runGpuKernel();

    // 頂点バッファオブジェクトによる描画
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, nMaxParticles_);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // フレームバッファを切り替えて再描画
    glutSwapBuffers();
    glutPostRedisplay();
}

// リサイズ処理
void resize(int width, int height)
{
    // ウィンドウサイズの取得
    window_width_ = width;
    window_height_ = height;

    // 座標範囲の読み取り
    init_left_ = init_bottom_ = 10000.0;
    init_right_ = init_top_ = -10000.0;
    for (size_t i = 0; i < nMaxParticles_; ++i)
    {
        init_left_ = std::min(static_cast<float>(init_left_), hostParticles_->getX(i));
        init_right_ = std::max(static_cast<float>(init_right_), hostParticles_->getX(i));
        init_bottom_ = std::min(static_cast<float>(init_bottom_), hostParticles_->getY(i));
        init_top_ = std::max(static_cast<float>(init_top_), hostParticles_->getY(i));
    }
    
    // マージンを5%分加える
    auto marginX = (init_right_ - init_left_) * 0.05;
    init_left_ -= marginX;
    init_right_ += marginX;

    auto marginY = (init_top_ - init_bottom_) * 0.05;
    init_bottom_ -= marginY;
    init_top_ += marginY;

    // 表示範囲のアスペクト比
    auto dx = init_right_ - init_left_;
    auto dy = init_top_ - init_bottom_;
    auto d_aspect = dy / dx;

    // ウィンドウのアスペクト比
    auto w_aspect = (double)height / (double)width;

    if (w_aspect > d_aspect)
    {
        // ウィンドウが表示範囲よりも縦長なので, 表示範囲を縦に広げる
        auto d = (dy * (w_aspect / d_aspect - 1.0)) * 0.5;

        // 必ず表示する範囲をベースに広げる
        left_ = init_left_;
        right_ = init_right_;
        bottom_ = init_bottom_ - d;
        top_ = init_top_ + d;
    }
    else
    {
        // ウィンドウが表示範囲よりも横長なので, 表示範囲を横に広げる
        auto d = (dx * (d_aspect / w_aspect - 1.0)) * 0.5;

        // 必ず表示する範囲をベースに広げる
        left_ = init_left_ - d;
        right_ = init_right_ + d;
        bottom_ = init_bottom_;
        top_ = init_top_;
    }
}

// キーボード入力処理
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'q':
        case 'Q':
        case '\033':
            std::exit(EXIT_SUCCESS);
    }
}

// 後処理
void cleanup(void)
{
    deleteVBO(vbo_);
}


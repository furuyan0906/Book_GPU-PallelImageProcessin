#include <iostream>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "CpuDiffusion2D.hpp"


static constexpr std::uint64_t init_pos_x = 128;
static constexpr std::uint64_t init_pos_y = 128;
static constexpr std::uint64_t window_width = 512;
static constexpr std::uint64_t window_height = 512;
static constexpr std::uint64_t grid_width = 1024;
static constexpr std::uint64_t grid_height = 1024;
static constexpr float x_min = -0.5f;
static constexpr float x_max = 0.5f;
static constexpr float y_min = -0.5f;
static constexpr float y_max = 0.5f;
static constexpr float kappa = 0.1f;
static constexpr float max_density = 1.0f;
static constexpr float dx = (x_max - x_min) / grid_width;
static constexpr float dy = (y_max - y_min) / grid_height;
static const float dt = 0.2f * std::fmin(dx * dx, dy * dy) / kappa;

static void display(void);
static void keyboard(std::uint8_t key, int x, int y);
static void resize(int width, int height);
static void cleanup(void);
static void initKernel();
static std::uint32_t* runKernel(void);

GLuint textureId;

int main(int argc, char** argv)
{
    // 2D-Diffusion initialization
    initKernel();

    // GLUT initialization
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition(init_pos_x, init_pos_y);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("2D-Diffusion Simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(resize);
    atexit(cleanup);

    // GL initialization
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0"))
    {
        throw std::runtime_error("Support for necessary OpenGL extensions missing.");
    }
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, grid_width, grid_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glutMainLoop();

	return EXIT_SUCCESS;
}


void display(void)
{
    auto graphicalResource = runKernel();

    // Move to texture
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, grid_width, grid_height, GL_RGBA, GL_UNSIGNED_BYTE, graphicalResource);

    // GL setup
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glViewport(0, 0, window_width, window_height);

    // Display texture image
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
    glEnd();

    // Move to next buffer
    glutSwapBuffers();
    glutPostRedisplay();
}

void keyboard(std::uint8_t key, int x, int y)
{
    switch (key)
    {
        case 'q':
        case 'Q':
        case '\033':
            std::exit(EXIT_SUCCESS);
    }
}

void resize(int width, int height)
{
    // do nothing
}

void cleanup(void)
{
    glDeleteTextures(1, &textureId);
    textureId = 0;
}

void initKernel()
{
    cpu::Initialize(
            grid_width,
            grid_height,
            kappa,
            max_density,
            std::pair<float, float>(x_max, y_max),
            std::pair<float, float>(x_min, y_min));
}

std::uint32_t* runKernel(void)
{
    return cpu::Launch(dt, dx, dy);
}


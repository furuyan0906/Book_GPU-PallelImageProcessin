#include  <iostream>
#include  <cstdlib>

#include  <GL/freeglut.h>


void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);

    glVertex3d(0.5f, 0.5f, 0.0f);
    glVertex3d(-0.5f, 0.5f, 0.0f);

    glVertex3d(-0.5f, 0.5f, 0.0f);
    glVertex3d(-0.5f, -0.5f, 0.0f);

    glVertex3d(-0.5f, -0.5f, 0.0f);
    glVertex3d(0.5f, -0.5f, 0.0f);

    glVertex3d(0.5f, -0.5f, 0.0f);
    glVertex3d(0.5f, 0.5f, 0.0f);

    glEnd();
    glFlush();
}

void initGL(void)
{
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutCreateWindow(argv[0]);
    glutDisplayFunc(display);
    initGL();
    glutMainLoop();

	return EXIT_SUCCESS;
}


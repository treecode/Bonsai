#include <stdio.h>

#include <GL/glew.h>
#ifdef WIN32
#include <GL/wglew.h>
#else
#include <GL/glxew.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime_api.h>

#include "renderloop.h"
#include "render_particles.h"

class BonsaiDemo
{
public:
  BonsaiDemo(octree *tree, octree::IterationData &idata) 
    : m_tree(tree), m_idata(idata), iterationsRemaining(true) {
      tree->iterate_setup(m_idata);
  }

  ~BonsaiDemo() {
    m_tree->iterate_teardown(m_idata);
    delete m_tree;
  }
  
  void step() { 
    if (iterationsRemaining)
      iterationsRemaining = !m_tree->iterate_once(m_idata); 
  }
  
  void display() { 
    getBodyData(); 
    m_renderer.display();
  }

private:
  void getBodyData() {
    m_tree->localTree.bodies_pos.d2h();
    //m_tree->localTree.bodies_vel.d2h();
    //m_tree->localTree.bodies_ids.d2h();

    m_renderer.setPositions((float*)&m_tree->localTree.bodies_pos[0], m_tree->localTree.n_active_particles);
  }

  octree *m_tree;
  octree::IterationData &m_idata;
  bool iterationsRemaining;

  ParticleRenderer m_renderer;
};

BonsaiDemo *theDemo = NULL;

void onexit() {
  if (theDemo) delete theDemo;
}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  theDemo->step();
  theDemo->display();
  
  glutSwapBuffers();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mouse(int button, int state, int x, int y)
{
    /*int mods;

    if (state == GLUT_DOWN) {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP) {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();*/
}

void motion(int x, int y)
{
  /*
    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3) {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    }
    else if (buttonState & 2) {
        // middle = translate
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    }
    else if (buttonState & 1) {
        // left = rotate
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();*/
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
        /*case ' ':
            bPause = !bPause;
            break;
            */
        case 27: // escape
        case 'q':
        case 'Q':
            cudaDeviceReset();
            exit(0);
            break;
            /*
        case 13: // return
            if (bSupportDouble) {
                if (fp64) {
                    switchDemoPrecision<float, double>();
                }
                else {
                    switchDemoPrecision<double, float>();
                }

                printf("> %s precision floating point simulation\n", fp64 ? "Double" : "Single");
            }

            break;

        case '`':
            bShowSliders = !bShowSliders;
            break;

        case 'g':
        case 'G':
            bDispInteractions = !bDispInteractions;
            break;

        case 'p':
        case 'P':
            displayMode = (ParticleRenderer::DisplayMode)
                          ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;

        case 'c':
        case 'C':
            cycleDemo = !cycleDemo;
            printf("Cycle Demo Parameters: %s\n", cycleDemo ? "ON" : "OFF");
            break;

        case '[':
            activeDemo = (activeDemo == 0) ? numDemos - 1 : (activeDemo - 1) % numDemos;
            selectDemo(activeDemo);
            break;

        case ']':
            activeDemo = (activeDemo + 1) % numDemos;
            selectDemo(activeDemo);
            break;

        case 'd':
        case 'D':
            displayEnabled = !displayEnabled;
            break;

        case 'o':
        case 'O':
            activeParams.print();
            break;

        case '1':
            if (fp64) {
                NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_SHELL);
            }
            else {
                NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_SHELL);
            }

            break;

        case '2':
            if (fp64) {
                NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_RANDOM);
            }
            else {
                NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_RANDOM);
            }

            break;

        case '3':
            if (fp64) {
                NBodyDemo<double>::reset(numBodies, NBODY_CONFIG_EXPAND);
            }
            else {
                NBodyDemo<float>::reset(numBodies, NBODY_CONFIG_EXPAND);
            }

            break;*/
    }

    glutPostRedisplay();
}

void special(int key, int x, int y)
{
    //paramlist->Special(key, x, y);
    glutPostRedisplay();
}

void idle(void)
{
    glutPostRedisplay();
}

void initGL(int argc, char** argv)
{  
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(720, 480);
  glutCreateWindow("Bonsai Tree-code Gravitational N-body Simulation");
  //if (bFullscreen)
  //  glutFullScreen();
  GLenum err = glewInit();

  if (GLEW_OK != err)
  {
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    cudaDeviceReset();
    exit(-1);
  }
  else if (!glewIsSupported("GL_VERSION_2_0 "
    "GL_VERSION_1_5 "
    "GL_ARB_multitexture "
    "GL_ARB_vertex_buffer_object")) 
  {
    fprintf(stderr, "Required OpenGL extensions missing.");
    exit(-1);
  }
  else
  {
#if   defined(WIN32)
    wglSwapIntervalEXT(0);
#elif defined(LINUX)
    glxSwapIntervalSGI(0);
#endif      
  }

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(key);
  glutSpecialFunc(special);
  glutIdleFunc(idle);

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  checkGLErrors("initGL");

  atexit(onexit);
}


void initAppRenderer(int argc, char** argv, octree *tree, octree::IterationData &idata) {
  initGL(argc, argv);
  theDemo = new BonsaiDemo(tree, idata);
  glutMainLoop();
}
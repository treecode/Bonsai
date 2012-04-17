#include <stdio.h>

#ifdef WIN32
#define NOMINMAX
#endif

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
#include "vector_math.h"

class BonsaiDemo
{
public:
  BonsaiDemo(octree *tree, octree::IterationData &idata) 
    : m_tree(tree), m_idata(idata), iterationsRemaining(true),
      m_displayMode(ParticleRenderer::PARTICLE_POINTS),
      m_ox(0), m_oy(0), m_buttonState(0), m_inertia(0.1f),
      m_paused(false)
  {
    m_windowDims = make_int2(720, 480);
    m_cameraTrans = make_float3(0, -2, -100);
    m_cameraTransLag = m_cameraTrans;
    m_cameraRot   = make_float3(0, 0, 0);
    m_cameraRotLag = m_cameraRot;

    fitCamera();
    
    float color[4] = { 0.8f, 0.7f, 0.95f, 1.0f};
    m_renderer.setBaseColor(color);
    m_renderer.setPointSize(0.00001f);
    tree->iterate_setup(m_idata);
  }

  ~BonsaiDemo() {
    m_tree->iterate_teardown(m_idata);
    delete m_tree;
  }

  void cycleDisplayMode() {
    m_displayMode = (ParticleRenderer::DisplayMode)
      ((m_displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
    // MJH todo: add body color support and remove this
    if (ParticleRenderer::PARTICLE_SPRITES_COLOR == m_displayMode)
      cycleDisplayMode();
  }

  void togglePause() {
    m_paused = !m_paused;
  }

  void step() { 
    if (!m_paused && iterationsRemaining)
      iterationsRemaining = !m_tree->iterate_once(m_idata); 
  }
  
  void display() { 
    getBodyData();

    // view transform
    {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      m_cameraTransLag += (m_cameraTrans - m_cameraTransLag) * m_inertia;
      m_cameraRotLag += (m_cameraRot - m_cameraRotLag) * m_inertia;
      
      glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);
      glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
      glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
    }

    m_renderer.display(m_displayMode);
  }

  void mouse(int button, int state, int x, int y)
  {
    int mods;

    if (state == GLUT_DOWN) {
        m_buttonState |= 1<<button;
    }
    else if (state == GLUT_UP) {
        m_buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        m_buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL) {
        m_buttonState = 3;
    }

    m_ox = x;
    m_oy = y;
  }

  void motion(int x, int y)
  {
    float dx = (float)(x - m_ox);
    float dy = (float)(y - m_oy);

    if (m_buttonState == 3) {
      // left+middle = zoom
      m_cameraTrans.z += (dy / 100.0f) * 0.5f * fabs(m_cameraTrans.z);
    }
    else if (m_buttonState & 2) {
      // middle = translate
      m_cameraTrans.x += dx / 100.0f;
      m_cameraTrans.y -= dy / 100.0f;
    }
    else if (m_buttonState & 1) {
      // left = rotate
      m_cameraRot.x += dy / 5.0f;
      m_cameraRot.y += dx / 5.0f;
    }

    m_ox = x;
    m_oy = y;
  }

  void reshape(int w, int h) {
    m_windowDims = make_int2(w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) m_windowDims.x / (float) m_windowDims.y, 0.1, 10000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, m_windowDims.x, m_windowDims.y);
  }

  void fitCamera() {
    // MJH: todo: figure out why I need these large fudge constants
    // just computing the distance using radius / sin(fov/2) should be enough!
    float3 boxMin = make_float3(m_tree->rMinLocalTree);
    float3 boxMax = make_float3(m_tree->rMaxLocalTree);

    const float pi = 3.1415926f;
    float3 center = 0.5f * (boxMin + boxMax);
    float radius = std::max(length(boxMax), length(boxMin));
    const float fovRads = (m_windowDims.y / (float)m_windowDims.x) * pi / 3.0f ; // 60 degrees

    float distanceToCenter = radius / sinf(0.5f * fovRads);
    
    m_cameraTrans = center + 20.0f * make_float3(0, 0, - distanceToCenter);

    m_cameraRot = make_float3(0, 0, 0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) m_windowDims.x / (float) m_windowDims.y, 0.005 * radius, 50 * distanceToCenter);
  }

private:
  void getBodyData() {
    m_tree->localTree.bodies_pos.d2h();
    //m_tree->localTree.bodies_vel.d2h();
    //m_tree->localTree.bodies_ids.d2h();

    m_renderer.setPositions((float*)&m_tree->localTree.bodies_pos[0], m_tree->localTree.n);
  }

  octree *m_tree;
  octree::IterationData &m_idata;
  bool iterationsRemaining;

  ParticleRenderer m_renderer;
  ParticleRenderer::DisplayMode m_displayMode; 

  // view params
  int m_ox; // = 0
  int m_oy; // = 0;
  int m_buttonState;     
  int2 m_windowDims;
  float3 m_cameraTrans;   
  float3 m_cameraRot;     
  float3 m_cameraTransLag;
  float3 m_cameraRotLag;
  const float m_inertia;

  bool m_paused;
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
  theDemo->reshape(w, h);
  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
  theDemo->mouse(button, state, x, y);
  glutPostRedisplay();
}

void motion(int x, int y)
{
  theDemo->motion(x, y);
  glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
  switch (key) {
  case ' ':
    theDemo->togglePause();
    break;
  case 27: // escape
  case 'q':
  case 'Q':
    cudaDeviceReset();
    exit(0);
    break;
  /*case '`':
     bShowSliders = !bShowSliders;
     break;*/
  case 'p':
  case 'P':
    theDemo->cycleDisplayMode();
    break;
  case 'd':
  case 'D':
    //displayEnabled = !displayEnabled;
    break;
  case 'f':
  case 'F':
    theDemo->fitCamera();
    break;
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

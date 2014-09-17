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

//#include <cuda_runtime_api.h>

#include <algorithm>
#include "renderloop.h"
#include "renderer.h"
#include "vector_math.h"
#include <cstdarg>


#include <sys/time.h>
static inline double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}

unsigned long long fpsCount;
double timeBegin;

void beginDeviceCoords(void)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT), -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
}

void glutStrokePrint(float x, float y, const char *s, void *font)
{
  glPushMatrix();
  glTranslatef(x, y, 0.0f);
  int len = (int) strlen(s);
  for (int i = 0; i < len; i++) {
    glutStrokeCharacter(font, s[i]);
  }
  glPopMatrix();
}



void glPrintf(float x, float y, const char* format, ...)                                
{                                                                                       
  //void *font = GLUT_STROKE_ROMAN;                                                     
  void *font = GLUT_STROKE_MONO_ROMAN;                                                  
  char buffer[256];                                                                     
  va_list args;                                                                         
  va_start (args, format);                                                              
  vsnprintf (buffer, 255, format, args);                                                
  glutStrokePrint(x, y, buffer, font);                                                  
  va_end(args);                                                                         
} 

void drawWireBox(float3 boxMin, float3 boxMax) 
{
  glLineWidth(1.0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);  
  glColor3f(0.0, 1.0, 0.0);
  glBegin(GL_QUADS);
    // Front Face
    glNormal3f( 0.0, 0.0, 1.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(boxMin.x, boxMin.y, boxMax.z);
    glTexCoord2f(1.0, 0.0); glVertex3f(boxMax.x, boxMin.y, boxMax.z);
    glTexCoord2f(1.0, 1.0); glVertex3f(boxMax.x, boxMax.y, boxMax.z);
    glTexCoord2f(0.0, 1.0); glVertex3f(boxMin.x, boxMax.y, boxMax.z);
    // Back Face
    glNormal3f( 0.0, 0.0,-1.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(boxMax.x, boxMin.y, boxMin.z);
    glTexCoord2f(1.0, 0.0); glVertex3f(boxMin.x, boxMin.y, boxMin.z);
    glTexCoord2f(1.0, 1.0); glVertex3f(boxMin.x, boxMax.y, boxMin.z);
    glTexCoord2f(0.0, 1.0); glVertex3f(boxMax.x, boxMax.y, boxMin.z);
    // Top Face
    glNormal3f( 0.0, 1.0, 0.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(boxMin.x, boxMax.y, boxMax.z);
    glTexCoord2f(1.0, 0.0); glVertex3f(boxMax.x, boxMax.y, boxMax.z);
    glTexCoord2f(1.0, 1.0); glVertex3f(boxMax.x, boxMax.y, boxMin.z);
    glTexCoord2f(0.0, 1.0); glVertex3f(boxMin.x, boxMax.y, boxMin.z);
    // Bottom Face
    glNormal3f( 0.0,-1.0, 0.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(boxMax.x, boxMin.y, boxMin.z);
    glTexCoord2f(1.0, 0.0); glVertex3f(boxMin.x, boxMin.y, boxMin.z);
    glTexCoord2f(1.0, 1.0); glVertex3f(boxMin.x, boxMin.y, boxMax.z);
    glTexCoord2f(0.0, 1.0); glVertex3f(boxMax.x, boxMin.y, boxMax.z);
    // Right face
    glNormal3f( 1.0, 0.0, 0.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(boxMax.x, boxMin.y, boxMax.z);
    glTexCoord2f(1.0, 0.0); glVertex3f(boxMax.x, boxMin.y, boxMin.z);
    glTexCoord2f(1.0, 1.0); glVertex3f(boxMax.x, boxMax.y, boxMin.z);
    glTexCoord2f(0.0, 1.0); glVertex3f(boxMax.x, boxMax.y, boxMax.z);
    // Left Face
    glNormal3f(-1.0, 0.0, 0.0);
    glTexCoord2f(0.0, 0.0); glVertex3f(boxMin.x, boxMin.y, boxMin.z);
    glTexCoord2f(1.0, 0.0); glVertex3f(boxMin.x, boxMin.y, boxMax.z);
    glTexCoord2f(1.0, 1.0); glVertex3f(boxMin.x, boxMax.y, boxMax.z);
    glTexCoord2f(0.0, 1.0); glVertex3f(boxMin.x, boxMax.y, boxMin.z);
  glEnd();
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);  
}



class Demo
{
public:
  Demo(RendererData &idata) 
    : m_idata(idata), iterationsRemaining(true),
      m_displayMode(ParticleRenderer::PARTICLE_SPRITES_COLOR),
      m_octreeDisplayLevel(3),
      m_ox(0), 
      m_oy(0), 
      m_buttonState(0), 
      m_inertia(0.1f),
      m_paused(false), 
      m_displayBoxes(false),
      m_displayFps(true),
      m_displaySliders(true),
      m_params(m_renderer.getParams())
  {
    m_windowDims = make_int2(720, 480);
    m_cameraTrans = make_float3(0, -2, -10);
    m_cameraTransLag = m_cameraTrans;
    m_cameraRot = make_float3(0, 0, 0);
    m_cameraRotLag = m_cameraRot;
            
    //float color[4] = { 0.8f, 0.7f, 0.95f, 0.5f};
	float color[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    m_renderer.setBaseColor(color);
    m_renderer.setPointSize(0.00001f);
    //tree->iterate_setup(m_idata);
  }

  ~Demo() {
    //m_tree->iterate_teardown(m_idata);
    //delete m_tree;
  }

  void cycleDisplayMode() {
    m_displayMode = (ParticleRenderer::DisplayMode)
      ((m_displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
    // MJH todo: add body color support and remove this
    //if (ParticleRenderer::PARTICLE_SPRITES_COLOR == m_displayMode)
    //  cycleDisplayMode();
  }

  void togglePause() { m_paused = !m_paused; }
  void toggleBoxes() { m_displayBoxes = !m_displayBoxes; }
  void toggleSliders() { m_displaySliders = !m_displaySliders; }                        
  void toggleFps() { m_displayFps = !m_displayFps; }                        
  void incrementOctreeDisplayLevel(int inc) { 
    m_octreeDisplayLevel += inc;
    m_octreeDisplayLevel = std::max(0, std::min(m_octreeDisplayLevel, 30));
  }

  void step() { 
#if 0
    if (!m_paused && iterationsRemaining)
      iterationsRemaining = !m_tree->iterate_once(m_idata); 
    if (!iterationsRemaining)
      printf("No iterations Remaining!\n");
#endif
  }

  void drawStats()
  {
#if 0
    if (!m_enableStats)
      return;
#endif

    const float fps = fpsCount/(rtc() - timeBegin);
    if (fpsCount > 10*fps)
    {
      fpsCount = 0;
      timeBegin = rtc();
    }

//    int bodies = m_tree->localTree.n;
//    int dust = m_tree->localTree.n_dust;

    beginDeviceCoords();
    glScalef(0.25f, 0.25f, 1.0f);

    glEnable(GL_LINE_SMOOTH);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);

    float x = 100.0f;
    //float y = 50.0f;
    float y = glutGet(GLUT_WINDOW_HEIGHT)*4.0f - 200.0f;
	const float lineSpacing = 140.0f;

#if 0
    float Myr = m_tree->get_t_current() * 9.78f;
    glPrintf(x, y, "MYears:    %.2f", Myr);
    y -= lineSpacing;

    glPrintf(x, y, "BODIES:    %d", bodies + dust);
    y -= lineSpacing;

    if (m_displayBodiesSec) {
	  double frameTime = 1.0 / fps;
      glPrintf(x, y, "BODIES/SEC:%.0f", bodies / frameTime);
	  y -= lineSpacing;
    }
#endif

    if (m_displayFps)
    {
      glPrintf(x, y, "FPS:       %.2f", fps);
      y -= lineSpacing;
    }

    glDisable(GL_BLEND);
    endWinCoords();

    char str[256];
    sprintf(str, "N-Body Renderer: %0.1f fps",
            fps);

    glutSetWindowTitle(str);
  }

  void display() { 
    getBodyData();
    fpsCount++;

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

    if (m_displayBoxes) {
//      displayOctree();  
    }

    if (m_displaySliders)
    {
      m_params->Render(0, 0);    
    }

    drawStats();
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

    if (m_displaySliders)
    {
      if (m_params->Motion(x, y))                                                       
        return;                                                                         
    }       

    if (m_buttonState == 3) {
      // left+middle = zoom
      m_cameraTrans.z += (dy / 100.0f) * 0.5f * fabs(m_cameraTrans.z);
    }
    else if (m_buttonState & 2) {
      // middle = translate
      m_cameraTrans.x += dx / 10.0f;
      m_cameraTrans.y -= dy / 10.0f;
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
    fitCamera();
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, m_windowDims.x, m_windowDims.y);
  }

  void fitCamera() {
    float3 boxMin = make_float3(m_idata.rmin());
    float3 boxMax = make_float3(m_idata.rmax());

    const float pi = 3.1415926f;
    float3 center = 0.5f * (boxMin + boxMax);
    float radius = std::max(length(boxMax), length(boxMin));
    const float fovRads = (m_windowDims.x / (float)m_windowDims.y) * pi / 3.0f ; // 60 degrees

    float distanceToCenter = radius / sinf(0.5f * fovRads);
    
    m_cameraTrans = center + make_float3(0, 0, - distanceToCenter);
	m_cameraTransLag = m_cameraTrans;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, 
                   (float) m_windowDims.x / (float) m_windowDims.y, 
                   0.0001 * distanceToCenter, 
                   4 * (radius + distanceToCenter));
  }

private:
  void getBodyData() {
    //m_tree->localTree.bodies_pos.d2h();
    //m_tree->localTree.bodies_ids.d2h();
    //m_tree->localTree.bodies_vel.d2h();

    int n = m_idata.n();

    float4 darkMatterColor = make_float4(1.0f, 0.5f, 0.1f, 0.1f);
    darkMatterColor = make_float4(0.0f, 0.0f, 0.9f, 0.4f);
//    darkMatterColor = make_float4(0.0f, 0.2f, 0.4f, 0.0f);      // blue

    float4 starColor =       make_float4(0.1f, 0.0f, 1.0f, 0.2f);

    float4 *colors = new float4[n];
    float4 *pos    = new float4[n];

    const float velMin = m_idata.attributeMin(RendererData::VEL);
    const float velMax = m_idata.attributeMax(RendererData::VEL);
    for (int i = 0; i < n; i++) {
      int type = m_idata.type(i); //m_tree->localTree.bodies_ids[i];
      const float vel = m_idata.attribute(RendererData::VEL,i);
      const float f = (vel - velMin)/velMax;
      float4 color;
      switch(type)
      {
        case 0:
          color = darkMatterColor;
          break;
        case 1:
//          color = starColor;
          color = starColor;
//          color.x = 0;
          color.y = f;
//          color.z = 0;
          break;
        default:
          assert(0);
          color = make_float4(0, 0, 0, 0); // dust -- not used yet
      }
      pos[i] = make_float4(m_idata.posx(i), m_idata.posy(i), m_idata.posz(i),0);
      colors[i] = color;
    }

    m_renderer.setPositions((float*)pos, n);
    m_renderer.setColors((float*)colors, n);

    delete [] colors;
    delete [] pos;
  }

#if 0
  void displayOctree() {
    float3 boxMin = make_float3(m_idata.rmin());
    float3 boxMax = make_float3(m_idata.rmax());

    drawWireBox(boxMin, boxMax);
      
    /m_tree->localTree.boxCenterInfo.d2h();
    m_tree->localTree.boxSizeInfo.d2h();
    m_tree->localTree.node_level_list.d2h(); //Should not be needed is created on host
           
    int displayLevel = min(m_octreeDisplayLevel, m_tree->localTree.n_levels);
      
    for(uint i=0; i < m_tree->localTree.level_list[displayLevel].y; i++)
    {
      float3 boxMin, boxMax;
      boxMin.x = m_tree->localTree.boxCenterInfo[i].x-m_tree->localTree.boxSizeInfo[i].x;
      boxMin.y = m_tree->localTree.boxCenterInfo[i].y-m_tree->localTree.boxSizeInfo[i].y;
      boxMin.z = m_tree->localTree.boxCenterInfo[i].z-m_tree->localTree.boxSizeInfo[i].z;

      boxMax.x = m_tree->localTree.boxCenterInfo[i].x+m_tree->localTree.boxSizeInfo[i].x;
      boxMax.y = m_tree->localTree.boxCenterInfo[i].y+m_tree->localTree.boxSizeInfo[i].y;
      boxMax.z = m_tree->localTree.boxCenterInfo[i].z+m_tree->localTree.boxSizeInfo[i].z;
      drawWireBox(boxMin, boxMax);
    }
  }
#endif

  RendererData &m_idata;
  bool iterationsRemaining;

  ParticleRenderer m_renderer;
  ParticleRenderer::DisplayMode m_displayMode; 
  int m_octreeDisplayLevel;

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
  bool m_displayBoxes;
  bool m_displayFps;
  bool m_displaySliders;
  ParamListGL *m_params;
};

Demo *theDemo = NULL;

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
//    cudaDeviceReset();
    exit(0);
    break;
  /*case '`':
     bShowSliders = !bShowSliders;
     break;*/
  case 'p':
  case 'P':
    theDemo->cycleDisplayMode();
    break;
  case 'b':
  case 'B':
    theDemo->toggleBoxes();
    break;
  case 'd':
  case 'D':
    //displayEnabled = !displayEnabled;
    break;
  case 'f':
  case 'F':
    theDemo->fitCamera();
    break;
  case ',':
  case '<':
    theDemo->incrementOctreeDisplayLevel(-1);
    break;
  case '.':
  case '>':
    theDemo->incrementOctreeDisplayLevel(+1);
    break;
  case 'h':
  case 'H':
    theDemo->toggleSliders();
//    m_enableStats = !m_displaySliders;
    break;
  case '0':
    theDemo->toggleFps();
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
  //glutInitWindowSize(720, 480);
  glutInitWindowSize(1024, 768);
  glutCreateWindow("Bonsai Tree-code Gravitational N-body Simulation");
  //if (bFullscreen)
  //  glutFullScreen();
  GLenum err = glewInit();

  if (GLEW_OK != err)
  {
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
//    cudaDeviceReset();
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


void initAppRenderer(
    int argc, char** argv, 
    RendererData &idata)
{
  initGL(argc, argv);
  theDemo = new Demo(idata);
  fpsCount = 0;
  timeBegin = rtc();
  glutMainLoop();
}

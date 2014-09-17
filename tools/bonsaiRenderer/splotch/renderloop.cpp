#include <stdio.h>
#include "GLSLProgram.h"

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

#include <algorithm>
#include "renderloop.h"
#include "renderer.h"
#include "vector_math.h"
#include <cstdarg>

#include "colorMap"
#include "Splotch.h"

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

const char passThruVS[] = 
{    
" void main()                                                        \n "
" {                                                                  \n "
"   gl_Position = gl_Vertex;                                         \n "
"   gl_TexCoord[0] = gl_MultiTexCoord0;                              \n "
"   gl_FrontColor = gl_Color;                                        \n "
" }                                                                  \n "
};

const char texture2DPS[]  =
{
" uniform sampler2D tex;                                             \n "
" void main()                                                        \n "
" {                                                                  \n "
"   vec4 c = texture2D(tex, gl_TexCoord[0].xy);                      \n "
"   gl_FragColor = c;                                                \n "
" }                                                                  \n "
};

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




class Demo
{
  public:
    Demo(RendererData &idata) 
      : m_idata(idata), 
      m_ox(0), 
      m_oy(0), 
      m_buttonState(0), 
      m_inertia(0.1f),
      m_paused(false), 
      m_displayFps(true),
      m_displaySliders(true),
      m_texture(0),
      m_displayTexProg(0)
//      m_fov(40.0f)k
//      m_params(m_renderer.getParams())
  {
    m_windowDims = make_int2(720, 480);
    m_cameraTrans = make_float3(0, -2, -10);
    m_cameraTransLag = m_cameraTrans;
    m_cameraRot = make_float3(0, 0, 0);
    m_cameraRotLag = m_cameraRot;

    //float color[4] = { 0.8f, 0.7f, 0.95f, 0.5f};
//    float4 color = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
//    m_renderer.setBaseColor(color);
//    m_renderer.setSpriteSizeScale(1.0f);
  
    m_displayTexProg = new GLSLProgram(passThruVS, texture2DPS);

    m_renderer.setColorMap(reinterpret_cast<float3*>(colorMap),256,256, 1.0f/255.0f);

    m_renderer.setWidth (m_windowDims.x);
    m_renderer.setHeight(m_windowDims.y);
    m_spriteSize = 0.1f;
  }
    ~Demo() 
    {
      delete m_displayTexProg;
    }

    void cycleDisplayMode() {}

    void toggleSliders() { m_displaySliders = !m_displaySliders; }                        
    void toggleFps()     { m_displayFps = !m_displayFps; }                        
    void drawQuad(const float s = 1.0f, const float z = 0.0f)
    {
      glBegin(GL_QUADS);
      glTexCoord2f(0.0, 0.0); glVertex3f(-s, -s, z);
      glTexCoord2f(1.0, 0.0); glVertex3f(s, -s, z);
      glTexCoord2f(1.0, 1.0); glVertex3f(s, s, z);
      glTexCoord2f(0.0, 1.0); glVertex3f(-s, s, z);
      glEnd();
    }

    void displayTexture(const GLuint tex)
    {
      m_displayTexProg->enable();
      m_displayTexProg->bindTexture("tex", tex, GL_TEXTURE_2D, 0);
      drawQuad();
      m_displayTexProg->disable();
    }

    GLuint createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format, void *data)
    {
      GLuint texid;
      glGenTextures(1, &texid);
      glBindTexture(target, texid);

      glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, data);
      return texid;
    }

    void drawStats()
    {
      const float fps = fpsCount/(rtc() - timeBegin);
      if (fpsCount > 10*fps)
      {
        fpsCount = 0;
        timeBegin = rtc();
      }


      beginDeviceCoords();
      glScalef(0.25f, 0.25f, 1.0f);

      glEnable(GL_LINE_SMOOTH);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glEnable(GL_BLEND);
      glDisable(GL_DEPTH_TEST);
      glColor4f(0.0f, 1.0f, 0.0f, 1.0f);

      float x = 100.0f;
      float y = glutGet(GLUT_WINDOW_HEIGHT)*4.0f - 200.0f;
      const float lineSpacing = 140.0f;

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

    void display() 
    { 
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

#if 0
      if (m_displaySliders)
      {
        m_params->Render(0, 0);    
      }
#endif

#if 1
      glGetDoublev( GL_MODELVIEW_MATRIX, (GLdouble*)m_modelView);
      glGetDoublev(GL_PROJECTION_MATRIX, (GLdouble*)m_projection);
      m_renderer. setModelViewMatrix(m_modelView);
      m_renderer.setProjectionMatrix(m_projection);
      m_renderer.genImage();
      fprintf(stderr, " --- frame done --- \n");
      const int width  = m_renderer.getWidth();
      const int height = m_renderer.getHeight();
      const float4 *img = &m_renderer.getImage()[0];
      m_texture = createTexture(GL_TEXTURE_2D, width, height, GL_RGBA, GL_RGBA, (void*)img);
#else
      const int width = 128;
      const int height = 128;
      std::vector<float> data(4*width*height);
      {
        using ShortVec3       = MathArray<float,3>;
        std::vector<ShortVec3> tex(256*256);
        int idx = 0;
        for (int i = 0; i < 256; i++)
          for (int j = 0; j < 256; j++)
          {
            tex[idx][0] = colorMap[i][j][0]/255.0f;
            tex[idx][1] = colorMap[i][j][1]/255.0f;
            tex[idx][2] = colorMap[i][j][2]/255.0f;
            idx++;
          }
        Texture2D<ShortVec3> texMap(&tex[0],256,256);
        for (int j = 0; j < height; j++)
          for (int i = 0; i < width; i++)
          {
            auto f = texMap(1.0f*i/width, j*1.0f/height);
//            f = texMap(1.0f, j*1.0f/height);
            data[0 + 4*(i + width*j)] = f[0];
            data[1 + 4*(i + width*j)] = f[1];
            data[2 + 4*(i + width*j)] = f[2];
            data[3 + 4*(i + width*j)] = 1.0f;
          }
      }
      m_texture = createTexture(GL_TEXTURE_2D, width, height, GL_RGBA, GL_RGBA, &data[0]);
#endif
      
      displayTexture(m_texture);
      drawStats();
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

#if 0
      if (m_displaySliders)
      {
        if (m_params->Motion(x, y))                                                       
          return;                                                                         
      }       
#endif

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
      m_renderer.setWidth (w);
      m_renderer.setHeight(h);
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
    void getBodyData() 
    {
      int n = m_idata.n();

      const float velMax = m_idata.attributeMax(RendererData::VEL);
      const float velMin = m_idata.attributeMin(RendererData::VEL);
      const float rhoMax = m_idata.attributeMax(RendererData::RHO);
      const float rhoMin = m_idata.attributeMin(RendererData::RHO);
      const bool hasRHO = rhoMax > 0.0f;
      const float scaleVEL =          1.0/(velMax - velMin);
      const float scaleRHO = hasRHO ? 1.0/(rhoMax - rhoMin) : 0.0;

      m_renderer.resize(n);
#pragma omp parallel for
      for (int i = 0; i < n; i++)
      {
        auto vtx = m_renderer.vertex_at(i);
        vtx.pos = Splotch::pos3d_t(m_idata.posx(i), m_idata.posy(i), m_idata.posz(i), m_spriteSize);
//        vtx.pos.h = m_idata.attribute(RendererData::H,i)*2;
//        vtx.pos.h *= m_spriteScale;
        vtx.color = make_float4(1.0f);
        float vel = m_idata.attribute(RendererData::VEL,i);
        float rho = m_idata.attribute(RendererData::RHO,i);
        vel = (vel - velMin) * scaleVEL;
        vel = std::pow(vel, 1.0f);
        rho = hasRHO ? (rho - rhoMin) * scaleRHO : 0.5f;
        assert(vel >= 0.0f && vel <= 1.0f);
        assert(rho >= 0.0f && rho <= 1.0f);
        vtx.attr  = Splotch::attr_t(rho, vel, m_spriteIntensity, m_idata.type(i));
      }
    }


    RendererData &m_idata;

    Splotch m_renderer;

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

    float m_spriteSize;
    float m_spriteIntensity;

    unsigned int m_texture;

    GLSLProgram *m_displayTexProg;

    double m_modelView [4][4];
    double m_projection[4][4];
};

Demo *theDemo = NULL;

void onexit() {
  if (theDemo) delete theDemo;
}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
//      theDemo->togglePause();
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
//      theDemo->toggleBoxes();
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
//      theDemo->incrementOctreeDisplayLevel(-1);
      break;
    case '.':
    case '>':
//      theDemo->incrementOctreeDisplayLevel(+1);
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
  //glutInitWindowSize(720, 480);
  glutInitWindowSize(1024, 768);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
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

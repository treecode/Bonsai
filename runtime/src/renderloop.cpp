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
#include <cstdarg>
#include <vector>

#include "renderloop.h"
#include "render_particles.h"
#include "SmokeRenderer.h"
#include "vector_math.h"
#include "timer.h"
#include "paramgl.h"
#include "depthSort.h"

extern void displayTimers();    // For profiling counter display

// fps
bool displayFps = false;
double fps = 0.0;
int fpsCount = 0;
int fpsLimit = 5;
cudaEvent_t startEvent, stopEvent;

void drawWireBox(float3 boxMin, float3 boxMax) {
#if 0
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);  
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
#else
  glBegin(GL_LINE_LOOP);
    glVertex3f(boxMin.x, boxMin.y, boxMin.z);
    glVertex3f(boxMax.x, boxMin.y, boxMin.z);
    glVertex3f(boxMax.x, boxMax.y, boxMin.z);
    glVertex3f(boxMin.x, boxMax.y, boxMin.z);
  glEnd();

  glBegin(GL_LINE_LOOP);
    glVertex3f(boxMin.x, boxMin.y, boxMax.z);
    glVertex3f(boxMax.x, boxMin.y, boxMax.z);
    glVertex3f(boxMax.x, boxMax.y, boxMax.z);
    glVertex3f(boxMin.x, boxMax.y, boxMax.z);
  glEnd();

  glBegin(GL_LINES);
    glVertex3f(boxMin.x, boxMin.y, boxMin.z);
    glVertex3f(boxMin.x, boxMin.y, boxMax.z);

    glVertex3f(boxMax.x, boxMin.y, boxMin.z);
    glVertex3f(boxMax.x, boxMin.y, boxMax.z);

    glVertex3f(boxMax.x, boxMax.y, boxMin.z);
    glVertex3f(boxMax.x, boxMax.y, boxMax.z);

    glVertex3f(boxMin.x, boxMax.y, boxMin.z);
    glVertex3f(boxMin.x, boxMax.y, boxMax.z);
  glEnd();
#endif
}

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
  char buffer[256];
  va_list args;
  va_start (args, format); 
  vsnprintf (buffer, 255, format, args);
  glutStrokePrint(x, y, buffer, GLUT_STROKE_ROMAN);
  va_end(args);
}

#define MAX_PARTICLES 700000
class BonsaiDemo
{
public:
  BonsaiDemo(octree *tree, octree::IterationData &idata) 
    : m_tree(tree), m_idata(idata), iterationsRemaining(true),
//       m_renderer(tree->localTree.n + tree->localTree.n_dust),
      m_renderer(tree->localTree.n + tree->localTree.n_dust, MAX_PARTICLES),
      //m_displayMode(ParticleRenderer::PARTICLE_SPRITES_COLOR),
	    m_displayMode(SmokeRenderer::VOLUMETRIC),
      m_ox(0), m_oy(0), m_buttonState(0), m_inertia(0.2f),
      m_paused(false),
      m_renderingEnabled(true),
  	  m_displayBoxes(false), 
      m_displaySliders(false),
      m_enableGlow(true),
      m_displayLightBuffer(false),
      m_directGravitation(false),
      m_octreeMinDepth(0),
      m_octreeMaxDepth(3),
      m_flyMode(false),
	    m_fov(60.0f),
      m_supernova(false),
      m_overBright(1.0f),
      m_params(m_renderer.getParams()),
      m_brightFreq(100)
  {
    m_windowDims = make_int2(1024, 768);
    m_cameraTrans = make_float3(0, -2, -100);
    m_cameraTransLag = m_cameraTrans;
    m_cameraRot = make_float3(0, 0, 0);
    m_cameraRotLag = m_cameraRot;
            
    //float color[4] = { 0.8f, 0.7f, 0.95f, 0.5f};
	  float color[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
    //m_renderer.setBaseColor(color);
    //m_renderer.setPointSize(0.00001f);
    tree->iterate_setup(m_idata);

   
    int arraySize = tree->localTree.n;
    arraySize    += tree->localTree.n_dust;
 
    //m_particleColors  = new float4[arraySize];
    m_particleColors  = new float4[MAX_PARTICLES];  

    cudaMalloc( &m_particleColorsDev, MAX_PARTICLES * sizeof(float4));  

    initBodyColors();

	  m_renderer.setFOV(m_fov);
	  m_renderer.setWindowSize(m_windowDims.x, m_windowDims.y);
	  m_renderer.setDisplayMode(m_displayMode);

    for(int i=0; i<256; i++) m_keyDown[i] = false;

    readCameras("cameras.txt");
    initColors();

    cudaEventCreate(&startEvent, 0);
    cudaEventCreate(&stopEvent, 0);
    cudaEventRecord(startEvent, 0);

    StartTimer();
  }

  ~BonsaiDemo() {
    m_tree->iterate_teardown(m_idata);
    delete m_tree;
    delete [] m_particleColors;
  }

  void cycleDisplayMode() {
    //m_displayMode = (ParticleRenderer::DisplayMode) ((m_displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
	  m_displayMode = (SmokeRenderer::DisplayMode) ((m_displayMode + 1) % SmokeRenderer::NUM_MODES);
	  m_renderer.setDisplayMode(m_displayMode);
      if (m_displayMode == SmokeRenderer::SPRITES) {
        //m_renderer.setAlpha(0.1f);
      } else {
        //m_renderer.setAlpha(1.0f);
      }
    // MJH todo: add body color support and remove this
    //if (ParticleRenderer::PARTICLE_SPRITES_COLOR == m_displayMode)
    //  cycleDisplayMode();
  }

  void toggleRendering() { m_renderingEnabled = !m_renderingEnabled; }
  void togglePause() { m_paused = !m_paused; }
  void toggleBoxes() { m_displayBoxes = !m_displayBoxes; }
  void toggleSliders() { m_displaySliders = !m_displaySliders; }
  void toggleGlow() { m_enableGlow = !m_enableGlow; m_renderer.setEnableFilters(m_enableGlow); }
  void toggleLightBuffer() { m_displayLightBuffer = !m_displayLightBuffer; m_renderer.setDisplayLightBuffer(m_displayLightBuffer); }

  void incrementOctreeMaxDepth(int inc) { 
    m_octreeMaxDepth += inc;
    m_octreeMaxDepth = std::max(m_octreeMinDepth+1, std::min(m_octreeMaxDepth, 30));
  }

  void incrementOctreeMinDepth(int inc) { 
    m_octreeMinDepth += inc;
    m_octreeMinDepth = std::max(0, std::min(m_octreeMinDepth, m_octreeMaxDepth-1));
  }

  void step() { 
    double startTime = GetTimer();
    if (!m_paused && iterationsRemaining)
    {
      iterationsRemaining = !m_tree->iterate_once(m_idata); 
    }
    m_simTime = GetTimer() - startTime;

    if (!iterationsRemaining)
      printf("No iterations Remaining!\n");
  }

  void drawStats(double fps)
  {
    int bodies = m_tree->localTree.n;
    int dust = m_tree->localTree.n_dust;

    beginDeviceCoords();
    glScalef(0.25f, 0.25f, 1.0f);

    glEnable(GL_LINE_SMOOTH);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);

    float x = 50.0f;
    float y = 50.0f;
    if (displayFps)
    {
      glPrintf(x, y, "FPS:   %.2f", fps);
      y += 150.0f;
    }

    glPrintf(x, y, "BODIES: %d", bodies + dust);

    float Myr = m_tree->get_t_current() * 9.78f;

    y += 150.0f;
    glPrintf(x, y, "Myears: %.2f", Myr);

    glDisable(GL_BLEND);
    endWinCoords();

    char str[256];
    sprintf(str, "Bonsai N-Body Tree Code (%d bodies, %d dust): %0.1f fps",
            bodies, dust, fps);

    glutSetWindowTitle(str);
  }

  void display() {
    double startTime = GetTimer();
    double getBodyDataTime = startTime;

    if (m_renderingEnabled)
    {
      //Check if we need to update the number of particles
      if((m_tree->localTree.n + m_tree->localTree.n_dust) > m_renderer.getNumberOfParticles())
      {
        //Update the particle count in the renderer
        m_renderer.setNumberOfParticles(m_tree->localTree.n + m_tree->localTree.n_dust);
        fitCamera(); //Try to get the model back in view
      }
      
      getBodyData();
      getBodyDataTime = GetTimer();

      moveCamera();
#if 1
      m_cameraTransLag += (m_cameraTrans - m_cameraTransLag) * m_inertia;
      m_cameraRotLag += (m_cameraRot - m_cameraRotLag) * m_inertia;
#else
	  m_cameraTransLag = m_cameraTrans;
	  m_cameraRotLag = m_cameraRot;
#endif
      // view transform
      {
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        if (m_flyMode) {
          glRotatef(m_cameraRotLag.z, 0.0, 0.0, 1.0);
          glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
          glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
          glRotatef(90.0f, 1.0f, 0.0f, 0.0f); // rotate galaxies into XZ plane
          glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);
          m_cameraRot.z *= 0.95f;
        } else {
          // orbit viwer - rotate around centre, then translate
          glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);
          glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
          glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
          glRotatef(90.0f, 1.0f, 0.0f, 0.0f); // rotate galaxies into XZ plane
        }

        glGetFloatv(GL_MODELVIEW_MATRIX, m_modelView);

        if (m_supernova) {
          if (m_overBright > 1.0f) {
            m_overBright -= 1.0f;
          } else {
            m_overBright = 1.0f;
            m_supernova = false;
          }
          m_renderer.setOverbright(m_overBright);
        }

        //m_renderer.display(m_displayMode);
        m_renderer.render();

        if (m_displayBoxes) {
          glEnable(GL_DEPTH_TEST);
          displayOctree();  
        }

        if (m_displaySliders) {
          m_params->Render(0, 0);
        }
      }
    }

    drawStats(fps);
  }

  void mouse(int button, int state, int x, int y)
  {
    int mods;

	  if (m_displaySliders) {
		  if (m_params->Mouse(x, y, button, state))
			  return;
	  }

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
    const float translateSpeed = 0.1f;
    const float zoomSpeed = 0.005f;
    const float rotateSpeed = 0.2f;

    float dx = (float)(x - m_ox);
    float dy = (float)(y - m_oy);

    if (m_displaySliders) {
      if (m_params->Motion(x, y))
        return;
    }

    if (m_buttonState == 3) {
      // left+middle = zoom
      float3 v = make_float3(0.0f, 0.0f, dy*zoomSpeed*fmaxf(fabs(m_cameraTrans.z), 1.0f));
      if (m_flyMode) {
		v = ixform(v, m_modelView);
      }
      m_cameraTrans += v;
    }
    else if (m_buttonState & 2) {
      // middle = translate
      float3 v = make_float3(dx * translateSpeed, -dy*translateSpeed, 0.0f);
      if (m_flyMode) {
		v = ixform(v, m_modelView);
      }
      m_cameraTrans += v;
    }
    else if (m_buttonState & 1) {
      // left = rotate
      m_cameraRot.x += dy * rotateSpeed;
      m_cameraRot.y += dx * rotateSpeed;
      m_cameraRot.z += dx * rotateSpeed * 0.5f;  // roll effect
    }

    m_ox = x;
    m_oy = y;
  }

  void moveCamera()
  {
    if (!m_flyMode)
      return;

    //const float flySpeed = 0.25f;
    //float flySpeed = (m_keyModifiers & GLUT_ACTIVE_SHIFT) ? 1.0f : 0.25f;
    float flySpeed = (m_buttonState & 4) ? 1.0f : 0.25f;

	// Z
    if (m_keyDown['w']) { //  || (m_buttonState & 1)) {
	  // foward
	  m_cameraTrans.x += m_modelView[2] * flySpeed;
	  m_cameraTrans.y += m_modelView[6] * flySpeed;
	  m_cameraTrans.z += m_modelView[10] * flySpeed;
    }
    if (m_keyDown['s']) {
	  // back
	  m_cameraTrans.x -= m_modelView[2] * flySpeed;
	  m_cameraTrans.y -= m_modelView[6] * flySpeed;
	  m_cameraTrans.z -= m_modelView[10] * flySpeed;
    }
	// X
    if (m_keyDown['a']) {
      // left
	  m_cameraTrans.x += m_modelView[0] * flySpeed;
	  m_cameraTrans.y += m_modelView[4] * flySpeed;
	  m_cameraTrans.z += m_modelView[8] * flySpeed;
    }
    if (m_keyDown['d']) {
	  // right
	  m_cameraTrans.x -= m_modelView[0] * flySpeed;
	  m_cameraTrans.y -= m_modelView[4] * flySpeed;
	  m_cameraTrans.z -= m_modelView[8] * flySpeed;
    }
	// Y
    if (m_keyDown['e']) {
      // up
	  m_cameraTrans.x += m_modelView[1] * flySpeed;
	  m_cameraTrans.y += m_modelView[5] * flySpeed;
	  m_cameraTrans.z += m_modelView[9] * flySpeed;
	}
    if (m_keyDown['q']) {
      // down
	  m_cameraTrans.x -= m_modelView[1] * flySpeed;
	  m_cameraTrans.y -= m_modelView[5] * flySpeed;
	  m_cameraTrans.z -= m_modelView[9] * flySpeed;
    }
  }

  // transform vector by inverse of matrix (assuming orthonormal)
  float3 ixform(float3 &v, float *m)
  {
    float3 r;
    r.x = v.x*m[0] + v.y*m[1] + v.z*m[2];
    r.y = v.x*m[4] + v.y*m[5] + v.z*m[6];
    r.z = v.x*m[8] + v.y*m[9] + v.z*m[10];
    return r;
  }

  void key(unsigned char key)
  {
    m_keyModifiers = glutGetModifiers();

    switch (key) {
    case ' ':
      togglePause();
      break;
    case 27: // escape
      displayTimers();
      exit(0);
      break;
    case '`':
       toggleSliders();
       break;
    case 'p':
    case 'P':
      cycleDisplayMode();
      break;
    case 'b':
    case 'B':
      toggleBoxes();
      break;
    case 'r':
    case 'R':
      toggleRendering();
      break;
    case 'l':
    case 'L':
      toggleLightBuffer();
      break;
    case 'c':
    case 'C':
      fitCamera();
      break;
    case '-':
      incrementOctreeMaxDepth(-1);
      break;
    case '=':
    case '+':
      incrementOctreeMaxDepth(+1);
      break;
    case '[':
      incrementOctreeMinDepth(-1);
      break;
    case ']':
      incrementOctreeMinDepth(+1);
      break;
    case 'h':
  	  toggleSliders();
      break;
    case 'g':
      toggleGlow();
      break;
    case 'f':
      m_flyMode = !m_flyMode;
      if (m_flyMode) {
        m_cameraTrans = m_cameraTransLag = ixform(m_cameraTrans, m_modelView);
        m_cameraRotLag.z = m_cameraRot.z = 0.0f;
      } else {
        fitCamera();
      }
      break;
    case 'n':
      m_supernova = true;
      m_overBright = 20.0f;
      break;
    case '1':
      m_directGravitation = !m_directGravitation;
      m_tree->setUseDirectGravity(m_directGravitation);
    case '0':
      break;
    case '.':
      m_renderer.setNumSlices(m_renderer.getNumSlices()*2);
      m_renderer.setNumDisplayedSlices(m_renderer.getNumSlices());
      break;
    case ',':
      m_renderer.setNumSlices(m_renderer.getNumSlices()/2);
      m_renderer.setNumDisplayedSlices(m_renderer.getNumSlices());
      break;
    case 'D':
      //printf("%f %f %f %f %f %f\n", m_cameraTrans.x, m_cameraTrans.y, m_cameraTrans.z, m_cameraRot.x, m_cameraRot.y, m_cameraRot.z);
      writeCameras("cameras.txt");
      break;
    case 'j':
      if (m_params == m_colorParams) {
        m_params = m_renderer.getParams();
      } else {
        m_params = m_colorParams;
      }
      break;
    default:
      break;
    }

    m_keyDown[key] = true;
  }

  void keyUp(unsigned char key) {
    m_keyDown[key] = false;
    m_keyModifiers = 0;
  }

  void special(int key)
  {
    int modifiers = glutGetModifiers(); // freeglut complains about this, but it seems to work

    switch (key) {
    case GLUT_KEY_F1:
    case GLUT_KEY_F2:
    case GLUT_KEY_F3:
    case GLUT_KEY_F4:
    case GLUT_KEY_F5:
    case GLUT_KEY_F6:
    case GLUT_KEY_F7:
    case GLUT_KEY_F8:
    //case GLUT_KEY_F9:
    //case GLUT_KEY_F10:
      {
        int cam = key - GLUT_KEY_F1;
        if (modifiers & GLUT_ACTIVE_SHIFT) {
          // save camera
          printf("Saved camera %d\n", cam);
          m_camera[cam].translate = m_cameraTrans;
          m_camera[cam].rotate = m_cameraRot;
          m_camera[cam].fly = m_flyMode;
        } else {
          // restore camera
          printf("Restored camera %d\n", cam);
          m_cameraTrans = m_camera[cam].translate;
          m_cameraRot = m_camera[cam].rotate;
          m_flyMode = m_camera[cam].fly;
        }
      }
      break;
    default:
      // for cursor keys on sliders
      m_params->Special(key, 0, 0);
      break;
    }
  }

  void reshape(int w, int h) {
    m_windowDims = make_int2(w, h);

	m_renderer.setFOV(m_fov);
	m_renderer.setWindowSize(m_windowDims.x, m_windowDims.y);

    fitCamera();
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, m_windowDims.x, m_windowDims.y);
  }

  void fitCamera() {
    float3 boxMin = make_float3(m_tree->rMinLocalTree);
    float3 boxMax = make_float3(m_tree->rMaxLocalTree);

    const float pi = 3.1415926f;
    float3 center = 0.5f * (boxMin + boxMax);
    float radius = std::max(length(boxMax), length(boxMin));
    const float fovRads = (m_windowDims.x / (float)m_windowDims.y) * pi / 3.0f ; // 60 degrees

    float distanceToCenter = radius / sinf(0.5f * fovRads);
    
    m_cameraTrans = center + make_float3(0, 0, -distanceToCenter*0.2f);
	m_cameraTransLag = m_cameraTrans;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(m_fov, 
                   (float) m_windowDims.x / (float) m_windowDims.y, 
                   0.0001 * distanceToCenter, 
                   4 * (radius + distanceToCenter));
  }

  //float  frand() { return rand() / (float) RAND_MAX; }
  //float4 randColor(float scale) { return make_float4(frand()*scale, frand()*scale, frand()*scale, 0.0f); }

// integer hash function (credit: rgba/iq)
  int ihash(int n)
  {
      n=(n<<13)^n;
      return (n*(n*n*15731+789221)+1376312589) & 0x7fffffff;
  }

  // returns random float between 0 and 1
  float frand(int n)
  {
          return ihash(n) / 2147483647.0f;
  }


  void initBodyColors()
  {
    int n = m_tree->localTree.n + m_tree->localTree.n_dust;   
    for(int i=0; i<n; i++) {
      m_particleColors[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
  }

  void getBodyData() {

   int n = m_tree->localTree.n + m_tree->localTree.n_dust;   
   //Above is safe since it is 0 if we dont use dust

    #ifdef USE_DUST
     //We move the dust data into the position data (on the device :) )
     m_tree->localTree.bodies_pos.copy_devonly(m_tree->localTree.dust_pos,
                           m_tree->localTree.n_dust, m_tree->localTree.n); 
     m_tree->localTree.bodies_ids.copy_devonly(m_tree->localTree.dust_ids,
                           m_tree->localTree.n_dust, m_tree->localTree.n);
    #endif    

    float4 color2 = make_float4(starColor2.x*starColor2.w, starColor2.y*starColor2.w, starColor2.z*starColor2.w, 1.0f);
    float4 color3 = make_float4(starColor3.x*starColor3.w, starColor3.y*starColor3.w, starColor3.z*starColor3.w, 1.0f);
    float4 color4 = make_float4(starColor4.x*starColor4.w, starColor4.y*starColor4.w, starColor4.z*starColor4.w, 1.0f);

    int sunIdx = -1;
    int m31Idx = -1;

#if 0
    m_tree->localTree.bodies_ids.d2h();   
    
    float4 *colors = m_particleColors;

#if 1
    for (int i = 0; i < n; i++)
    {
      int id =  m_tree->localTree.bodies_ids[i];
			if (id == 30000000 - 1) sunIdx = i;
			if (id == 30000000 - 2) m31Idx = i;
      //printf("%d: id %d, mass: %f\n", i, id, m_tree->localTree.bodies_pos[i].w);
#if 1
      float r = frand(id);
#if 0 /* eg: original version */
      if (id >= 0 && id < 50000000)     //Disk
      {
        //colors[i] = make_float4(0, 1, 0, 1);
          colors[i] = ((id % 100) != 0) ? 
						//starColor * make_float4(r, r, r, 1.0f):
                        starColor :
						((id / 100) & 1) ? starColor2 : starColor3;
      } 
#else /* eg: adds glowing massive particles */
      if (id >= 0 && id < 40000000)     //Disk
      {
        colors[i] = ((id % m_brightFreq) != 0) ? 
        starColor :
        ((id / m_brightFreq) & 1) ? color2 : color3;
      } else if (id >= 40000000 && id < 50000000)     // Glowing stars in spiral arms
      {
        colors[i] = ((id%4) == 0) ? color4 : color3;
      }
#endif
#if 0 /* eg: original version */
      else if (id >= 50000000 && id < 100000000) //Dust
      {
        //colors[i] = starColor;
		colors[i] = dustColor * make_float4(r, r, r, 1.0f);
      } 
#else /* eg: adds glowing massless particles */
				else if (id >= 50000000 && id < 70000000) //Dust
				{
					colors[i] = dustColor * make_float4(r, r, r, 1.0f);
				} 
				else if (id >= 70000000 && id < 100000000) // Glow massless dust particles
				{
					colors[i] = color3;  /*  adds glow in purple */
				}
#endif
      else if (id >= 100000000 && id < 200000000) //Bulge
      {
		  //colors[i] = starColor;
        colors[i] = bulgeColor;
	  } 
      else //>= 200000000, Dark matter
      {
        colors[i] = darkMatterColor;
		  //colors[i] = darkMatterColor * make_float4(r, r, r, 1.0f);
      }            
      
#else
      // test sorting
      colors[i] = make_float4(frand(), frand(), frand(), 1.0f);
#endif
    }
#endif

		LOGF(stderr, "sunIdx= %d  m31Idx= %d \n", sunIdx, m31Idx);

	m_renderer.setColors((float*)colors);
#else
	 assignColors( m_particleColorsDev, (int*)m_tree->localTree.bodies_ids.d(), n, 
		 color2, color3, color4, starColor, bulgeColor, darkMatterColor, dustColor, m_brightFreq );
	 m_renderer.setColorsDevice( (float*)m_particleColorsDev );
#endif

    m_renderer.setNumParticles( m_tree->localTree.n + m_tree->localTree.n_dust);    
    m_renderer.setPositionsDevice((float*) m_tree->localTree.bodies_pos.d());   // use d2d copy
    //m_tree->localTree.bodies_pos.d2h(); m_renderer.setPositions((float *) &m_tree->localTree.bodies_pos[0]);

  }


  void displayOctree() {
    float3 boxMin = make_float3(m_tree->rMinLocalTree);
    float3 boxMax = make_float3(m_tree->rMaxLocalTree);

    glLineWidth(0.8f);
    //glColor3f(0.0, 1.0, 0.0);
    glEnable(GL_LINE_SMOOTH);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    //drawWireBox(boxMin, boxMax);
      
    m_tree->localTree.boxCenterInfo.d2h();
    m_tree->localTree.boxSizeInfo.d2h();
    m_tree->localTree.node_level_list.d2h(); //Should not be needed is created on host
           
    uint displayMax = m_tree->localTree.level_list[min(m_octreeMaxDepth, m_tree->localTree.n_levels)].y;
    uint displayMin = m_tree->localTree.level_list[max(0, m_octreeMinDepth)].y;

    float alpha = std::min(1.0f, 1.0f - (float)(displayMax - displayMin) / m_tree->localTree.n_nodes);
        
    glColor4f(0.0f, 0.5f, 0.0f, std::max(alpha, 0.2f));
    //glColor4f(0.0f, 0.5f, 0.0f, 1.0f / m_octreeMaxDepth);

      
    for(uint i=displayMin; i < displayMax; i++)
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
    
    
#if 0
    m_tree->specialParticles.d2h();
    
    //Draw a line from sun to M31. Sun is [0] M31 center is [1]
    glBegin(GL_LINES);
      glVertex3f(m_tree->specialParticles[0].x, m_tree->specialParticles[0].y, m_tree->specialParticles[0].z);
      glVertex3f(m_tree->specialParticles[1].x, m_tree->specialParticles[1].y, m_tree->specialParticles[1].z);
    glEnd();
#endif    
    

    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);    
  }

  void addColorParam(ParamListGL *list, std::string name, float4 &color, bool intensity=false)
  {
    list->AddParam(new Param<float>((name + " r").c_str(), color.x, 0.0f, 1.0f, 0.01f, &color.x));
    list->AddParam(new Param<float>((name + " g").c_str(), color.y, 0.0f, 1.0f, 0.01f, &color.y));
    list->AddParam(new Param<float>((name + " b").c_str(), color.z, 0.0f, 1.0f, 0.01f, &color.z));
    if (intensity) {
      list->AddParam(new Param<float>((name + " i").c_str(), color.w, 0.0f, 100.0f, 1.0f, &color.w));
    }
  }

  void initColors()
  {
    //starColor = make_float4(1.0f, 1.0f, 0.5f, 1.0f);  // yellowish
    starColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);  // white
    starColor2 = make_float4(1.0f, 0.2f, 0.5f, 100.0f); // bright redish (w is brightness)
    starColor3 = make_float4(0.1f, 0.1f, 1.0f, 100.0f); // bright bluish
    starColor4 = make_float4(0.0f, 1.0f, 0.0f, 100.0f);  // green

    bulgeColor = make_float4(1.0f, 1.0f, 0.5f, 2.0f);  // yellowish

    //dustColor = make_float4(0.0f, 0.0f, 0.1f, 0.0f);      // blue
    //dustColor =  make_float4(0.1f, 0.1f, 0.1f, 0.0f);    // grey
    dustColor = make_float4(0.05f, 0.02f, 0.0f, 0.0f);  // brownish
    //dustColor = make_float4(0.0f, 0.2f, 0.1f, 0.0f);  // green
    //dustColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // black

    darkMatterColor = make_float4(0.0f, 0.2f, 0.4f, 3.0f);      // blue

    m_colorParams = new ParamListGL("colors");
    addColorParam(m_colorParams, "star color", starColor);
    addColorParam(m_colorParams, "bulge color", bulgeColor);
    addColorParam(m_colorParams, "star color2", starColor2, true);
    addColorParam(m_colorParams, "star color3", starColor3, true);
    addColorParam(m_colorParams, "star color4", starColor4, true);
    addColorParam(m_colorParams, "dust color", dustColor);
    addColorParam(m_colorParams, "dark matter color", darkMatterColor);
    m_colorParams->AddParam(new Param<int>("bright star freq", m_brightFreq, 1, 1000, 1, &m_brightFreq));
  }

  octree *m_tree;
  octree::IterationData &m_idata;
  bool iterationsRemaining;

  //ParticleRenderer m_renderer;
  //ParticleRenderer::DisplayMode m_displayMode; 
  SmokeRenderer m_renderer;
  SmokeRenderer ::DisplayMode m_displayMode; 
  int m_octreeMinDepth;
  int m_octreeMaxDepth;

  float4 *m_particleColors;
  float4 *m_particleColorsDev;

  // view params
  int m_ox; // = 0
  int m_oy; // = 0;
  int m_buttonState;     
  int2 m_windowDims;
  float m_fov;
  float3 m_cameraTrans;   
  float3 m_cameraRot;     
  float3 m_cameraTransLag;
  float3 m_cameraRotLag;
  float m_modelView[16];
  const float m_inertia;

  bool m_paused;
  bool m_displayBoxes;
  bool m_displaySliders;
  bool m_enableGlow;
  bool m_displayLightBuffer;
  bool m_renderingEnabled;
  bool m_flyMode;
  bool m_directGravitation;

  bool m_keyDown[256];
  int m_keyModifiers;

  double m_simTime, m_renderTime;
  double m_fps;
  int m_fpsCount, m_fpsLimit;

  bool m_supernova;
  float m_overBright;

  float4 starColor;
  float4 starColor2;
  float4 starColor3;
  float4 starColor4;
  float4 bulgeColor;
  float4 dustColor;
  float4 darkMatterColor;
  int m_brightFreq;

  ParamListGL *m_colorParams;
  ParamListGL *m_params;    // current

  // saved cameras
  struct Camera {
    Camera() {
      translate = make_float3(0.f);
      rotate = make_float3(0.0f);
      fly = false;
    }

    void write(FILE *fp)
    {
      fprintf(fp, "%d %f %f %f %f %f %f\n", fly, translate.x, translate.y, translate.z, rotate.x, rotate.y, rotate.z);
    }

    bool read(FILE *fp)
    {
      return fscanf(fp, "%d %f %f %f %f %f %f", &fly, &translate.x, &translate.y, &translate.z, &rotate.x, &rotate.y, &rotate.z) == 7;
    }

    float3 translate;
    float3 rotate;
    bool fly;
  };

  static const int maxCameras = 8;
  Camera m_camera[maxCameras];

  void writeCameras(char *filename)
  {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
      fprintf(stderr, "Error writing camera file '%s'\n", filename);
      return;
    }
    for(int i=0; i<maxCameras; i++) {
      m_camera[i].write(fp);
    }
    fclose(fp);
    printf("Wrote camera file '%s'\n", filename);
  }

  bool readCameras(char *filename)
  {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
      fprintf(stderr, "Couldn't open camera file '%s'\n", filename);
      return false;
    }
    for(int i=0; i<maxCameras; i++) {
      if (!m_camera[i].read(fp))
        break;
    }
    fclose(fp);
    printf("Read camera file '%s'\n", filename);
    return true;
  }
};

BonsaiDemo *theDemo = NULL;

void onexit() {
  if (theDemo) delete theDemo;
  if (glutGameModeGet(GLUT_GAME_MODE_ACTIVE) != 0) {
    glutLeaveGameMode();
  }
  cudaDeviceReset();
}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  theDemo->step();
  theDemo->display();

  //glutReportErrors();
  glutSwapBuffers();

  fpsCount++;

  // this displays the frame rate updated every second (independent of frame rate)
  if (fpsCount >= fpsLimit)
  {
    static double gflops = 0;
    static double interactionsPerSecond = 0;

    float milliseconds = 1;

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    
    milliseconds /= (float)fpsCount;
    
    fps = 1.f / (milliseconds / 1000.f);
    theDemo->drawStats(fps);
    
    fpsCount = 0;
    fpsLimit = (fps > 1.f) ? (int)fps : 1;

    if (theDemo->m_paused)
    {
      fpsLimit = 0;
    }

    cudaEventRecord(startEvent, 0);
  }
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
  theDemo->key(key);

  switch (key) {
  case '0':
    displayFps = !displayFps;
    break;
  default:
    break;
  }
  glutPostRedisplay();
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
  theDemo->keyUp(key);
}

void special(int key, int x, int y)
{
    theDemo->special(key);
    glutPostRedisplay();
}

void idle(void)
{
    glutPostRedisplay();
}

void initGL(int argc, char** argv, const char *fullScreenMode)
{  
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);

  if (fullScreenMode[0]) {
      printf("fullScreenMode: %s\n", fullScreenMode);
      glutGameModeString(fullScreenMode);
      if (glutGameModeGet(GLUT_GAME_MODE_POSSIBLE)) {
          int win = glutEnterGameMode();
      } else {
          printf("mode is not available\n");
          exit(-1);
      }
  } else {
    glutInitWindowSize(1024, 768);
    glutCreateWindow("Bonsai Tree-code Gravitational N-body Simulation");
  }

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(key);
  glutKeyboardUpFunc(keyUp);
  glutSpecialFunc(special);
  glutIdleFunc(idle);

  glutIgnoreKeyRepeat(GL_TRUE);

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

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  checkGLErrors("initGL");

  atexit(onexit);
}


void initAppRenderer(int argc, char** argv, octree *tree, 
                     octree::IterationData &idata, bool showFPS) {
  displayFps = showFPS;
  //initGL(argc, argv);
  theDemo = new BonsaiDemo(tree, idata);
  glutMainLoop();
}

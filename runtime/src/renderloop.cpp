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
#include <cassert>

#include "renderloop.h"
#include "render_particles.h"
#include "SmokeRenderer.h"
#include "vector_math.h"
#include "timer.h"
#include "paramgl.h"
#include "depthSort.h"
#include "tr.h"
#include "WOGSocketManager.h"

float TstartGlow;
float dTstartGlow;

#define DEG2RAD(a) ((a)/57.295)
//for stereo
enum EYE
{
  LEFT_EYE = 0,
  RIGHT_EYE = 1
};

struct Rand48
{
	double drand()
	{
		update();
		return (stat&0xFFFFFFFFFFFF)*(1.0/281474976710656.0);
	}
	long lrand()
	{
		update();
		return (long)(stat>>17)&0x7FFFFFFF;
	}
	long mrand()
	{
		update();
		return(long)(stat>>16)&0xFFFFFFFF;
	}
	void srand(const long seed)
	{
		stat = (seed<<16)+0x330E;
	}

	private:
	long long stat;
	void update()
	{
		stat = stat*0x5DEECE66D + 0xB;
	}
};

namespace StarSamplerData
{
#if 0
	const int N = 7;
	const float4 Colours[N] = 
	{  /* colours for different spectral classes: Oh Be A Fine Girl Kiss Me */
		make_float4(189.0, 188.0, 239.0, 1.0),  /* O-star */
		make_float4(203.0, 214.0, 228.0, 1.0),  /* B-star */
		make_float4(210.0, 211.0, 206.0, 1.0),  /* A-star */
		make_float4(229.0, 219.0, 169.0, 1.0),  /* F-star */
		make_float4(215.0, 211.0, 125.0, 1.0),  /* G-star, Sun-like */
		make_float4(233.0, 187.0, 116.0, 1.0),  /* K-star */
		make_float4(171.0,  49.0,  57.0, 1.0)   /* M-star, red-dwarfs */
	};
	const double Masses[N+1] =
	{  /* masses for each of the spectra type */
		/* O     B    A    F    G    K     M */
		150.0, 18.0, 3.2, 1.7, 1.1, 0.78, 0.47, 0.1
	};
#else
	const int N = 16;
	const float4 Colours[N] = 
	{  /* colours for different spectral classes: Oh Be A Fine Girl Kiss Me */
		make_float4( 32.0f,  78.0f, 255.0f, 1.0f),  /* O0 */
		make_float4( 62.0f, 108.0f, 255.0f, 1.0f),  /* O5 */
		make_float4( 68.0f, 114.0f, 255.0f, 1.0f),  /* B0 */
		make_float4( 87.0f, 133.0f, 255.0f, 1.0f),  /* B5 */
		make_float4(124.0f, 165.0f, 255.0f, 1.0f),  /* A0 */
		make_float4(156.0f, 189.0f, 255.0f, 1.0f),  /* A5 */
		make_float4(177.0f, 204.0f, 255.0f, 1.0f),  /* F0 */
		make_float4(212.0f, 228.0f, 255.0f, 1.0f),  /* F5 */
		make_float4(237.0f, 244.0f, 255.0f, 1.0f),  /* G0 */
		make_float4(253.0f, 254.0f, 255.0f, 1.0f),  /* G2 -- the Sun */
		make_float4(255.0f, 246.0f, 233.0f, 1.0f),  /* G5 */
		make_float4(255.0f, 233.0f, 203.0f, 1.0f),  /* K0 */
		make_float4(255.0f, 203.0f, 145.0f, 1.0f),  /* K5 */
		make_float4(255.0f, 174.0f,  98.0f, 1.0f),  /* M0 */
		make_float4(255.0f, 138.0f,  56.0f, 1.0f),  /* M5 */
		make_float4(240.0f,   0.0f,   0.0f, 1.0f)   /* M8 */
	};
	double Masses[N+1] =
	{  /* masses for each of the spectra type */
		150.0, 40.0f, 18.0, 6.5, 3.2, 2.1, 1.7, 1.29, 1.1, 1.0, 0.93, 0.78, 0.69, 0.47, 0.21, 0.1, 0.05
	};
#endif
}

class StarSampler
{
	private:
		double slope;
		double slope1;
		double slope1inv;
		double Mu_lo;
		double C;
		Rand48 rnd;

	public:

		StarSampler(const double _slope = -2.35, const long seed = 12345) : slope(_slope)
	{
		rnd.srand(seed);
		slope1    = slope + 1.0f;
		assert(slope1 != 0.0f);
	  slope1inv	= 1.0f/slope1;

		const double Mhi = StarSamplerData::Masses[0];
		const double Mlo = StarSamplerData::Masses[StarSamplerData::N];
		Mu_lo = std::pow(Mlo, slope1);
		C = (std::pow(Mhi, slope1) - std::pow(Mlo, slope1));
	}

		double sampleMass() 
		{
			const double Mu = C*rnd.drand() + Mu_lo;
			assert(Mu > 0.0);
			const double M   = std::pow(Mu, slope1inv);
			const double Mhi = StarSamplerData::Masses[0];
			const double Mlo = StarSamplerData::Masses[StarSamplerData::N];
			assert(M >= Mlo);
			assert(M <= Mhi);
			return M;
		}

		float4 getColour(const float M)
		{
			const float Mhi = StarSamplerData::Masses[0];
			const float Mlo = StarSamplerData::Masses[StarSamplerData::N];
			assert(M >= Mlo);
			assert(M <= Mhi);
			int beg = 0;
			int end = StarSamplerData::N-1;
			int mid = (beg + end) >> 1;
			while (end - beg > 1)
			{
				if (StarSamplerData::Masses[mid] > M)
					beg = mid;
				else 
					end = mid;
				mid = (beg + end) >> 1;
			}
			assert(mid >= 0);
			assert(mid <  StarSamplerData::N-1);

			return StarSamplerData::Colours[mid];
		}
};

extern void displayTimers();    // For profiling counter display

/* thread safe drand48(), man drand48 */

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
  //void *font = GLUT_STROKE_ROMAN;
  void *font = GLUT_STROKE_MONO_ROMAN;
  char buffer[256];
  va_list args;
  va_start (args, format); 
  vsnprintf (buffer, 255, format, args);
  glutStrokePrint(x, y, buffer, font);
  va_end(args);
}

// reducing to improve perf
#define MAX_PARTICLES 700000

class BonsaiDemo
{
public:
  BonsaiDemo(octree *tree, octree::IterationData &idata, GalaxyStore const& galaxyStore,
    int wogPort, real wogCameraDistance, real wogDeletionRadiusFactor)
    : m_tree(tree), m_idata(idata), iterationsRemaining(true),
      //m_renderer(tree->localTree.n + tree->localTree.n_dust),
      m_renderer(tree->localTree.n + tree->localTree.n_dust, MAX_PARTICLES),
      //m_displayMode(ParticleRenderer::PARTICLE_SPRITES_COLOR),
	  m_displayMode(SmokeRenderer::VOLUMETRIC),
      m_ox(0), m_oy(0), m_buttonState(0), m_inertia(0.2f),
      m_paused(false),
      m_renderingEnabled(true),
  	  m_displayBoxes(false), 
      m_displaySliders(false),
      m_displayCursor(1),
      m_cursorSize(0.5),
      m_enableGlow(true),
      m_displayLightBuffer(false),
      m_directGravitation(false),
      m_octreeMinDepth(0),
      m_octreeMaxDepth(3),
      m_flyMode(false),
	  m_fov(60.0f),
	  m_nearZ(0.2),
	  m_screenZ(450.0),
#ifdef WAR_OF_GALAXIES
	  m_farZ(1000),
#else
	  m_farZ(2000),
#endif
	  m_IOD(4.0),
	  m_stereoEnabled(false), //SV TODO Must be false, never make it true
      m_supernova(false),
      m_overBright(1.0f),
      m_params(m_renderer.getParams()),
      m_brightFreq(100),
      m_displayBodiesSec(true),
      m_cameraRollHome(0.0f),
      m_cameraRoll(0.0f),
      m_enableStats(true),
      m_galaxyStore(galaxyStore),
      m_wogSocketManager(wogPort, 1024, 768, m_fov, m_farZ, wogCameraDistance, wogDeletionRadiusFactor)
  {
    m_windowDims = make_int2(1024, 768);
    m_cameraTrans = make_float3(0, -2, -100);
    m_cameraTransLag = m_cameraTrans;
    m_cameraRot = make_float3(0, 0, 0);
    m_cameraRotLag = m_cameraRot;
    //m_cursorPos = make_float3(-41.043961, 37.102409,-42.675949);//the ogl cursor position, hardcoding it based on the treemin & max
            
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

    initColors();

    readCameras("cameras.txt");
    readParams(m_renderer.getParams(), "params.txt");
    readParams(m_colorParams, "colorparams.txt");

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
  void toggleStereo() {
    m_stereoEnabled = !m_stereoEnabled;
  }
  void togglePause() { m_paused = !m_paused; }
  void toggleBoxes() { m_displayBoxes = !m_displayBoxes; }
  void toggleSliders() { m_displaySliders = !m_displaySliders; }
  void toggleGlow() { m_enableGlow = !m_enableGlow; m_renderer.setEnableFilters(m_enableGlow); }
  void toggleLightBuffer() { m_displayLightBuffer = !m_displayLightBuffer; m_renderer.setDisplayLightBuffer(m_displayLightBuffer); }

  void incrementOctreeMaxDepth(int inc) { 
    m_octreeMaxDepth += inc;
    m_octreeMaxDepth = std::max(m_octreeMinDepth, std::min(m_octreeMaxDepth, m_tree->localTree.n_levels));
  }

  void incrementOctreeMinDepth(int inc) { 
    m_octreeMinDepth += inc;
    m_octreeMinDepth = std::max(0, std::min(m_octreeMinDepth, m_octreeMaxDepth));
  }

  void step() { 
    double startTime = GetTimer();
    if (!m_paused && iterationsRemaining)
    {
      iterationsRemaining = !m_tree->iterate_once(m_idata); 
    }
    m_simTime = GetTimer() - startTime;

    if (!iterationsRemaining)
    {
      //printf("No iterations Remaining!\n");
    }
  }

  void drawStats(double fps)
  {
    if (!m_enableStats)
      return;

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

    float x = 100.0f;
    //float y = 50.0f;
    float y = glutGet(GLUT_WINDOW_HEIGHT)*4.0f - 200.0f;
	const float lineSpacing = 140.0f;

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

    if (displayFps)
    {
      glPrintf(x, y, "FPS:       %.2f", fps);
      y -= lineSpacing;
    }

    glDisable(GL_BLEND);
    endWinCoords();

    char str[256];
    sprintf(str, "Bonsai N-Body Tree Code (%d bodies, %d dust): %0.1f fps",
            bodies, dust, fps);

    glutSetWindowTitle(str);
  }

  //calculate position of software 3D cursor, not so useful right now but helps for picking in future
  void calculateCursorPos() {
    //need modelview in double, so convert what we have in float
    //idenity mat
    GLdouble  mviewd[16] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    //GLdouble  mviewd[16];
    //for (int i=0;i<16;i++)
    //  mviewd[i] = m_modelView[i];
    GLdouble projPos[3];

    if (m_stereoEnabled) { //STEREO get the position from both eyes and calculate midpoint
      GLdouble posLeft[3], posRight[3];
      //=== Get the left eye cursor position ===
      gluProject(m_cursorPos[0], m_cursorPos[1], m_cursorPos[2],mviewd,m_projectionLeft,m_viewport,&projPos[0],&projPos[1],&projPos[2]);
      //Unproject 3D Screen coordinates into wonderful world coordinates
      //viewport[3]-y = conversion from upper left (0,0) to lower left (0,0)
      gluUnProject(m_ox, m_viewport[3]-m_oy, projPos[2], mviewd, m_projectionLeft, m_viewport, &posLeft[0], &posLeft[1], &posLeft[2]);

      //=== Get the right eye cursor position ===
      gluProject(m_cursorPos[0], m_cursorPos[1], m_cursorPos[2],mviewd,m_projectionRight,m_viewport,&projPos[0],&projPos[1],&projPos[2]);
      //Unproject 3D Screen coordinates into wonderful world coordinates
      //viewport[3]-y = conversion from upper left (0,0) to lower left (0,0)
      gluUnProject(m_ox, m_viewport[3]-m_oy, projPos[2], mviewd, m_projectionRight, m_viewport, &posRight[0], &posRight[1], &posRight[2]);

      m_cursorPos[0] = 0.5*(posLeft[0] + posRight[0]);
      m_cursorPos[1] = 0.5*(posLeft[1] + posRight[1]);
      m_cursorPos[2] = 0.5*(posLeft[2] + posRight[2]);
    }
    else { //MONO
      GLdouble pos[3];
      //project to screen to get z
      gluProject(m_cursorPos[0], m_cursorPos[1], m_cursorPos[2],mviewd,m_projection,m_viewport,&projPos[0],&projPos[1],&projPos[2]);
      ////Unproject 3D Screen coordinates into wonderful world coordinates
      ////viewport[3]-y = conversion from upper left (0,0) to lower left (0,0)
      gluUnProject(m_ox, m_viewport[3]-m_oy, projPos[2], mviewd, m_projection, m_viewport, &pos[0], &pos[1], &pos[2]);
      m_cursorPos[0] = pos[0];
      m_cursorPos[1] = pos[1];
      m_cursorPos[2] = pos[2];
    }
  }
  //This is the main render routine that is called by display for each eye (in stereo case), assumes mview and proj are already setup
  void mainRender(EYE whichEye)
  {
    //m_renderer.display(m_displayMode);
    m_renderer.render();

    if (m_displayBoxes) {
      glEnable(GL_DEPTH_TEST);
      displayOctree();
    }
    if (m_displayCursor) {

      //JB Hack TODO, I disable cursor in non-stereo mode since doesnt seem to do anything
      //in my non-stereo setup
      if(m_stereoEnabled)
      {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glEnable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glPushMatrix();
        {
          glLoadIdentity();
          glTranslatef(m_cursorPos[0],m_cursorPos[1],m_cursorPos[2]);
          glutSolidSphere(m_cursorSize,40,40);
        }
        glPopMatrix();
        glDisable(GL_LIGHTING);
        glDisable(GL_LIGHT0);
        glDisable(GL_BLEND);
      }
    }

    if (m_displaySliders) {
      m_params->Render(0, 0);
    }
    drawStats(fps);

  } //end of mainRender


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

    //Stereo setup +  get the left and right projection matrices and store it sv
    float frustumShift = 0.0;
    float top, right;
    if (m_stereoEnabled) { //STEREO
      float aspect = (float)m_windowDims.x/m_windowDims.y;
      if (aspect > 1.0) {
        top = m_nearZ * float(tan(DEG2RAD(m_fov)/2.0));
        right = top * aspect;
      } else {
        right   = m_nearZ * float(tan(DEG2RAD(m_fov)/2.0));
        top = right / aspect;
      }

      frustumShift = (m_IOD/2)*m_nearZ/m_screenZ;
      //Get left projection matrix and store it
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glFrustum( -right+frustumShift, right+frustumShift, -top, top, m_nearZ, m_farZ);
      glTranslatef(m_IOD/2, 0.0, 0.0);        //translate to cancel parallax
        glGetDoublev(GL_PROJECTION_MATRIX, m_projectionLeft);

      //Get right projection matrix and store it
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glFrustum( -right-frustumShift, right-frustumShift, -top, top, m_nearZ, m_farZ);
      glTranslatef(-m_IOD/2, 0.0, 0.0);        //translate to cancel parallax
        glGetDoublev(GL_PROJECTION_MATRIX, m_projectionRight);

    }
    else { //MONO
      //Get the projection matrix and store it
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(m_fov,
        (float) m_windowDims.x / (float) m_windowDims.y,
        m_nearZ, m_farZ);
        glGetDoublev(GL_PROJECTION_MATRIX, m_projection);

    }


      // view transform
      //{
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        //lighting for cursor
        GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
        GLfloat mat_shininess[] = { 50.0 };
        GLfloat light_position[] = { 0.0, 0.0, 1.0, 0.0 };
        // glShadeModel (GL_SMOOTH);
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
        glLightfv(GL_LIGHT0, GL_POSITION, light_position);
        //end cursor lighting
        glGetIntegerv( GL_VIEWPORT, m_viewport);
        if (m_displayCursor) {
          calculateCursorPos();
        }

        if (m_flyMode) {
          glRotatef(m_cameraRotLag.z, 0.0, 0.0, 1.0);
          glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
          glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
          glRotatef(m_cameraRoll, 0.0, 0.0, 1.0);
          glRotatef(90.0f, 1.0f, 0.0f, 0.0f); // rotate galaxies into XZ plane
          glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);

          m_cameraRot.z *= 0.95f;
          //m_cameraRot.z = (m_cameraRollHome - m_cameraRot.z)*0.1f;

        } else {
          // orbit viwer - rotate around centre, then translate
          glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);
          glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
          glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
          glRotatef(m_cameraRoll, 0.0, 0.0, 1.0);
          //glRotatef(90.0f, 1.0f, 0.0f, 0.0f); // rotate galaxies into XZ plane
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

        //Start drawing
        if (m_stereoEnabled) { //STEREO
          //draw left
          glDrawBuffer(GL_BACK_LEFT);                                   //draw into back left buffer
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          glMatrixMode(GL_PROJECTION);
          glLoadMatrixd(m_projectionLeft);
          glMatrixMode(GL_MODELVIEW);
          glLoadMatrixf(m_modelView);
          mainRender(LEFT_EYE);

          //draw right
          glDrawBuffer(GL_BACK_RIGHT);                              //draw into back right buffer
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          glMatrixMode(GL_PROJECTION);
          glLoadMatrixd(m_projectionRight);
          glMatrixMode(GL_MODELVIEW);
          glLoadMatrixf(m_modelView);
          mainRender(RIGHT_EYE);
        } //end of draw back right
        else { //MONO
          //draw left
          glDrawBuffer(GL_BACK);                                   //draw into back left buffer
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          glMatrixMode(GL_PROJECTION);
          glLoadMatrixd(m_projection);
          glMatrixMode(GL_MODELVIEW);
          glLoadMatrixf(m_modelView);
          mainRender(LEFT_EYE);
        }
      }
      else //rendering disabled just draw stats
        drawStats(fps);



//        //m_renderer.display(m_displayMode);
//        m_renderer.render();
//
//        if (m_displayBoxes) {
//          glEnable(GL_DEPTH_TEST);
//          displayOctree();
//        }
//
//        if (m_displaySliders) {
//          m_params->Render(0, 0);
//        }
//      }
//    }
//
//    drawStats(fps);
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

  void passiveMotion(int x, int y) {
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
    float flySpeed = (m_buttonState & 4) ? 0.5f : 0.1f;

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
    case '`':
    case 'h':
  	  toggleSliders();
      m_enableStats = !m_displaySliders;
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
      break;
    case '0':
      printf("==== SIM TIME %f\n",m_simTime);
      break;
    case '3': //toggle stereo
      toggleStereo();
      break;
    case '4':
      m_screenZ -= 25;
      printf("SCREENZ %f\n",m_screenZ);
      break;
    case '5':
      m_screenZ += 25;
      printf("SCREENZ %f\n",m_screenZ);
      break;
    case '6':
      m_IOD += 1;
      printf("IOD %f\n",m_IOD);
      break;
    case '7':
      m_IOD -= 1;
      printf("IOD %f\n",m_IOD);
      break;
    case '9':
      m_displayBodiesSec = !m_displayBodiesSec;
      break;
    case '8':
      m_enableStats = !m_enableStats;
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
      writeParams(m_renderer.getParams(), "params.txt");
      writeParams(m_colorParams, "colorparams.txt");
      glClearColor(0.0f, 1.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT); glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glutSwapBuffers();
      break;
    case 'j':
      if (m_params == m_colorParams) {
        m_params = m_renderer.getParams();
      } else {
        m_params = m_colorParams;
      }
      break;
    case 'm':
      m_renderer.setCullDarkMatter(!m_renderer.getCullDarkMatter());
      break;
    case '>':
      m_cameraRoll += 2.0f;
      break;
    case '<':
      m_cameraRoll -= 2.0f;
      break;
    case 'W':
      break;
    default:
      break;
    }

    m_keyDown[key] = true;
  }

  void writeParams(ParamList *params, char *filename)
  {
    ofstream stream;
    stream.open(filename);
    if (stream.is_open()) {
      params->Write(stream);
      printf("Wrote parameters '%s'\n", filename);
    }
    stream.close();
  }

  void readParams(ParamList *params, char *filename)
  {
    ifstream stream;
    stream.open(filename);
    if (stream.is_open()) {
      params->Read(stream);
      stream.close();
      printf("Read parameters '%s'\n", filename);
    }
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

	m_wogSocketManager.reshape(w, h);

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
    
    //m_cameraTrans = center + make_float3(0, 0, -distanceToCenter*0.2f);
    m_cameraTrans = make_float3(0, 0, -500);
    printf("camera trans %f %f %f \n",m_cameraTrans.x, m_cameraTrans.y, m_cameraTrans.z);

#if 0
    /* JB This came with stereo, seems to break rotation */
    //ignore above and read what we have in the cameras.txt file - dirty SV TODO
    m_cameraTrans = m_camera[0].translate;
    m_cameraRot = m_camera[0].rotate;

    m_cameraTransLag = m_cameraTrans;
    m_nearZ = 0.0001 * distanceToCenter;
    m_screenZ = distanceToCenter*0.6; //around 450 or so
    m_IOD = m_screenZ/100.0; //around 4 or so
    m_farZ = 4 * (radius + distanceToCenter);
    //set the cursor position to the center of the scene
    //m_cursorPos[0] = center.x;
    //m_cursorPos[1] = center.y;
    //m_cursorPos[2] = center.z;

    m_cursorPos[0] = 0.0;
    m_cursorPos[1] = 0.0;
    m_cursorPos[2] = -m_screenZ;

//    printf("NearZ %f farZ %f center %f %f %f radius %f distancetocenter %f\n",m_nearZ,m_farZ,center.x,center.y, center.z,radius, distanceToCenter);
//    printf("Box min %f %f %f Box max %f %f %f \n",boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z);
//    printf("camera trans %f %f %f \n",m_cameraTrans.x, m_cameraTrans.y, m_cameraTrans.z);
//    printf("stereo params screenZ %f IOD %f %f \n",m_screenZ, m_IOD);


    
#else
    /* JB this was original  */
    m_cameraTransLag = m_cameraTrans;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    std::cout << "m_fov = " << m_fov << std::endl;
    gluPerspective(m_fov, 
                   (float) m_windowDims.x / (float) m_windowDims.y, 
                   0.0001 * distanceToCenter, 
                   4 * (radius + distanceToCenter));
#endif
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

#if 0  /*eg:  assign colours on the host*/

#if 0
		const float slope = -1.35; // mass function slope, Salpeter
#elif 0
		const float slope = -0.35; // top heavy mass function
#elif 0
		const float slope = -0.01; // nearly uniform MF
#elif 1
		const float slope = -0.1; // reversed MF
#elif 1
		const float slope = +1.35; // reversed MF, low mass depleted
#endif

		StarSampler sSampler (slope-1);
		StarSampler sSampler1(-0.1-1, 1234556);

		int sunIdx = -1;
		int m31Idx = -1;

		m_tree->localTree.bodies_ids.d2h();   
		float4 *colors = m_particleColors;

		for (int i = 0; i < n; i++)
		{
			int id =  m_tree->localTree.bodies_ids[i];
			if (id == 30000000 - 1) sunIdx = i;
			if (id == 30000000 - 2) m31Idx = i;
			//printf("%d: id %d, mass: %f\n", i, id, m_tree->localTree.bodies_pos[i].w);
#if 1
			float r = frand(id);
			if (id >= 0 && id < 40000000)     //Disk
			{
				colors[i] = ((id % m_brightFreq) != 0) ? 
					starColor :
					((id / m_brightFreq) & 1) ? color2 : color3;
				const float  Mstar = sSampler.sampleMass();
				const float4 Cstar = sSampler.getColour(Mstar);
				const float fdim = 4.0;
				colors[i] = ((id%1000) == 0) ?   /* one in 1000 stars glows a bit */
					make_float4(Cstar.x/fdim,  Cstar.y/fdim,  Cstar.z/fdim, Cstar.w) : 
#if 0
					colors[i];  /* use Simon's colours, make inner galaxy differently coloured */
#else
					make_float4(Cstar.x/100.0, Cstar.y/100.0, Cstar.z/100.0, Cstar.w);
#endif
			}
		 	else if (id >= 40000000 && id < 50000000)     // Glowing stars in spiral arms
			{
				colors[i] = ((id%4) == 0) ? color4 : color3;
				/* sample colours form the MF */
				const float  Mstar = sSampler1.sampleMass();
				const float4 Cstar = sSampler1.getColour(Mstar);
				colors[i] = Cstar;
			}
			else if (id >= 50000000 && id < 70000000) //Dust
			{
				colors[i] = dustColor * make_float4(r, r, r, 1.0f);
			} 
			else if (id >= 70000000 && id < 100000000) // Glow massless dust particles
			{
				colors[i] = color3;  /*  adds glow in purple */
				colors[i] = dustColor * make_float4(r, r, r, 1.0f);
			}
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

		LOGF(stderr, "sunIdx= %d  m31Idx= %d \n", sunIdx, m31Idx);

		m_renderer.setColors((float*)colors);
#else  /* eg: assign colours on the device */
		const float Tcurrent = m_tree->get_t_current() * 9.78f;
		assignColors( m_particleColorsDev, (int*)m_tree->localTree.bodies_ids.d(), n, 
				color2, color3, color4, starColor, bulgeColor, darkMatterColor, dustColor, m_brightFreq, 
				make_float4(
					Tcurrent, TstartGlow,
					std::max(0.0f, std::min(1.0f, (Tcurrent - TstartGlow)/dTstartGlow)),
					Tcurrent > TstartGlow ? 1.0f : 3.0f)
				);

		m_renderer.setColorsDevice( (float*)m_particleColorsDev );
#endif

    m_renderer.setNumParticles( m_tree->localTree.n + m_tree->localTree.n_dust);    
    m_renderer.setPositionsDevice((float*) m_tree->localTree.bodies_pos.d());   // use d2d copy
    //m_tree->localTree.bodies_pos.d2h(); m_renderer.setPositions((float *) &m_tree->localTree.bodies_pos[0]);
    
    m_renderer.depthSort((float4*)m_tree->localTree.bodies_pos.d());

  }


  void displayOctree() {
    float3 boxMin = make_float3(m_tree->rMinLocalTree);
    float3 boxMax = make_float3(m_tree->rMaxLocalTree);

    glLineWidth(0.8f);
    //glLineWidth(3.2f);
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
    uint displayMin = m_tree->localTree.level_list[max(0, m_octreeMinDepth)].x;

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
#if 0
    addColorParam(m_colorParams, "star color", starColor);
    addColorParam(m_colorParams, "bulge color", bulgeColor);
    addColorParam(m_colorParams, "star color2", starColor2, true);
    addColorParam(m_colorParams, "star color3", starColor3, true);
    addColorParam(m_colorParams, "star color4", starColor4, true);
    m_colorParams->AddParam(new Param<int>("bright star freq", m_brightFreq, 1, 1000, 1, &m_brightFreq));
#endif
    addColorParam(m_colorParams, "dust color", dustColor);
    addColorParam(m_colorParams, "dark matter color", darkMatterColor);
    m_colorParams->AddParam(new Param<float>("camera roll", m_cameraRoll, -180.0f, 180.0f, 1, &m_cameraRoll));

    m_colorParams->AddParam(new Param<float>("screen Z", m_screenZ, 100.0, 2000.0, 450.0, &m_screenZ)); //I know  the scene bounds
    m_colorParams->AddParam(new Param<float>("iod", m_IOD, 1.0f, 20.0f, 4.0, &m_IOD));
    m_colorParams->AddParam(new Param<float>("cursor size", m_cursorSize, 0.0, 5.0, 0.5, &m_cursorSize));

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
  float m_cameraRollHome;
  float m_cameraRoll;
  float m_modelView[16];

  //SV TODO combine left and mono later
  double m_projection[16]; //mono projection
  double m_projectionLeft[16]; //left projection
  double m_projectionRight[16]; //right projection
  GLint m_viewport[4];//viewport dimensions+pos for 3d cursor & picking

  const float m_inertia;
  double m_cursorPos[3]; //the sw cursor position
  int m_displayCursor; //to show cursor
  float m_cursorSize; //size of cursor
  // stereo params
  float m_IOD; //Interocular distance
  float m_nearZ;
  float m_farZ;
  float m_screenZ;
  bool m_stereoEnabled;

  bool m_paused;
  bool m_displayBoxes;
  bool m_displaySliders;
  bool m_enableGlow;
  bool m_displayLightBuffer;
  bool m_renderingEnabled;
  bool m_flyMode;
  bool m_directGravitation;
  bool m_displayBodiesSec;
  bool m_enableStats;

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

  GalaxyStore const& m_galaxyStore;
  WOGSocketManager m_wogSocketManager;

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

void storeImage()
{
  #define FILENAME "tileimg.ppm"


  //The trick is to make sure that there are only full sizes tiles to be 
  //rendered, the 512,512,192 gives full sized tiles. But requires many
  //many many renders
  // imageW should be dividable by (tileW-border*2)
  // imageH should be dividable by (tileH-border*2)

  //imageW = 2000, border = 256, 2000 / (x-512) = int
  //with x > 512  -> 1012 would work

  //int finalSizeW = 1920;
  //int finalSizeH = 1080;

  int finalSizeW = 4096;
  int finalSizeH = 3072;
  
  
  
  const int border = 256;

  //Tile size is going to start at window dimensions 
  //and then decreased till its multiple
  int tileW = theDemo->m_windowDims.x;
  int tileH = theDemo->m_windowDims.y;

  while((finalSizeW % (tileW - 2*border)) != 0)
  {
    tileW--;
  }

  while((finalSizeH % (tileH - 2*border)) != 0)
  {
    tileH--;
  }

  int TILE_WIDTH  = tileW;
  int TILE_HEIGHT = tileH;
  int TILE_BORDER = border;

  int IMAGE_WIDTH  = finalSizeW;
  int IMAGE_HEIGHT = finalSizeH;

  //We have to increase the sprite size to make the image look
  //the same as the display
  double curSize  = theDemo->m_renderer.getParticleRadius();
  float  increase = IMAGE_HEIGHT / (float)theDemo->m_windowDims.y;
  increase = std::max(increase, IMAGE_WIDTH / (float)theDemo->m_windowDims.x);
  theDemo->m_renderer.setParticleRadius(curSize*increase);


  TRcontext *tr;
  GLubyte *buffer;
  GLubyte *tile;
  
  tile   = (GLubyte *)malloc(TILE_WIDTH  * TILE_HEIGHT  * 3 * sizeof(GLubyte));
  buffer = (GLubyte *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(GLubyte));

  tr = trNew();

  trTileSize(tr, TILE_WIDTH, TILE_HEIGHT, TILE_BORDER);
  trTileBuffer(tr, GL_RGB, GL_UNSIGNED_BYTE, tile);
  trImageSize(tr, IMAGE_WIDTH, IMAGE_HEIGHT);
  trRowOrder(tr, TR_TOP_TO_BOTTOM);

  //Get the values that went into glPerspective
  float3 boxMin = make_float3(theDemo->m_tree->rMinLocalTree);
  float3 boxMax = make_float3(theDemo->m_tree->rMaxLocalTree);

  const float pi      = 3.1415926f;
  float3 center       = 0.5f * (boxMin + boxMax);
  float radius        = std::max(length(boxMax), length(boxMin));
  const float fovRads = (theDemo->m_windowDims.x / (float)theDemo->m_windowDims.y) * pi / 3.0f ; // 60 degrees

  float distanceToCenter = radius / sinf(0.5f * fovRads);  
  trPerspective(tr, theDemo->m_fov,  
                (float) theDemo->m_windowDims.x / (float) theDemo->m_windowDims.y, 
                0.0001 * distanceToCenter, 4 * (radius + distanceToCenter));
  
  /* Prepare ppm output file */
  FILE *f = fopen(FILENAME, "wb");
  if (!f) {
    printf("Couldn't open image file: %s\n", FILENAME);
    return;
  }
  fprintf(f,"P6\n");
  fprintf(f,"# ppm-file created by %s\n", "Bonsai");
  fprintf(f,"%i %i\n", IMAGE_WIDTH, IMAGE_HEIGHT);
  fprintf(f,"255\n");
  //fclose(f);
  //f = fopen(FILENAME, "ab");  /* now append binary data */

  int count = 0;
  int more  = 1;
  while (more) 
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    fprintf(stderr, "Count: %d || Current w: %d  Current h: %d \n", 
        count++, trGet(tr, TR_CURRENT_TILE_WIDTH), trGet(tr, TR_CURRENT_TILE_HEIGHT));

    trBeginTile(tr);
    int curColumn = trGet(tr, TR_CURRENT_COLUMN);

    //Important, set the correct 'windowSize' otherwise aspects
    //of the tiles are messed up
    theDemo->m_renderer.setWindowSize(TILE_WIDTH,TILE_HEIGHT);
    theDemo->m_renderer.render();

    if (theDemo->m_displayBoxes) {
        glEnable(GL_DEPTH_TEST);
        theDemo->displayOctree();  
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    more = trEndTile(tr);

    /* save tile into tile row buffer*/
    {
       int curTileWidth = trGet(tr, TR_CURRENT_TILE_WIDTH);
       int bytesPerImageRow = IMAGE_WIDTH*3*sizeof(GLubyte);
       int bytesPerTileRow = (TILE_WIDTH-2*TILE_BORDER) * 3*sizeof(GLubyte);
       int xOffset = curColumn * bytesPerTileRow;
       int bytesPerCurrentTileRow = (curTileWidth-2*TILE_BORDER)*3*sizeof(GLubyte);
       int i;
       int curTileHeight = trGet(tr, TR_CURRENT_TILE_HEIGHT);
       for (i=0;i<curTileHeight;i++) {
          memcpy(buffer + i*bytesPerImageRow + xOffset, /* Dest */
                 tile + i*bytesPerTileRow,              /* Src */
                 bytesPerCurrentTileRow);               /* Byte count*/
       }
    }
    
    if (curColumn == trGet(tr, TR_COLUMNS)-1) {
       /* write this buffered row of tiles to the file */
       int curTileHeight = trGet(tr, TR_CURRENT_TILE_HEIGHT);
       int bytesPerImageRow = IMAGE_WIDTH*3*sizeof(GLubyte);
       int i;
       GLubyte *rowPtr;
       /* The arithmetic is a bit tricky here because of borders and
        * the up/down flip.  Thanks to Marcel Lancelle for fixing it.
        */
       for (i=2*TILE_BORDER;i<curTileHeight;i++) {
          /* Remember, OpenGL images are bottom to top.  Have to reverse. */
          rowPtr = buffer + (curTileHeight-1-i) * bytesPerImageRow;
          fwrite(rowPtr, 1, IMAGE_WIDTH*3, f);
       }
      }
    }//end while

  //Restore our window settings :)
  theDemo->m_renderer.setWindowSize( theDemo->m_windowDims.x , theDemo->m_windowDims.y);
  theDemo->m_renderer.setParticleRadius(curSize);

  trDelete(tr);
  free(tile);
  free(buffer);
  fclose(f);


  fprintf(stderr, "Writing done! \n");
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
void passiveMotion(int x, int y)
{
  theDemo->passiveMotion(x, y);
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
  case 'u':
    storeImage();
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
  theDemo->m_wogSocketManager.execute(theDemo->m_tree, theDemo->m_galaxyStore);
  glutPostRedisplay();
}

void initGL(int argc, char** argv, const char *fullScreenMode, bool &stereo)
{  
  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
  glutInit(&argc, argv);

  if (stereo) {
    printf("===== Checking Stereo Pixel Format \n===========");
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_STEREO |GLUT_DOUBLE);
  }
  else
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

  //Make sure we got stereo if we asked for it, this must happen after glutCreateWindow
  if (stereo) {
    GLboolean bStereoEnabled = false;
    glGetBooleanv(GL_STEREO, &bStereoEnabled);
    if (bStereoEnabled)
      printf("======= yay! STEREO ENABLED ========\n");
    else //we asked for stereo but didnt get it, set the stereo to false
      stereo = false;
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

  //shalini
  glutSetCursor(GLUT_CURSOR_CROSSHAIR);
  //glutSetCursor(GLUT_CURSOR_NONE);
  //JB I turned cursor back on

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

void initAppRenderer(int argc, char** argv, octree *tree, octree::IterationData &idata,
		             bool showFPS, bool stereo, GalaxyStore const& galaxyStore, int wogPort,
		             real wogCameraDistance, real wogDeletionRadiusFactor)
{
  displayFps = showFPS;
  //initGL(argc, argv);
  theDemo = new BonsaiDemo(tree, idata, galaxyStore, wogPort, wogCameraDistance, wogDeletionRadiusFactor);
  if (stereo)
    theDemo->toggleStereo(); //SV assuming stereo is set to disable by default.
  glutMainLoop();
}

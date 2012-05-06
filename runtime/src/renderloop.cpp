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
#include "SmokeRenderer.h"
#include "vector_math.h"

#include "../add_dust/DustRing.h"
#define M_PI        3.14159265358979323846264338328

extern void displayTimers();    // For profiling counter display



extern double rot[3][3];

extern void rotmat(double i,double w);

extern void rotate(double rot[3][3],float *vin);

extern void euler(vector<real4> &bodyPositions,
           vector<real4> &bodyVelocities,
           double inc, double omega);

extern double centerGalaxy(vector<real4> &bodyPositions,
                    vector<real4> &bodyVelocities);

extern int setupMergerModel(vector<real4> &bodyPositions1,
                     vector<real4> &bodyVelocities1,
                     vector<int>   &bodyIDs1,
                     vector<real4> &bodyPositions2,
                     vector<real4> &bodyVelocities2,
                     vector<int>   &bodyIDs2,
                     double ds = -1,
                     double ms = -1,
                     double b  = -1,
                     double rsep = -1,
                     double inc1 = -1,
                     double omega1 = -1,
                     double inc2  = -1,
                     double omega2 = -1);


bool modelDoubled = false;
bool setupMergerComplete = false;

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

#define MAX_PARTICLES 5000000
class BonsaiDemo
{
public:
  BonsaiDemo(octree *tree, octree::IterationData &idata) 
    : m_tree(tree), m_idata(idata), iterationsRemaining(true),
    //Set max particles to 5Million, should be enough for demos
      m_renderer(tree->localTree.n + tree->localTree.n_dust, MAX_PARTICLES),
      //m_displayMode(ParticleRenderer::PARTICLE_SPRITES_COLOR),
	    m_displayMode(SmokeRenderer::SPRITES),
      m_ox(0), m_oy(0), m_buttonState(0), m_inertia(0.2f),
      m_paused(false),
      m_renderingEnabled(true),
  	  m_displayBoxes(false), 
      m_displaySliders(false),
      m_enableGlow(true),
      m_displayLightBuffer(false),
      m_octreeDisplayLevel(3),
      m_flyMode(false),
	  m_fov(60.0f)
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
 
   m_particleColors  = new float4[MAX_PARTICLES];
 
	  m_renderer.setFOV(m_fov);
	  m_renderer.setWindowSize(m_windowDims.x, m_windowDims.y);
	  m_renderer.setDisplayMode(m_displayMode);

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

  void incrementOctreeDisplayLevel(int inc) { 
    m_octreeDisplayLevel += inc;
    m_octreeDisplayLevel = std::max(0, std::min(m_octreeDisplayLevel, 30));
  }

  void step() { 

    if(!setupMergerComplete)
      return;

    if (!m_paused && iterationsRemaining)
    {
      iterationsRemaining = !m_tree->iterate_once(m_idata); 
    }
    if (!iterationsRemaining)
      printf("No iterations Remaining!\n");
  }

  vector<real4> original_bodyPositions;
  vector<real4> original_bodyVelocities;
  vector<int>   original_bodyIDs; 


  vector<real4> bodyPositions;
  vector<real4> bodyVelocities;
  vector<int>   bodyIDs; 

  vector<real4> bodyPositions2;
  vector<real4> bodyVelocities2;
  vector<int>   bodyIDs2; 

  void mergeDustWithNormalParticles()
  {
    //If we've put the dust into a seperate array
    //we are now going to merge it with the original galaxy
    //so that the orbit calculations apply to the dust as well
    #ifdef USE_DUST
      int n_dust = m_tree->localTree.n_dust;
      if(n_dust > 0)
      {
        int n_particles = m_tree->localTree.n;
        m_tree->localTree.bodies_ids.d2h();   
        m_tree->localTree.bodies_pos.d2h();  
        m_tree->localTree.bodies_vel.d2h();  
        m_tree->localTree.dust_ids.d2h();   
        m_tree->localTree.dust_pos.d2h();  
        m_tree->localTree.dust_vel.d2h();  

        original_bodyPositions.clear();
        original_bodyVelocities.clear();
        original_bodyIDs.clear();

        original_bodyPositions.insert(original_bodyPositions.begin(),  
            &m_tree->localTree.bodies_pos[0],  
            &m_tree->localTree.bodies_pos[0]+n_particles);
        original_bodyVelocities.insert(original_bodyVelocities.begin(),  
            &m_tree->localTree.bodies_vel[0], 
            &m_tree->localTree.bodies_vel[0]+n_particles);
        original_bodyIDs.insert(original_bodyIDs.begin(),  
            &m_tree->localTree.bodies_ids[0], 
            &m_tree->localTree.bodies_ids[0]+n_particles);


        m_tree->localTree.setN(n_dust + m_tree->localTree.n);
        m_tree->allocateParticleMemory(m_tree->localTree);
        
        original_bodyPositions.insert(original_bodyPositions.end(),
          &m_tree->localTree.dust_pos[0], 
          &m_tree->localTree.dust_pos[0]+n_dust);

        original_bodyVelocities.insert(original_bodyVelocities.end(),
          &m_tree->localTree.dust_vel[0], 
          &m_tree->localTree.dust_vel[0]+n_dust);

        original_bodyIDs.insert(original_bodyIDs.end(),
          &m_tree->localTree.dust_ids[0], 
          &m_tree->localTree.dust_ids[0]+n_dust);

        fprintf(stderr,"original_bodyPositions : %d \n", original_bodyPositions.size());
        memcpy(&m_tree->localTree.bodies_pos[0], &original_bodyPositions[0], sizeof(real4)*original_bodyPositions.size());
        memcpy(&m_tree->localTree.bodies_vel[0], &original_bodyVelocities[0], sizeof(real4)*original_bodyVelocities.size());
        memcpy(&m_tree->localTree.bodies_ids[0], &original_bodyIDs[0], sizeof(int)*original_bodyIDs.size());
          
     /*   memcpy(&m_tree->localTree.bodies_pos[(int)bodyPositions.size()], &bodyPositions2[0], sizeof(real4)*bodyPositions.size());
        memcpy(&m_tree->localTree.bodies_Ppos[(int)bodyPositions.size()], &bodyPositions2[0], sizeof(real4)*bodyPositions.size());
        memcpy(&m_tree->localTree.bodies_vel[(int)bodyPositions.size()], &bodyVelocities2[0], sizeof(real4)*bodyPositions.size());
        memcpy(&m_tree->localTree.bodies_Pvel[(int)bodyPositions.size()], &bodyVelocities2[0], sizeof(real4)*bodyPositions.size());
        memcpy(&m_tree->localTree.bodies_ids[(int)bodyPositions.size()], &bodyIDs2[0], sizeof(int)*bodyPositions.size());

*/

        m_tree->localTree.bodies_ids.h2d();   
        m_tree->localTree.bodies_pos.h2d();  
        m_tree->localTree.bodies_vel.h2d(); 

        m_tree->localTree.setNDust(0);


      }
    #endif  
  }


 //Put the current galaxy on an orbit (with a copy of itself)
  void doubleGalaxyModel()
  {
    //Make a full copy of the current galaxy and set it up
    //using some defaults
    int n_particles = m_tree->localTree.n;

    original_bodyPositions.clear();
    original_bodyVelocities.clear();
    original_bodyIDs.clear();


    mergeDustWithNormalParticles();


     m_tree->localTree.bodies_ids.d2h();   
    m_tree->localTree.bodies_pos.d2h();  
    m_tree->localTree.bodies_vel.d2h();  

    original_bodyPositions.insert(original_bodyPositions.begin(),  
      &m_tree->localTree.bodies_pos[0],  
      &m_tree->localTree.bodies_pos[0]+n_particles);
    original_bodyVelocities.insert(original_bodyVelocities.begin(),  
      &m_tree->localTree.bodies_vel[0], 
      &m_tree->localTree.bodies_vel[0]+n_particles);
    original_bodyIDs.insert(original_bodyIDs.begin(),  
      &m_tree->localTree.bodies_ids[0], 
      &m_tree->localTree.bodies_ids[0]+n_particles);

    float sizeRatio = 1.52f;
    float massRatio = 1;
    float impact    = 10;
    float seperation= 168;
    float angle1  = 0;
    float angle1b = 0;
    float angle2  = 180;
    float angle2b = 0;

    modelDoubled = true;
    fprintf(stderr, "size %d %d\n", (int)original_bodyPositions.size()*2, m_tree->localTree.n);
    m_tree->localTree.setN((int)original_bodyPositions.size()*2);
    m_tree->allocateParticleMemory(m_tree->localTree);

    updateMergerConfiguration(sizeRatio, massRatio, impact, seperation, 
                              angle1, angle1b, angle2, angle2b);

    m_renderer.setNumberOfParticles(m_tree->localTree.n + m_tree->localTree.n_dust);
  }

  void updateMergerConfiguration(float sizeRatio, float massRatio, float impact,
                                 float seperation, float angle1, float angle1b,
                                 float angle2, float angle2b)
  {

    bodyPositions.clear();
    bodyVelocities.clear();
    bodyIDs.clear();
    bodyPositions2.clear();
    bodyVelocities2.clear();
    bodyIDs2.clear();


    //Copy the original galaxy, overwriting the modified ones
    bodyPositions.insert(bodyPositions.begin(), original_bodyPositions.begin(), original_bodyPositions.end());
    bodyVelocities.insert(bodyVelocities.begin(), original_bodyVelocities.begin(), original_bodyVelocities.end());
    bodyIDs.insert(bodyIDs.begin(), original_bodyIDs.begin(), original_bodyIDs.end());

    bodyPositions2.insert(bodyPositions2.begin(), bodyPositions.begin(), bodyPositions.end());
    bodyVelocities2.insert(bodyVelocities2.begin(), bodyVelocities.begin(), bodyVelocities.end());
    bodyIDs2.insert(bodyIDs2.begin(), bodyIDs.begin(), bodyIDs.end());

    setupMergerModel(bodyPositions,  bodyVelocities,  bodyIDs,
                       bodyPositions2, bodyVelocities2, bodyIDs2,
                       sizeRatio, massRatio, impact, seperation,
                       angle1, angle1b, angle2, angle2b);

    //Load data onto the device
    memcpy(&m_tree->localTree.bodies_pos[0], &bodyPositions[0], sizeof(real4)*bodyPositions.size());
    memcpy(&m_tree->localTree.bodies_vel[0], &bodyVelocities[0], sizeof(real4)*bodyPositions.size());
    memcpy(&m_tree->localTree.bodies_ids[0], &bodyIDs[0], sizeof(int)*bodyPositions.size());
      
    memcpy(&m_tree->localTree.bodies_pos[(int)bodyPositions.size()], &bodyPositions2[0], sizeof(real4)*bodyPositions.size());
    memcpy(&m_tree->localTree.bodies_Ppos[(int)bodyPositions.size()], &bodyPositions2[0], sizeof(real4)*bodyPositions.size());
    memcpy(&m_tree->localTree.bodies_vel[(int)bodyPositions.size()], &bodyVelocities2[0], sizeof(real4)*bodyPositions.size());
    memcpy(&m_tree->localTree.bodies_Pvel[(int)bodyPositions.size()], &bodyVelocities2[0], sizeof(real4)*bodyPositions.size());
    memcpy(&m_tree->localTree.bodies_ids[(int)bodyPositions.size()], &bodyIDs2[0], sizeof(int)*bodyPositions.size());

    //Only copy whats needed for visualization
    m_tree->localTree.bodies_pos.h2d();
    m_tree->localTree.bodies_ids.h2d();   
  }

  void startSimulation()
  {
    //m_tree->localTree.setN((int)bodyPositions.size());
    //m_tree->allocateParticleMemory(m_tree->localTree);

    m_tree->localTree.bodies_Ppos.h2d();
    m_tree->localTree.bodies_Pvel.h2d();
    m_tree->localTree.bodies_vel.h2d();
    m_tree->localTree.bodies_pos.h2d();
    m_tree->localTree.bodies_ids.h2d();  

    m_tree->localTree.bodies_acc0.zeroMem();

    //Stupid fix for the predict / set GrpID problem. JB Need to FIX this
    m_tree->localTree.active_group_list.cresize(m_tree->localTree.n, false);
    m_tree->localTree.body2group_list.zeroMem();

    m_tree->reset_energy();
  
    setupMergerComplete = true;
  }


  void display() { 
    if (m_renderingEnabled)
    {
      
      //Jeroen 
      #ifdef USE_DUST //Only works if dust is in seperate array that we can modify
        
        int updateRingRegen  = 0;
        int updateRingRotate = 0;
        updateRingRotate += (m_renderer.m_ringInclination != m_renderer.m_ringInclination_old);
        updateRingRotate += (m_renderer.m_ringPhi != m_renderer.m_ringPhi_old);
        updateRingRegen  += (m_renderer.m_ringShiftFromCenter != m_renderer.m_ringShiftFromCenter_old);
        updateRingRegen  += (m_renderer.m_ringZscale != m_renderer.m_ringZscale_old);
        updateRingRegen  += (m_renderer.m_ringRscale != m_renderer.m_ringRscale_old);
        updateRingRegen  += (m_renderer.m_nDustParticles != m_renderer.m_nDustParticles_old);

        m_renderer.m_ringZscale_old = m_renderer.m_ringZscale;
        m_renderer.m_ringRscale_old = m_renderer.m_ringRscale;
        m_renderer.m_ringShiftFromCenter_old = m_renderer.m_ringShiftFromCenter;


        if(updateRingRotate)
        {
          fprintf(stderr, "old %f %f new: %f %f\n",
          m_renderer.m_ringInclination_old, m_renderer.m_ringPhi_old,
          m_renderer.m_ringInclination, m_renderer.m_ringPhi);
          float incnew = (float)( m_renderer.m_ringInclination * M_PI/180.0);
          float phinew = (float)( m_renderer.m_ringPhi         * M_PI/180.0);
          float incold = (float)( m_renderer.m_ringInclination_old * M_PI/180.0);
          float phiold = (float)( m_renderer.m_ringPhi_old         * M_PI/180.0);

          const Rotation MatOld(-incold, phiold);
          const Rotation MatNew(incnew,phinew);
          m_tree->localTree.dust_pos.d2h(); 
          m_tree->localTree.dust_vel.d2h(); 

          for(int i=0; i < m_tree->localTree.n_dust; i++)
          {   
            //Rotate back to standard
            m_tree->localTree.dust_pos[i] = MatOld.rotate(m_tree->localTree.dust_pos[i]);
            m_tree->localTree.dust_vel[i] = MatOld.rotate(m_tree->localTree.dust_vel[i]);
        
            //New rotation
            m_tree->localTree.dust_pos[i] = MatNew.rotate(m_tree->localTree.dust_pos[i]);
            m_tree->localTree.dust_vel[i] = MatNew.rotate(m_tree->localTree.dust_vel[i]);
          }
          m_tree->localTree.dust_pos.h2d(); 
          m_tree->localTree.dust_vel.h2d(); 

        }

       m_renderer.m_ringInclination_old     = m_renderer.m_ringInclination;
       m_renderer.m_ringPhi_old    = m_renderer.m_ringPhi;
       m_renderer.m_nDustParticles_old =  m_renderer.m_nDustParticles;

        if(updateRingRegen)
        {
          fprintf(stderr, "Update ring\n");

          //Generate a new ring
          real dRshift  = m_renderer.m_ringShiftFromCenter;
          real Rscale   = (real) m_renderer.m_ringRscale;
          real Zscale   = (real) m_renderer.m_ringZscale;
          real nrScale  = (real) 4.0;
          real nzScale  = (real) 4.0;
          real inclination = (real) m_renderer.m_ringInclination;
          real phi         = (real) m_renderer.m_ringPhi;
          DustRing::RingType ring_type = DustRing::CYLINDER;
          //DustRing::RingType ring_type = DustRing::TORUS;
          int Ndust = m_renderer.m_nDustParticles;
          const int Ndisk = 150000;

          fprintf(stderr, " Adding %s dust ring: \n", ring_type == DustRing::CYLINDER ? "CYLINDER" : "TORUS");
          fprintf(stderr, "   N=       %d \n", Ndust);
          fprintf(stderr, "   dRshift= %g \n", dRshift);
          fprintf(stderr, "   Rscale=  %g \n", Rscale);
          fprintf(stderr, "   Zscale=  %g \n", Zscale);
          fprintf(stderr, "   incl=    %g  degrees \n", inclination);
          fprintf(stderr, "   phi=     %g  degrees \n", phi);
          fprintf(stderr, "   nrScale= %g \n", nrScale);
          fprintf(stderr, "   nzScale= %g \n", nzScale);

          Vel1D::Vector VelCurve;
          VelCurve.reserve(Ndisk);
          vec3 L(0.0);
          Real Mtot = 0.0;
          Real Rmin = (Real) HUGE;
          Real Rmax = 0.0;

          m_tree->localTree.bodies_ids.d2h();   
          m_tree->localTree.bodies_pos.d2h();   

          for (int i = 0; i < m_tree->localTree.n; i++)
          {
            //Only if its a disk particle
            if(m_tree->localTree.bodies_ids[i] < 50000000)
            {   
              real4 pos4 =  m_tree->localTree.bodies_pos[i];
              real4 vel4 =  m_tree->localTree.bodies_pos[i];
              const vec3 pos(pos4.x, pos4.y, pos4.z);
              const vec3 vel(vel4.x, vel4.y, vel4.z);
              const Real V = std::sqrt(vel.x*vel.x + vel.y*vel.y);
              if(0.01*V > std::abs(vel.z))
              {
                L    += pos4.w * (pos%vel);
                Mtot += pos4.w;
                const Real R = std::sqrt(pos.x*pos.x + pos.y*pos.y);
        
                VelCurve.push_back(Vel1D(R, V));
                Rmin = std::min(Rmin, R);
                Rmax = std::max(Rmax, R);
              }
            }
          }
          L *= 1.0f/Mtot;
          fprintf(stderr, " Ncurve= %d :: L= %g %g %g \n", (int)VelCurve.size(), L.x, L.y, L.z);
          fprintf(stderr, "  Rmin= %g  Rmax= %g \n", Rmin, Rmax);

          /* setting radial scale height of the dust ring */

          const Real dR = (Real)((Rmax - Rmin)*0.5);
          const Real D  = (Real) Rscale*dR;
          
          dRshift      = m_renderer.m_ringShiftFromCenter;
          //const Real Ro = (Real)( (Rmax + Rmin)*0.5 + dRshift * D);
          const Real Ro = (Real)( (Rmax + Rmin)*dRshift);
         
          /* determining vertical scale-height of the disk */
          Real Zmin = (Real) HUGE;
          Real Zmax = 0.0;

          for (int i = 0; i < m_tree->localTree.n; i++)
          {
            //Only if its a disk particle
            if(m_tree->localTree.bodies_ids[i] < 50000000)
            {  
              real4 pos =  m_tree->localTree.bodies_pos[i];
              const Real R = std::sqrt(pos.x*pos.x + pos.y*pos.y);
              if(R > Ro - nrScale*D && R < Ro + nrScale*D)
              {
                Zmin = std::min(Zmin, pos.z);
                Zmax = std::max(Zmax, pos.z);
              }
            }
          }

          const real dZ = Zmax - Zmin;
          fprintf(stderr, "Zmin= %g Zmax= %g \n", Zmin, Zmax);

          /* setting vertical scale height of the dust ring */

          const Real H = Zscale*dZ;

          /** Generating dust ring **/

          const DustRing ring(Ndust, Ro, D, H, inclination, VelCurve, phi, nrScale, nzScale, ring_type);

          fprintf(stderr, "Generation complete \n");
          //Update the dust particles
          m_tree->localTree.dust_pos.d2h(); 
          m_tree->localTree.dust_vel.d2h(); 

         // if(Ndust > m_tree->localTree.n_dust)
          {
            m_tree->localTree.setNDust(Ndust);
            m_tree->resizeDustMemory(m_tree->localTree);
          }

          int dustID = 50000000;
          for(int i=0; i < Ndust; i++)
          {   
              m_tree->localTree.dust_pos[i].x = ring.ptcl[i].pos.x;
              m_tree->localTree.dust_pos[i].y = ring.ptcl[i].pos.y;
              m_tree->localTree.dust_pos[i].z = ring.ptcl[i].pos.z;
              m_tree->localTree.dust_pos[i].w = 0;
              m_tree->localTree.dust_ids[i] = dustID++;;
              m_tree->localTree.dust_vel[i].x = ring.ptcl[i].vel.x;
              m_tree->localTree.dust_vel[i].y = ring.ptcl[i].vel.y;
              m_tree->localTree.dust_vel[i].z = ring.ptcl[i].vel.z;
          }

          m_tree->localTree.dust_pos.h2d(); 
          m_tree->localTree.dust_vel.h2d(); 
          m_renderer.setNumberOfParticles(m_tree->localTree.n + m_tree->localTree.n_dust);
        }//UpdateRing
      #endif //USE_DUST

      #if 1
        if(modelDoubled)
        {
          //Only check for these settings after we are sure we 
          //doubled the galaxies
          int updateMergerConfigurationRequest = 0;
          updateMergerConfigurationRequest += (m_renderer.m_mergSizeRatio != m_renderer.m_mergSizeRatio_old);
          updateMergerConfigurationRequest += (m_renderer.m_mergMassRatio != m_renderer.m_mergMasRatio_old);
          updateMergerConfigurationRequest += (m_renderer.m_merImpact != m_renderer.m_mergImpact_old);
          updateMergerConfigurationRequest += (m_renderer.m_mergSeperation != m_renderer.m_mergSeperation_old);
          updateMergerConfigurationRequest += (m_renderer.m_inclination1 != m_renderer.m_inclination1_old);
          updateMergerConfigurationRequest += (m_renderer.m_inclination2 != m_renderer.m_inclination2_old);
          updateMergerConfigurationRequest += (m_renderer.m_omega1 != m_renderer.m_omega1_old);
          updateMergerConfigurationRequest += (m_renderer.m_omega2 != m_renderer.m_omega2_old);

          m_renderer.m_mergSizeRatio_old = m_renderer.m_mergSizeRatio;
          m_renderer.m_mergMasRatio_old = m_renderer.m_mergMassRatio;
          m_renderer.m_mergImpact_old = m_renderer.m_merImpact;
          m_renderer.m_mergSeperation_old = m_renderer.m_mergSeperation;
          m_renderer.m_inclination1_old = m_renderer.m_inclination1;
          m_renderer.m_inclination2_old = m_renderer.m_inclination2;
          m_renderer.m_omega1_old = m_renderer.m_omega1;
          m_renderer.m_omega2_old = m_renderer.m_omega2;

          if(updateMergerConfigurationRequest)
          {
            fprintf(stderr, "Updating! \n");
            float sizeRatio = 1.52f;
            float massRatio = 1;
            float impact    = 10;
            float seperation= 168;
            float angle1  = 0;
            float angle1b = 0;
            float angle2  = 180;
            float angle2b = 0;

            sizeRatio =  m_renderer.m_mergSizeRatio;
            massRatio =  m_renderer.m_mergMassRatio;
            impact =  m_renderer.m_merImpact;
            seperation =  m_renderer.m_mergSeperation;
            angle1 =  m_renderer.m_inclination1;
            angle2 =  m_renderer.m_inclination2;
            angle1b =  m_renderer.m_omega1;
            angle2b =  m_renderer.m_omega2;

            updateMergerConfiguration(sizeRatio, massRatio, impact, seperation, 
                              angle1, angle1b, angle2, angle2b);
          } //updateMergerConfigurationRequest
        } //modelDoubled

      #endif 



      //end Jeroen


      getBodyData();

      moveCamera();
      m_cameraTransLag += (m_cameraTrans - m_cameraTransLag) * m_inertia;
      m_cameraRotLag += (m_cameraRot - m_cameraRotLag) * m_inertia;

      // view transform
      {
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

	    if (m_flyMode) {
		  glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
		  glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
          glRotatef(90.0f, 1.0f, 0.0f, 0.0f); // rotate galaxies into XZ plane
		  glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);
	    } else {
		  // orbit viwer - rotate around centre, then translate
          glTranslatef(m_cameraTransLag.x, m_cameraTransLag.y, m_cameraTransLag.z);
          glRotatef(m_cameraRotLag.x, 1.0, 0.0, 0.0);
          glRotatef(m_cameraRotLag.y, 0.0, 1.0, 0.0);
          glRotatef(90.0f, 1.0f, 0.0f, 0.0f); // rotate galaxies into XZ plane
        }

        glGetFloatv(GL_MODELVIEW_MATRIX, m_modelView);

        //m_renderer.display(m_displayMode);
	      m_renderer.render();

        if (m_displayBoxes) {
          glEnable(GL_DEPTH_TEST);
          displayOctree();  
        }

        if (m_displaySliders) {
	        m_renderer.getParams()->Render(0, 0);
        }
      }
    }
  }

  void mouse(int button, int state, int x, int y)
  {
    int mods;

	  if (m_displaySliders) {
		  if (m_renderer.getParams()->Mouse(x, y, button, state))
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
      if (m_renderer.getParams()->Motion(x, y))
        return;
    }

    if (m_buttonState == 3) {
      // left+middle = zoom
      float3 v = make_float3(0.0f, 0.0f, dy*zoomSpeed*fabs(m_cameraTrans.z));
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
    }

    m_ox = x;
    m_oy = y;
  }

  void moveCamera()
  {
    if (!m_flyMode)
      return;

    const float flySpeed = 1.0f;
    //float flySpeed = (m_keyModifiers & GLUT_ACTIVE_SHIFT) ? 4.0f : 1.0f;

	// Z
    if (m_keyDown['w']) {
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
    //case 'q':
    //case 'Q':
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
    case ',':
    case '<':
      incrementOctreeDisplayLevel(-1);
      break;
    case '.':
    case '>':
      incrementOctreeDisplayLevel(+1);
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
      } else {
        fitCamera();
      }
      break;
    case 'M':
    case 'm':
      doubleGalaxyModel();
      break;
    case 'N':
    case 'n':
      startSimulation();
      break;
    case 'J':
    case 'j':
      mergeDustWithNormalParticles();
      break;
      
    }

    m_keyDown[key] = true;
  }

  void keyUp(unsigned char key) {
    m_keyDown[key] = false;
    m_keyModifiers = 0;
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

  ParamListGL *getParams() { return m_renderer.getParams(); }

  float  frand() { return rand() / (float) RAND_MAX; }
  float4 randColor(float scale) { return make_float4(frand()*scale, frand()*scale, frand()*scale, 0.0f); }

#if 0
 void getBodyData() {
    //m_tree->localTree.bodies_pos.d2h();
    m_tree->localTree.bodies_ids.d2h();
    //m_tree->localTree.bodies_vel.d2h();

    int n = m_tree->localTree.n;

    float4 starColor = make_float4(1.0f, 1.0f, 0.5f, 1.0f);	// yellowish
    //float4 starColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);		// white
    float4 starColor2 = make_float4(1.0f, 0.2f, 0.5f, 1.0f) * make_float4(100.0f, 100.0f, 100.0f, 1.0f);		// purplish

    float overbright = 1.0f;
    starColor *= make_float4(overbright, overbright, overbright, 1.0f);

    float4 dustColor = make_float4(0.0f, 0.0f, 0.1f, 0.0f);	// blue
    //float4 dustColor = make_float4(0.1f, 0.1f, 0.1f, 0.0f);	// grey

    //float4 *colors = new float4[n];
    float4 *colors = m_particleColors;

    for (int i = 0; i < n; i++) {
      int id = m_tree->localTree.bodies_ids[i];
	    //printf("%d: id %d, mass: %f\n", i, id, m_tree->localTree.bodies_pos[i].w);
	    srand(id*1783);
#if 1
	    float r = frand();
	    if (id >= 0 && id < 100000000) {
		    // dust -- not used yet
		    colors[i] = make_float4(1, 0, 0, 1);
	    } else if (id >= 100000000 && id < 200000000) {
	      // dark matter
//         colors[i] = starColor + randColor(0.1f);
//         colors[i] = starColor; // * powf(r, 2.0f);
		  colors[i] = (frand() < 0.999f) ? starColor * r : starColor2;
        //colors[i] = starColor;
	    } else {
		    // stars
		    colors[i] = dustColor * make_float4(r, r, r, 1.0f);
	    }
#else
	    // test sorting
	    colors[i] = make_float4(frand(), frand(), frand(), 1.0f);
#endif
    }

    //m_renderer.setPositions((float*)&m_tree->localTree.bodies_pos[0], n);
    //m_renderer.setColors((float*)colors, n);
    m_renderer.setNumParticles(n);
    //m_renderer.setPositions((float*)&m_tree->localTree.bodies_pos[0]);
    m_renderer.setPositionsDevice((float*) m_tree->localTree.bodies_pos.d());
    m_renderer.setColors((float*)colors);

    //delete [] colors;
  }//end getBodyData
#else
 void getBodyData() {

   int n = m_tree->localTree.n + m_tree->localTree.n_dust;   
   //Above is save since it is 0 if we dont use dust
 

    #ifdef USE_DUST
     //We move the dust data into the position data (on the device :) )
     if(m_tree->localTree.n_dust > 0)
     {
       m_tree->localTree.bodies_pos.copy_devonly(m_tree->localTree.dust_pos,
                             m_tree->localTree.n_dust, m_tree->localTree.n); 
       m_tree->localTree.bodies_ids.copy_devonly(m_tree->localTree.dust_ids,
                             m_tree->localTree.n_dust, m_tree->localTree.n);
     }
    #endif    

    m_tree->localTree.bodies_ids.d2h();   
  
    
    float4 starColor = make_float4(1.0f, 1.0f, 0.5f, 1.0f);  // yellowish
    //float4 starColor = make_float4(1.0f, 1.0f, 0.0f, 1.0f);               // white
    float4 starColor2 = make_float4(1.0f, 0.2f, 0.5f, 1.0f) * make_float4(100.0f, 100.0f, 100.0f, 1.0f);             // purplish

    float overbright = 1.0f;
    starColor *= make_float4(overbright, overbright, overbright, 1.0f);

    float4 dustColor = make_float4(0.0f, 0.0f, 0.1f, 0.0f);      // blue
    //float4 dustColor =  make_float4(0.1f, 0.1f, 0.1f, 0.0f);    // grey

    float4 *colors = m_particleColors;

    for (int i = 0; i < n; i++)
    {
      int id =  m_tree->localTree.bodies_ids[i];
            //printf("%d: id %d, mass: %f\n", i, id, m_tree->localTree.bodies_pos[i].w);
            srand(id*1783);
#if 1
      float r = frand();
            
      if (id >= 0 && id < 50000000)     //Disk
      {
        colors[i] = make_float4(0, 0, 1, 1);        
      } 
      else if (id >= 50000000 && id < 100000000) //Dust
      {
        colors[i] = starColor;

      } 
      else if (id >= 100000000 && id < 200000000) //Bulge
      {
          colors[i] = (frand() < 0.99f) ? starColor : starColor2;          
      } 
      else //>= 200000000, Dark matter
      {
         colors[i] = dustColor;
      }            
      
#else
            // test sorting
            colors[i] = make_float4(frand(), frand(), frand(), 1.0f);
#endif
    }

    m_renderer.setNumParticles( m_tree->localTree.n + m_tree->localTree.n_dust);    
    m_renderer.setPositionsDevice((float*) m_tree->localTree.bodies_pos.d());
    m_renderer.setColors((float*)colors);
  }
#endif 



  void displayOctree() {
    float3 boxMin = make_float3(m_tree->rMinLocalTree);
    float3 boxMax = make_float3(m_tree->rMaxLocalTree);

    glLineWidth(1.0);
    //glColor3f(0.0, 1.0, 0.0);
    glColor3f(0.0, 0.5, 0.0);
    glEnable(GL_LINE_SMOOTH);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    drawWireBox(boxMin, boxMax);
      
    m_tree->localTree.boxCenterInfo.d2h();
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

    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
  }

  octree *m_tree;
  octree::IterationData &m_idata;
  bool iterationsRemaining;

  //ParticleRenderer m_renderer;
  //ParticleRenderer::DisplayMode m_displayMode; 
  SmokeRenderer m_renderer;
  SmokeRenderer ::DisplayMode m_displayMode; 
  int m_octreeDisplayLevel;

  float4 *m_particleColors;

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

  bool m_keyDown[256];
  int m_keyModifiers;
};

BonsaiDemo *theDemo = NULL;

void onexit() {
  if (theDemo) delete theDemo;
  cudaDeviceReset();
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
  theDemo->key(key);
  glutPostRedisplay();
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
  theDemo->keyUp(key);
}

void special(int key, int x, int y)
{
	theDemo->getParams()->Special(key, x, y);
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
  glutKeyboardUpFunc(keyUp);
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

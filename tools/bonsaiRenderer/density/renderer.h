/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Smoke particle renderer with volumetric shadows

#ifndef SMOKE_RENDERER_H
#define SMOKE_RENDERER_H

#include <mpi.h>
#include <GL/glew.h>
#include "framebufferObject.h"
#include "GLSLProgram.h"
#include "nvMath.h"
#include "paramgl.h"
#include <cuda_runtime.h>
#include <array>
//#include "GpuArray.h"
//
class SmokeRendererParams
{
  public:
    enum DisplayMode
    {
      POINTS,
#if 0
      SPRITES,
      SPRITES_SORTED,
#endif      
      VOLUMETRIC,
      SPLOTCH,
      SPLOTCH_SORTED,
      VOLUMETRIC_NEW,
      NUM_MODES
    };
  protected:
    DisplayMode	        mDisplayMode;

    // window
    unsigned int        mWindowW, mWindowH;
    float               mAspect, mInvFocalLen;
    float               mFov;

    int                 m_downSample;
    int                 m_blurDownSample;
    int                 m_imageW, m_imageH;
    int                 m_downSampledW, m_downSampledH;

    int                 m_batchSize;
    int                 m_sliceNo;

    // parameters
    bool                m_doBlur;
    bool                m_displayLightBuffer;
    bool                m_domainView;
    int                 m_domainViewIdx;

    nv::vec3f               m_lightVector, m_lightPos, m_lightTarget;
    nv::vec3f               m_colorOpacity;
    float               m_lightDistance;

    nv::matrix4f            m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
    nv::vec3f               m_viewVector, m_halfVector;
    bool                m_invertedView;
    nv::vec4f               m_eyePos;
    nv::vec4f               m_halfVectorEye;
    nv::vec4f               m_lightPosEye;

    float				m_minDepth, m_maxDepth;
    bool                m_enableAA;
    bool				m_enableVolume;
    bool				m_enableFilters;

    /***********************
     *			   *
     *  SPLOTCH PARAMETERS *
     *			   *
     ***********************
    */  

    float m_starScaleLog;
    float m_starAlpha;

    float m_dmScaleLog;
    float m_dmAlpha;

    float m_spriteSizeMaxLog;
    float m_spriteAlpha;
    float m_transmission;

    float m_imageBrightnessPre;
    float m_gammaPre;
    float m_imageBrightnessPost;
    float m_gammaPost;

    /**************************
     *			      *
     *  VOLUMETRIC PARAMETERS *
     *			      *
     **************************
    */ 


    int                 m_numSlices;
    int                 m_numDisplayedSlices;

    float               mParticleRadius;
    float               mParticleScaleLog;

    float               m_dustAlpha;
    float m_ageScale;

    nv::vec3f           m_lightColor;
    float 		m_spriteAlpha_volume;
    float               m_shadowAlpha;

    float m_indirectAmount;
    float m_fog;
    float m_overBright;
    float m_overBrightThreshold;
    float m_imageBrightness;

    float m_gamma;
    int m_blurPasses;
    float               m_blurRadius;

    float m_sourceIntensity;
    float m_starBlurRadius;
    float m_starPower;
    float m_starIntensity;
    float m_starThreshold;
    
    float m_glowRadius;
    float m_glowIntensity;

    float m_flareIntensity;
    float m_flareThreshold;
    float m_flareRadius;

    float m_skyboxBrightness;



    /********************************/




    float m_volumeAlpha;
    nv::vec3f m_volumeColor;
    float m_noiseFreq;
    float m_noiseAmp;
    float m_volumeIndirect;
    float m_volumeStart;
    float m_volumeWidth;

    bool m_cullDarkMatter;
    bool   m_doClipping;
    SmokeRendererParams();
  public:
    void setDomainView(const bool m) {  m_domainView = m;}
    void setDomainViewIdx(const int idx) { m_domainViewIdx = idx; }
    void enableClipping() {m_doClipping = true;}
    void disableClipping() {m_doClipping = false;}
};

class SmokeRenderer : public SmokeRendererParams
{
  const int rank, nrank;
  const MPI_Comm &comm;

  int getMaster() const { return 0;}
  bool isMaster() const { return rank == getMaster(); };

  using uint = unsigned int;
  public:
    SmokeRenderer(int numParticles, int maxParticles, const int , const int, const MPI_Comm&);
    ~SmokeRenderer();


    enum Target
    {
      LIGHT_BUFFER,
      SCENE_BUFFER
    };

    void setDisplayMode(DisplayMode mode) { mDisplayMode = mode; }
    DisplayMode getDisplayMode() const { return mDisplayMode; }

    void setNumParticles(unsigned int x) { mNumParticles = x; }

    void setPositionBuffer(GLuint vbo) { mPosVbo = vbo; }
    void setVelocityBuffer(GLuint vbo) { mVelVbo = vbo; }
    void setColorBuffer(GLuint vbo) { mColorVbo = vbo; }
    void setSizeBuffer(GLuint vbo) { mSizeVbo = vbo; }
    void setIndexBuffer(GLuint ib) { mIndexBuffer = ib; }

    void setPositions(float *pos);
    void setPositionsDevice(float *posD);
    void setColors(float *color);
    void setSizes(float *sizes);
    void setColorsDevice(float *colorD);

    void setWindowSize(int w, int h);
    void setFOV(float fov) { mFov = fov; }

    // params
    void setParticleRadius(float x) { mParticleRadius = x; }
    float getParticleRadius() { return mParticleRadius;}

    void setNumSlices(int x) { m_numSlices = x; }
    int getNumSlices() { return m_numSlices; }
    void setNumDisplayedSlices(int x) { m_numDisplayedSlices = x; }

    void setAlpha(float x) { m_spriteAlpha = x; }
    void setShadowAlpha(float x) { m_shadowAlpha = x; }
    void setColorOpacity(nv::vec3f c) { m_colorOpacity = c; }
    void setLightColor(nv::vec3f c);
    nv::vec3f getLightColor() { return m_lightColor; }

    void setOverbright(float x) { m_overBright = x; }

    void setDoBlur(bool b) { m_doBlur = b; }
    void setBlurRadius(float x) { m_blurRadius = x; }
    void setDisplayLightBuffer(bool b) { m_displayLightBuffer = b; }

    void setDepthMinMax(float min, float max) { m_minDepth = min; m_maxDepth = max; }
    void setEnableAA(bool b) { m_enableAA = b; }

    void setEnableVolume(bool b) { m_enableVolume = b; }
    bool getEnableVolume() { return m_enableVolume; }

    void setEnableFilters(bool b) { m_enableFilters = b; }
    bool getEnableFilters() { return m_enableFilters; }

    void beginSceneRender(Target target);
    void endSceneRender(Target target);

    void setLightPosition(nv::vec3f v) { m_lightPos = v; }
    void setLightTarget(nv::vec3f v) { m_lightTarget = v; }

    void setRampTex(GLuint tex) { m_rampTex = tex; }

    nv::vec4f getLightPositionEyeSpace() { return m_lightPosEye; }
    nv::matrix4f getShadowMatrix() { return m_shadowMatrix; }

    GLuint getShadowTexture() { return m_lightTexture[m_srcLightTexture]; }

    void calcVectors();
    nv::vec3f getSortVector() { return m_halfVector; }
    //    nv::vec3f getSortVector() { return m_viewVector; }

    bool getCullDarkMatter() { return m_cullDarkMatter; }
    void setCullDarkMatter(bool b) { m_cullDarkMatter = b; }

    ParamListGL *getParams() { return m_params[(int)mDisplayMode]; }
    ParamListGL **getAllParams() { return m_params; }

    void render();
    void debugVectors();

    void depthSort(float4 *posD);

    //By JB to modify particle count while running
    void setNumberOfParticles(uint n_particles); 
    int  getNumberOfParticles() {return this->mNumParticles; }

  private:
    //GLuint loadTexture(char *filename);
    //void loadSmokeTextures(int nImages, int offset, char* sTexturePrefix);
    GLuint createRainbowTexture();

    void depthSortCopy();

    void drawPoints(int start, int count, bool sorted);
    void drawPointSprites(GLSLProgram *prog, int start, int count, bool shadowed, bool sorted);

    void drawSlice(int i);
    void drawSliceLightView(int i);
    void drawSliceLightViewAA(int i);

    void drawVolumeSlice(int i, bool shadowed);

    void drawSlices();
    void renderSprites(bool sort);

    void displayTexture(GLuint tex, float scale);
    void doStarFilter();
    void downSample();
    void doFlare();
    void doGlowFilter();
    void compositeResult();
    void blurLightBuffer();
    void processImage(GLSLProgram *prog, GLuint src, GLuint dest);

    void splotchDraw    ();
    void splotchDrawSort();
    void volumetricNew  ();

    GLuint createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format, void *data = 0);
    GLuint createNoiseTexture(int w, int h, int d);
    float *createSplatImage(int n);
    GLuint createSpriteTexture(int size);
    GLuint createSphTexture(int size);

    void createBuffers(int w, int h);
    void createLightBuffer();

    void drawQuad(float s=1.0f, float z=0.0f);
    void drawVector(nv::vec3f v);
    void drawBounds();
    void drawSkybox(GLuint tex);

    void initParams();

    // particle data
    unsigned int        mMaxParticles;
    unsigned int		mNumParticles;

    GLuint              mPosVbo;
    GLuint              mVelVbo;
    GLuint              mColorVbo;
    GLuint              mSizeVbo;
    GLuint              mSizeVao;
    GLuint              mIndexBuffer;
    GLuint              mPosBufferTexture;

    //	GpuArray<float4>    mParticlePos;
    //	GpuArray<float>     mParticleDepths;
    //	GpuArray<uint>      mParticleIndices;

    unsigned int m_pbo;

#if 0
    struct 
    {
      float               mParticleRadius;
      float               mParticleScaleLog;
      DisplayMode	        mDisplayMode;

      // window
      unsigned int        mWindowW, mWindowH;
      float               mAspect, mInvFocalLen;
      float               mFov;

      int                 m_downSample;
      int                 m_blurDownSample;
      int                 m_imageW, m_imageH;
      int                 m_downSampledW, m_downSampledH;

      int                 m_numSlices;
      int                 m_numDisplayedSlices;
      int                 m_batchSize;
      int                 m_sliceNo;

      // parameters
      float               m_shadowAlpha;
      float               m_dustAlpha;
      bool                m_doBlur;
      float               m_blurRadius;
      bool                m_displayLightBuffer;

      nv::vec3f               m_lightVector, m_lightPos, m_lightTarget;
      nv::vec3f               m_lightColor;
      nv::vec3f               m_colorOpacity;
      float               m_lightDistance;

      nv::matrix4f            m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
      nv::vec3f               m_viewVector, m_halfVector;
      bool                m_invertedView;
      nv::vec4f               m_eyePos;
      nv::vec4f               m_halfVectorEye;
      nv::vec4f               m_lightPosEye;

      float				m_minDepth, m_maxDepth;
      bool                m_enableAA;
      bool				m_enableVolume;
      bool				m_enableFilters;

      /****************/
      float m_starScaleLog;
      float m_starAlpha;

      float m_dmScaleLog;
      float m_dmAlpha;

      float m_spriteSizeMaxLog;
      float m_spriteAlpha;
      float m_transmission;

      float m_imageBrightnessPre;
      float m_gammaPre;
      float m_imageBrightnessPost;
      float m_gammaPost;

      /****************/

      float m_overBright;
      float m_overBrightThreshold;
      float m_imageBrightness;
      int m_blurPasses;
      float m_indirectAmount;

      float m_starBlurRadius;
      float m_starPower;
      float m_starIntensity;
      float m_starThreshold;

      float m_glowRadius;
      float m_glowIntensity;
      float m_gamma;

      float m_sourceIntensity;
      float m_flareIntensity;
      float m_flareThreshold;
      float m_flareRadius;

      float m_ageScale;
      float m_fog;
      float m_skyboxBrightness;

      float m_volumeAlpha;
      nv::vec3f m_volumeColor;
      float m_noiseFreq;
      float m_noiseAmp;
      float m_volumeIndirect;
      float m_volumeStart;
      float m_volumeWidth;

      bool m_cullDarkMatter;
    } params;
#endif

    ParamListGL         *m_params[NUM_MODES];

    // programs
    GLSLProgram         *m_simpleProg;
    GLSLProgram         *m_particleProg, *m_particleAAProg, *m_particleShadowProg;
    GLSLProgram         *m_displayTexProg, *m_blurProg;
    GLSLProgram         *m_starFilterProg;
    GLSLProgram         *m_compositeProg;
    GLSLProgram         *m_volumeProg;
    GLSLProgram         *m_downSampleProg;
    GLSLProgram         *m_gaussianBlurProg;
    GLSLProgram         *m_skyboxProg;
    GLSLProgram         *m_thresholdProg;
    GLSLProgram         *m_splotchProg;
    GLSLProgram         *m_splotch2texProg;
    GLSLProgram         *m_volnewProg;
    GLSLProgram         *m_volnew2texProg;

    // image buffers
    FramebufferObject   *m_fbo;
    int                 m_lightTextureW, m_lightTextureH;   // texture size
    int                 m_lightBufferSize;                  // actual buffer size
    GLuint              m_lightTexture[2];
    int                 m_srcLightTexture;
    GLuint              m_lightDepthTexture;

    GLuint              m_imageTex[5], m_depthTex;
    GLuint              m_downSampledTex[3];

    GLuint              m_rampTex;
    GLuint              m_rainbowTex;
    GLuint              m_textureArrayID;
    GLuint              m_spriteTex;
    GLuint              m_sphTex;
    GLuint              m_noiseTex;
    //    GLuint              m_cubemapTex;

    cudaStream_t        m_copyStreamPos;
    cudaStream_t        m_copyStreamColor;

    cudaStream_t      m_copyStreamSortPos;
    cudaStream_t      m_copyStreamSortDepth;
    cudaStream_t      m_copyStreamSortIndices;

    float4 *mParticlePos;
    float4 *mParticleColors;
    float  *mParticleDepths;
    uint   *mParticleIndices;

    float4 m_clippingPlane[6];  
    float3 m_xhigh, m_xlow;
    double m_modelViewWin[16], m_projectionWin[16];

    std::vector<int> compositingOrder;
    std::array<int,4> getVisibleViewport() const;
    void composeImages(const GLuint imgTex, const GLuint depthTex = 0);


  public:

    void setCompositingOrder(const std::vector<int> &order)
    {
      compositingOrder = order;
    }
    void setClippingPlane(const int i, const float4 &plane)
    {
      assert(i>=0 && i<6);
      m_clippingPlane[i] = plane;
    }
    float4 getClippingPlane(const int i) const
    {
      return m_clippingPlane[i];
    }
    void setXhighlow(const float3 xlow, const float3 xhigh)
    {
      m_xlow  = xlow;
      m_xhigh = xhigh;
    }
    void setMVP(const double modelViewWin[16], const double projectionWin[16])
    {
      for (int i = 0; i < 16; i++)
      {
        m_modelViewWin [i] = modelViewWin [i];
        m_projectionWin[i] = projectionWin[i];
      }
    }
};

#endif

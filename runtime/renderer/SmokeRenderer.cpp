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

/*
    This class renders particles using OpenGL and GLSL shaders
*/

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "SmokeRenderer.h"
#include "SmokeShaders.h"
//#include <nvImage.h>
#include "depthSort.h"
#include "Cubemap.h"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#define COLOR_ATTENUATION 0
#define USE_MRT 0
#define USE_HALF_ANGLE 0
#define MOTION_BLUR 0

extern int renderDevID;
extern int devID;

using namespace nv;

SmokeRenderer::SmokeRenderer(int numParticles, int maxParticles) :
    mMaxParticles(maxParticles),
    mNumParticles(numParticles),
    mPosVbo(0),
    mVelVbo(0),
    mColorVbo(0),
    mIndexBuffer(0),
    mParticleRadius(0.1f),
	mDisplayMode(SPRITES),
    mWindowW(800),
    mWindowH(600),
    mFov(40.0f),
    m_downSample(1),
    m_blurDownSample(2),
    m_numSlices(64),
	m_numDisplayedSlices(m_numSlices),
    m_sliceNo(0),
    m_shadowAlpha(0.1f),
    m_spriteAlpha(0.1f),
    m_volumeAlpha(0.2f),
	m_dustAlpha(1.0f),
    m_volumeColor(0.5f, 0.0f, 0.0f),
    m_doBlur(true),
	m_blurRadius(1.0f),
    m_blurPasses(2),
    m_displayLightBuffer(false),
    m_lightPos(5.0f, 5.0f, -5.0f),
	m_lightTarget(0.0f, 0.5f, 0.0f),
	m_lightColor(0.0f, 0.0f, 0.0f),
	m_colorOpacity(0.1f, 0.2f, 0.3f),
    //m_lightBufferSize(256),
	m_lightBufferSize(512),
    m_srcLightTexture(0),
    m_lightDepthTexture(0),
    m_fbo(0),
    m_depthTex(0),
    m_rampTex(0),
    m_overBright(1.0f),
    m_overBrightThreshold(0.005f),
    m_imageBrightness(1.0f),
    m_invertedView(false),
	m_minDepth(0.0f),
	m_maxDepth(1.0f),
	m_enableAA(false),
	m_starBlurRadius(40.0f),
	m_starThreshold(1.0f),
    m_starPower(1.0f),
	m_starIntensity(0.5f),
	m_glowRadius(10.0f),
    m_glowIntensity(0.5f),
	m_ageScale(10.0f),
	m_enableVolume(false),
	m_enableFilters(true),
    m_noiseFreq(0.05f),
    m_noiseAmp(1.0f),
    m_indirectAmount(0.5f),
    m_volumeIndirect(0.5f),
	m_volumeStart(0.5f),
	m_volumeWidth(0.1f),
	m_gamma(1.0f / 2.2f),
	m_fog(0.001f),
    m_cubemapTex(0),
    m_flareThreshold(0.5f),
    m_flareIntensity(0.0f),
    m_sourceIntensity(0.5f),
    m_flareRadius(50.0f),
    m_skyboxBrightness(0.5f),
    m_transmission(0.0f),
    m_cullDarkMatter(true)
{
	// load shader programs
	m_simpleProg = new GLSLProgram(simpleVS, simplePS);
#if MOTION_BLUR
    m_particleProg = new GLSLProgram(mblurVS, mblurGS, particlePS);
    m_particleAAProg = new GLSLProgram(mblurVS, mblurGS, particleAAPS);
    m_particleShadowProg = new GLSLProgram(mblurVS, mblurGS, particleShadowPS);
#else
    m_particleProg = new GLSLProgram(particleVS, particlePS);
    m_particleAAProg = new GLSLProgram(particleVS, particleAAPS);
    m_particleShadowProg = new GLSLProgram(particleVS, particleShadowPS);
#endif

    //m_blurProg = new GLSLProgram(passThruVS, blur2PS);
    m_blurProg = new GLSLProgram(passThruVS, blur3x3PS);
    m_displayTexProg = new GLSLProgram(passThruVS, texture2DPS);
	m_compositeProg = new GLSLProgram(passThruVS, compositePS);

	m_starFilterProg = new GLSLProgram(passThruVS, starFilterPS);
    m_volumeProg = new GLSLProgram(volumeVS, volumePS);

    //m_downSampleProg = new GLSLProgram(passThruVS, downSample4PS);
	m_downSampleProg = new GLSLProgram(passThruVS, downSample2PS);
    m_gaussianBlurProg = new GLSLProgram(passThruVS, gaussianBlurPS);
	m_thresholdProg = new GLSLProgram(passThruVS, thresholdPS);

    m_skyboxProg = new GLSLProgram(skyboxVS, skyboxPS);

    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    m_fbo = new FramebufferObject();
    // create buffer for light shadows
    //createLightBuffer();

    // textures
//    loadSmokeTextures(32,0,"perlinNoiseTextures2/noise2_");
//    loadSmokeTextures(8, 0, "noiseTextures3/noise");
//    m_rainbowTex = loadTexture("data/rainbow.png");
	m_rainbowTex = createRainbowTexture();

	glGenTextures(1, &mPosBufferTexture);
	m_noiseTex = createNoiseTexture(64, 64, 64);

    m_cubemapTex = loadCubemapCross("../images/Carina_cross.ppm");
    //m_cubemapTex = loadCubemap("../images/deepfield%d.ppm");
    //m_cubemapTex = loadCubemap("../images/deepfield%d_1k.ppm");

	m_spriteTex = createSpriteTexture(256);

    initParams();

	//initCUDA();

	cudaGLSetGLDevice(renderDevID);

	mParticlePos.alloc(mMaxParticles, true, false, false);
	mParticleDepths.alloc(mMaxParticles, false, false, false);
	mParticleIndices.alloc(mMaxParticles, true, false, true);
	for(uint i=0; i<mMaxParticles; i++) {
		mParticleIndices.getHostPtr()[i] = i;
	}
	mParticleIndices.copy(GpuArray<uint>::HOST_TO_DEVICE);

	cudaStreamCreate(&m_copyStreamPos);
    	cudaStreamCreate(&m_copyStreamColor);

	cudaSetDevice(devID);

	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));

    glutReportErrors();
}

SmokeRenderer::~SmokeRenderer()
{
    delete m_particleProg;
	delete m_particleAAProg;
    delete m_particleShadowProg;
    delete m_blurProg;
    delete m_displayTexProg;
	delete m_simpleProg;
	delete m_starFilterProg;
	delete m_compositeProg;
    delete m_volumeProg;
    delete m_skyboxProg;

    delete m_fbo;
    glDeleteTextures(2, m_lightTexture);
    glDeleteTextures(1, &m_lightDepthTexture);
	glDeleteTextures(1, &mPosBufferTexture);

    glDeleteTextures(4, m_imageTex);
    glDeleteTextures(1, &m_depthTex);
    glDeleteTextures(3, m_downSampledTex);

	glDeleteTextures(1, &m_noiseTex);
	glDeleteTextures(1, &m_cubemapTex);

	mParticleDepths.free();
	mParticleIndices.free();
}

GLuint SmokeRenderer::createRainbowTexture()
{
	vec4f colors[] = {
		vec4f(1.0f, 0.0f, 0.0f, 1.0f),	// r
		vec4f(1.0f, 1.0f, 0.0f, 1.0f),	// y
		vec4f(0.0f, 1.0f, 0.0f, 1.0f),	// g
		vec4f(0.0f, 0.0f, 1.0f, 1.0f),	// b
		vec4f(1.0f, 0.0f, 1.0f, 1.0f),	// m
	};

	GLuint tex;
    glGenTextures(1, &tex);

    GLenum target = GL_TEXTURE_2D;
    glBindTexture( target, tex);

    glTexParameteri( target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri( target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri( target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(target, 0, GL_RGBA, sizeof(colors) / sizeof(vec4f), 1, 0, GL_RGBA, GL_FLOAT, &colors[0]);
    return tex;
}

#if 0
GLuint SmokeRenderer::loadTexture(char *filename)
{
    nv::Image image;
    if (!image.loadImageFromFile(filename))  {
        printf( "Failed to load image '%s'\n", filename);
        return 0;
    }

    GLuint tex;
    glGenTextures(1, &tex);

    GLenum target = GL_TEXTURE_2D;
    glBindTexture( target, tex);

    glTexParameteri( target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri( target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri( target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(target, 0, image.getInternalFormat(), image.getWidth(), image.getHeight(), 0, image.getFormat(), image.getType(), image.getLevel(0));
    return tex;
}

void SmokeRenderer::loadSmokeTextures(int nImages, int offset, char* sTexturePrefix)
{
    nv::Image* images  = new nv::Image[nImages];
    char textureName[260];
    std::string resolved_path;
    for(int i=0; i<nImages; i++) 
    {
        sprintf_s(textureName, "%s%.3d.png", sTexturePrefix, i+offset);
        //if ( pathHelper.getFilePath( textureName, resolved_path)) 
        {
            //if (! images[i].loadImageFromFile( resolved_path.c_str())) 
            if (! images[i].loadImageFromFile(textureName)) 
            {
                printf( "Failed to load smoke texture\n");
                exit(-1);
            }
            printf("Loaded '%s'\n", textureName);
        }
    }

    // load images as 2d texture array
    glGenTextures(1, &m_textureArrayID);
    glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, m_textureArrayID);

    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    // 2D Texture arrays a loaded just like 3D textures
    glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_LUMINANCE8, images[0].getWidth(), images[0].getHeight(), nImages, 0, images[0].getFormat(), images[0].getType(), NULL);
    for (int i = 0; i < nImages; i++) 
        glTexSubImage3D( GL_TEXTURE_2D_ARRAY_EXT, 0, 0, 0, i, images[i].getWidth(), images[i].getHeight(), 1, images[i].getFormat(), images[i].getType(), images[i].getLevel(0));

}
#endif

void SmokeRenderer::setNumberOfParticles(uint n_particles)
{
  if(n_particles > this->mMaxParticles)
  {
    //Uhohhh too many particles
    fprintf(stderr, "Sorry increase the number of maxParticles \n");
    this->mNumParticles = this->mMaxParticles;
  }
  else
  {
    this->mNumParticles = n_particles;
  }
}

void SmokeRenderer::setPositions(float *pos)
{
#if 0
	//memcpy(mParticlePos.getHostPtr(), pos, mNumParticles*4*sizeof(float));
	//ParticlePos.copy(GpuArray<float4>::HOST_TO_DEVICE);
#else
    // XXX - why is this so much faster?
    int posVbo = mParticlePos.getVbo();
    glBindBuffer(GL_ARRAY_BUFFER_ARB, posVbo);
    glBufferSubData(GL_ARRAY_BUFFER_ARB, 0, mNumParticles * 4 * sizeof(float), pos);
    glBindBuffer( GL_ARRAY_BUFFER_ARB, 0);
#endif
}

void SmokeRenderer::setPositionsDevice(float *posD)
{
	cudaSetDevice(renderDevID);

    mParticlePos.map();
//  cudaMemcpy(mParticlePos.getDevicePtr(), posD, mNumParticles*4*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpyPeerAsync(mParticlePos.getDevicePtr(), renderDevID, posD, devID, mNumParticles*4*sizeof(float), m_copyStreamPos);
    mParticlePos.unmap();

	cudaSetDevice(devID);
}

void SmokeRenderer::setColors(float *color)
{
	if (!mColorVbo)
	{
		// allocate
		glGenBuffers(1, &mColorVbo);
		glBindBuffer(GL_ARRAY_BUFFER_ARB, mColorVbo);
// 		glBufferData(GL_ARRAY_BUFFER_ARB, mNumParticles * 4 * sizeof(float), color, GL_DYNAMIC_DRAW);
        //Jeroen, I allocate the maximum number of particles
        glBufferData(GL_ARRAY_BUFFER_ARB, mMaxParticles * 4 * sizeof(float), color, GL_DYNAMIC_DRAW);                
	}

	glBindBuffer(GL_ARRAY_BUFFER_ARB, mColorVbo);
	//glBufferData(GL_ARRAY_BUFFER_ARB, mNumParticles * 4 * sizeof(float), color, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER_ARB, 0, mNumParticles * 4 * sizeof(float), color);
	glBindBuffer( GL_ARRAY_BUFFER_ARB, 0);
}

void SmokeRenderer::setColorsDevice(float *colorD)
{
	cudaSetDevice(renderDevID);

 	if (!mColorVbo)
	{
		// allocate
		glGenBuffers(1, &mColorVbo);
		glBindBuffer(GL_ARRAY_BUFFER_ARB, mColorVbo);
// 		glBufferData(GL_ARRAY_BUFFER_ARB, mNumParticles * 4 * sizeof(float), color, GL_DYNAMIC_DRAW);
        //Jeroen, I allocate the maximum number of particles
        glBufferData(GL_ARRAY_BUFFER_ARB, mMaxParticles * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);    
		cutilSafeCall(cudaGLRegisterBufferObject(mColorVbo));
		cutilSafeCall(cudaGLSetBufferObjectMapFlags(mColorVbo, cudaGLMapFlagsWriteDiscard));    // CUDA writes, GL consumes
	}

	
	void *ptr;
	cutilSafeCall(cudaGLMapBufferObject((void **) &ptr, mColorVbo));
//	cudaMemcpy( ptr, colorD, mNumParticles * 4 * sizeof(float), cudaMemcpyDeviceToDevice );
	cudaMemcpyPeerAsync( ptr, renderDevID, colorD, devID, mNumParticles * 4 * sizeof(float), m_copyStreamColor );
	cutilSafeCall(cudaGLUnmapBufferObject(mColorVbo));

	cudaSetDevice(devID);
}

void SmokeRenderer::depthSort()
{
	cudaSetDevice(renderDevID);

    mParticleIndices.map();
    mParticlePos.map();

    float4 modelViewZ = make_float4(m_modelView._array[2], m_modelView._array[6], m_modelView._array[10], m_modelView._array[14]);
    depthSortCUDA(mParticlePos.getDevicePtr(), mParticleDepths.getDevicePtr(), (int *) mParticleIndices.getDevicePtr(), modelViewZ, mNumParticles);

    mParticlePos.unmap();
    mParticleIndices.unmap();

	cudaSetDevice(devID);
}

// draw points from vertex buffer objects
void SmokeRenderer::drawPoints(int start, int count, bool sorted)
{
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, mParticlePos.getVbo());
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);                

    if (mColorVbo) {
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, mColorVbo);
        glColorPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_COLOR_ARRAY);
    }

    if (mVelVbo) {
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, mVelVbo);
        glClientActiveTexture(GL_TEXTURE0);
        glTexCoordPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if (sorted) {
        //glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndexBuffer);
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mParticleIndices.getVbo());
        glDrawElements(GL_POINTS, count, GL_UNSIGNED_INT, (void*) (start*sizeof(unsigned int)) );
        glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    } else {
        glDrawArrays(GL_POINTS, start, count);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    glClientActiveTexture(GL_TEXTURE0);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

// draw points using given shader program
void SmokeRenderer::drawPointSprites(GLSLProgram *prog, int start, int count, bool shadowed, bool sorted)
{
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);  // don't write depth
    glEnable(GL_BLEND);

    prog->enable();
    prog->setUniform1f("pointRadius", mParticleRadius);
	prog->setUniform1f("ageScale", m_ageScale);
	prog->setUniform1f("dustAlpha", m_dustAlpha);
    prog->setUniform1f("overBright", m_overBright);
    //prog->setUniform1f("overBrightThreshold", m_overBrightThreshold);
	prog->setUniform1f("fogDist", m_fog);
    prog->setUniform1f("cullDarkMatter", (float) m_cullDarkMatter);

    //prog->bindTexture("rampTex", m_rampTex, GL_TEXTURE_2D, 0);
    //prog->bindTexture("rampTex", m_rainbowTex, GL_TEXTURE_2D, 0);
    //prog->bindTexture("spriteTex",  m_textureArrayID, GL_TEXTURE_2D_ARRAY_EXT, 1);
	prog->bindTexture("spriteTex",  m_spriteTex, GL_TEXTURE_2D, 1);
	if (shadowed) {
		prog->bindTexture("shadowTex", m_lightTexture[m_srcLightTexture], GL_TEXTURE_2D, 2);
#if USE_MRT
        //prog->setUniform2f("shadowTexScale", m_lightBufferSize / (float) m_imageW, m_lightBufferSize / (float) m_imageH);
#else
        //prog->setUniform2f("shadowTexScale", 1.0f, 1.0f);
#endif
        prog->setUniform1f("indirectAmount", m_indirectAmount);
		prog->setUniform1f("alphaScale", m_spriteAlpha);

	} else {
		prog->setUniform1f("alphaScale", m_shadowAlpha);
        prog->setUniform1f("transmission", m_transmission);
	}

#if MOTION_BLUR==0
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    //prog->setUniform1f("pointScale", mWindowH / mInvFocalLen);
    prog->setUniform1f("pointScale", viewport[3] / mInvFocalLen);

    //glClientActiveTexture(GL_TEXTURE0);
    glActiveTexture(GL_TEXTURE0);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glEnable(GL_POINT_SPRITE_ARB);
#endif

    // draw points
    drawPoints(start, count, sorted);

    prog->disable();

    glDisable(GL_POINT_SPRITE_ARB);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

// calculate vectors for half-angle slice rendering
void SmokeRenderer::calcVectors()
{
    // get model view matrix
    glGetFloatv(GL_MODELVIEW_MATRIX, (float *) m_modelView.get_value());

    // calculate eye space light vector
    m_lightVector = normalize(m_lightPos);
    m_lightPosEye = m_modelView * vec4f(m_lightPos, 1.0);

    m_viewVector = -vec3f(m_modelView.get_row(2));
#if USE_HALF_ANGLE
    // calculate half-angle vector between view and light
    if (dot(m_viewVector, m_lightVector) > 0) {
        m_halfVector = normalize(m_viewVector + m_lightVector);
        m_invertedView = false;
    } else {
        m_halfVector = normalize(-m_viewVector + m_lightVector);
        m_invertedView = true;
    }

    // calculate light view matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(m_lightPos[0], m_lightPos[1], m_lightPos[2], 
              m_lightTarget[0], m_lightTarget[1], m_lightTarget[2],
              0.0, 1.0, 0.0);

    // calculate light projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 1.0, 200.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, (float *) m_lightView.get_value());
    glGetFloatv(GL_PROJECTION_MATRIX, (float *) m_lightProj.get_value());

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
#else
	// camera-aligned slices
	m_halfVector = m_viewVector;
	m_lightView = m_modelView;
    glGetFloatv(GL_PROJECTION_MATRIX, (float *) m_lightProj.get_value());
	m_invertedView = false;
#endif

    // construct shadow matrix
    matrix4f scale;
    scale.set_scale(vec3f(0.5, 0.5, 0.5));
    matrix4f translate;
    translate.set_translate(vec3f(0.5, 0.5, 0.5));

    m_shadowMatrix = translate * scale * m_lightProj * m_lightView * inverse(m_modelView);

    // calc object space eye position
    m_eyePos = inverse(m_modelView) * vec4f(0.0, 0.0, 0.0, 1.0);

    // calc half vector in eye space
    m_halfVectorEye = m_modelView * vec4f(m_halfVector, 0.0);
}

// draw quad for volume rendering
void SmokeRenderer::drawVolumeSlice(int i, bool shadowed)
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	//glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    //glMatrixMode(GL_MODELVIEW);
    //glPushMatrix();
	//glLoadIdentity();

    //glColor4f(1.0, 1.0, 1.0, m_volumeAlpha);
    glColor4f(m_volumeColor[0], m_volumeColor[1], m_volumeColor[2], m_volumeAlpha);

    m_volumeProg->enable();
    m_volumeProg->bindTexture("noiseTex", m_noiseTex, GL_TEXTURE_3D, 0);
	if (shadowed) {
		m_volumeProg->bindTexture("shadowTex", m_lightTexture[m_srcLightTexture], GL_TEXTURE_2D, 1);
    }
    m_volumeProg->setUniform1f("noiseFreq", m_noiseFreq);
    m_volumeProg->setUniform1f("noiseAmp", m_noiseAmp);
    m_volumeProg->setUniform1f("indirectLighting", shadowed ? m_volumeIndirect: 0.0f);
    m_volumeProg->setUniform1f("volumeStart", m_volumeStart);
    m_volumeProg->setUniform1f("volumeWidth", m_volumeWidth);

	float t = i / (float) m_numSlices;
	float z = m_minDepth + (m_maxDepth - m_minDepth) * t;
    drawQuad(0.5f, z);

    m_volumeProg->disable();

    //glPopMatrix();
	glDisable(GL_BLEND);
}

// draw slice of particles from camera view
void SmokeRenderer::drawSlice(int i)
{
#if USE_MRT
    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
    glDrawBuffers(2,buffers);

    glColorMaskIndexedEXT(0, true, true, true, true);
    glColorMaskIndexedEXT(1, false, false, false, false);
#else
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_imageTex[0], GL_COLOR_ATTACHMENT0_EXT);
    //m_fbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
	m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_DEPTH_ATTACHMENT_EXT);
#endif
    glViewport(0, 0, m_imageW, m_imageH);

	if (m_enableVolume) {
		drawVolumeSlice(i, true);
	}

    glColor4f(1.0, 1.0, 1.0, m_spriteAlpha);
    if (m_invertedView) {
        // front-to-back
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE);
    } else {
        // back-to-front
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_ONE, GL_ONE);
    }
    drawPointSprites(m_particleShadowProg, i*m_batchSize, m_batchSize, true, true);
}

// draw slice of particles from light's point of view
void SmokeRenderer::drawSliceLightView(int i)
{
#if USE_MRT
    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
    glDrawBuffers(2, buffers);

    glColorMaskIndexedEXT(0, false, false, false, false);
    glColorMaskIndexedEXT(1, true, true, true, true);
#else
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
    //m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightDepthTexture, GL_DEPTH_ATTACHMENT_EXT);
	m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_DEPTH_ATTACHMENT_EXT);
#endif

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf((GLfloat *) m_lightView.get_value());

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf((GLfloat *) m_lightProj.get_value());

    glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

	if (m_enableVolume) {
		drawVolumeSlice(i, false);
	}

#if COLOR_ATTENUATION
    glColor4f(m_colorOpacity[0], m_colorOpacity[1], m_colorOpacity[2], m_shadowAlpha);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
#else
    glColor4f(1.0, 1.0, 1.0, m_shadowAlpha);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
//    glBlendFunc(GL_ONE, GL_ONE);	// additive
#endif

	drawPointSprites(m_particleProg, i*m_batchSize, m_batchSize, false, true);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// draw slice of particles from light's point of view
// version with anti-aliasing
void SmokeRenderer::drawSliceLightViewAA(int i)
{
#if USE_MRT
    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
    glDrawBuffers(2, buffers);

    glColorMaskIndexedEXT(0, false, false, false, false);
    glColorMaskIndexedEXT(1, true, true, true, true);
#else
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
    //m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightDepthTexture, GL_DEPTH_ATTACHMENT_EXT);
	m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_DEPTH_ATTACHMENT_EXT);
#endif

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf((GLfloat *) m_lightView.get_value());

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf((GLfloat *) m_lightProj.get_value());

    glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

#if 0
    glColor4f(m_colorOpacity[0] * m_shadowAlpha, m_colorOpacity[1] * m_shadowAlpha, m_colorOpacity[2] * m_shadowAlpha, 1.0);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
#else
    glColor4f(1.0, 1.0, 1.0, m_shadowAlpha);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
#endif

	/*
	m_particleProg->enable();
	m_particleProg->setUniform1f("sliceNo", i);
	m_particleProg->setUniformfv("sortVector", &m_halfVector[0], 3, 1);
	m_particleProg->setUniform1f("numParticles", mNumParticles);
	m_particleProg->setUniform1f("numSlices", m_numSlices);
	m_particleProg->bindTexture("positionSampler", mPosBufferTexture, GL_TEXTURE_BUFFER_EXT, 3);
	*/

	float sliceWidth = (m_maxDepth - m_minDepth) / (float) m_numSlices;
	float sliceZ = m_minDepth + (sliceWidth * i);
	//printf("%d: z = %f\n", i, sliceZ);
	m_particleAAProg->enable();
	m_particleAAProg->setUniform1f("sliceZ", sliceZ);
	m_particleAAProg->setUniform1f("invSliceWidth", 1.0f / sliceWidth);

	//drawPointSprites(m_particleProg, i*m_batchSize, m_batchSize, false);

	// render previous and current batch
	int start = (i-1)*m_batchSize;
	int end = start + m_batchSize*2;
	start = max(start, 0);
	end = min(end, (int) mNumParticles);
	drawPointSprites(m_particleAAProg, start, end - start, false, true);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// draw particles as slices with shadowing
void SmokeRenderer::drawSlices()
{
    m_batchSize = mNumParticles / m_numSlices;
    m_srcLightTexture = 0;

    setLightColor(m_lightColor);

    // clear light buffer
    m_fbo->Bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_COLOR_ATTACHMENT1_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_DEPTH_ATTACHMENT_EXT);
	//glClearColor(1.0 - m_lightColor[0], 1.0 - m_lightColor[1], 1.0 - m_lightColor[2], 0.0);
    glClearColor(m_lightColor[0], m_lightColor[1], m_lightColor[2], 0.0);
	//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);	// clear to transparent
    glClear(GL_COLOR_BUFFER_BIT);

    // clear volume image
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_imageTex[0], GL_COLOR_ATTACHMENT0_EXT);
    glClearColor(0.0, 0.0, 0.0, 0.0); 
    glClear(GL_COLOR_BUFFER_BIT);

    drawSkybox(m_cubemapTex);

	/*
	// bind vbo as buffer texture
	glBindTexture(GL_TEXTURE_BUFFER_EXT, mPosBufferTexture);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, mPosVbo);
	*/

#if USE_MRT
    // write to both color attachments
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_imageTex[0], GL_COLOR_ATTACHMENT0_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture], GL_COLOR_ATTACHMENT1_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);

    GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
    glDrawBuffers(2, buffers);
#endif

    glActiveTexture(GL_TEXTURE0);
    glMatrixMode(GL_TEXTURE);
    glLoadMatrixf((GLfloat *) m_shadowMatrix.get_value());

    // render slices
	if (m_numDisplayedSlices > m_numSlices) m_numDisplayedSlices = m_numSlices;

    for(int i=0; i<m_numDisplayedSlices; i++) {
#if 0
        // draw slice from camera view, sampling light buffer
        drawSlice(i);
        // draw slice from light view to light buffer, accumulating shadows
        drawSliceLightView(i);
        if (m_doBlur) {
            blurLightBuffer();
        }
#else
		// opposite order
		if (m_enableAA) {
			drawSliceLightViewAA(i);
		} else {
			drawSliceLightView(i);
		}
        if (m_doBlur) {
            blurLightBuffer();
        }

        drawSlice(i);
#endif
    }

#if USE_MRT
    glColorMaskIndexedEXT(0, true, true, true, true);
    glColorMaskIndexedEXT(1, false, false, false, false);
#endif
    m_fbo->Disable();

    glActiveTexture(GL_TEXTURE0);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
}

// blur light buffer to simulate scattering effects
void SmokeRenderer::blurLightBuffer()
{
    glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

    m_blurProg->enable();
    m_blurProg->setUniform2f("texelSize", 1.0f / (float) m_lightBufferSize, 1.0f / (float) m_lightBufferSize);
    glDisable(GL_DEPTH_TEST);

    for(int i=0; i<m_blurPasses; i++) {
#if 1
        // single pass
        m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[1 - m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);

        m_blurProg->bindTexture("tex", m_lightTexture[m_srcLightTexture], GL_TEXTURE_2D, 0);
		//m_blurProg->setUniform1f("blurRadius", m_blurRadius);
        m_blurProg->setUniform1f("blurRadius", m_blurRadius*(i+1));
		//m_blurProg->setUniform1f("blurRadius", m_blurRadius*powf(2.0f, (float) i));
	    drawQuad();

        m_srcLightTexture = 1 - m_srcLightTexture;
#else
        // separable
        m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[1 - m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
        m_blurProg->bindTexture("tex", m_lightTexture[m_srcLightTexture], GL_TEXTURE_2D, 0);
		//m_blurProg->setUniform1f("blurRadius", m_blurRadius);
		m_blurProg->setUniform1f("blurRadius", m_blurRadius*powf(2.0f, (float) i));
        m_blurProg->setUniform2f("texelSize", 1.0f / (float) m_lightBufferSize, 0.0f);
	    drawQuad();
        m_srcLightTexture = 1 - m_srcLightTexture;

        m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[1 - m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
        m_blurProg->bindTexture("tex", m_lightTexture[m_srcLightTexture], GL_TEXTURE_2D, 0);
        m_blurProg->setUniform2f("texelSize", 0.0f, 1.0f / (float) m_lightBufferSize);
	    drawQuad();
        m_srcLightTexture = 1 - m_srcLightTexture;
#endif
    }
    m_blurProg->disable();
}

// post-process final volume image
void SmokeRenderer::processImage(GLSLProgram *prog, GLuint src, GLuint dest)
{
	m_fbo->Bind();
    m_fbo->AttachTexture(GL_TEXTURE_2D, dest, GL_COLOR_ATTACHMENT0_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_DEPTH_ATTACHMENT_EXT);

    prog->enable();
    prog->bindTexture("tex", src, GL_TEXTURE_2D, 0);

    glDisable(GL_DEPTH_TEST);
	drawQuad();

	prog->disable();
	m_fbo->Disable();
}

// display texture to screen
void SmokeRenderer::displayTexture(GLuint tex, float scale)
{
    m_displayTexProg->enable();
    m_displayTexProg->bindTexture("tex", tex, GL_TEXTURE_2D, 0);
    m_displayTexProg->setUniform1f("scale", scale);
	m_displayTexProg->setUniform1f("gamma", m_gamma);
    drawQuad();
    m_displayTexProg->disable();
}

#define DIAGONAL_STARS 0

void SmokeRenderer::doStarFilter()
{
    glViewport(0, 0, m_imageW, m_imageH);

#if 1
	// threshold
	m_thresholdProg->enable();
	m_thresholdProg->setUniform1f("scale", m_starPower);
	m_thresholdProg->setUniform1f("threshold", m_starThreshold);
	processImage(m_thresholdProg, m_imageTex[0], m_imageTex[3]);
#endif

    // star filter
	// horizontal
	m_starFilterProg->enable();
	m_starFilterProg->setUniform1f("radius", m_starBlurRadius);
#if DIAGONAL_STARS
  m_starFilterProg->setUniform2f("texelSize", 2.0f / (float) m_imageW, 2.0f / (float) m_imageH);	// diagonal
#else
	m_starFilterProg->setUniform2f("texelSize", 2.0f / (float) m_imageW, 0.0f);	// axis aligned
#endif
    m_starFilterProg->bindTexture("kernelTex", m_rainbowTex, GL_TEXTURE_2D, 1);
	processImage(m_starFilterProg, m_imageTex[3], m_imageTex[1]);
	//processImage(m_starFilterProg, m_imageTex[0], m_imageTex[1]);

	// vertical
	m_starFilterProg->enable();
#if DIAGONAL_STARS
	m_starFilterProg->setUniform2f("texelSize", -2.0f / (float) m_imageW, 2.0f / (float) m_imageH);	// diagonal
#else
	m_starFilterProg->setUniform2f("texelSize", 0.0f, 2.0f / (float) m_imageW);	// axis aligned
#endif
    processImage(m_starFilterProg, m_imageTex[3], m_imageTex[2]);
	//processImage(m_starFilterProg, m_imageTex[0], m_imageTex[2]);
}

void SmokeRenderer::downSample()
{
    // downsample
    glViewport(0, 0, m_downSampledW, m_downSampledH);
    m_downSampleProg->enable();
	//m_downSampleProg->setUniform2f("texelSize", 1.0f / (float) m_imageW, 1.0f / (float) m_imageH);
        processImage(m_downSampleProg, m_imageTex[0], m_downSampledTex[0]);
	m_downSampleProg->disable();
}

// anamorphic flare?
void SmokeRenderer::doFlare()
{
#if 1
	// threshold
	m_thresholdProg->enable();
	m_thresholdProg->setUniform1f("scale", 1.0f);
	m_thresholdProg->setUniform1f("threshold", m_flareThreshold);
	processImage(m_thresholdProg, m_downSampledTex[0], m_downSampledTex[1]);
#endif

#if 1
    m_gaussianBlurProg->enable();
    m_gaussianBlurProg->setUniform1f("radius", m_flareRadius);
	m_gaussianBlurProg->setUniform2f("texelSize", 2.0f / (float) m_downSampledW, 0.0f);
	//m_gaussianBlurProg->setUniform2f("texelSize", 1.0f / (float) m_downSampledW, 0.0f);
    processImage(m_gaussianBlurProg, m_downSampledTex[1], m_downSampledTex[2]);
#else
	m_starFilterProg->enable();
	m_starFilterProg->setUniform1f("radius", m_flareRadius);
	m_starFilterProg->setUniform2f("texelSize", 2.0f / (float) m_downSampledW, 0.0f);	// axis aligned
    m_starFilterProg->bindTexture("kernelTex", m_rainbowTex, GL_TEXTURE_2D, 1);
	processImage(m_starFilterProg, m_downSampledTex[1], m_downSampledTex[2]);
#endif
}

void SmokeRenderer::doGlowFilter()
{
    // blur
    m_gaussianBlurProg->enable();
    m_gaussianBlurProg->setUniform1f("radius", m_glowRadius);
	m_gaussianBlurProg->setUniform2f("texelSize", 2.0f / (float) m_downSampledW, 0.0f);
	//m_gaussianBlurProg->setUniform2f("texelSize", 1.0f / (float) m_downSampledW, 0.0f);
    processImage(m_gaussianBlurProg, m_downSampledTex[0], m_downSampledTex[1]);

    m_gaussianBlurProg->enable();
    m_gaussianBlurProg->setUniform2f("texelSize", 0.0f, 2.0f / (float) m_downSampledH);
	//m_gaussianBlurProg->setUniform2f("texelSize", 0.0f, 1.0f / (float) m_downSampledH);

    processImage(m_gaussianBlurProg, m_downSampledTex[1], m_downSampledTex[0]);
    m_gaussianBlurProg->disable();
}

// composite final volume image on top of scene
void SmokeRenderer::compositeResult()
{
    if (m_enableFilters) {
	    if (m_starBlurRadius > 0.0f && m_starIntensity > 0.0f) {
            doStarFilter();
	    }

        if (m_glowIntensity > 0.0f || m_flareIntensity > 0.0f) {
          downSample();
        }
        if (m_flareIntensity > 0.0f) {
          doFlare();
        }
        if (m_glowRadius > 0.0f && m_glowIntensity > 0.0f) {
            doGlowFilter();
        }
    }

    glViewport(0, 0, mWindowW, mWindowH);
    glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable(GL_BLEND);

	if (m_enableFilters) {
		m_compositeProg->enable();
		m_compositeProg->bindTexture("tex", m_imageTex[0], GL_TEXTURE_2D, 0);
		m_compositeProg->bindTexture("blurTexH", m_imageTex[1], GL_TEXTURE_2D, 1);
		m_compositeProg->bindTexture("blurTexV", m_imageTex[2], GL_TEXTURE_2D, 2);
		m_compositeProg->bindTexture("glowTex", m_downSampledTex[0], GL_TEXTURE_2D, 3);
        m_compositeProg->bindTexture("flareTex", m_downSampledTex[2], GL_TEXTURE_2D, 4);
		m_compositeProg->setUniform1f("scale", m_imageBrightness);
		m_compositeProg->setUniform1f("sourceIntensity", m_sourceIntensity);
		m_compositeProg->setUniform1f("glowIntensity", m_glowIntensity);
		m_compositeProg->setUniform1f("starIntensity", m_starIntensity);
		m_compositeProg->setUniform1f("flareIntensity", m_flareIntensity);
		m_compositeProg->setUniform1f("gamma", m_gamma);
		drawQuad();
		m_compositeProg->disable();
	} else {
		displayTexture(m_imageTex[0], m_imageBrightness);
        //displayTexture(m_downSampledTex[0], m_imageBrightness);
	}

    glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);
}

void SmokeRenderer::drawBounds()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glPushMatrix();
	glTranslatef(0.0f, 0.0f, m_minDepth);
	glColor3f(0.0f, 1.0f, 0.0f);
	drawQuad();
	glPopMatrix();

	glPushMatrix();
	glTranslatef(0.0f, 0.0f, m_maxDepth);
	glColor3f(1.0f, 0.0f, 0.0f);
	drawQuad();
	glPopMatrix();

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glPopMatrix();
}

void SmokeRenderer::renderSprites(bool sort)
{
#if 1
	// post
    m_fbo->Bind();
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_imageTex[0], GL_COLOR_ATTACHMENT0_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_DEPTH_ATTACHMENT_EXT);
	//m_fbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
    glViewport(0, 0, m_imageW, m_imageH);
    glClearColor(0.0, 0.0, 0.0, 0.0); 
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_BLEND);
    drawSkybox(m_cubemapTex);
#endif

	glColor4f(1.0, 1.0, 1.0, m_spriteAlpha);
    if (sort) {
	    calcVectors();
	    depthSort();
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	    drawPointSprites(m_particleProg, 0, mNumParticles, false, true);	
    } else {
        glBlendFunc(GL_ONE, GL_ONE);
	    drawPointSprites(m_particleProg, 0, mNumParticles, false, false);
    }

#if 1
    m_fbo->Disable();
	compositeResult();
#endif
}

void SmokeRenderer::render()
{
	switch(mDisplayMode) {
	case POINTS:
		glPointSize(2.0f);
		glEnable(GL_DEPTH_TEST);
		glColor4f(1.0, 1.0, 1.0, 1.0f);
		m_simpleProg->enable();
		drawPoints(0, mNumParticles, false);
		m_simpleProg->disable();
        glPointSize(1.0f);
		break;

	case SPRITES:
		renderSprites(false);
		break;

    case SPRITES_SORTED:
        renderSprites(true);
        break;

	case VOLUMETRIC:
		calcVectors();
		depthSort();
		drawSlices();
		compositeResult();
		//drawBounds();
		break;

	case NUM_MODES:
		break;
	}

    if (m_displayLightBuffer) {
        // display light buffer to screen
        glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);
        glDisable(GL_DEPTH_TEST);
        displayTexture(m_lightTexture[m_srcLightTexture], 1.0f);
        glViewport(0, 0, mWindowW, mWindowH);
    }

    //glutReportErrors();
}

// render scene depth to texture
// (this is to ensure that particles are correctly occluded in the low-resolution render buffer)
void SmokeRenderer::beginSceneRender(Target target)
{
    m_fbo->Bind();
    if (target == LIGHT_BUFFER) {
        m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
        m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightDepthTexture, GL_DEPTH_ATTACHMENT_EXT);

        glViewport(0, 0, m_lightBufferSize, m_lightBufferSize);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf((GLfloat *) m_lightView.get_value());

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf((GLfloat *) m_lightProj.get_value());
    } else {
        m_fbo->AttachTexture(GL_TEXTURE_2D, m_imageTex[0], GL_COLOR_ATTACHMENT0_EXT);
        m_fbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);

        glViewport(0, 0, m_imageW, m_imageH);
    }
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void SmokeRenderer::endSceneRender(Target target)
{
    m_fbo->Disable();
    if (target == LIGHT_BUFFER) {
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }
    glViewport(0, 0, mWindowW, mWindowH);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
}

// create an OpenGL texture
GLuint
SmokeRenderer::createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format, void *data)
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

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

inline float sfrand()
{
    return frand()*2.0f-1.0f;
}

GLuint SmokeRenderer::createNoiseTexture(int w, int h, int d)
{
	int size = w*h*d;
    float *data = new float [size];
    float *ptr = data;
    for(int i=0; i<size; i++) {
        *ptr++ = sfrand();
        //*ptr++ = sfrand();
        //*ptr++ = sfrand();
        //*ptr++ = sfrand();
    }

    GLuint texid;
    glGenTextures(1, &texid);
	GLenum target = GL_TEXTURE_3D;
    glBindTexture(target, texid);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE16F_ARB, w, h, d, 0, GL_LUMINANCE, GL_FLOAT, data);
	delete [] data;

	return texid;
}

float * SmokeRenderer::createSplatImage(int n)
{
	float *img = new float[n*n];
	for(int y=0; y<n; y++) {
		float v = (y / (float) (n-1))*2.0f-1.0f;
		for(int x=0; x<n; x++) {
			float u = (x / (float) (n-1))*2.0f-1.0f;
			float d = sqrtf(u*u + v*v);
			if (d > 1.0f) d = 1.0f;
			float i = 1.0f - d*d*(3.0f - 2.0f*d);	// smoothstep
			img[y*n+x] = i;
		}
	}
	return img;
}

GLuint SmokeRenderer::createSpriteTexture(int size)
{
	float *img = createSplatImage(size);

	GLuint tex = createTexture(GL_TEXTURE_2D, size, size, GL_LUMINANCE8, GL_LUMINANCE, img);
	delete [] img;
	glGenerateMipmapEXT(GL_TEXTURE_2D);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	return tex;
}

// create textures for off-screen rendering
void SmokeRenderer::createBuffers(int w, int h)
{
    if (m_imageTex[0]) {
        glDeleteTextures(4, m_imageTex);
        glDeleteTextures(1, &m_depthTex);

        glDeleteTextures(3, m_downSampledTex);
    }

    mWindowW = w;
    mWindowH = h;

    m_imageW = w / m_downSample;
    m_imageH = h / m_downSample;
	printf("image size: %d %d\n", m_imageW, m_imageH);

    // create texture for image buffer
	GLint format = GL_RGBA16F_ARB;
	//GLint format = GL_LUMINANCE16F_ARB;
	//GLint format = GL_RGBA8;
    m_imageTex[0] = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
    m_imageTex[1] = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
    m_imageTex[2] = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
    m_imageTex[3] = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);

    m_depthTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    m_downSampledW = m_imageW / m_blurDownSample;
    m_downSampledH = m_imageH / m_blurDownSample;
    printf("downsampled size: %d %d\n", m_downSampledW, m_downSampledH);

    m_downSampledTex[0] = createTexture(GL_TEXTURE_2D, m_downSampledW, m_downSampledH, format, GL_RGBA);
    m_downSampledTex[1] = createTexture(GL_TEXTURE_2D, m_downSampledW, m_downSampledH, format, GL_RGBA);
    m_downSampledTex[2] = createTexture(GL_TEXTURE_2D, m_downSampledW, m_downSampledH, format, GL_RGBA);

    createLightBuffer();
}

// create textures for light buffer
void
SmokeRenderer::createLightBuffer()
{
    if (m_lightTexture[0]) {
        glDeleteTextures(1, &m_lightTexture[0]);
        glDeleteTextures(1, &m_lightTexture[1]);
        glDeleteTextures(1, &m_lightDepthTexture);
    }

    GLint format = GL_RGBA16F_ARB;
    //GLint format = GL_RGBA8;
    //GLint format = GL_LUMINANCE16F_ARB;

#if USE_MRT
    // textures must be same size to be bound to same FBO at same time
    m_lightTextureW = std::max(m_lightBufferSize, m_imageW);
    m_lightTextureH = std::max(m_lightBufferSize, m_imageH);
#else
    m_lightTextureW = m_lightBufferSize;
    m_lightTextureH = m_lightBufferSize;
#endif

    m_lightTexture[0] = createTexture(GL_TEXTURE_2D, m_lightTextureW, m_lightTextureH, format, GL_RGBA);
    // make shadows clamp to light color at edges
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    m_lightTexture[1] = createTexture(GL_TEXTURE_2D, m_lightTextureW, m_lightTextureH, format, GL_RGBA);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    m_lightDepthTexture = createTexture(GL_TEXTURE_2D, m_lightTextureW, m_lightTextureH, GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);

    m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightTexture[m_srcLightTexture], GL_COLOR_ATTACHMENT0_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, 0, GL_COLOR_ATTACHMENT1_EXT);
    m_fbo->AttachTexture(GL_TEXTURE_2D, m_lightDepthTexture, GL_DEPTH_ATTACHMENT_EXT);
    m_fbo->IsValid();
}

void
SmokeRenderer::setLightColor(vec3f c)
{
    m_lightColor = c;

    // set light texture border color
//    GLfloat borderColor[4] = { 1.0 - m_lightColor[0], 1.0 - m_lightColor[1], 1.0 - m_lightColor[2], 0.0 };
    GLfloat borderColor[4] = { m_lightColor[0], m_lightColor[1], m_lightColor[2], 0.0 };

    glBindTexture(GL_TEXTURE_2D, m_lightTexture[0]);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindTexture(GL_TEXTURE_2D, m_lightTexture[1]);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void SmokeRenderer::setWindowSize(int w, int h)
{
    mAspect = (float) mWindowW / (float) mWindowH;
    mInvFocalLen = (float) tan(mFov*0.5*NV_PI/180.0);

    createBuffers(w, h);
}

void SmokeRenderer::drawQuad(float s, float z)
{
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-s, -s, z);
    glTexCoord2f(1.0, 0.0); glVertex3f(s, -s, z);
    glTexCoord2f(1.0, 1.0); glVertex3f(s, s, z);
    glTexCoord2f(0.0, 1.0); glVertex3f(-s, s, z);
    glEnd();
}

void SmokeRenderer::drawVector(vec3f v)
{
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3fv((float *) &v[0]);
    glEnd();
}

// render vectors to screen for debugging
void SmokeRenderer::debugVectors()
{
    glColor3f(1.0, 1.0, 0.0);
    drawVector(m_lightVector);

    glColor3f(0.0, 1.0, 0.0);
    drawVector(m_viewVector);

    glColor3f(0.0, 0.0, 1.0);
    drawVector(-m_viewVector);

    glColor3f(1.0, 0.0, 0.0);
    drawVector(m_halfVector);
}

void SmokeRenderer::drawSkybox(GLuint tex)
{
    if (!m_cubemapTex)
      return;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    m_skyboxProg->enable();
    m_skyboxProg->bindTexture("tex", tex, GL_TEXTURE_CUBE_MAP, 0);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glColor3f(m_skyboxBrightness, m_skyboxBrightness, m_skyboxBrightness);
    //glColor3f(0.25f, 0.25f, 0.25f);
    //glColor3f(0.5f, 0.5f, 0.5f);
    //glColor3f(1.0f, 1.0f, 1.0f);

    glutSolidCube(2.0);

    m_skyboxProg->disable();

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void SmokeRenderer::initParams()
{
    m_params = new ParamListGL("render params");

	  m_params->AddParam(new Param<int>("slices", m_numSlices, 0, 256, 1, &m_numSlices));
	  m_params->AddParam(new Param<int>("displayed slices", m_numDisplayedSlices, 0, 256, 1, &m_numDisplayedSlices));

    m_params->AddParam(new Param<float>("sprite size", mParticleRadius, 0.0f, 2.0f, 0.01f, &mParticleRadius));
    m_params->AddParam(new Param<float>("dust scale", m_ageScale, 0.0f, 50.0f, 0.1f, &m_ageScale));
    m_params->AddParam(new Param<float>("dust alpha", m_dustAlpha, 0.0f, 1.0f, 0.1f, &m_dustAlpha));

    m_params->AddParam(new Param<float>("light color r", m_lightColor[0], 0.0f, 1.0f, 0.01f, &m_lightColor[0]));
    m_params->AddParam(new Param<float>("light color g", m_lightColor[1], 0.0f, 1.0f, 0.01f, &m_lightColor[1]));
    m_params->AddParam(new Param<float>("light color b", m_lightColor[2], 0.0f, 1.0f, 0.01f, &m_lightColor[2]));

#if 0
    m_params->AddParam(new Param<float>("color opacity r", m_colorOpacity[0], 0.0f, 1.0f, 0.01f, &m_colorOpacity[0]));
    m_params->AddParam(new Param<float>("color opacity g", m_colorOpacity[1], 0.0f, 1.0f, 0.01f, &m_colorOpacity[1]));
    m_params->AddParam(new Param<float>("color opacity b", m_colorOpacity[2], 0.0f, 1.0f, 0.01f, &m_colorOpacity[2]));
#endif

    m_params->AddParam(new Param<float>("alpha", m_spriteAlpha, 0.0f, 1.0f, 0.001f, &m_spriteAlpha));
    m_params->AddParam(new Param<float>("shadow alpha", m_shadowAlpha, 0.0f, 1.0f, 0.001f, &m_shadowAlpha));
    m_params->AddParam(new Param<float>("transmission", m_transmission, 0.0f, 1.0f, 0.001f, &m_transmission));
    m_params->AddParam(new Param<float>("indirect lighting", m_indirectAmount, 0.0f, 1.0f, 0.001f, &m_indirectAmount));

#if 0
	// volume stuff
    m_params->AddParam(new Param<float>("volume alpha", m_volumeAlpha, 0.0f, 1.0f, 0.01f, &m_volumeAlpha));
    m_params->AddParam(new Param<float>("volume indirect", m_volumeIndirect, 0.0f, 1.0f, 0.01f, &m_volumeIndirect));

    m_params->AddParam(new Param<float>("volume color r", m_volumeColor[0], 0.0f, 1.0f, 0.01f, &m_volumeColor[0]));
    m_params->AddParam(new Param<float>("volume color g", m_volumeColor[1], 0.0f, 1.0f, 0.01f, &m_volumeColor[1]));
    m_params->AddParam(new Param<float>("volume color b", m_volumeColor[2], 0.0f, 1.0f, 0.01f, &m_volumeColor[2]));
    m_params->AddParam(new Param<float>("volume noise freq", m_noiseFreq, 0.0f, 1.0f, 0.01f, &m_noiseFreq));
    m_params->AddParam(new Param<float>("volume noise amp", m_noiseAmp, 0.0f, 2.0f, 0.01f, &m_noiseAmp));
    m_params->AddParam(new Param<float>("volume start", m_volumeStart, 0.0f, 1.0f, 0.01f, &m_volumeStart));
    m_params->AddParam(new Param<float>("volume width", m_volumeWidth, 0.0f, 1.0f, 0.01f, &m_volumeWidth));
#endif

    m_params->AddParam(new Param<float>("fog", m_fog, 0.0f, 0.1f, 0.001f, &m_fog));

    m_params->AddParam(new Param<float>("over bright multiplier", m_overBright, 0.0f, 100.0f, 1.0f, &m_overBright));
    m_params->AddParam(new Param<float>("over bright threshold", m_overBrightThreshold, 0.0f, 1.0f, 0.001f, &m_overBrightThreshold));
    m_params->AddParam(new Param<float>("image brightness", m_imageBrightness, 0.0f, 10.0f, 0.1f, &m_imageBrightness));
    m_params->AddParam(new Param<float>("image gamma", m_gamma, 0.0f, 2.0f, 0.0f, &m_gamma));

    m_params->AddParam(new Param<float>("blur radius", m_blurRadius, 0.0f, 10.0f, 0.1f, &m_blurRadius));
    m_params->AddParam(new Param<int>("blur passes", m_blurPasses, 0, 10, 1, &m_blurPasses));

    m_params->AddParam(new Param<float>("source intensity", m_sourceIntensity, 0.0f, 1.0f, 0.01f, &m_sourceIntensity));
    m_params->AddParam(new Param<float>("star blur radius", m_starBlurRadius, 0.0f, 100.0f, 1.0f, &m_starBlurRadius));
    m_params->AddParam(new Param<float>("star threshold", m_starThreshold, 0.0f, 10.0f, 0.1f, &m_starThreshold));
    m_params->AddParam(new Param<float>("star power", m_starPower, 0.0f, 100.0f, 0.1f, &m_starPower));
    m_params->AddParam(new Param<float>("star intensity", m_starIntensity, 0.0f, 1.0f, 0.1f, &m_starIntensity));
    m_params->AddParam(new Param<float>("glow radius", m_glowRadius, 0.0f, 100.0f, 1.0f, &m_glowRadius));
    m_params->AddParam(new Param<float>("glow intensity", m_glowIntensity, 0.0f, 1.0f, 0.01f, &m_glowIntensity));
    m_params->AddParam(new Param<float>("flare intensity", m_flareIntensity, 0.0f, 1.0f, 0.01f, &m_flareIntensity));
    m_params->AddParam(new Param<float>("flare threshold", m_flareThreshold, 0.0f, 10.0f, 0.01f, &m_flareThreshold));
    m_params->AddParam(new Param<float>("flare radius", m_flareRadius, 0.0f, 100.0f, 0.01f, &m_flareRadius));

    m_params->AddParam(new Param<float>("skybox brightness", m_skyboxBrightness, 0.0f, 1.0f, 0.01f, &m_skyboxBrightness));

}

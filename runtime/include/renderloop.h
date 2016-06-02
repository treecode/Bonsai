#ifndef _RENDERLOOP_H_
#define _RENDERLOOP_H_

#include "octree.h"
#include "GalaxyStore.h"

extern float TstartGlow;
extern float dTstartGlow;

void initGL(int argc, char** argv, const char *fullScreenMode, bool &stereo);
void initAppRenderer(int argc, char** argv, octree *tree, 
                     octree::IterationData &idata, bool showFPS, bool stereo,
                     GalaxyStore const& galaxyStore, int wogPort,
                     real wogCameraDistance, real wogDeletionRadiusFactor);

#endif // _RENDERLOOP_H_

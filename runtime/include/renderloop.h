#ifndef _RENDERLOOP_H_
#define _RENDERLOOP_H_

#include "octree.h"

extern float TstartGlow;
extern float dTstartGlow;

void initGL(int argc, char** argv, const char *gameModeString, bool &stereo, bool fullscreen);

void initAppRenderer(int argc, char** argv, octree *tree, 
                     octree::IterationData &idata, bool showFPS, bool stereo,
                     std::string const& wogPath, int wogPort,
                     real wogCameraDistance, real wogDeletionRadiusFactor);

#endif // _RENDERLOOP_H_

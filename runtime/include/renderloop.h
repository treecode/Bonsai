#ifndef _RENDERLOOP_H_
#define _RENDERLOOP_H_

#include "octree.h"
extern float TstartGlow;

void initGL(int argc, char** argv, const char *fullScreenMode);
void initAppRenderer(int argc, char** argv, octree *tree, 
                     octree::IterationData &idata,
                     bool showFPS);

#endif // _RENDERLOOP_H_

#ifndef _RENDERLOOP_H_
#define _RENDERLOOP_H_

#include "octree.h"

void initGL(int argc, char** argv, const char *fullScreenMode);
void initAppRenderer(int argc, char** argv, octree *tree, octree::IterationData &idata);

#endif // _RENDERLOOP_H_

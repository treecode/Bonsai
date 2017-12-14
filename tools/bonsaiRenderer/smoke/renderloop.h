#ifndef _RENDERLOOP_H_
#define _RENDERLOOP_H_

#include "RendererData.h"

void initAppRenderer(int argc, char** argv, 
                     RendererData &data,
                     const char *fulleScreenMode = "",
                     const bool stereo = false);
#endif // _RENDERLOOP_H_

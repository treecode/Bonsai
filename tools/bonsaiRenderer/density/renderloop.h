#ifndef _RENDERLOOP_H_
#define _RENDERLOOP_H_

#include "RendererData.h"
#include <functional>

void initAppRenderer(int argc, char** argv, 
                     const int rank, const int nrank, const MPI_Comm &comm,
                     RendererData &data,
                     const char *fulleScreenMode /* = "" */,
                     const bool stereo /* = false */,
                     std::function<void(int)> &updateFunc);
#endif // _RENDERLOOP_H_

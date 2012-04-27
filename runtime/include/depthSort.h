#include "vector_math.h"

extern "C" void initCUDA();
extern "C" void depthSortCUDA(float4 *pos, float *depth, int *indices, float4 modelViewZ, int numParticles);

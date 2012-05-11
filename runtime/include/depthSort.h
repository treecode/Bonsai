#include "vector_math.h"

extern "C" void initCUDA();
extern "C" void depthSortCUDA(float4 *pos, float *depth, int *indices, float4 modelViewZ, int numParticles);
extern "C" void assignColors(float4 *colors, int *ids, int numParticles, 
	float4 color2, float4 color3, float4 color4, 
	float4 starColor, float4 bulgeColor, float4 darkMatterColor, float4 dustColor,
	int m_brightFreq);


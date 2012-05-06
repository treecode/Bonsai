#include "bonsai.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "vector_math.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

// calculate eye-space depth for each particle
KERNEL_DECLARE(calcDepthKernel)(float4 *pos, float *depth, int *indices, float4 modelViewZ, int numParticles)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= numParticles) return;

	float4 p = pos[i];
	float z = dot(make_float4(p.x, p.y, p.z, 1.0f), modelViewZ);
	
	depth[i] = z;
	indices[i] = i;
}

void thrustSort(float* keys, int* values, int count)
{
    thrust::device_ptr<float> dkeys(keys);
    thrust::device_ptr<int> dvalues(values);
    thrust::sort_by_key(dkeys, dkeys + count, dvalues);
}

extern "C"
void initCUDA()
{
    cudaGLSetGLDevice(0);
}

extern "C"
void depthSortCUDA(float4 *pos, float *depth, int *indices, float4 modelViewZ, int numParticles)
{
	int numThreads = 256;
	int numBlocks = (numParticles + numThreads - 1) / numThreads;
    calcDepthKernel<<< numBlocks, numThreads >>>(pos, depth, indices, modelViewZ, numParticles);

	thrustSort(depth, indices, numParticles);
}
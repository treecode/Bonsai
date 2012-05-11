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

// integer hash function (credit: rgba/iq)
  __device__
  int ihash(int n)
  {
      n=(n<<13)^n;
      return (n*(n*n*15731+789221)+1376312589) & 0x7fffffff;
  }

  // returns random float between 0 and 1
  __device__
  float frand(int n)
  {
	  return ihash(n) / 2147483647.0f;
  }

__global__
void assignColorsKernel(float4 *colors, int *ids, int numParticles, 
	float4 color2, float4 color3, float4 color4, 
	float4 starColor, float4 bulgeColor, float4 darkMatterColor, float4 dustColor,
	int m_brightFreq)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= numParticles ) return;

	int id =  ids[tid];

	float r = frand(id);
	//float4 color = { r, 1-r, 0.5f, 1.0f };
	//float4 color = { 1.0f, 0.0f, 0.0f, 1.0f };

	float4 color;

      if (id >= 0 && id < 40000000)     //Disk
      {
        color = ((id % m_brightFreq) != 0) ? 
        starColor :
        ((id / m_brightFreq) & 1) ? color2 : color3;
      } else if (id >= 40000000 && id < 50000000)     // Glowing stars in spiral arms
      {
        color = ((id%4) == 0) ? color4 : color3;
      }
				else if (id >= 50000000 && id < 70000000) //Dust
				{
					color = dustColor * make_float4(r, r, r, 1.0f);
				} 
				else if (id >= 70000000 && id < 100000000) // Glow massless dust particles
				{
					color = color3;  /*  adds glow in purple */
				}
      else if (id >= 100000000 && id < 200000000) //Bulge
      {
		  //colors[i] = starColor;
        color = bulgeColor;
	  } 
      else //>= 200000000, Dark matter
      {
        color = darkMatterColor;
		  //colors[i] = darkMatterColor * make_float4(r, r, r, 1.0f);
      }            
      
  
	colors[tid] = color;
}

extern "C"
void assignColors(float4 *colors, int *ids, int numParticles, 
	float4 color2, float4 color3, float4 color4, 
	float4 starColor, float4 bulgeColor, float4 darkMatterColor, float4 dustColor,
	int m_brightFreq)
{
	int numThreads = 256;
	int numBlocks = (numParticles + numThreads - 1) / numThreads;
    assignColorsKernel<<< numBlocks, numThreads >>>(colors, ids, numParticles, 
		color2, color3, color4, starColor, bulgeColor, darkMatterColor, dustColor, m_brightFreq);
}
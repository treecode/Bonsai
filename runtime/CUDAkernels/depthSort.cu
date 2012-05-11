#include "bonsai.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "vector_math.h"
#include <cassert>

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

#if 0  /* Simon's code */
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
#else  /* Ev's code :) */

class StarSampler
{
	private:
		float slope;
		float slope1;
		float slope1inv;
		float Mu_lo;
		float C;
		int   N;
		float *Masses;

	public:

		__device__ StarSampler(const float _N, float *_Masses, const float _slope = -2.35) : slope(_slope)
	{
		Masses = _Masses;
		N      = _N;
		const float Mhi = Masses[0];
		const float Mlo = Masses[N-1];
		slope1    = slope + 1.0f;
		assert(slope1 != 0.0f);
	  slope1inv	= 1.0f/slope1;

		Mu_lo = __powf(Mlo, slope1);
		C = (powf(Mhi, slope1) - powf(Mlo, slope1));
	}

		__device__ float sampleMass(const int id)  const
		{
			const float Mu = C*frand(id) + Mu_lo;
			assert(Mu > 0.0);
			const float M   = __powf(Mu, slope1inv);
			return M;
		}

		__device__ int getColour(const float M) const
		{
			int beg = 0;
			int end = N;
			int mid = (beg + end) >> 1;
			while (end - beg > 1)
			{
				if (Masses[mid] > M)
					beg = mid;
				else 
					end = mid;
				mid = (beg + end) >> 1;
			}

			return mid;
		}
};

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

	const int N = 7;
	const float4 Colours[N] = 
	{  /* colours for different spectral classes: Oh Be A Fine Girl Kiss Me */
		make_float4(189.0f, 188.0f, 239.0, 1.0f),  /* O-star */
		make_float4(203.0f, 214.0f, 228.0, 1.0f),  /* B-star */
		make_float4(210.0f, 211.0f, 206.0, 1.0f),  /* A-star */
		make_float4(229.0f, 219.0f, 169.0, 1.0f),  /* F-star */
		make_float4(215.0f, 211.0f, 125.0, 1.0f),  /* G-star, Sun-like */
		make_float4(233.0f, 187.0f, 116.0, 1.0f),  /* K-star */
		make_float4(171.0f,  49.0f,  57.0, 1.0f)   /* M-star, red-dwarfs */
	};
	float Masses[N+1] =
	{  /* masses for each of the spectra type */
		/* O     B    A    F    G    K     M */
		150.0f, 18.0f, 3.2f, 1.7f, 1.1f, 0.78f, 0.47f, 0.1f
	};
	const float slope_disk = 0.35f;
	const float slope_glow = 1.35f;
	StarSampler sDisk(N, Masses, slope_disk);
	StarSampler sGlow(N, Masses, slope_glow);

	float4 color;

	if (id >= 0 && id < 40000000)     //Disk
	{
		color = ((id % m_brightFreq) != 0) ? 
			starColor :
			((id / m_brightFreq) & 1) ? color2 : color3;

		const float  Mstar = sDisk.sampleMass(id);
		const float4 Cstar = Colours[sDisk.getColour(Mstar)];
		const float fdim = 0.5f;
		color = ((id & 1023) == 0) ?   /* one in 1000 stars glows a bit */
			              make_float4(Cstar.x*fdim,  Cstar.y*fdim,  Cstar.z*fdim,  Cstar.w) : 
			(0) ? color : make_float4(Cstar.x*0.01f, Cstar.y*0.01f, Cstar.z*0.01f, Cstar.w);
	} else if (id >= 40000000 && id < 50000000)     // Glowing stars in spiral arms
	{
		color = ((id%4) == 0) ? color4 : color3;
#if 1
		/* sample colours form the MF */
		const float  Mstar = sGlow.sampleMass(id);
		const float4 Cstar = Colours[sGlow.getColour(Mstar)];
		color = Cstar;
#endif
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
#endif

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

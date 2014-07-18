#include "bonsai.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "vector_math.h"
#include <cassert>

#ifdef USE_OPENGL
  #include <GL/glew.h>
  #include <cuda_gl_interop.h>
#endif

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
  #ifdef USE_OPENGL
    cudaGLSetGLDevice(0);
  #endif
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
		float4 *Colours;

	public:

		__device__ StarSampler(
				const float _N, 
				float  *_Masses, 
				float4 *_Colours,
				const float _slope = -2.35) : 
			slope(_slope), N(_N), Masses(_Masses), Colours(_Colours)
	{
		const float Mhi = Masses[0];
		const float Mlo = Masses[N-1];
		slope1    = slope + 1.0f;
//		assert(slope1 != 0.0f);
	  slope1inv	= 1.0f/slope1;

		Mu_lo = __powf(Mlo, slope1);
		C = (powf(Mhi, slope1) - powf(Mlo, slope1));
	}

		__device__ float sampleMass(const int id)  const
		{
			const float Mu = C*frand(id) + Mu_lo;
//			assert(Mu > 0.0);
			const float M   = __powf(Mu, slope1inv);
			return M;
		}

		__device__ float4 getColour(const float M) const
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

			return Colours[mid];
		}
};

KERNEL_DECLARE(assignColorsKernel) (float4 *colors, ulonglong1 *ids, 
		int numParticles,
		float2 *density, float maxDensity,
		float4 color2, float4 color3, float4 color4, 
		float4 starColor, float4 bulgeColor, float4 darkMatterColor, float4 dustColor,
		int m_brightFreq, float4 t_current)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= numParticles ) return;

	unsigned long long id =  ids[tid].x;

	float r = frand(id);
	//float4 color = { r, 1-r, 0.5f, 1.0f };
	//float4 color = { 1.0f, 0.0f, 0.0f, 1.0f };

#if 0
	const int N = 7;
	float4 Colours[N] = 
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
#else
	const int N = 16;
	float4 Colours[N] = 
	{  /* colours for different spectral classes: Oh Be A Fine Girl Kiss Me */
		make_float4( 32.0f,  78.0f, 255.0f, 1.0f),  /* O0 */
		make_float4( 62.0f, 108.0f, 255.0f, 1.0f),  /* O5 */
		make_float4( 68.0f, 114.0f, 255.0f, 1.0f),  /* B0 */
		make_float4( 87.0f, 133.0f, 255.0f, 1.0f),  /* B5 */
		make_float4(124.0f, 165.0f, 255.0f, 1.0f),  /* A0 */
		make_float4(156.0f, 189.0f, 255.0f, 1.0f),  /* A5 */
		make_float4(177.0f, 204.0f, 255.0f, 1.0f),  /* F0 */
		make_float4(212.0f, 228.0f, 255.0f, 1.0f),  /* F5 */
		make_float4(237.0f, 244.0f, 255.0f, 1.0f),  /* G0 */
		make_float4(253.0f, 254.0f, 255.0f, 1.0f),  /* G2 -- the Sun */
		make_float4(255.0f, 246.0f, 233.0f, 1.0f),  /* G5 */
		make_float4(255.0f, 233.0f, 203.0f, 1.0f),  /* K0 */
		make_float4(255.0f, 203.0f, 145.0f, 1.0f),  /* K5 */
		make_float4(255.0f, 174.0f,  98.0f, 1.0f),  /* M0 */
		make_float4(255.0f, 138.0f,  56.0f, 1.0f),  /* M5 */
		make_float4(240.0f,   0.0f,   0.0f, 1.0f)   /* M8 */
	};
	float Masses[N+1] =
	{  /* masses for each of the spectra type */
		150.0f, 40.0f, 18.0f, 6.5f, 3.2f, 2.1f, 1.7f, 1.29f, 1.1f, 1.0f, 0.93f, 0.78f, 0.69f, 0.47f, 0.21f, 0.1f, 0.05f
	};
#endif

#if 0 /* use ad'hpc MF to sample stellar colours */

	float slope_disk  = -2.35f;  /* salpeter MF */
	float slope_glow  = -2.35f;
	float slope_bulge = -1.35f;
#if 1
	slope_disk  = +0.1;  /* gives disk stars nice Blue tint */
	slope_glow  = +0.1;  /* give  glowing stars blue ting as well*/
	slope_bulge = -1.35; /* bulge remains yellowish */
#endif
	StarSampler sDisk (N, Masses, Colours, slope_disk -1);
	StarSampler sGlow (N, Masses, Colours, slope_glow -1);
	StarSampler sBulge(N, Masses, Colours, slope_bulge-1);

#else /* use realistic mass & luminosity functions to sample stars */

	const float MoL_bulge   = 4.0;         /* mass-to-light ratio */
	const float MoL_disk    = 3.5;         
	const float MoL_glow    = 1.0;
	const float slope_bulge = -1.35f + MoL_bulge;
	const float slope_disk  = -1.35f + MoL_disk;  /* salpeter MF slope + MoL to get light distribution function*/
	const float slope_glow  = -1.35f + MoL_glow;
	StarSampler sBulge(7,  Masses+10, Colours+10, slope_bulge-1);  /* only include old GKM stars */
	StarSampler sDisk (15, Masses+2,  Colours+2,  slope_disk -1);  /* limit only to BAFGKM stars */
	StarSampler sGlow (6,  Masses,    Colours,    slope_glow -1);  /* only OBA stars */

#endif

	float4 color;

	if (id < 40000000)     //Disk
	{
		color = ((id % m_brightFreq) != 0) ? 
			starColor :
			((id / m_brightFreq) & 1) ? color2 : color3;

		const float  Mstar = sDisk.sampleMass(id);
		const float4 Cstar = sDisk.getColour(Mstar);
#if 0
		const float fdim = 1.0;
		color = ((id & 1023) == 0) ?   /* one in 1000 stars glows a bit */
			              make_float4(Cstar.x*fdim,  Cstar.y*fdim,  Cstar.z*fdim,  Cstar.w) : 
			(0) ? color : make_float4(Cstar.x*0.01f, Cstar.y*0.01f, Cstar.z*0.01f, Cstar.w);
#else
		color = ((id & 1023) == 0) ?   /* one in 1000 stars glows a bit */
			              sGlow.getColour(sGlow.sampleMass(id)) :
			(0) ? color : make_float4(Cstar.x*0.01f, Cstar.y*0.01f, Cstar.z*0.01f, Cstar.w);
#endif
	} else if (id >= 40000000 && id < 50000000)     // Glowing stars in spiral arms
	{
		color = ((id%4) == 0) ? color4 : color3;
#if 1
		/* sample colours form the MF */
		const float  Mstar = sGlow.sampleMass(id);
		const float4 Cstar = sGlow.getColour(Mstar);
		color = Cstar;

		//We need to tune this parameter, this disabled glowing stars uptill a certain time
		color.x *= t_current.z;
		color.y *= t_current.z;
		color.z *= t_current.z;
		color.w  = t_current.w;
#if 0
		if(t_current.x < t_current.y)    
			color.w = 3.0f;
#endif
#endif
	}
	else if (id >= 50000000 && id < 70000000) //Dust
	{
		color = dustColor * make_float4(r, r, r, 1.0f);
	} 
	else if (id >= 70000000 && id < 2000000000000000000) // Glow massless dust particles
	//else if (id >= 70000000 && id < 100000000) // Glow massless dust particles
	{
		color = color3;  /*  adds glow in purple */
	}
	else if (id >= 2000000000000000000 && id < 3000000000000000000) //Bulge
	//else if (id >= 100000000 && id < 200000000) //Bulge
	{
		//colors[i] = starColor;
		color = bulgeColor;
#if 1
		const float  Mstar = sBulge.sampleMass(id);
		const float4 Cstar = sBulge.getColour(Mstar);
		const float fdim = 0.01f;
		color = Cstar * make_float4(fdim, fdim, fdim, 2.0f);
#endif
	} 
	else //>= 3000000000000000000, Dark matter
	{
		color = darkMatterColor;
		//colors[i] = darkMatterColor * make_float4(r, r, r, 1.0f);
	}            


	colors[tid] = color;

	//Density hack, turn particles with density less than
	//some limit into dark-matter (disables rendering)
	//float tempDens = log10(density[tid].x);
	float tempDens = density[tid].x;
//	tempDens /= maxDensity;
//	int densTest = (int) tempDens* 100; //percentage
//	if(tid == 0) printf("Limit: %d cur: %f  %f\n",	densLimit, density[tid].x, maxDensity);

	if(log10(tempDens) < maxDensity)
	{
		colors[tid].w = 3.0f;
	}



}
#endif

	extern "C"
void assignColors(float4 *colors, ulonglong1 *ids, int numParticles,
		float2 *density, float maxDensity, 
		float4 color2, float4 color3, float4 color4, 
		float4 starColor, float4 bulgeColor, float4 darkMatterColor, float4 dustColor,
		int m_brightFreq, float4  t_current)
{
	int numThreads = 256;
	int numBlocks = (numParticles + numThreads - 1) / numThreads;
	assignColorsKernel<<< numBlocks, numThreads >>>(colors, ids, numParticles,
		        density, maxDensity, 	
			color2, color3, color4, starColor, bulgeColor, darkMatterColor, dustColor, m_brightFreq, t_current);
}

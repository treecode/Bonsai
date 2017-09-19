#include <cassert>
#include <algorithm>
#include <parallel/algorithm>
#include <vector>
#include "vector_math.h"


#if 0
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
#endif

typedef std::pair<float,int> KeyVal_t;
struct Cmp
{
  bool operator()(const KeyVal_t &lhs, const KeyVal_t &rhs)
  {
    return lhs.first < rhs.first;
  }
};
void sort_by_key(float* keys, int* values, int count)
{
  std::vector<KeyVal_t> pairs(count);
#pragma omp parallel for
  for (int i = 0; i < count; i++)
    pairs[i] = std::make_pair(keys[i], values[i]);
#if 1
  __gnu_parallel::sort(pairs.begin(), pairs.end(), Cmp());
#endif
#pragma omp parallel for
  for (int i = 0; i < count; i++)
  {
    keys  [i] = pairs[i].first;
    values[i] = pairs[i].second;
  }
}

extern "C"
void depthSortCUDA(float4 *pos, float *depth, int *indices, float4 modelViewZ, int numParticles)
{
  for (int i = 0; i < numParticles; i++)
  {
    float4 p = pos[i];
    float z = dot(make_float4(p.x, p.y, p.z, 1.0f), modelViewZ);

    depth[i] = z;
    indices[i] = i;
  }
	sort_by_key(depth, indices, numParticles);
}

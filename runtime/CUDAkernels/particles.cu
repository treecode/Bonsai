#include "bonsai.h"
// #include "support_kernels.cu"
#include "../profiling/bonsai_timing.h"
PROF_MODULE(parallel);

#include <stdio.h>
#include "node_specs.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <cassert>
#include <algorithm>


#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/version.h>

#if THRUST_VERSION >=  100700


#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>

//Thrust cached allocator, note this assumes that the passed buffer has sufficient
//size for the requested operations!!!!!!!!!!!!!!
// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator
{

private:
  int memOffset;
  int *currentPointer;

  public:

  typedef char value_type;

    cached_allocator(my_dev::dev_mem<uint> &temporaryBuffer, int startOffset)
    {
      memOffset      = 0;
      currentPointer = (int*)temporaryBuffer.a(startOffset);
    }

    ~cached_allocator(){}
    ///////////
    //Return the number of elements (of type uint) to be padded
    //to get to the correct address boundary
     int getGlobalMemAllignmentPadding2(int n)
    {
      const int allignBoundary = 128*sizeof(uint); //CC 2.X and 3.X ,128 bytes

      int offset = 0;
      //Compute the number of bytes
      offset = n*sizeof(uint);
      //Compute number of allignBoundary byte blocks
      offset = (offset / allignBoundary) + (((offset % allignBoundary) > 0) ? 1 : 0);
      //Compute the number of bytes padded / offset
      offset = (offset * allignBoundary) - n*sizeof(uint);
      //Back to the actual number of elements
      offset = offset / sizeof(uint);

      return offset;
    }


    char *allocate(std::ptrdiff_t num_bytes)
    {
      char *result = (char*)(void*)(size_t)(currentPointer + memOffset);

      //Convert num_bytes to integer offset
      int numIntItems = (int)(num_bytes / sizeof(int));
      numIntItems++;

      //Increase the offset, make sure it is a multiple of predefined number
      int currentOffset = memOffset + numIntItems;
      int padding       = getGlobalMemAllignmentPadding2(currentOffset);
      memOffset         = currentOffset + padding;

//      std::cout << "Allocating: bytes: " << num_bytes << std::endl;
//      std::cout << "Allocating: ints : " << numIntItems<< std::endl;
//      std::cout << "memOffset: " <<  memOffset  << std::endl;
//      std::cout << "currentOffset: " << currentOffset<< std::endl;
//      std::cout << "padding: " << padding << std::endl;

      return result;
    }

    void deallocate(char *ptr, size_t n) {}
};

#endif


struct isInOurDomain
{
  __host__ __device__
  bool operator()(const uint2 &val)
  {
    return (val.x >> 31);
  }
};

//struct isInOurDomain2
//{
//  __host__ __device__
//  bool operator()(const uint &val)
//  {
//    return (val >> 31);
//  }
//};

// user-defined comparison operator that acts like less<int>,
// except even numbers are considered to be smaller than odd numbers
struct domainCompare
{
  __host__ __device__
  bool operator()(uint2 x, uint2 y)
  {
    return x.x < y.x;
  }
};


 struct domainCompare2 : public binary_function<uint2,uint2,bool>
{
__host__ __device__ bool operator()(const uint2 &lhs, const uint2 &rhs) const {return lhs.x == rhs.x;}
}; // end domainCompare2

#include <sys/time.h>
 double get_time() {

   struct timeval Tvalue;
   struct timezone dummy;

   gettimeofday(&Tvalue,&dummy);
   return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
 }

extern "C" uint2 thrust_partitionDomains( my_dev::dev_mem<uint2> &validList,
                                          my_dev::dev_mem<uint2> &validList2, //Unsorted compacted list
                                          my_dev::dev_mem<uint> &idList,
                                          my_dev::dev_mem<uint2> &outputKeys,
                                          my_dev::dev_mem<uint> &outputValues,
                                          const int N,
                                          my_dev::dev_mem<uint> &generalBuffer,
                                          const int currentOffset)
{
  thrust::device_ptr<uint2> values      = thrust::device_pointer_cast(validList.raw_p());
  thrust::device_ptr<uint>  listofones  = thrust::device_pointer_cast(idList.raw_p());
  thrust::device_ptr<uint2> outKeys     = thrust::device_pointer_cast(outputKeys.raw_p());
  thrust::device_ptr<uint>  outValues   = thrust::device_pointer_cast(outputValues.raw_p());
  thrust::plus<int> binary_op;


#if THRUST_VERSION >=  100700
  cached_allocator alloc(generalBuffer, currentOffset);

  double t1 = get_time();
  //Partition the values by in or out of domain. Result: [[outside],[inside ids]]
  thrust::device_ptr<uint2>  res = thrust::partition(thrust::cuda::par(alloc), values, values + N, isInOurDomain());
  const int remoteParticles      = (int) (res-values);
  double t2 = get_time();

  validList2.copy_devonly(validList, remoteParticles); //Copy the list before sorting, needed for internal move

  //Sort the outside our domain particles by their domain index
  //Result: [[ids domain0],[ids domain1], [ids domain2], ...]
#if 0
  //Although it is faster it puts particles in the wrong order because we have marked
  //them with a high bit integer. Also does not work to put domains in right order. So
  //not use this for now. Can look at it at a later point
  unsigned long long *tempPtr = (unsigned long long*)validList.raw_p();
  thrust::device_ptr<unsigned long long> valuesLL      = thrust::device_pointer_cast(tempPtr);
  thrust::stable_sort(thrust::cuda::par(alloc),
                      valuesLL,
                      valuesLL + remoteParticles,
                      thrust::greater<unsigned long long>());
#else
  thrust::stable_sort(thrust::cuda::par(alloc),
                      values,
                      values + remoteParticles,
                      domainCompare());
//  cudaDeviceSynchronize();
#endif
  double t3 = get_time();
  //Reduce the domains. The result is that we get per domain the number of particles
  //that will send to that process. These are stored into the output buffers
  thrust::pair<thrust::device_ptr<uint2>,thrust::device_ptr<uint> > new_end;
  new_end = thrust::reduce_by_key(thrust::cuda::par(alloc),
                                  values,                   //inputIterator1
                                  values + remoteParticles, //InputIterator1
                                  listofones,              //InputIterator2
                                  outKeys,                  //OutputIterator1
                                  outValues,                  //OutputIterator2
                                  domainCompare2(),
                                  binary_op);

#else
  //Partition the values by in or out of domain. Result: [[outside],[inside ids]]
   double t1 = get_time();
  thrust::device_ptr<uint2>  res = thrust::partition(values, values + N, isInOurDomain());
  const int remoteParticles      = (int) (res-values);
  double t2 = get_time();
  validList2.copy_devonly(validList, remoteParticles); //Copy the list before sorting, needed for internal move

  //Sort the outside our domain particles by their domain index
  //Result: [[ids domain0],[ids domain1], [ids domain2], ...]
  thrust::stable_sort(values,  values + remoteParticles, domainCompare());
  //cudaDeviceSynchronize();
  double t3 = get_time();
  //Reduce the domains. The result is that we get per domain the number of particles
  //that will send to that process. These are stored into the output buffers
  thrust::pair<thrust::device_ptr<uint2>,thrust::device_ptr<uint> > new_end;
  new_end = thrust::reduce_by_key(values,                   //inputIterator1
                                  values + remoteParticles, //InputIterator1
                                  listofones,              //InputIterator2
                                  outKeys,                  //OutputIterator1
                                  outValues,                  //OutputIterator2
                                  domainCompare2(),
                                  binary_op);
#endif

  LOGF(stderr,"Sorting detail: N: %d partition: %lg sort: %lg reduce: %lg \n",remoteParticles, t2-t1,t3-t2,get_time()-t3);

  const int nValues = (int)(new_end.first  - outKeys);
 //return the number of remote particles and the number of remote domains
 return make_uint2(remoteParticles, nValues);

}











static __device__ inline int isinbox(real4 pos, double4 xlow, double4 xhigh)
{  
    if((pos.x < xlow.x)||(pos.x > xhigh.x))          
      return 0;
    if((pos.y < xlow.y)||(pos.y > xhigh.y))          
      return 0;
    if((pos.z < xlow.z)||(pos.z > xhigh.z))          
      return 0;
    
    return 1;
}

static __device__ inline int cmp_uint4(uint4 a, uint4 b) {
  if      (a.x < b.x) return -1;
  else if (a.x > b.x) return +1;
  else {
    if       (a.y < b.y) return -1;
    else  if (a.y > b.y) return +1;
    else {
      if       (a.z < b.z) return -1;
      else  if (a.z > b.z) return +1;
      return 0;
    } //end z
  }  //end y
} //end x, function



KERNEL_DECLARE(doDomainCheck)(int    n_bodies,
                                           double4  xlow,
                                           double4  xhigh,
                                           real4  *body_pos,
                                           int    *validList    //Valid is 1 if particle is outside domain
){
  CUXTIMER("doDomainCheck");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;
  
  if (id >= n_bodies) return;

  real4 pos = body_pos[id];

  int valid      = isinbox(pos, xlow, xhigh);
  valid = !valid;
  validList[id] = id | ((valid) << 31);
}
  

//Checks the domain and computes the key list
//if a particle is outside the domain it gets a special key
//otherwise the normal key is used
KERNEL_DECLARE(doDomainCheckAdvanced)(int    n_bodies,
                                           double4  xlow,
                                           double4  xhigh,
                                           real4  *body_pos,
                                           int    *validList    //Valid is 1 if particle is outside domain
){
  CUXTIMER("doDomainCheckAdvanced");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;
  
  if (id >= n_bodies) return;

  real4 pos = body_pos[id];

  int valid      = isinbox(pos, xlow, xhigh);
  valid = !valid;
  validList[id] = id | ((valid) << 31);
}
  

KERNEL_DECLARE(gpu_extractSampleParticles)(int    n_bodies,
                                                  int    sample_freq,
                                                  real4  *body_pos,
                                                  real4  *samplePosition
){
  CUXTIMER("extractSampleParticles");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;
  

  int idx  = id*sample_freq;
  if  (idx >= n_bodies) return;

  samplePosition[id] =  body_pos[idx];
}

KERNEL_DECLARE(gpu_extractSampleParticlesSFC)(int     n_bodies,
                                              int     nSamples,
                                              float   sample_freq,
                                              uint4  *body_pos,
                                              uint4  *samplePosition
){
  CUXTIMER("extractSampleParticles");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if(id >= nSamples) return;

  int idx  = (int)(id*sample_freq);
  if  (idx >= n_bodies) return;

  samplePosition[id] =  body_pos[idx];
}

//KERNEL_DECLARE(extractOutOfDomainParticlesR4)(int n_extract,
//                                                       int *extractList,
//                                                       real4 *source,
//                                                       real4 *destination)
//{
//  CUXTIMER("extractOutOfDomainParticlesR4");
//  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
//  uint tid = threadIdx.x;
//  uint id  = bid * blockDim.x + tid;
//
//  if(id >= n_extract) return;
//
//  destination[id] = source[extractList[id]];
//
//}


typedef struct bodyStruct
{
  real4  acc0;
  real4  Ppos;
  real4  Pvel;
  float2 time;
  unsigned long long id;
//  float    id;
//  int    temp;

#ifdef DO_BLOCK_TIMESTEP_EXCHANGE_MPI
  uint4 key;
  real4 pos;
  real4 vel;
  real4 acc1;
#endif
} bodyStruct;



KERNEL_DECLARE(gpu_internalMove)(int       n_extract,
                                        int       n_bodies,
                                        double4  xlow,
                                        double4  xhigh,
                                        int       *extractList,
                                        int       *indexList,
                                        real4     *Ppos,
                                        real4     *Pvel,
                                        real4     *pos,
                                        real4     *vel,
                                        real4     *acc0,
                                        real4     *acc1,
                                        float2    *time,
                                        int       *body_id)
{
  CUXTIMER("internalMove");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if(id >= n_extract) return;

  int srcIdx     = (n_bodies-n_extract) + id;
  real4 testpos  = Ppos[srcIdx];

  if(isinbox(testpos, xlow, xhigh))
  {
    int dstIdx = atomicAdd(indexList, 1);    
    dstIdx     = extractList[dstIdx];

    //Move!
    Ppos[dstIdx] = Ppos[srcIdx];
    Pvel[dstIdx] = Pvel[srcIdx];
    pos[dstIdx]  = pos[srcIdx];
    vel[dstIdx]  = vel[srcIdx];
    acc0[dstIdx] = acc0[srcIdx];
    acc1[dstIdx] = acc1[srcIdx];
    time[dstIdx] = time[srcIdx];
    body_id[dstIdx] = body_id[srcIdx];
  }//if isinbox

}



//Check if a particles key is within the min and max boundaries
KERNEL_DECLARE(gpu_domainCheckSFC)(int    n_bodies,
                               uint4  lowBoundary,
                               uint4  highBoundary,
                               uint4  *body_key,
                               int    *validList    //Valid is 1 if particle is outside domain
){
  CUXTIMER("domainCheckSFC");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if (id >= n_bodies) return;

  uint4 key = body_key[id];

  int bottom = cmp_uint4(key, lowBoundary);
  int top    = cmp_uint4(key, highBoundary);

  int valid = 0;
  if(bottom >= 0 && top < 0)
  {
    //INside
  }
  else
  {
    //    outside
    valid = 1;
  }
  validList[id] = id | ((valid) << 31);
}


//Binary search of the key within certain bounds (cij.x, cij.y)
//Note this is the same as 'find_key'
static __device__ int find_domain(uint4 key, uint2 cij, uint4 *keys) {
  int l = cij.x;
  int r = cij.y - 1;
  while (r - l > 1) {
    int m = (r + l) >> 1;
    int cmp = cmp_uint4(keys[m], key);
    //if(cmp == 0) return m;
    if(0) {}
    else if (cmp == -1) {
      l = m;
    } else {
      r = m;
    }
  }

  //if (cmp_uint4(keys[r], key) == 0) return r;
  //return l;

  if (cmp_uint4(keys[l], key) >= 0) return l;
  return r;
}
//Check if a particles key is within the min and max boundaries
KERNEL_DECLARE(gpu_domainCheckSFCAndAssign)(int    n_bodies,
                                            int    nProcs,
                                            uint4  lowBoundary,
                                            uint4  highBoundary,
                                            uint4  *boundaryList, //The full list of boundaries
                                            uint4  *body_key,
                                            uint2  *validList,    //Valid is 1 if particle is outside domain,
                                            uint   *idList, int procId
){
  CUXTIMER("domainCheckSFCAndAssign");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if (id >= n_bodies) return;

  uint4 key = body_key[id];

  int bottom = cmp_uint4(key, lowBoundary);
  int top    = cmp_uint4(key, highBoundary);

  uint valid = 0;
  if(bottom >= 0 && top < 0)
  {
    //Inside
//    valid = 0x0;
  }
  else
  {
    //    outside
    //Search the box that this particle belongs to. Note we start at idx[1] that
    //way we get the top-end values of the domain
    uint2 cij;
    cij.x = 0; cij.y = nProcs+1;
    int domain = find_domain(key, cij, &boundaryList[1]);
    //int domain = find_domain(key, cij, &boundaryList[0]);

    if(procId == domain) domain = domain + 1;

    valid = domain | ((1) << 31);
  }

  validList[id] = make_uint2(valid, id);
  idList[id]    = 1;
}

#if 0
//Check if a particles key is within the min and max boundaries
KERNEL_DECLARE(gpu_domainCheckSFCAndAssign)(int    n_bodies,
                                            int    nProcs,
                                            uint4  lowBoundary,
                                            uint4  highBoundary,
                                            uint4  *boundaryList, //The full list of boundaries
                                            uint4  *body_key,
                                            int    *validList    //Valid is 1 if particle is outside domain
){
  CUXTIMER("domainCheckSFCAndAssign");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if (id >= n_bodies) return;

  uint4 key = body_key[id];

  int bottom = cmp_uint4(key, lowBoundary);
  int top    = cmp_uint4(key, highBoundary);

  int valid = 0;
  if(bottom >= 0 && top < 0)
  {
    //Inside
    valid = -1;
  }
  else
  {
    //    outside
    valid = 1;

//    if(id == 4992)
    {
      //Search the box that this particle belongs to
      uint2 cij;
      cij.x = 0; cij.y = nProcs+1;
      int domain = find_domainBox(key, cij, boundaryList);


//      printf("XXX Particles: %d outside, namely in domain: %d \n", id, domain);
      valid = domain;
    }


  }
  //validList[id] = id | ((valid) << 31);
  validList[id] = valid;
}
#endif

KERNEL_DECLARE(gpu_internalMoveSFC2) (int       n_extract,
                                  int       n_bodies,
                                  uint4  lowBoundary,
                                  uint4  highBoundary,
                                  int2       *extractList,
                                  int       *indexList,
                                  real4     *Ppos,
                                  real4     *Pvel,
                                  real4     *pos,
                                  real4     *vel,
                                  real4     *acc0,
                                  real4     *acc1,
                                  float2    *time,
                                  unsigned long long       *body_id,
                                  uint4     *body_key,
				  float *h)
{
  CUXTIMER("internalMoveSFC2");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if(id >= n_extract) return;

  int srcIdx     = (n_bodies-n_extract) + id;


  uint4 key  = body_key[srcIdx];
  int bottom = cmp_uint4(key, lowBoundary);
  int top    = cmp_uint4(key, highBoundary);


  if((bottom >= 0 && top < 0))
  {
    int dstIdx = atomicAdd(indexList, 1);
    dstIdx     = extractList[dstIdx].y;

    //Move!
    Ppos[dstIdx] = Ppos[srcIdx];
    Pvel[dstIdx] = Pvel[srcIdx];
    pos[dstIdx]  = pos[srcIdx];
    vel[dstIdx]  = vel[srcIdx];
    acc0[dstIdx] = acc0[srcIdx];
    acc1[dstIdx] = acc1[srcIdx];
    time[dstIdx] = time[srcIdx];
    body_key[dstIdx] = body_key[srcIdx];
    body_id[dstIdx]  = body_id[srcIdx];
    h[dstIdx]     = h[srcIdx];
  }//if inside

}

KERNEL_DECLARE(gpu_internalMoveSFC) (int       n_extract,
                                  int       n_bodies,
                                  uint4  lowBoundary,
                                  uint4  highBoundary,
                                  int       *extractList,
                                  int       *indexList,
                                  real4     *Ppos,
                                  real4     *Pvel,
                                  real4     *pos,
                                  real4     *vel,
                                  real4     *acc0,
                                  real4     *acc1,
                                  float2    *time,
                                  unsigned long long        *body_id,
                                  uint4     *body_key
				  )
{
  CUXTIMER("internalMoveSFC");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if(id >= n_extract) return;

  int srcIdx     = (n_bodies-n_extract) + id;


  uint4 key  = body_key[srcIdx];
  int bottom = cmp_uint4(key, lowBoundary);
  int top    = cmp_uint4(key, highBoundary);


  if((bottom >= 0 && top < 0))
  {
    int dstIdx = atomicAdd(indexList, 1);
    dstIdx     = extractList[dstIdx];

    //Move!
    Ppos[dstIdx] = Ppos[srcIdx];
    Pvel[dstIdx] = Pvel[srcIdx];
    pos[dstIdx]  = pos[srcIdx];
    vel[dstIdx]  = vel[srcIdx];
    acc0[dstIdx] = acc0[srcIdx];
    acc1[dstIdx] = acc1[srcIdx];
    time[dstIdx] = time[srcIdx];
    body_key[dstIdx] = body_key[srcIdx];
    body_id[dstIdx]  = body_id[srcIdx];
  }//if inside

}

KERNEL_DECLARE(gpu_extractOutOfDomainParticlesAdvancedSFC2)(
                                                       int offset,
                                                       int n_extract,
                                                       uint2 *extractList,
                                                       real4 *Ppos,
                                                       real4 *Pvel,
                                                       real4 *pos,
                                                       real4 *vel,
                                                       real4 *acc0,
                                                       real4 *acc1,
                                                       float2 *time,
                                                       unsigned long long    *body_id,
                                                       uint4 *body_key,
						       float *h,
                                                       bodyStruct *destination)
{
  CUXTIMER("extractOutOfDomainParticlesAdvancedSFC2");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;
#if 0
  //slowest
  if(id >= n_extract) return;

  //copy the data from a struct of arrays into a array of structs
  destination[id].Ppos = Ppos[extractList[offset+id].y];
  destination[id].Pvel = Pvel[extractList[offset+id].y];
  destination[id].pos  = pos[extractList[offset+id].y];
  destination[id].vel  = vel[extractList[offset+id].y];
  destination[id].acc0  = acc0[extractList[offset+id].y];
  destination[id].acc1  = acc1[extractList[offset+id].y];
  destination[id].time  = time[extractList[offset+id].y];
  destination[id].id    = body_id[extractList[offset+id].y];
  destination[id].key   = body_key[extractList[offset+id].y];

#elif 1
  //5x faster than original (above)
  __shared__ bodyStruct shmem[128];

  if((bid * blockDim.x) >= n_extract) return;

  if(id < n_extract)
  {
    shmem[threadIdx.x].Ppos  = Ppos[extractList[offset+id].y];
    shmem[threadIdx.x].Pvel  = Pvel[extractList[offset+id].y];
    shmem[threadIdx.x].acc0  = acc0[extractList[offset+id].y];
    shmem[threadIdx.x].time  = time[extractList[offset+id].y];

    shmem[threadIdx.x].id    = body_id[extractList[offset+id].y];
    shmem[threadIdx.x].Pvel.w = h[extractList[offset+id].y];


#ifdef DO_BLOCK_TIMESTEP_EXCHANGE_MPI
    shmem[threadIdx.x].key   = body_key[extractList[offset+id].y];
    shmem[threadIdx.x].pos   = pos[extractList[offset+id].y];
    shmem[threadIdx.x].vel   = vel[extractList[offset+id].y];
    shmem[threadIdx.x].acc1  = acc1[extractList[offset+id].y];
#endif
  }
  __syncthreads();

  int startWrite  = bid * blockDim.x;
  float4 *shdata4 = (float4*)shmem;
  float4 *output  = (float4*)&destination[startWrite];


  //We have blockDim.x thread, each thread writes a float4. Compute number of items per thread-block
  //and number of loops and remaining items
  const float nThreadsPerItem = sizeof(bodyStruct) / sizeof(float4);
  const int   nItemsPerLoop   = (int)(blockDim.x / nThreadsPerItem);

  const int nExtractThisBlock = min(n_extract-startWrite, (int)blockDim.x);

  const int   nLoops          = (nExtractThisBlock/nItemsPerLoop);
  int         nExtra          = (nExtractThisBlock - nLoops*nItemsPerLoop)*nThreadsPerItem;


  int startOut = 0;
  for(int i=0; i < nLoops; i++)
  {
    output[startOut + threadIdx.x] = shdata4[threadIdx.x  + startOut]; //Write blockDim.x * float4 items
    startOut += blockDim.x;
  }

  //Write the remaining items
  if(threadIdx.x < nExtra)
  {
    output[startOut + threadIdx.x] = shdata4[threadIdx.x  + startOut]; //Write remaining float4 items
  }



#endif

#if 0
  Do not use the below kernels without first checking
  that the offsets are correct. There were problems with the ones
  above.

#elif 0
  //
  __shared__ bodyStruct shmem[64];
  #define WARP_SIZE2 5
  #define WARP_SIZE  32
  #define laneId (threadIdx.x & (WARP_SIZE - 1))
  #define warpId (threadIdx.x >> WARP_SIZE2)

  int startOut = 0;
  for(int j=0; j < 2; j++)
  {
    int nExtractThisBlock = min(n_extract-(bid * blockDim.x), (int)blockDim.x);
    nExtractThisBlock    -= j*64;
    nExtractThisBlock     = min(64, nExtractThisBlock);

    int readIdx = j*64+offset+id;

    __syncthreads();
    if(warpId < 2)
    {
      if(readIdx < n_extract)
      {
        readIdx                  = extractList[readIdx].y;
        shmem[threadIdx.x].Ppos  = Ppos[readIdx];
        shmem[threadIdx.x].Pvel  = Pvel[readIdx];
        shmem[threadIdx.x].pos   = pos[readIdx];
        shmem[threadIdx.x].vel   = vel[readIdx];
      }
    }
    else
    {
      if(readIdx-64 < n_extract)
      {
        readIdx                       = extractList[readIdx- 64].y;
        shmem[threadIdx.x - 64].acc0  = acc0[readIdx];
        shmem[threadIdx.x - 64].acc1  = acc1[readIdx];
        shmem[threadIdx.x - 64].time  = time[readIdx];
        shmem[threadIdx.x - 64].id    = body_id[readIdx];
        shmem[threadIdx.x - 64].key   = body_key[readIdx];
      }
    }
    __syncthreads();


    int startWrite  = bid * blockDim.x;
    float4 *shdata4 = (float4*)shmem;
    float4 *output  = (float4*)&destination[startWrite];

    //We have blockDim.x thread, each thread writes a float4. Compute number of items per thread-block
    //and number of loops and remaining items
    const float nThreadsPerItem = sizeof(bodyStruct) / sizeof(float4);
    const int   nItemsPerLoop   = (int)(blockDim.x / nThreadsPerItem);

    const int   nLoops          = (nExtractThisBlock/nItemsPerLoop);
    const int   nExtra          = (nExtractThisBlock - nLoops*nItemsPerLoop)*nThreadsPerItem;


  #pragma unroll
    for(int i=0; i < nLoops; i++)
    {
      output[startOut + threadIdx.x] = shdata4[threadIdx.x  + i*blockDim.x]; //Write first blockDim.x * float4 items
      startOut += blockDim.x;
    }

    //Write the remaining items
    if(threadIdx.x < nExtra)
    {
      output[startOut + threadIdx.x] = shdata4[threadIdx.x  + nLoops*blockDim.x]; //Write first blockDim.x * float4 items
    }
  } //for j

#elif 0

//Comparable to one below

  __shared__ bodyStruct shmem[32];
  #define WARP_SIZE2 5
  #define WARP_SIZE  32
  #define laneId (threadIdx.x & (WARP_SIZE - 1))
  #define warpId (threadIdx.x >> WARP_SIZE2)

  int startWrite  = bid * blockDim.x;
  float4 *shdata4 = (float4*)shmem;
  float4 *output  = (float4*)&destination[startWrite];
  int startOut    = 0;

  float4 temp1;
  float4 temp2;
  uint4  temp3;
  float2 temp4;
  int    temp5;

  for(int loop=0; loop < 4; loop++) //4 = 128 threads / 32 items
  {
    int readIdx = (bid * blockDim.x) + loop*32 + laneId + offset;
    //We only read as much as we can write in two transaction
    readIdx = min(readIdx, n_extract-1);

    if(warpId == 0)
    {
      temp1 = pos[extractList[readIdx].y];
      temp2 = Ppos[extractList[readIdx].y];
      temp3 = body_key[extractList[readIdx].y];
    }
    else if (warpId == 1)
    {
      temp1 = Pvel[extractList[readIdx].y];
      temp2 = vel[extractList[readIdx].y];
    }
    else if (warpId == 2)
    {
      temp1 = acc0[extractList[readIdx].y];
      temp2 = acc1[extractList[readIdx].y];
    }
    else if (warpId == 3)
    {
      temp5   = body_id[extractList[readIdx].y];
      temp4 = time[extractList[readIdx].y];
    }

    __syncthreads();

    if(warpId == 0)
    {
      shmem[laneId].pos = temp1;
      shmem[laneId].Ppos = temp2;
      shmem[laneId].key = temp3;
    }
    else if (warpId == 1)
    {
      shmem[laneId].Pvel = temp1;
      shmem[laneId].vel = temp2;
    }
    else if (warpId == 2)
    {
      shmem[laneId].acc0 = temp1;
      shmem[laneId].acc1 = temp2;
    }
    else if (warpId == 3)
    {
      shmem[laneId].id   = temp5;
      shmem[laneId].time = temp4;
    }
    __syncthreads();

    for(int i=0; i < 2; i++) //2 is 32 items / 16 items write per 128 threads
    {
      if(startOut + threadIdx.x < (8*n_extract)) //8* sincce 8 float4 in bodystruct
        output[startOut + threadIdx.x] = shdata4[threadIdx.x  + i*blockDim.x]; //Write first blockDim.x * float4 items
      startOut += blockDim.x;
    }

  }
#elif 0
//Second fastest
  __shared__ bodyStruct shmem[32];
  #define WARP_SIZE2 5
  #define WARP_SIZE  32
  #define laneId (threadIdx.x & (WARP_SIZE - 1))
  #define warpId (threadIdx.x >> WARP_SIZE2)

  int startWrite  = bid * blockDim.x;
  float4 *shdata4 = (float4*)shmem;
  float4 *output  = (float4*)&destination[startWrite];
  int startOut    = 0;

  for(int loop=0; loop < 4; loop++) //4 = 128 threads / 32 items
  {
    int readIdx = (bid * blockDim.x) + loop*32 + laneId  + offset;
    //We only read as much as we can write in two transaction
    readIdx = min(readIdx, n_extract-1);

    __syncthreads();
    if(warpId == 0)
    {
      shmem[laneId].pos = pos[extractList[readIdx].y];
      shmem[laneId].Ppos = Ppos[extractList[readIdx].y];
      shmem[laneId].key = body_key[extractList[readIdx].y];
    }
    else if (warpId == 1)
    {
      shmem[laneId].Pvel = Pvel[extractList[readIdx].y];
      shmem[laneId].vel = vel[extractList[readIdx].y];
    }
    else if (warpId == 2)
    {
      shmem[laneId].acc0 = acc0[extractList[readIdx].y];
      shmem[laneId].acc1 = acc1[extractList[readIdx].y];
    }
    else if (warpId == 3)
    {
      shmem[laneId].id   = body_id[extractList[readIdx].y];
      shmem[laneId].time = time[extractList[readIdx].y];
    }

    __syncthreads();

    for(int i=0; i < 2; i++) //2 is 32 items / 16 items write per 128 threads
    {
      if(startOut + threadIdx.x < (8*n_extract)) //8* sincce 8 float4 in bodystruct
        output[startOut + threadIdx.x] = shdata4[threadIdx.x  + i*blockDim.x]; //Write first blockDim.x * float4 items
      startOut += blockDim.x;
    }

  }

#endif


}

KERNEL_DECLARE(gpu_insertNewParticlesSFC)(int       n_extract,
                                              int       n_insert,
                                              int       n_oldbodies,
                                              int       offset,
                                              real4     *Ppos,
                                              real4     *Pvel,
                                              real4     *pos,
                                              real4     *vel,
                                              real4     *acc0,
                                              real4     *acc1,
                                              float2    *time,
                                              unsigned long long        *body_id,
                                              uint4     *body_key,
					      float     *h,
                                              bodyStruct *source)
{
  CUXTIMER("insertNewParticlesSFC");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  if(id >= n_insert) return;

  //The newly added particles are added at the end of the array
  int idx = (n_oldbodies-n_extract) + id + offset;

  //copy the data from a struct of arrays into a array of structs
  Ppos[idx]     = source[id].Ppos;
  Pvel[idx]     = source[id].Pvel;
  acc0[idx]     = source[id].acc0;
  time[idx]     = source[id].time;
  body_id[idx]  = source[id].id;
	  
  h[idx]        = source[id].Pvel.w; 
 

#ifdef DO_BLOCK_TIMESTEP_EXCHANGE_MPI
  body_key[idx] = source[id].key;
  acc1[idx]     = source[id].acc1;
  pos[idx]      = source[id].pos;
  vel[idx]      = source[id].vel;
#endif
}

//KERNEL_DECLARE(gpu_insertNewParticles)(int       n_extract,
//                                              int       n_insert,
//                                              int       n_oldbodies,
//                                              int       offset,
//                                              real4     *Ppos,
//                                              real4     *Pvel,
//                                              real4     *pos,
//                                              real4     *vel,
//                                              real4     *acc0,
//                                              real4     *acc1,
//                                              float2    *time,
//                                              int       *body_id,
//                                              bodyStruct *source)
//{
//  CUXTIMER("insertNewParticles");
//  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
//  uint tid = threadIdx.x;
//  uint id  = bid * blockDim.x + tid;
//
//  if(id >= n_insert) return;
//
//  //The newly added particles are added at the end of the array
//  int idx = (n_oldbodies-n_extract) + id + offset;
//
//  //copy the data from a struct of arrays into a array of structs
//  Ppos[idx]     = source[id].Ppos;
//  Pvel[idx]     = source[id].Pvel;
//  pos[idx]      = source[id].pos;
//  vel[idx]      = source[id].vel;
//  acc0[idx]     = source[id].acc0;
//  acc1[idx]     = source[id].acc1;
//  time[idx]     = source[id].time;
//  body_id[idx]  = source[id].id;
//}

//KERNEL_DECLARE(extractOutOfDomainParticlesAdvanced)(int n_extract,
//                                                       int *extractList,
//                                                       real4 *Ppos,
//                                                       real4 *Pvel,
//                                                       real4 *pos,
//                                                       real4 *vel,
//                                                       real4 *acc0,
//                                                       real4 *acc1,
//                                                       float2 *time,
//                                                       int   *body_id,
//                                                       bodyStruct *destination)
//{
//  CUXTIMER("extractOutOfDomainParticlesAdvanced");
//  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
//  uint tid = threadIdx.x;
//  uint id  = bid * blockDim.x + tid;
//
//  if(id >= n_extract) return;
//
//  //copy the data from a struct of arrays into a array of structs
//  destination[id].Ppos = Ppos[extractList[id]];
//  destination[id].Pvel = Pvel[extractList[id]];
//
//  destination[id].acc0  = acc0[extractList[id]];
//  destination[id].acc1  = acc1[extractList[id]];
//  destination[id].pos  = pos[extractList[id]];
//  destination[id].vel  = vel[extractList[id]];
//  destination[id].time  = time[extractList[id]];
//  destination[id].id    = body_id[extractList[id]];
//
//}



//KERNEL_DECLARE(gpu_extractOutOfDomainParticlesAdvancedSFC)(
//                                                       int offset,
//                                                       int n_extract,
//                                                       int *extractList,
//                                                       real4 *Ppos,
//                                                       real4 *Pvel,
//                                                       real4 *pos,
//                                                       real4 *vel,
//                                                       real4 *acc0,
//                                                       real4 *acc1,
//                                                       float2 *time,
//                                                       int   *body_id,
//                                                       uint4 *body_key,
//                                                       bodyStruct *destination)
//{
//  CUXTIMER("extractOutOfDomainParticlesAdvancedSFC");
//  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
//  uint tid = threadIdx.x;
//  uint id  = bid * blockDim.x + tid;
//
//  if(id >= n_extract) return;
//
//  //copy the data from a struct of arrays into a array of structs
//  destination[id].Ppos = Ppos[extractList[offset+id]];
//  destination[id].Pvel = Pvel[extractList[offset+id]];
//  destination[id].pos  = pos[extractList[offset+id]];
//  destination[id].vel  = vel[extractList[offset+id]];
//  destination[id].acc0  = acc0[extractList[offset+id]];
//  destination[id].acc1  = acc1[extractList[offset+id]];
//  destination[id].time  = time[extractList[offset+id]];
//  destination[id].id    = body_id[extractList[offset+id]];
//  destination[id].key   = body_key[extractList[offset+id]];
//}


// KERNEL_DECLARE(insertNewParticles)(int       n_extract,
//                                               int       n_insert,
//                                               int       n_oldbodies,
//                                               int       *extractList,
//                                               real4     *Ppos,
//                                               real4     *Pvel,
//                                               real4     *pos,
//                                               real4     *vel,
//                                               real4     *acc0,
//                                               real4     *acc1,
//                                               float2    *time,
//                                               int       *body_id,
//                                               bodyStruct *source)
// {
//   uint bid = blockIdx.y * gridDim.x + blockIdx.x;
//   uint tid = threadIdx.x;
//   uint id  = bid * blockDim.x + tid;
// 
//   int idx, srcidx = -1; 
// /*
// 
// //Situaties:
// - n_insert > n_extract -> particles moeten aan einde worden toegevoegd (meer toevoegen dan weggehaald)
//     id < n_extract -> idx = extractList[id]  ; uit source[id]
//     id >= n_extract & id < n_insert  --> idx = n_oldbodies + (id-n_extract); uit source[id]
//   
// - n_insert <= n_exract -> particles moeten van het einde naar het begin (meer verwijderd dan toegevoegd)
//     id < n_extract -> idx = extractList[id] ; uit source[id]
//     id >= n_extract & id < n_insert -> idx = extractList[id] ; uit dest[n_bodies-(n_extract-n_insert) + (id - n_insert)]
// 
//   */
// 
//   if(n_insert > n_extract)
//   {
//     if(id < n_extract)
//     {
//        idx = extractList[id];
//     }
//     else if(id >= n_extract && id < n_insert)
//     {
//       //Insert particles at the end of the array
//       idx = n_oldbodies + (id-n_extract);
//     }
//     else
//     {
//       return;
//     }
//   }
//   else
//   {
//     //n_insert <= n_extract
// 
//     if(id < n_insert)
//     {
//        idx = extractList[id];
//     }
//     else if(id >= n_insert && id < n_extract)
//     {
//       //Move particles from the back of the array to the empty spots
//       idx    = extractList[id];
//       srcidx = extractList[n_oldbodies-(n_extract-n_insert) + (id - n_insert)];
//     //  srcidx = n_oldbodies-(n_extract-n_insert) + (id - n_insert);
//     }
//     else
//     {
//       return;
//     }
//   }
// /*
// Gaat niet goed als n_insert < n_extract
// omdat we als we gaan moven we ook kans hebben dat we iets moven
// van het begin naar het eind als daar iets is uitgehaald
// we zouden dus de laatste verwijderde moeten vinden en zorgen dat er neits achter komt ofzo
// 
// 
// 
// */
// 
// /*
//   if(id < n_extract)
//   {
//     idx = extractList[id];
//   }
//   else if(id >= n_extract && id < n_insert)
//   {
//     if(n_insert > n_extract)
//     {
//       //Insert particles at the end of the array
//       idx = n_oldbodies + (id-n_extract);
//     }
//     else
//     {
//       //Move particles from the back of the array to the empty spots
//       idx    = extractList[id];
//       srcidx = n_oldbodies-(n_extract-n_insert) + (id - n_insert);
//     }
//   }
//   else
//   {
//     //Outside all array ranges
//     return;
//   }*/
// 
// 
//   if(srcidx < 0)
//   {
//     //copy the data from a struct of arrays into a array of structs
//     Ppos[idx] = source[id].Ppos;
//     Pvel[idx] = source[id].Pvel;
//     pos[idx]  = source[id].pos;
//     vel[idx]  = source[id].vel;
//     acc0[idx] = source[id].acc0;
//     acc1[idx] = source[id].acc1;
//     time[idx] = source[id].time;
//     body_id[idx] = source[id].id;
// 
// printf("%d  (CMOVE external %d) goes to: %d \n", source[id].id,n_insert, idx);
// 
// 
//   }
//   else
//   {
//     Ppos[idx] = Ppos[srcidx];
//     Pvel[idx] = Pvel[srcidx];
//     pos[idx]  = pos[srcidx];
//     vel[idx]  = vel[srcidx];
//     acc0[idx] = acc0[srcidx];
//     acc1[idx] = acc1[srcidx];
//     time[idx] = time[srcidx];
//   int temp = body_id[idx];
//     body_id[idx] = body_id[srcidx];
// 
// printf("%d stored at: %d (CMOVE internal %d) goes to: %d  overwr: %d \n", body_id[srcidx],srcidx, n_insert, idx, temp);
// 
// 
//   }//if srcidx < 0
// 
// 
// }



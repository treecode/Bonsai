#include "bonsai.h"
#include "scanKernels.cu"
// #include "support_kernels.cu"
#include "../profiling/bonsai_timing.h"
PROF_MODULE(sortKernels);

#include "node_specs.h"

//Helper functions
//Reorders data 

#ifdef USE_B40C
  #include <b40c/util/multi_buffer.cuh>
  #include <b40c/radix_sort/enactor.cuh>

  #include <thrust/device_ptr.h>
  #include <thrust/copy.h>
  #include <thrust/sort.h>
  #include <thrust/gather.h>
  #include <thrust/device_vector.h> 
  #include <thrust/iterator/transform_iterator.h>

  #include "../include/my_cuda_rt.h"
  #include "sort.h"

  Sort90::Sort90(uint N) 
  {
    // Allocate reusable ping-pong buffers on device.
    double_buffer = new b40c::util::DoubleBuffer<uint, uint>;
    sort_enactor = new b40c::radix_sort::Enactor;
  
    // The key buffers are opaque.
    cudaMalloc((void**) &double_buffer->d_keys[0], sizeof(uint) * N);
    cudaMalloc((void**) &double_buffer->d_keys[1], sizeof(uint) * N);
  
    // The current value buffer (double_buffer.d_values[double_buffer.selector])
    // backs the desired permutation array.
    cudaMalloc((void**) &double_buffer->d_values[0], sizeof(uint) * N);
    cudaMalloc((void**) &double_buffer->d_values[1], sizeof(uint) * N);

    selfAllocatedMemory = true;
  }

  Sort90::Sort90(uint N, void* generalBuffer) 
  {
    // Allocate reusable ping-pong buffers on device.
    double_buffer = new b40c::util::DoubleBuffer<uint, uint>;
    sort_enactor = new b40c::radix_sort::Enactor;
  
    // The key buffers are opaque.
    // Here we reuse the memory buffer already created by bonsai
    // Note that we need to make sure the allignment is correct
    // Initial offset is 4*N since we use the buffer in sort_bodies_gpu 
    int initialPaddingElements = my_dev::dev_mem<uint>::getGlobalMemAllignmentPadding(4*N);
    int paddingElements        = my_dev::dev_mem<uint>::getGlobalMemAllignmentPadding(N);

    int offset = (4*N) + initialPaddingElements;


    double_buffer->d_keys[0] = ((uint*)generalBuffer)+offset;
    offset     += (1*N) + paddingElements;
    double_buffer->d_keys[1] = ((uint*)generalBuffer)+offset;
    offset     += (1*N) + paddingElements;
    // The current value buffer (double_buffer.d_values[double_buffer.selector])
    // backs the desired permutation array.
    double_buffer->d_values[0] = ((uint*)generalBuffer)+offset;
    offset     += (1*N) + paddingElements;
    double_buffer->d_values[1] = ((uint*)generalBuffer)+offset;

    selfAllocatedMemory = false;
  }

  Sort90::~Sort90() 
  {
    if(selfAllocatedMemory)
    {
      //Only free if not-allocated by Bonsai internally
      cudaFree(double_buffer->d_keys[0]);
      cudaFree(double_buffer->d_keys[1]);
      cudaFree(double_buffer->d_values[0]);
      cudaFree(double_buffer->d_values[1]);
    }
    delete double_buffer;
    delete sort_enactor;
  }

  // Apply thrust_permutation
  template<typename KeyPtr, typename PermutationPtr, typename OutputPtr>
  void apply_permutation(KeyPtr& thrust_in,
                         PermutationPtr& thrust_permutation,
                         OutputPtr& thrust_out,
                         int N)
  {
    // permute the keys into out vector
    thrust::gather(thrust_permutation, thrust_permutation + N, thrust_in, thrust_out);
  }


  // Extract 32-bit word from uint4
  template<int keyIdx>
  struct ExtractBits: public thrust::unary_function<uint4, uint>
  {
    __host__ __device__ __forceinline__ uint operator()(uint4 key) const
    {
      if      (keyIdx == 0) return key.x;
      else if (keyIdx == 1) return key.y;
      else                  return key.z;
    }
  };


  // Update thrust_permutation
  template<int keyIdx, typename KeyPtr>
  void update_permutation(KeyPtr& thrust_src_keys, 
                          int N,
                          b40c::util::DoubleBuffer<uint, uint> &double_buffer,
		                      b40c::radix_sort::Enactor &sort_enactor)
  {
    // thrust ptr to thrust_permutation buffer
    thrust::device_ptr<uint> thrust_permutation = 
      thrust::device_pointer_cast(double_buffer.d_values[double_buffer.selector]);

    // thrust ptr to temporary 32-bit keys
    thrust::device_ptr<uint> thust_32bit_temp = 
      thrust::device_pointer_cast(double_buffer.d_keys[double_buffer.selector]);

    // gather into temporary keys with the current reordering
    thrust::gather(thrust_permutation,
                   thrust_permutation + N,
                   thrust::make_transform_iterator(thrust_src_keys, ExtractBits<keyIdx>()),
                   thust_32bit_temp);

    // Stable-sort the top 30 bits of the temp keys (and
    // associated thrust_permutation values)
    sort_enactor.Sort<30, 0>(double_buffer, N);
  }


  // Back40 90-bit sorting: sorts the lower 30 bits in uint4's key
  void Sort90::sort(my_dev::dev_mem<uint4> &srcKeys,
                    my_dev::dev_mem<uint4> &sortedKeys,
                    int N)
  {
    // thrust ptr to srcKeys
    thrust::device_ptr<uint4> thrust_src_keys = 
      thrust::device_pointer_cast(srcKeys.raw_p());

    // thrust ptr to sortedKeys
    thrust::device_ptr<uint4> thrust_out_keys = 
      thrust::device_pointer_cast(sortedKeys.raw_p());

    // thrust ptr to permutation buffer
    thrust::device_ptr<uint> thrust_permutation = 
      thrust::device_pointer_cast(double_buffer->d_values[double_buffer->selector]);

    // initialize values (thrust_permutation) to [0, 1, 2, ... ,N-1]
    thrust::sequence(thrust_permutation, thrust_permutation + N);

    // sort z, y, x
    // careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
    update_permutation<2>(thrust_src_keys, N, *double_buffer, *sort_enactor);
    update_permutation<1>(thrust_src_keys, N, *double_buffer, *sort_enactor);
    update_permutation<0>(thrust_src_keys, N, *double_buffer, *sort_enactor);

    // refresh thrust ptr to permutation buffer (may have changed inside ping-pong)
    thrust_permutation = 
      thrust::device_pointer_cast(double_buffer->d_values[double_buffer->selector]);

    // Note: thrust_permutation now maps unsorted keys to sorted order
    apply_permutation(thrust_src_keys, thrust_permutation, thrust_out_keys, N);
  }
#endif


KERNEL_DECLARE(gpu_dataReorderR4)(const int n_particles,
                                         real4 *source,
                                         real4 *destination,
                                         uint  *permutation) {
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;
  
  int idx = bid * dim + tid;
  if (idx >= n_particles) return;

   int newIndex = permutation[idx];
   destination[idx] = source[newIndex];  
}

// KERNEL_DECLARE(dataReorderF2)(const int n_particles,
//                                          float2 *source,
//                                          float2 *destination,
//                                          uint  *permutation) {
//   const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
//   const int tid = threadIdx.y * blockDim.x + threadIdx.x;
//   const int dim =  blockDim.x * blockDim.y;
//   
//   int idx = bid * dim + tid;
//   if (idx >= n_particles) return;
// 
//   int newIndex = permutation[idx];
//   destination[idx] = source[newIndex];  
// }

KERNEL_DECLARE(gpu_dataReorderI1)(const int n_particles,
                                         int *source,
                                         int *destination,
                                         uint  *permutation) {
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;
  
  int idx = bid * dim + tid;
  if (idx >= n_particles) return;

   int newIndex = permutation[idx];
   destination[idx] = source[newIndex];  
}


//Convert a 64bit key uint2 key into a 96key with a permutation value build in
KERNEL_DECLARE(gpu_convertKey64to96)(uint4 *keys,  uint4 *newKeys, const int N)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;

  if (idx >= N) return;

  uint4 temp = keys[idx];
  newKeys[idx] = make_uint4(temp.x, temp.y, temp.z, idx);
}

KERNEL_DECLARE(gpu_extractKeyAndPerm)(uint4 *newKeys, uint4 *keys, uint *permutation, const int N)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;

  if (idx >= N) return;

  uint4 temp = newKeys[idx];
  
  keys[idx]        = make_uint4(temp.x, temp.y, temp.z, temp.w);
  permutation[idx] = temp.w;
}

KERNEL_DECLARE(gpu_dataReorderCombined)(const int N, uint4 *keyAndPerm,
                                      real4 *source1, real4* destination1,
                                      real4 *source2, real4* destination2,
                                      real4 *source3, real4* destination3) {
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;
  
  int idx = bid * dim + tid;
  if (idx >= N) return;

  int newIndex      = keyAndPerm[idx].w;
  destination1[idx] = source1[newIndex];
  destination2[idx] = source2[newIndex];  
  destination3[idx] = source3[newIndex];  
//   destination1[idx] = source1[newIndex];  
//   destination1[idx] = source1[newIndex];    
}


KERNEL_DECLARE(dataReorderCombined4)(const int N, 
                                     uint4 *keyAndPerm,
                                     real4 *source1,  real4* destination1,
                                     ulonglong1 *source2,    ulonglong1 *   destination2,
                                     int *oldOrder) {
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;
  
  int idx = bid * dim + tid;
  if (idx >= N) return;

  int newIndex      = keyAndPerm[idx].w;
  destination1[idx] = source1[newIndex];
  destination2[idx] = source2[newIndex];  
  oldOrder[idx]     = newIndex;
}





KERNEL_DECLARE(dataReorderCombined2)(const int N, uint4 *keyAndPerm,
                                      real4 *source1, real4* destination1,
                                      real4 *source2, real4* destination2,
                                      real4 *source3, real4* destination3) {

  const int blockSize   = blockDim.x;
  unsigned int idx        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;
  

  while (idx < N)
  {
    if (idx             < N)
    {
      int newIndex      = keyAndPerm[idx].w;
      destination1[idx] = source1[newIndex];
      destination2[idx] = source2[newIndex];  
      destination3[idx] = source3[newIndex]; 
    }
    if (idx + blockSize < N)
    {
      int newIndex      = keyAndPerm[idx + blockSize].w;
      destination1[idx + blockSize] = source1[newIndex];
      destination2[idx + blockSize] = source2[newIndex];  
      destination3[idx + blockSize] = source3[newIndex]; 
    }
    idx += gridSize;
  }
}

KERNEL_DECLARE(dataReorderCombined3)(const int N, uint4 *keyAndPerm,
                                      real4 *source1, real4* destination1,
                                      real4 *source2, real4* destination2,
                                      real4 *source3, real4* destination3) {

  const int blockSize   = blockDim.x;
  unsigned int idx        = blockIdx.x*(blockSize) + threadIdx.x;
  unsigned int gridSize = blockSize*gridDim.x;
  

  while (idx < N)
  {
    if (idx             < N)
    {
      int newIndex      = keyAndPerm[idx].w;
      destination1[idx] = source1[newIndex];
      destination2[idx] = source2[newIndex];  
      destination3[idx] = source3[newIndex]; 
    }
    idx += gridSize;
  }
}


KERNEL_DECLARE(gpu_dataReorderF2)(const int N, uint4 *keyAndPerm,
                                         float2 *source1, float2 *destination1,
                                         ulonglong1    *source2, ulonglong1 *destination2) {
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;
  
  int idx = bid * dim + tid;
  if (idx >= N) return;


  int newIndex      = keyAndPerm[idx].w;
  destination1[idx] = source1[newIndex];
  destination2[idx] = source2[newIndex];  
}


KERNEL_DECLARE(gpu_dataReorderF1)(const int N, uint4 *keyAndPerm,
                                float *source1, float *destination1)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;
  
  int idx = bid * dim + tid;
  if (idx >= N) return;
  
  int newIndex      = keyAndPerm[idx].w;
  destination1[idx] = source1[newIndex];
}



//Extract 1 of the 4 items of an uint4 key and move it into a 32bit array
KERNEL_DECLARE(extractInt2)(uint4 *keys,  uint *simpleKeys, const int N, int keyIdx)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;

  if (idx >= N) return;

  uint4 temp = keys[idx];
  int  simpleTemp;

  if(keyIdx == 0)
      simpleTemp = temp.x;
  else if(keyIdx == 1)
      simpleTemp = temp.y;
  else if(keyIdx == 2)
      simpleTemp = temp.z;

  simpleKeys[idx] = simpleTemp;
}

KERNEL_DECLARE(extractInt_kernel)(uint4 *keys,  uint *simpleKeys,
                                      uint *sequence,
                                      const int N, int keyIdx)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;

  if (idx >= N) return;

  uint4 temp = keys[idx];
  int  simpleTemp;

  if(keyIdx == 0)
      simpleTemp = temp.x;
  else if(keyIdx == 1)
      simpleTemp = temp.y;
  else if(keyIdx == 2)
      simpleTemp = temp.z;

  simpleKeys[idx] = simpleTemp;
  sequence[idx] = idx;
}



//Create range of 0 to N
KERNEL_DECLARE(fillSequence)(uint *sequence, const int N)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;

  if (idx >= N) return;

  sequence[idx] = idx;
}

//Reorder the data in the arrays according to a given permutation
KERNEL_DECLARE(reOrderKeysValues_kernel)(uint4 *keysSrc, uint4 *keysDest, uint *permutation, const int N)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;

  if (idx >= N) return;

  int newIndex = permutation[idx];
  keysDest[idx] = keysSrc[newIndex];
}

KERNEL_DECLARE(sort_count)(volatile uint2 *valid, int *counts, const int N, setupParams sParam, int bitIdx/*, int2 *blaat*/)
{
  const int tid    =  threadIdx.x;
  const int bid    =  blockDim.y *  blockIdx.x + threadIdx.y;

  int totalNumThreads = gridDim.x*blockDim.y*blockDim.x; //120*4*32 // gridDim.x * blockDim.y; //2D !!!!

  volatile __shared__ int shmemSC[128];
  volatile __shared__ int shmemSCTEST[128];

  //Determine the parameters and loop over the particles
  int jobSize = (N / 2) / totalNumThreads;
  int offSet  = jobSize * bid;
  int count   = 0;

  jobSize = sParam.jobs;
  if(bid < sParam.blocksWithExtraJobs)
    jobSize++;

  if(bid <= sParam.blocksWithExtraJobs)
    offSet = (sParam.jobs+1)*64*bid;
  else
  {
    offSet = sParam.blocksWithExtraJobs*(sParam.jobs+1)*64;
    offSet += (bid-sParam.blocksWithExtraJobs)*(sParam.jobs)*64;
  }

  offSet /= 2;  //Divide by two since we do double loads (uint2)

  for(int i=0; i < jobSize; i++)
  {   
    count  += !(valid[offSet + tid].x & (1u<<bitIdx));
    count  += !(valid[offSet + tid].y & (1u<<bitIdx));
    offSet += blockDim.x;
  }

  //Reduce to get the count of this block
  shmemSC[32*threadIdx.y + tid] = count;
  reduce_block2(tid, &shmemSC[32*threadIdx.y], count);

  //Save the values / count of the current block
  if(threadIdx.x == 0)
    counts[bid] = shmemSC[32*threadIdx.y];

  //Block 0 handles any extra elements that couldn't be divided equally
  if(bid == 0)
  {
    //Here i use single element reads for ease of boundary conditions and steps
    count   = 0;
    offSet  = sParam.extraOffset;

    uint* valid2 = (uint*) valid;

    for(int i=0 ; i < sParam.extraElements;  i += blockDim.x)
    {
      if((offSet + i +  tid) < (N))  //Make sure we dont read more than there are items
      {
        count += !(valid2[offSet + i +  tid] & (1u<<bitIdx));
      }
    }

    //Reduce
    shmemSCTEST[tid] = count;

    __syncthreads();

    if(tid < 16){
      shmemSCTEST[tid] = count = count + shmemSCTEST[tid+16];
      shmemSCTEST[tid] = count = count + shmemSCTEST[tid+8];
      shmemSCTEST[tid] = count = count + shmemSCTEST[tid+4];
      shmemSCTEST[tid] = count = count + shmemSCTEST[tid+2];
      shmemSCTEST[tid] = count = count + shmemSCTEST[tid+1]; 
    }

    //Save the count
    if(tid == 0)
    {
      counts[gridDim.x*blockDim.y] = shmemSCTEST[0];
    }

    __syncthreads();

  }//end if  bid==0 
}//end compact_count


// static __device__  __forceinline__ int testTest(volatile unsigned int tmp[], uint val, const int idx, long test)
// {
//   tmp[idx-16] = 0; tmp[idx] = val;
// 
//   // Since we set half the array to 0 we don't need ifs!
//   tmp[idx] = val = tmp[idx -  1]  + val;
//   tmp[idx] = val = tmp[idx -  2]  + val;
//   tmp[idx] = val = tmp[idx -  4]  + val;
//   tmp[idx] = val = tmp[idx -  8]  + val;
//   tmp[idx] = val = tmp[idx -  16] + val;
// 
//   return (idx > 0) ? tmp[idx-1] : 0;
// }


/*
For sorting it turns out that the stage kernels works faster than the non-staged
Might depend on how much has to be sorted/moved, have to do timings in the actual code
*/
KERNEL_DECLARE(sort_move_stage_key_value)(uint2 *valid, int *output,
                                          uint2 *srcValues, uint *valuesOut,
                                          int *counts,
                                          const int N, setupParams sParam, int bitIdx)
{
  //Walk the values of this block
  const int tid    =  threadIdx.x;
  const int bid    =  blockDim.y *  blockIdx.x + threadIdx.y;

  volatile __shared__ unsigned int shmemSMSKV[192];
  volatile __shared__ int stage[64*4];
  volatile __shared__ int stage_values[64*4];

  //Determine the parameters and loop over the particles
  int jobSize, offSet;


  jobSize = sParam.jobs;
  if(bid < sParam.blocksWithExtraJobs)
    jobSize++;

  if(bid <= sParam.blocksWithExtraJobs)
    offSet = (sParam.jobs+1)*64*bid;
  else
  {
    offSet = sParam.blocksWithExtraJobs*(sParam.jobs+1)*64;
    offSet += (bid-sParam.blocksWithExtraJobs)*(sParam.jobs)*64;
  }

  int outputOffset = counts[bid];

  //Get the start of the output offset of the invalid items
  //this is calculated as follows:
  //totalValidItems + startReadOffset - startOutputOffset
  //startReadOffset - startOutputOffset <- is the total number of invalid items from any blocks
  //before the current block
  int rightOutputOffset = counts[gridDim.x*blockDim.y+1];
  rightOutputOffset     = rightOutputOffset + offSet - outputOffset;

  offSet /= 2;  //Divide by two since we do double loads (uint2) TODO what happens if offSet is uneven...?

  int curCount;
  int idx, ridx;

  outputOffset      += threadIdx.x;
  rightOutputOffset += threadIdx.x;

  //Do per step the prefix scan to determine the output locations
  for(int i=0; i < jobSize; i++)
  {
    uint2  validBase  = valid[offSet + tid];
    uint2  valuesBase = srcValues[offSet + tid];
    int value         = !(validBase.x  & (1u<<bitIdx));
    value            += !(validBase.y  & (1u<<bitIdx));

    idx  = hillisSteele5(&shmemSMSKV[48*threadIdx.y+16], curCount, value, threadIdx.x);

    ridx = curCount + threadIdx.x*2 - idx; //lane*2 - idx , *2 since we read 2 items a time

    if(!(validBase.x  & (1u<<bitIdx)))
    {
      stage[idx + threadIdx.y*64]          = validBase.x;
      stage_values[idx++ + threadIdx.y*64] = valuesBase.x;
    }
    else
    {
      stage[ridx + threadIdx.y*64]          = validBase.x;
      stage_values[ridx++ + threadIdx.y*64] = valuesBase.x;
    }

    if(!(validBase.y  & (1u<<bitIdx)))
    {
      stage[idx + threadIdx.y*64]        = validBase.y;
      stage_values[idx + threadIdx.y*64] = valuesBase.y;
    }
    else
    {
      stage[ridx + threadIdx.y*64]        = validBase.y;
      stage_values[ridx + threadIdx.y*64] = valuesBase.y;
    }

    //Reuse value as index
    value = outputOffset;
    //Flush output, first 32
    if(threadIdx.x >= curCount)
      value = rightOutputOffset-curCount;
    output[value]    = stage       [threadIdx.x + threadIdx.y*64];
    valuesOut[value] = stage_values[threadIdx.x + threadIdx.y*64];

    //2nd 32
    value = outputOffset + blockDim.x;
    if(threadIdx.x + blockDim.x >= curCount)
      value = rightOutputOffset + blockDim.x - curCount;

    output[value]    = stage       [threadIdx.x + blockDim.x + threadIdx.y*64];
    valuesOut[value] = stage_values[threadIdx.x + blockDim.x + threadIdx.y*64];

    outputOffset      += curCount;      //Increase the output offset
    rightOutputOffset += 64 - curCount; //64 (32*2) since we do 2 items a time
    offSet            += blockDim.x;    //Step to the next N threads
  }

  //Block 0 handles any extra elements that couldn't be divided equally
  if(bid == 0)
  {
    //Here i use single element reads for ease of boundary conditions and steps
    offSet              = sParam.extraOffset;
    outputOffset        = counts[gridDim.x*blockDim.y];
    rightOutputOffset   = counts[gridDim.x*blockDim.y+1];
    rightOutputOffset   = rightOutputOffset + offSet - outputOffset;

    uint* valid2 = (uint*) valid;
    uint* srcValues2 = (uint*) srcValues;

    for(int i=0; i < sParam.extraElements;  i += blockDim.x)
    {
      uint value = 0;
      uint srcValueItem = 0;
    
      if((offSet + i +  tid) < (N)){  //Make sure we dont read more than there are items
        value        = valid2[offSet + i +  tid];
        srcValueItem = srcValues2[offSet + i +  tid];
      }

      idx  = hillisSteele5(&shmemSMSKV[48*threadIdx.y+16], curCount, !(value & (1u<<bitIdx)), threadIdx.x);
      ridx = threadIdx.x - idx;

      if((offSet + i +  tid) < N)
        if(!(value & (1u<<bitIdx)))
        {
          output[idx + outputOffset]    = value;
          valuesOut[idx + outputOffset] = srcValueItem;
        }
        else
        {
          output[ridx + rightOutputOffset]     = value;
          valuesOut[ridx + rightOutputOffset]  = srcValueItem;
        }

      outputOffset      += curCount;       //Increase the output offset
      rightOutputOffset += 32-curCount;    //32 since we do only 1 at a time
    }
  }//end if bid==0 
}//end sort_move_stage_key_value


#ifndef TREEWALK_INCLUDE_H
#define TREEWALK_INCLUDE_H


#define WARP_SIZE2 5
#define WARP_SIZE  32

#if NCRIT > 4*WARP_SIZE
#error "NCRIT in include/node_specs.h must be <= WARP_SIZE"
#endif

#define laneId (threadIdx.x & (WARP_SIZE - 1))
#define warpId (threadIdx.x >> WARP_SIZE2)

#define BTEST(x) (-(int)(x))

#if 0
/* For CUDA8 compilation we require the old functions */
    #define __ballot_sync(mask, pred)           __ballot(pred)
    #define __shfl_sync(mask, val, src, width)  __shfl(val, src, width)
    #define __shfl_xor_sync(mask, val, src)     __shfl_xor(val, src)
    #define __shfl_down_sync(mask, val, src)    __shfl_down(val, src)
#endif



#define FULL_MASK 0xffffffff



  template<int SHIFT>
__forceinline__ static __device__ int ringAddr(const int i)
{
  return (i & ((CELL_LIST_MEM_PER_WARP<<SHIFT) - 1));
}



/************************************/
/*********   PREFIX SUM   ***********/
/************************************/

static __device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
  uint result;
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.sync.up.b32 r0|p, %1, %2, 0, 0xffffffff;"
      "@p add.u32 r0, r0, %3;"
      "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
  return result;
}

static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__ int lanemask_le()
{
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__ int ShflSegScanStepB(
            int partial,
            uint distance,
            uint up_offset)
{
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.sync.up.b32 r0, %1, %2, 0, 0xffffffff;"
      "setp.le.u32 p, %2, %3;"
      "@p add.u32 %1, r0, %1;"
      "mov.u32 %0, %1;}"
      : "=r"(partial) : "r"(partial), "r"(up_offset), "r"(distance));
  return partial;
}

  template<const int SIZE2>
static __device__ __forceinline__ int inclusive_segscan_warp_step(int value, const int distance)
{
  for (int i = 0; i < SIZE2; i++)
    value = ShflSegScanStepB(value, distance, 1<<i);
  return value;
}

  template <const int levels>
static __device__ __forceinline__ uint inclusive_scan_warp(const int sum)
{
  uint mysum = sum;
#pragma unroll
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

/*********************/

static __device__ __forceinline__ int2 warpIntExclusiveScan(const int value)
{
  const int sum = inclusive_scan_warp<WARP_SIZE2>(value);
  return make_int2(sum-value, __shfl_sync(FULL_MASK, sum, WARP_SIZE-1, WARP_SIZE));
}

static __device__ __forceinline__ int2 warpBinExclusiveScan(const bool p)
{
  const unsigned int b = __ballot_sync(FULL_MASK, p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}


static __device__ __forceinline__ int2 inclusive_segscan_warp(
    const int packed_value, const int carryValue)
{
  const int  flag = packed_value < 0;
  const int  mask = -flag;
  const int value = (~mask & packed_value) + (mask & (-1-packed_value));
  
  const int flags = __ballot_sync(FULL_MASK, flag);

  const int dist_block = __clz(__brev(flags));

  const int distance = __clz(flags & lanemask_le()) + laneId - 31;
  
  const int val = inclusive_segscan_warp_step<WARP_SIZE2>(value, min(distance, laneId));
  
  return make_int2(val + (carryValue & (-(laneId < dist_block))), __shfl_sync(FULL_MASK, val, WARP_SIZE-1, WARP_SIZE));
}

/**** binary scans ****/


#if 0
static __device__ int warp_exclusive_scan(const bool p, int &psum)
{
  const unsigned int b = __ballot(p);
  psum = __popc(b & lanemask_lt());
  return __popc(b);
}
static __device__ int warp_exclusive_scan(const bool p)
{
  const int b = __ballot(p);
  return __popc(b & lanemask_lt());
}
#endif


__inline__ __device__
float warpAllReduceMax(float val)
{
  for (int mask = warpSize/2; mask > 0; mask /= 2) 
       val = max(val, __shfl_xor_sync(FULL_MASK, val, mask)); 
  return val;
}
__inline__ __device__
float warpAllReduceMin(float val)
{
  for (int mask = warpSize/2; mask > 0; mask /= 2)
       val = min(val, __shfl_xor_sync(FULL_MASK, val, mask));
  return val;
}

__inline__ __device__
float warpGroupReduce(float val)
{
    for(int i=NCRIT; i < WARP_SIZE; i*=2)
        val += __shfl_down_sync(FULL_MASK, val,  i);
    return val;
}

__inline__ __device__
float warpGroupReduceMin(float val)
{
    for(int i=NCRIT; i < WARP_SIZE; i*=2)
        val = min(val, __shfl_down_sync(FULL_MASK, val,  i));
    return val;
}

__inline__ __device__
float warpGroupReduceMax(float val)
{
    for(int i=NCRIT; i < WARP_SIZE; i*=2)
        val = max(val, __shfl_down_sync(FULL_MASK, val,  i));
    return val;
}


#endif

// #include "support_kernels.cu"
#include <stdio.h>
#include "../profiling/bonsai_timing.h"
PROF_MODULE(dev_approximate_gravity);

#include "node_specs.h"

#ifdef WIN32
#define M_PI        3.14159265358979323846264338328
#endif

#define WARP_SIZE2 5
#define WARP_SIZE  32

#if 1
#define laneId (threadIdx.x & (WARP_SIZE - 1))
#define warpId (threadIdx.x >> WARP_SIZE2)
#endif

__forceinline__ __device__ float Wkernel(const float q)
{
  const float sigma = 8.0f/M_PI;

  const float qm = 1.0f - q;
  const float f1 = sigma * (1.0f + (-6.0f)*q*q*qm);
  const float f2 = sigma * 2.0f*qm*qm*qm;

  return fmaxf(0.0f, fminf(f1, f2));
}

__forceinline__ __device__ float interact(
    const float3 ipos,
    const float  h,
    const float  hinv,
    const float3 jpos,
    const float  jmass)
{
  const float3 dr = make_float3(jpos.x - ipos.x, jpos.y - ipos.y, jpos.z - ipos.z);
  const float  r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  if (r2 >= h*h) return 0.0f;
  const float q  = sqrtf(r2) * hinv;
  const float hinv3 = hinv*hinv*hinv;

  return jmass * Wkernel(q) * hinv3;
}


/***
**** --> prefix calculation via Horn(2005) data-parallel algoritm
***/
#define BTEST(x) (-(int)(x))

  template<int DIM2>
__device__ int calc_prefix1(int* prefix, int tid, int value)
{
  int  x;

  const int DIM = 1 << DIM2;

  prefix[tid] = value;
  __syncthreads();

  x = prefix[tid -  1]; __syncthreads(); prefix[tid] += x & BTEST(tid >=  1); __syncthreads();
  x = prefix[tid -  2]; __syncthreads(); prefix[tid] += x & BTEST(tid >=  2); __syncthreads();
  x = prefix[tid -  4]; __syncthreads(); prefix[tid] += x & BTEST(tid >=  4); __syncthreads();
  x = prefix[tid -  8]; __syncthreads(); prefix[tid] += x & BTEST(tid >=  8); __syncthreads();
  x = prefix[tid - 16]; __syncthreads(); prefix[tid] += x & BTEST(tid >= 16); __syncthreads();
  if (DIM2 >= 6) {x = prefix[tid - 32]; __syncthreads(); prefix[tid] += x & BTEST(tid >= 32); __syncthreads();}
  if (DIM2 >= 7) {x = prefix[tid - 64]; __syncthreads(); prefix[tid] += x & BTEST(tid >= 64); __syncthreads();}
  if (DIM2 >= 8) {x = prefix[tid -128]; __syncthreads(); prefix[tid] += x & BTEST(tid >=128); __syncthreads();}

  x = prefix[DIM - 1];
  __syncthreads();
  return x;
}

#if 1
#define _KEPLERCODE_
#endif

#ifdef _KEPLERCODE_

/****** KEPLER __shfl prefix sum ******/

__device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
  uint result;
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0|p, %1, %2, 0;"
      "@p add.u32 r0, r0, %3;"
      "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
  return result;
}

  template <const int levels>
__device__ __forceinline__ uint inclusive_scan_warp(int mysum)
{
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

template<int DIM2>
__device__ __forceinline__ int calc_prefix(int* prefix, int tid, int value) 
{
  if (DIM2 != 6)  /* should never be called */
    return calc_prefix1<DIM2>(prefix, tid, value);
  else
  {
    prefix[tid] = inclusive_scan_warp<WARP_SIZE2>(value);
    __syncthreads();

    prefix[tid] += prefix[WARP_SIZE - 1] & BTEST(tid >= WARP_SIZE);
    __syncthreads();

    const int x = prefix[(1 << DIM2)- 1];
    __syncthreads();       /* must be here, otherwise the code crashes */

    return x;
  }
}
#else
  template<int DIM2>
__device__ __forceinline__ int calc_prefix(int* prefix, int tid, int value) 
{
    return calc_prefix1<DIM2>(prefix, tid, value);
}
#endif


  template<int DIM2>
__device__ int calc_prefix(int N, int* prefix_in, int tid) 
{
  const int DIM = 1 << DIM2;

  int y = calc_prefix<DIM2>(prefix_in, tid, prefix_in[tid]);
  if (N <= DIM) return y;

  for (int p = DIM; p < N; p += DIM) 
  {
    int *prefix = &prefix_in[p];
    const int y1 = calc_prefix<DIM2>(prefix, tid, prefix[tid]);
    prefix[tid] += y;
    y += y1;
  }
  __syncthreads();

  return y;
} 

/************************************/
/********* SEGMENTED SCAN ***********/
/************************************/

 
#ifdef _KEPLERCODE_
  template<const int SIZE2>
__device__ __forceinline__ int inclusive_segscan_warp(int value, const int distance)
{
#if 0
  const unsigned int laneId = threadIdx.x & (WARP_SIZE - 1);
#endif
  
  const int SIZE = 1 << SIZE2; 

#if 0
  for (int i = 1; i <= SIZE; i <<= 1) 
    value += __shfl_up(value, i, SIZE) & BTEST(laneId >= i && i <= distance);
#else
  for (int i = 0; i < SIZE2; i++)
    value += __shfl_up(value, 1 << i, SIZE) & BTEST(laneId >= (1<<i)) & BTEST((1<<i) <= distance);
#endif

  return value;
}
#else /* !_KEPLERCODE_ */
#error "Only _KEPLERCODE_ is supported at this time"
#endif


__device__ __forceinline__ int bfi(const int x, const int y, const int bit, const int numBits) 
{
	int ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}

template<const int DIM2>
__device__ __forceinline__ int inclusive_segscan_block64(
    int *shmem, const int tid, const int packed_value, int &dist_block, int &nseg)
{
#if 0
  const int laneId = tid & (WARP_SIZE - 1);
  const int warpId = tid >> WARP_SIZE2;
#endif

  const int  flag = packed_value < 0;
  const int  mask = BTEST(flag);
  const int value = (mask & (-1-packed_value)) + (~mask & 1);
  
  shmem[(warpId << WARP_SIZE2) + WARP_SIZE - 1 - laneId] = flag;
  const int flags1 = __ballot(shmem[tid]);
  shmem[tid] = __popc(flags1);
  __syncthreads();
  nseg += shmem[0] + shmem[WARP_SIZE];
  __syncthreads();
  
  shmem[tid] = __clz (flags1);
  __syncthreads();

  const int flags = __ballot(flag) & bfi(0, 0xffffffff, 0, laneId + 1);
	const int distance = __clz(flags) + laneId - 31;

  int dist0  = shmem[WARP_SIZE];
  dist_block = shmem[0] + (BTEST(shmem[0] == WARP_SIZE) & dist0);
  
  __syncthreads();

  shmem[tid] = inclusive_segscan_warp<WARP_SIZE2>(value, distance);
  __syncthreads();
 
  shmem[tid] += shmem[WARP_SIZE - 1] & BTEST(tid >= WARP_SIZE) & BTEST(tid < WARP_SIZE + dist0);
  __syncthreads();

  const int val = shmem[(1 << DIM2) - 1];
  __syncthreads();

  return val;
}

/* does not work if segment size > 64 (= thead block size) */
template<const int DIM2>
__device__ __forceinline__ int inclusive_segscan_array(int *shmem_in, const int N, const int tid)
{
  const int DIM = 1 << DIM2;

  int dist, nseg = 0;
  int y  = inclusive_segscan_block64<DIM2>(shmem_in, tid, shmem_in[tid], dist, nseg);
  if (N <= DIM) return nseg;

  for (int p = DIM; p < N; p += DIM)
  {
    int *shmem = shmem_in + p;
    int y1  = inclusive_segscan_block64<DIM2>(shmem, tid, shmem[tid], dist, nseg);
    shmem[tid] += y & BTEST(tid < dist);
    y = y1;
  }

  return nseg;
}


/*************** Tree walk ************/


  template<int SHIFT>
__forceinline__ __device__ int ACCS(const int i)
{
  return (i & ((LMEM_STACK_SIZE << SHIFT) - 1))*blockDim.x + threadIdx.x;
}


#define BTEST(x) (-(int)(x))

texture<float4, 1, cudaReadModeElementType> texNodeSize;
texture<float4, 1, cudaReadModeElementType> texNodeCenter;
texture<float4, 1, cudaReadModeElementType> texMultipole;
texture<float4, 1, cudaReadModeElementType> texBody;

/*********** Forces *************/

__device__ float4 add_acc(
    float4 acc,  const float4 pos,
    const float massj, const float3 posj,
    const float eps2)
{
#if 1  /* to test performance of a tree-walk */
  const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);

  const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;
  const float rinv   = rsqrtf(r2);
  const float rinv2  = rinv*rinv;
  const float mrinv  = massj * rinv;
  const float mrinv3 = mrinv * rinv2;

  acc.w -= mrinv;
  acc.x += mrinv3 * dr.x;
  acc.y += mrinv3 * dr.y;
  acc.z += mrinv3 * dr.z;
#endif

  return acc;
}


//Improved Barnes Hut criterium
__device__ bool split_node_grav_impbh(
    const float4 nodeCOM, 
    const float4 groupCenter, 
    const float4 groupSize)
{
  //Compute the distance between the group and the cell
  float3 dr = make_float3(
      fabsf(groupCenter.x - nodeCOM.x) - (groupSize.x),
      fabsf(groupCenter.y - nodeCOM.y) - (groupSize.y),
      fabsf(groupCenter.z - nodeCOM.z) - (groupSize.z)
      );

  dr.x += fabsf(dr.x); dr.x *= 0.5f;
  dr.y += fabsf(dr.y); dr.y *= 0.5f;
  dr.z += fabsf(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  const float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  return (ds2 <= fabsf(nodeCOM.w));
}



#define TEXTURES


template<int DIM2, int SHIFT>
__device__ float4 approximate_gravity(int DIM2x, int DIM2y,
    int tid, int tx, int ty,
    int body_i, float4 pos_i,
    real4 group_pos,
    float eps2,
    uint2 node_begend,
    real4 *multipole_data,
    real4 *body_pos,
    int *shmem,
    int *lmem,
    int &ngb,
    int &apprCount, int &direCount,
    volatile float4 *boxSizeInfo,
    float4 groupSize,
    volatile float4 *boxCenterInfo,
    float group_eps,
    real4 *body_vel) {

  float4 acc_i = {0.0f, 0.0f, 0.0f, 0.0f};


  /*********** set necessary thread constants **********/

  const int DIMx = 1  << DIM2x;
  const int DIMy = 1  << DIM2y;
  const int DIM  = 1  << DIM2;
  const int offs = ty << DIM2x;

  /*********** shared memory distribution **********/

  //  begin,    end,   size
  // -----------------------
  const int stack_sz = (LMEM_STACK_SIZE << SHIFT) << DIM2;  /* stack allocated per thread-block */
  int *approxL = lmem + stack_sz; 

  int *directS = shmem;                              //  0*DIM,  1*DIM,  1*DIM
  int *nodesS  = directS + DIM;                      //  1*DIM, 10*DIM,  9*DIM
  int *prefix  = nodesS  + DIM*9;                    // 10*DIM, 11*DIM,  1*DIM

  const int NJMAX = DIM*3;
  int    *body_list = (int*   )&nodesS   [  DIM]; //  2*DIM,   5*DIM,  2*DIM
  float  *sh_mass   = (float* )&body_list[NJMAX]; //  5*DIM,   6*DIM,  1*DIM
  float3 *sh_pos    = (float3*)&sh_mass  [  DIM]; //  6*DIM,   9*DIM   3*DIM

  int *approxM = approxL;
  int *directM = directS;
  int * nodesM =  nodesS;


  float  *node_mon0 = sh_mass;
  float3 *node_mon1 = sh_pos; 

  float  *sh_pot = sh_mass;
  float3 *sh_acc = sh_pos;

  /*********** stack **********/

  int *nstack = lmem;

  /*********** begin tree-walk **********/

  int n_approx = 0;
  int n_direct = 0;


  for (int root_node = node_begend.x; root_node < node_begend.y; root_node += DIM) 
  {
    int n_nodes0 = min(node_begend.y - root_node, DIM);
    int n_stack0 = 0;
    int n_stack_pre = 0;

    { nstack[ACCS<SHIFT>(n_stack0)] = root_node + tid;   n_stack0++; }

    /*********** walk each level **********/
    while (n_nodes0 > 0) {


      int n_nodes1 = 0;
      int n_offset = 0;

      int n_stack1 = n_stack0;
      int c_stack0 = n_stack_pre;

      /*********** walk a level **********/
      while(c_stack0 < n_stack0) 
      {

        /***
         **** --> fetch the list of nodes rom LMEM
         ***/
        bool use_node = tid <  n_nodes0;
#if 0
        { prefix[tid] = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
        __syncthreads();
        int node  = prefix[min(tid, n_nodes0 - 1)];
#else  /* eg: seems to work, but I do not remember if that will *always* work */
        int node;
        { node  = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
#endif

#if 0
        if(n_nodes0 > 0){       //Work around pre 4.1 compiler bug
          n_nodes0 -= DIM;
        }
#else
        n_nodes0 -= DIM;
#endif

        /***
         **** --> process each of the nodes in the list in parallel
         ***/

#ifndef TEXTURES
        float4 nodeSize = boxSizeInfo[node];                   //Fetch the size of the box. Size.w = child info
        float4 node_pos = boxCenterInfo[node];                 //Fetch the center of the box. center.w = opening info
#else
        float4 nodeSize =  tex1Dfetch(texNodeSize, node);
        float4 node_pos =  tex1Dfetch(texNodeCenter, node);
#endif

        int node_data = __float_as_int(nodeSize.w);

        //Check if a cell has to be opened
#ifndef TEXTURES
        float4 nodeCOM = multipole_data[node*3];
#else
        float4 nodeCOM = tex1Dfetch(texMultipole,node*3);
#endif

        nodeCOM.w      = node_pos.w;
        bool   split   = split_node_grav_impbh(nodeCOM, group_pos, groupSize);


        bool leaf       = node_pos.w <= 0;  //Small AND equal incase of a 1 particle cell       //Check if it is a leaf
        //         split = true;

        uint mask    = BTEST((split && !leaf) && use_node);               // mask = #FFFFFFFF if use_node+split+not_a_leaf==true, otherwise zero
        int child    =    node_data & 0x0FFFFFFF;                         //Index to the first child of the node
        int nchild   = (((node_data & 0xF0000000) >> 28)) & mask;         //The number of children this node has

        /***
         **** --> calculate prefix
         ***/


        int n_total = calc_prefix<DIM2>(prefix, tid,  nchild);              // inclusive scan to compute memory offset of each child (return total # of children)
        int offset  = prefix[tid];
        offset     += n_offset - nchild;                                  // convert inclusive into exclusive scan for referencing purpose
        __syncthreads();                                                   // thread barrier

        for (int i = n_offset; i < n_offset + n_total; i += DIM)         //nullify part of the array that will be filled with children
          nodesM[tid + i] = 0;                                          //but do not touch those parts which has already been filled
        __syncthreads();                                                 //Thread barrier to make sure all warps finished writing data

        bool flag = (split && !leaf) && use_node;                        //Flag = use_node + split + not_a_leaf;Use only non_leaf nodes that are to be split
        if (flag) nodesM[offset] = child;                            //Thread with the node that is about to be split
        //writes the first child in the array of nodes
        /*** in the following 8 lines, we calculate indexes of all the children that have to be walked from the index of the first child***/
        if (flag && nodesM[offset + 1] == 0) nodesM[offset + 1] = child + 1; 
        if (flag && nodesM[offset + 2] == 0) nodesM[offset + 2] = child + 2;
        if (flag && nodesM[offset + 3] == 0) nodesM[offset + 3] = child + 3;
        if (flag && nodesM[offset + 4] == 0) nodesM[offset + 4] = child + 4;
        if (flag && nodesM[offset + 5] == 0) nodesM[offset + 5] = child + 5;
        if (flag && nodesM[offset + 6] == 0) nodesM[offset + 6] = child + 6;
        if (flag && nodesM[offset + 7] == 0) nodesM[offset + 7] = child + 7;
        __syncthreads();

        n_offset += n_total;    //Increase the offset in the array by the number of newly added nodes


        /***
         **** --> save list of nodes to LMEM
         ***/

        /*** if half of shared memory or more is filled with the the nodes, dump these into slowmem stack ***/
        while(n_offset >= DIM) 
        {
          n_offset -= DIM;
          const int offs1 = ACCS<SHIFT>(n_stack1);
          nstack[offs1] = nodesM[n_offset + tid];   n_stack1++;
          n_nodes1 += DIM;

          if((n_stack1 - c_stack0) >= (LMEM_STACK_SIZE << SHIFT))
          {
            //We overwrote our current stack
            apprCount = -1; 
            return acc_i;	 
          }
        }

        __syncthreads();



        /******************************/
        /******************************/
        /*****     EVALUATION     *****/
        /******************************/
        /******************************/
#if 1
        /***********************************/
        /******       APPROX          ******/
        /***********************************/

        n_total = calc_prefix<DIM2>(prefix, tid,  1 - (split || !use_node));
        offset = prefix[tid];

        if (!split && use_node) approxM[n_approx + offset - 1] = node;
        __syncthreads();
        n_approx += n_total;

        while (n_approx >= DIM) 
        {
          n_approx -= DIM;
          const int address      = (approxM[n_approx + tid] << 1) + approxM[n_approx + tid];
#ifndef TEXTURES
          const float4 monopole  = multipole_data[address    ];
#if 0
          float4 octopole0 = multipole_data[address + 1];
          float4 octopole1 = multipole_data[address + 2];
#endif
#else
          const float4 monopole  = tex1Dfetch(texMultipole, address);
#if 0
          float4 octopole0 = tex1Dfetch(texMultipole, address + 1);
          float4 octopole1 = tex1Dfetch(texMultipole, address + 2);
#endif
#endif

          node_mon0[tid] = monopole.w;
          node_mon1[tid] = make_float3(monopole.x,  monopole.y,  monopole.z);
          __syncthreads();

#if 1
#pragma unroll 16
          for (int i = 0; i < DIMx; i++)
            acc_i = add_acc(acc_i, pos_i, node_mon0[offs + i], node_mon1[offs+i], eps2);
          apprCount += DIMx;
          __syncthreads();
#endif
        }
        __syncthreads();
#endif

#if 1
        /***********************************/
        /******       DIRECT          ******/
        /***********************************/


        flag            = split && leaf && use_node;                                //flag = split + leaf + use_node
        const int jbody = node_data & BODYMASK;                                     //the first body in the leaf
        const int nbody = (((node_data & INVBMASK) >> LEAFBIT)+1) & BTEST(flag);    //number of bodies in the leaf masked with the flag

        body_list[tid] = directM[tid];                                            //copy list of bodies from previous pass to body_list

        // step 1
        calc_prefix<DIM2>(prefix, tid, flag);                                // inclusive scan on flags to construct array
        const int offset1 = prefix[tid];
        
        // step 2
        int n_bodies  = calc_prefix<DIM2>(prefix, tid, nbody);              // inclusive scan to compute memory offset for each body
        offset = prefix[tid];
        __syncthreads();

        if (flag) prefix[offset1 - 1] = tid;                             //with tidś whose leaves have to be opened
        __syncthreads();                                                      //thread barrier, make sure all warps completed the job

        directM[tid]  = offset;                                       //Store a copy of inclusive scan in direct
        offset       -= nbody;                                              //convert inclusive int oexclusive scan
        offset       += 1;                                                  //add unity, since later prefix0[tid] == 0 used to check barrier

        int nl_pre = 0;                                                     //Number of leaves that have already been processed

        while (n_bodies > 0) 
        {
          int nb    = min(n_bodies, NJMAX - n_direct);                    //Make sure number of bides to be extracted does not exceed
          //the amount of allocated shared memory

          // step 0                                                      //nullify part of the body_list that will be filled with bodies
          for (int i = n_direct; i < n_direct + nb; i += DIM)            //from the leaves that are being processed
            body_list[i + tid] = 0;
          __syncthreads();

          //step 1:
          if (flag && (directM[tid] <= nb) && (offset > 0))        //make sure that the thread indeed carries a leaf
            body_list[n_direct + offset- 1] = -1-jbody;            //whose bodies will be extracted
          __syncthreads();

          // step 2:
          const int nl = inclusive_segscan_array<DIM2>(&body_list[n_direct], nb, tid);
          nb = directM[prefix[nl_pre + nl - 1]];                       // number of bodies stored in these leaves
          __syncthreads();


          /*****************************************************************************
           *  example of what is accomplished in steps 0-2                             *
           *       ---------------------------                                         *
           * step 0: body_list = 000000000000000000000                                 *
           * step 1: body_list = n000m000p000000q00r00 n,m,.. = -1-jbody_n,m...        *
           * step 2: body_list = n n+1 n+2 n+3 m m+1 m+2 m+3 p p+1 p+2 p+3 p+4 p+5 ... *
           *****************************************************************************/

          n_bodies     -= nb;                                   //subtract from n_bodies number of bodies that have been extracted
          nl_pre       += nl;                                   //increase the number of leaves that where processed
          directM[tid] -= nb;                                   //subtract the number of extracted bodies in this pass
          offset        = max(offset - nb, 0);
          n_direct     += nb;                                  //increase the number of bodies to be procssed

          while(n_direct >= DIM) 
          {
            n_direct -= DIM;


            const float4 posj  = body_pos[body_list[n_direct + tid]];
#if 0
            const float4 posj  = tex1Dfetch(texBody, body_list[n_direct + tid]);
#endif
            sh_mass[tid] = posj.w;
            sh_pos [tid] = make_float3(posj.x, posj.y, posj.z);

            __syncthreads();
#if 1
#pragma unroll 16
            for (int j = 0; j < DIMx; j++)
              acc_i = add_acc(acc_i, pos_i, sh_mass[offs + j], sh_pos[offs + j], eps2);
#if 0
            direCount += DIMx;
#endif
            __syncthreads();
#endif
          }

        }
        directM[tid] = body_list[tid];
        __syncthreads();
#endif
      } //end lvl


      n_nodes1 += n_offset;
      if (n_offset > 0)
      { 
        nstack[ACCS<SHIFT>(n_stack1)] = nodesM[tid];   n_stack1++; 
        if((n_stack1 - c_stack0) >= (LMEM_STACK_SIZE << SHIFT))
        {
          //We overwrote our current stack
          apprCount = -1; 
          return acc_i;	 
        }
      }
      __syncthreads();


      /***
       **** --> copy nodes1 to nodes0: done by reassigning the pointers
       ***/
      n_nodes0    = n_nodes1;

      n_stack_pre = n_stack0;
      n_stack0    = n_stack1;

    }//end while   levels
  }//end for


  if(n_approx > 0)
  {
    if (tid < n_approx) 
    {
      const int address = (approxM[tid] << 1) + approxM[tid];
#ifndef TEXTURES
      float4 monopole  = multipole_data[address    ];
      float4 octopole0 = multipole_data[address + 1];
      float4 octopole1 = multipole_data[address + 2];
#else
      float4 monopole  = tex1Dfetch(texMultipole, address);
      float4 octopole0 = tex1Dfetch(texMultipole, address + 1);
      float4 octopole1 = tex1Dfetch(texMultipole, address + 2);
#endif

      node_mon0[tid] = monopole.w;
      node_mon1[tid] = make_float3(monopole.x,  monopole.y,  monopole.z);

    } else {

      //Set non-active memory locations to zero
      node_mon0[tid] = 0.0f;
      node_mon1[tid] = make_float3(1.0e10f, 1.0e10f, 1.0e10f);

    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < DIMx; i++)
      acc_i = add_acc(acc_i, pos_i, node_mon0[offs + i], node_mon1[offs+i],eps2);
    apprCount += DIMx;

    __syncthreads();
  } //if n_approx > 0

  if(n_direct > 0)
  {
    if (tid < n_direct) 
    {
      const float4 posj = body_pos[directM[tid]];
#if 0
      const float4 posj  = tex1Dfetch(texBody, direct[tid]);
#endif
      sh_mass[tid] = posj.w;
      sh_pos [tid] = make_float3(posj.x, posj.y, posj.z);
    } else {
      sh_mass[tid] = 0.0f;
      sh_pos [tid] = make_float3(1.0e10f, 1.0e10f, 1.0e10f);
    }

    __syncthreads();
#pragma unroll
    for (int j = 0; j < DIMx; j++) 
      acc_i = add_acc(acc_i, pos_i, sh_mass[offs + j], sh_pos[offs + j], eps2);
#if 0
    direCount += DIMx;
#endif
    __syncthreads();
  }

  /***
   **** --> reduce data between threads
   ***/
  sh_pot[tid] = acc_i.w;
  sh_acc[tid] = make_float3(acc_i.x, acc_i.y, acc_i.z);
  __syncthreads();

  if (ty == 0) 
#pragma unroll
    for (int i = 1; i < DIMy; i++) 
    {
      const int idx = (i << DIM2x) + tx;
      acc_i.w += sh_pot[idx];
      acc_i.x += sh_acc[idx].x;
      acc_i.y += sh_acc[idx].y;
      acc_i.z += sh_acc[idx].z;
    }
  __syncthreads();


  //Sum the interaction counters
  float  *sh_ds2 = (float*)&sh_acc[DIM];
  int    *sh_ngb = (int*  )&sh_ds2[DIM];
  sh_ds2[tid] = direCount;
  sh_ngb[tid] = apprCount;

  __syncthreads();


  if (ty == 0) {
#pragma unroll
    for (int i = 1; i < DIMy; i++){
      int idx = (i << DIM2x) + tx;
      direCount  += sh_ds2[idx];
      apprCount  += sh_ngb[idx];
    }
  }
  __syncthreads();

  return acc_i;
}


  extern "C" __global__ void
__launch_bounds__(NTHREAD)
  dev_approximate_gravity(const int n_active_groups,
      int    n_bodies,
      float eps2,
      uint2 node_begend,
      int    *active_groups,
      real4  *body_pos,
      real4  *multipole_data,
      float4 *acc_out,
      int    *ngb_out,
      int    *active_inout,
      int2   *interactions,
      float4  *boxSizeInfo,
      float4  *groupSizeInfo,
      float4  *boxCenterInfo,
      float4  *groupCenterInfo,
      real4   *body_vel,
      int     *MEM_BUF) {
    //                                                    int     grpOffset){


    const int blockDim2 = NTHREAD2;
    __shared__ int shmem[11*(1 << blockDim2)];
    //    __shared__ int shmem[24*(1 << blockDim2)]; is possible on FERMI
    //    int             lmem[LMEM_STACK_SIZE];



    /*********** check if this block is linked to a leaf **********/

    int bid = gridDim.x * blockIdx.y + blockIdx.x;

    while(true)
    {

      if(threadIdx.x == 0)
      {
        bid         = atomicAdd(&active_inout[n_bodies], 1);
        shmem[0]    = bid;
      }
      __syncthreads();

      bid   = shmem[0];

      if (bid >= n_active_groups) return;


      int tid = threadIdx.y * blockDim.x + threadIdx.x;

      int grpOffset = 0;

      //   volatile int *lmem = &MEM_BUF[blockIdx.x*LMEM_STACK_SIZE*blockDim.x + threadIdx.x*LMEM_STACK_SIZE];
      //   int *lmem = &MEM_BUF[blockIdx.x*LMEM_STACK_SIZE*blockDim.x + threadIdx.x*LMEM_STACK_SIZE];
      int *lmem = &MEM_BUF[blockIdx.x*(LMEM_STACK_SIZE*blockDim.x + LMEM_EXTRA_SIZE)];


      /*********** set necessary thread constants **********/
#ifdef DO_BLOCK_TIMESTEP
      real4 curGroupSize    = groupSizeInfo[active_groups[bid + grpOffset]];
#else
      real4 curGroupSize    = groupSizeInfo[bid + grpOffset];
#endif
      int   groupData       = __float_as_int(curGroupSize.w);
      uint body_i           =   groupData & CRITMASK;
      uint nb_i             = ((groupData & INVCMASK) >> CRITBIT) + 1;

#ifdef DO_BLOCK_TIMESTEP
      real4 group_pos       = groupCenterInfo[active_groups[bid + grpOffset]];
#else
      real4 group_pos       = groupCenterInfo[bid + grpOffset];
#endif
      //   if(tid == 0)
      //   printf("[%f %f %f %f ] \n [%f %f %f %f ] %d %d \n",
      //           curGroupSize.x, curGroupSize.y, curGroupSize.z, curGroupSize.w,
      //           group_pos.x, group_pos.y, group_pos.z, group_pos.w, body_i, nb_i);


      int DIM2x = 0;
      while (((nb_i - 1) >> DIM2x) > 0) DIM2x++;

      DIM2x     = max(DIM2x,4);
      int DIM2y = blockDim2 - DIM2x;

      int tx = tid & ((1 << DIM2x) - 1);
      int ty = tid >> DIM2x;

      body_i += tx%nb_i;

      //float4 pos_i = tex1Dfetch(bodies_pos_ref, body_i);   // texture read: 4 floats


      float4 pos_i = body_pos[body_i];


      int ngb_i;

      float4 acc_i = {0.0f, 0.0f, 0.0f, 0.0f};

#ifdef INDSOFT
      eps2 = body_vel[body_i].w;
      float group_eps = eps2;

      volatile float *reduc = (float*) &shmem[0];
      reduc[threadIdx.x] = eps2;

      //Find the maximum softening value for the particles in this group
      __syncthreads();
      // do reduction in shared mem
      if(blockDim.x >= 512) if (tid < 256) {reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 256]);} __syncthreads();
      if(blockDim.x >= 256) if (tid < 128) {reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 128]);} __syncthreads();
      if(blockDim.x >= 128) if (tid < 64)  {reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 64]);} __syncthreads();
      if(blockDim.x >= 64) if (tid < 32) { reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 32]);}
      if(blockDim.x >= 32) if (tid < 16) { reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 16]);}

      if(tid < 8)
      {
        reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 8]);
        reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 4]);
        reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 2]);
        reduc[threadIdx.x] = group_eps = fmaxf(group_eps, reduc[threadIdx.x + 1]);
      }
      __syncthreads();

      group_eps = reduc[0];
#else
      float group_eps  = 0;
#endif

      int apprCount = 0;
      int direCount = 0;


      acc_i = approximate_gravity<blockDim2, 0>( DIM2x, DIM2y, tid, tx, ty,
          body_i, pos_i, group_pos,
          eps2, node_begend,
          multipole_data, body_pos,
          shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
          group_eps, body_vel);
      if(apprCount < 0)
      {

        //Try to get access to the big stack, only one block per time is allowed
        if(threadIdx.x == 0)
        {
          int res = atomicExch(&active_inout[n_bodies+1], 1); //If the old value (res) is 0 we can go otherwise sleep
          int waitCounter  = 0;
          while(res != 0)
          {
            //Sleep
            for(int i=0; i < (1024); i++)
            {
              waitCounter += 1;
            }
            //Test again
            shmem[0] = waitCounter;
            res = atomicExch(&active_inout[n_bodies+1], 1); 
          }
        }

        __syncthreads();

        lmem = &MEM_BUF[gridDim.x*(LMEM_STACK_SIZE*blockDim.x + LMEM_EXTRA_SIZE)];    //Use the extra large buffer
        apprCount = direCount = 0;
        acc_i = approximate_gravity<blockDim2, 8>( DIM2x, DIM2y, tid, tx, ty,
            body_i, pos_i, group_pos,
            eps2, node_begend,
            multipole_data, body_pos,
            shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
            group_eps, body_vel);
        lmem = &MEM_BUF[blockIdx.x*(LMEM_STACK_SIZE*blockDim.x + LMEM_EXTRA_SIZE)];

        if(threadIdx.x == 0)
        {
          atomicExch(&active_inout[n_bodies+1], 0); //Release the lock
        }
      }//end if apprCount < 0

      if (tid < nb_i) {
        acc_out     [body_i] = acc_i;
        ngb_out     [body_i] = ngb_i;
        active_inout[body_i] = 1;
        interactions[body_i].x = apprCount;
        interactions[body_i].y = direCount ;
      }


    }     //end while
  }

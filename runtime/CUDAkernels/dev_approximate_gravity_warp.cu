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

#if NCRIT > 2*WARP_SIZE
#error "NCRIT in include/node_specs.h must be <= WARP_SIZE"
#endif


#define laneId (threadIdx.x & (WARP_SIZE - 1))
#define warpId (threadIdx.x >> WARP_SIZE2)

#define BTEST(x) (-(int)(x))

/************************************/
/*********   PREFIX SUM   ***********/
/************************************/

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

__device__ __forceinline__ int calc_prefix(int* prefix, int value) 
{
  prefix[laneId] = inclusive_scan_warp<WARP_SIZE2>(value);
  return prefix[WARP_SIZE-1];
}


__device__ int calc_prefix(int N, int* prefix_in) 
{

  int y = calc_prefix(prefix_in, prefix_in[laneId]);
  if (N <= WARP_SIZE) return y;

  for (int p = WARP_SIZE; p < N; p += WARP_SIZE) 
  {
    int *prefix = &prefix_in[p];
    const int y1 = calc_prefix(prefix, prefix[laneId]);
    prefix[laneId] += y;
    y += y1;
  }

  return y;
} 

/************************************/
/********* SEGMENTED SCAN ***********/
/************************************/

  template<const int SIZE2>
__device__ __forceinline__ int inclusive_segscan_warp(int value, const int distance)
{

  const int SIZE = 1 << SIZE2; 

  for (int i = 0; i < SIZE2; i++)
    value += __shfl_up(value, 1 << i, SIZE) & BTEST(laneId >= (1<<i)) & BTEST((1<<i) <= distance);

  return value;
}

__device__ __forceinline__ int lanemask_le()
{
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}

__device__ __forceinline__ int inclusive_segscan_block64(
    int *shmem, const int packed_value, int &dist_block, int &nseg)
{
  const int  flag = packed_value < 0;
  const int  mask = BTEST(flag);
  const int value = (mask & (-1-packed_value)) + (~mask & 1);

  const int flags = __ballot(flag);

  nseg      += __popc      (flags) ;
  dist_block = __clz(__brev(flags));

  const int distance = __clz(flags & lanemask_le()) + laneId - 31;
  shmem[laneId] = inclusive_segscan_warp<WARP_SIZE2>(value, distance);
  const int val = shmem[WARP_SIZE - 1];
  return val;
}

/* does not work if segment size > WARP_SIZE */
__device__ __forceinline__ int inclusive_segscan_array(int *shmem_in, const int N)
{
  int dist, nseg = 0;
  int y  = inclusive_segscan_block64(shmem_in, shmem_in[laneId], dist, nseg);
  if (N <= WARP_SIZE) return nseg;

  for (int p = WARP_SIZE; p < N; p += WARP_SIZE)
  {
    int *shmem = shmem_in + p;
    int y1  = inclusive_segscan_block64(shmem, shmem[laneId], dist, nseg);
    shmem[laneId] += y & BTEST(laneId < dist);
    y = y1;
  }

  return nseg;
}


/**************************************/
/*************** Tree walk ************/
/**************************************/

  template<int SHIFT>
__forceinline__ __device__ int ACCS(const int i)
{
  return (i & ((LMEM_STACK_SIZE << SHIFT) - 1))*blockDim.x + threadIdx.x;
}

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


/*******************************/
/****** Opening criterion ******/
/*******************************/

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

/*******************************/
/******  Force tree-walk  ******/
/*******************************/


template<const int SHIFT, const int BLOCKDIM2, const int NI>
__device__ __forceinline__ void approximate_gravity(
    float4 pos_i[NI],
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
    real4 acc_i[NI])
{


  /*********** set necessary thread constants **********/

  const int offs = 0;
  const int DIM2 = WARP_SIZE2;
  const int DIM  = WARP_SIZE;

  /*********** shared memory distribution **********/

  //  begin,    end,   size
  // -----------------------
  const int stack_sz = (LMEM_STACK_SIZE << SHIFT) << DIM2;  /* stack allocated per thread-block */
  const int nWarps2 = BLOCKDIM2 - WARP_SIZE2;
  int *approxL = lmem + stack_sz + (LMEM_EXTRA_SIZE >> nWarps2) * warpId;

  int *directS = shmem;                              //  0*DIM,  1*DIM,  1*DIM
  int *nodesS  = directS + DIM;                      //  1*DIM, 10*DIM,  9*DIM
  int *prefix  = nodesS  + DIM*8;                    //  9*DIM, 10*DIM,  1*DIM

  const int NJMAX = DIM*3;
  int    *body_list = (int*   )&nodesS   [  DIM]; //  2*DIM,   5*DIM,  2*DIM
  float  *sh_mass   = (float* )&body_list[NJMAX]; //  5*DIM,   6*DIM,  1*DIM
  float3 *sh_pos    = (float3*)&sh_mass  [  DIM]; //  6*DIM,   9*DIM   3*DIM

  int *approxM = approxL;
  int *directM = directS;
  int * nodesM =  nodesS;


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

    { nstack[ACCS<SHIFT>(n_stack0)] = root_node + laneId;   n_stack0++; }

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
        bool use_node = laneId <  n_nodes0;
#if 0
        { prefix[laneId] = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
        int node  = prefix[min(laneId, n_nodes0 - 1)];
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


        int n_total = calc_prefix(prefix,  nchild);              // inclusive scan to compute memory offset of each child (return total # of children)
        int offset  = prefix[laneId];
        offset     += n_offset - nchild;                                  // convert inclusive into exclusive scan for referencing purpose

        for (int i = n_offset; i < n_offset + n_total; i += DIM)         //nullify part of the array that will be filled with children
          nodesM[laneId + i] = 0;                                          //but do not touch those parts which has already been filled

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

        n_offset += n_total;    //Increase the offset in the array by the number of newly added nodes


        /***
         **** --> save list of nodes to LMEM
         ***/

        /*** if half of shared memory or more is filled with the the nodes, dump these into slowmem stack ***/
        while(n_offset >= DIM) 
        {
          n_offset -= DIM;
          const int offs1 = ACCS<SHIFT>(n_stack1);
          nstack[offs1] = nodesM[n_offset + laneId];   n_stack1++;
          n_nodes1 += DIM;

          if((n_stack1 - c_stack0) >= (LMEM_STACK_SIZE << SHIFT))
          {
            //We overwrote our current stack
            apprCount = -1; 
            return;
          }
        }




        /******************************/
        /******************************/
        /*****     EVALUATION     *****/
        /******************************/
        /******************************/
#if 1
        /***********************************/
        /******       APPROX          ******/
        /***********************************/

        n_total = calc_prefix(prefix, 1 - (split || !use_node));
        offset = prefix[laneId];

        if (!split && use_node) approxM[n_approx + offset - 1] = node;
        n_approx += n_total;

        while (n_approx >= DIM) 
        {
          n_approx -= DIM;
          const int address      = (approxM[n_approx + laneId] << 1) + approxM[n_approx + laneId];
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

          sh_mass[laneId] = monopole.w;
          sh_pos [laneId] = make_float3(monopole.x,  monopole.y,  monopole.z);

#pragma unroll 16
          for (int i = 0; i < WARP_SIZE; i++)
            for (int k = 0; k < NI; k++)
              acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[offs + i], sh_pos[offs+i], eps2);
          apprCount += WARP_SIZE*NI;
        }
#endif

#if 1
        /***********************************/
        /******       DIRECT          ******/
        /***********************************/


        flag            = split && leaf && use_node;                                //flag = split + leaf + use_node
        const int jbody = node_data & BODYMASK;                                     //the first body in the leaf
        const int nbody = (((node_data & INVBMASK) >> LEAFBIT)+1) & BTEST(flag);    //number of bodies in the leaf masked with the flag

        body_list[laneId] = directM[laneId];                                            //copy list of bodies from previous pass to body_list

        // step 1
        calc_prefix(prefix, flag);                                // inclusive scan on flags to construct array
        const int offset1 = prefix[laneId];

        // step 2
        int n_bodies  = calc_prefix(prefix, nbody);              // inclusive scan to compute memory offset for each body
        offset = prefix[laneId];

        if (flag) prefix[offset1 - 1] = laneId;                             //with tidś whose leaves have to be opened

        directM[laneId]  = offset;                                       //Store a copy of inclusive scan in direct
        offset       -= nbody;                                              //convert inclusive int oexclusive scan
        offset       += 1;                                                  //add unity, since later prefix0[tid] == 0 used to check barrier

        int nl_pre = 0;                                                     //Number of leaves that have already been processed

        while (n_bodies > 0) 
        {
          int nb    = min(n_bodies, NJMAX - n_direct);                    //Make sure number of bides to be extracted does not exceed
          //the amount of allocated shared memory

          // step 0                                                      //nullify part of the body_list that will be filled with bodies
          for (int i = n_direct; i < n_direct + nb; i += DIM)            //from the leaves that are being processed
            body_list[i + laneId] = 0;

          //step 1:
          if (flag && (directM[laneId] <= nb) && (offset > 0))        //make sure that the thread indeed carries a leaf
            body_list[n_direct + offset- 1] = -1-jbody;            //whose bodies will be extracted

          // step 2:
          const int nl = inclusive_segscan_array(&body_list[n_direct], nb);
          nb = directM[prefix[nl_pre + nl - 1]];                       // number of bodies stored in these leaves


          /*****************************************************************************
           *  example of what is accomplished in steps 0-2                             *
           *       ---------------------------                                         *
           * step 0: body_list = 000000000000000000000                                 *
           * step 1: body_list = n000m000p000000q00r00 n,m,.. = -1-jbody_n,m...        *
           * step 2: body_list = n n+1 n+2 n+3 m m+1 m+2 m+3 p p+1 p+2 p+3 p+4 p+5 ... *
           *****************************************************************************/

          n_bodies     -= nb;                                   //subtract from n_bodies number of bodies that have been extracted
          nl_pre       += nl;                                   //increase the number of leaves that where processed
          directM[laneId] -= nb;                                   //subtract the number of extracted bodies in this pass
          offset        = max(offset - nb, 0);
          n_direct     += nb;                                  //increase the number of bodies to be procssed

          while(n_direct >= DIM) 
          {
            n_direct -= DIM;


            const float4 posj  = body_pos[body_list[n_direct + laneId]];
#if 0
            const float4 posj  = tex1Dfetch(texBody, body_list[n_direct + tid]);
#endif
            sh_mass[laneId] = posj.w;
            sh_pos [laneId] = make_float3(posj.x, posj.y, posj.z);

#pragma unroll 16
            for (int j = 0; j < WARP_SIZE; j++)
              for (int k = 0; k < NI; k++)
                acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[offs + j], sh_pos[offs + j], eps2);
            direCount += WARP_SIZE*NI;
          }

        }
        directM[laneId] = body_list[laneId];
#endif
      } //end lvl


      n_nodes1 += n_offset;
      if (n_offset > 0)
      { 
        nstack[ACCS<SHIFT>(n_stack1)] = nodesM[laneId];   n_stack1++; 
        if((n_stack1 - c_stack0) >= (LMEM_STACK_SIZE << SHIFT))
        {
          //We overwrote our current stack
          apprCount = -1; 
          return;
        }
      }


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
    if (laneId < n_approx) 
    {
      const int address = (approxM[laneId] << 1) + approxM[laneId];
#ifndef TEXTURES
      float4 monopole  = multipole_data[address    ];
      float4 octopole0 = multipole_data[address + 1];
      float4 octopole1 = multipole_data[address + 2];
#else
      float4 monopole  = tex1Dfetch(texMultipole, address);
      float4 octopole0 = tex1Dfetch(texMultipole, address + 1);
      float4 octopole1 = tex1Dfetch(texMultipole, address + 2);
#endif

      sh_mass[laneId] = monopole.w;
      sh_pos [laneId] = make_float3(monopole.x,  monopole.y,  monopole.z);

    } else {

      //Set non-active memory locations to zero
      sh_mass[laneId] = 0.0f;
      sh_pos [laneId] = make_float3(1.0e10f, 1.0e10f, 1.0e10f);

    }
#pragma unroll 16
    for (int i = 0; i < WARP_SIZE; i++)
      for (int k = 0; k < NI; k++)
        acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[offs + i], sh_pos[offs+i],eps2);
    apprCount += WARP_SIZE*NI;

  } //if n_approx > 0

  if(n_direct > 0)
  {
    if (laneId < n_direct) 
    {
      const float4 posj = body_pos[directM[laneId]];
#if 0
      const float4 posj  = tex1Dfetch(texBody, direct[tid]);
#endif
      sh_mass[laneId] = posj.w;
      sh_pos [laneId] = make_float3(posj.x, posj.y, posj.z);
    } else {
      sh_mass[laneId] = 0.0f;
      sh_pos [laneId] = make_float3(1.0e10f, 1.0e10f, 1.0e10f);
    }

#pragma unroll 16
    for (int j = 0; j < WARP_SIZE; j++) 
      for (int k = 0; k < NI; k++)
        acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[offs + j], sh_pos[offs + j], eps2);
    direCount += WARP_SIZE*NI;
  }
}


extern "C" __global__ void
#if 0 /* casues 164 bytes spill to lmem with NTHREAD = 128 */
__launch_bounds__(NTHREAD)
#endif
  dev_approximate_gravity(
      const int n_active_groups,
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
      int     *MEM_BUF) 
{
  const int blockDim2 = NTHREAD2;
  const int shMemSize = 10 * (1 << blockDim2);
  __shared__ int shmem_pool[shMemSize];

  const int nWarps2 = blockDim2 - WARP_SIZE2;
  const int sh_offs = (shMemSize >> nWarps2) * warpId;
  int *shmem = shmem_pool + sh_offs;

  /*********** check if this block is linked to a leaf **********/

  int *lmem = &MEM_BUF[blockIdx.x*(LMEM_STACK_SIZE*blockDim.x + LMEM_EXTRA_SIZE)];
  int  bid  = gridDim.x * blockIdx.y + blockIdx.x;

  while(true)
  {

    if(laneId == 0)
    {
      bid         = atomicAdd(&active_inout[n_bodies], 1);
      shmem[0]    = bid;
    }

    bid   = shmem[0];

    if (bid >= n_active_groups) return;

    int grpOffset = 0;

    /*********** set necessary thread constants **********/
#ifdef DO_BLOCK_TIMESTEP
    real4 curGroupSize    = groupSizeInfo[active_groups[bid + grpOffset]];
#else
    real4 curGroupSize    = groupSizeInfo[bid + grpOffset];
#endif
    const int   groupData       = __float_as_int(curGroupSize.w);
    const uint body_addr        =   groupData & CRITMASK;
    const uint nb_i             = ((groupData & INVCMASK) >> CRITBIT) + 1;

#ifdef DO_BLOCK_TIMESTEP
    real4 group_pos       = groupCenterInfo[active_groups[bid + grpOffset]];
#else
    real4 group_pos       = groupCenterInfo[bid + grpOffset];
#endif

    uint body_i[2];
    int ni = nb_i <= WARP_SIZE ? 1 : 2;
    body_i[0] = body_addr + laneId%nb_i;
    body_i[1] = body_addr + WARP_SIZE + laneId%(nb_i - WARP_SIZE);

    float4 pos_i[2];
    float4 acc_i[2];

    pos_i[0] = body_pos[body_i[0]];
    pos_i[1] = body_pos[body_i[1]];
    acc_i[0] = acc_i[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    int ngb_i;

    const float group_eps  = 0;

    int apprCount = 0;
    int direCount = 0;

    if (ni == 1)
      approximate_gravity<0, blockDim2, 1>(
          pos_i, group_pos,
          eps2, node_begend,
          multipole_data, body_pos,
          shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
          group_eps, 
          acc_i);
    else
      approximate_gravity<0, blockDim2, 2>(
          pos_i, group_pos,
          eps2, node_begend,
          multipole_data, body_pos,
          shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
          group_eps, 
          acc_i);

#if 0 /* this increase lmem spill count */
    if(apprCount < 0)
    {

      //Try to get access to the big stack, only one block per time is allowed
      if(laneId == 0)
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

      lmem = &MEM_BUF[gridDim.x*(LMEM_STACK_SIZE*blockDim.x + LMEM_EXTRA_SIZE)];    //Use the extra large buffer
      apprCount = direCount = 0;
      acc_i[0] = acc_i[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (ni == 1)
        approximate_gravity<0, blockDim2, 1>(
            pos_i, group_pos,
            eps2, node_begend,
            multipole_data, body_pos,
            shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
            group_eps, 
            acc_i);
      else
        approximate_gravity<0, blockDim2, 2>(
            pos_i, group_pos,
            eps2, node_begend,
            multipole_data, body_pos,
            shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
            group_eps, 
            acc_i);

      lmem = &MEM_BUF[blockIdx.x*(LMEM_STACK_SIZE*blockDim.x + LMEM_EXTRA_SIZE)];

      if(threadIdx.x == 0)
      {
        atomicExch(&active_inout[n_bodies+1], 0); //Release the lock
      }
    }//end if apprCount < 0
#endif

    if (laneId < nb_i) 
    {
      const int addr = body_i[0];
      acc_out     [addr] = acc_i[0];
      ngb_out     [addr] = ngb_i;
      active_inout[addr] = 1;
      interactions[addr].x = apprCount;
      interactions[addr].y = direCount ;
      if (ni == 2)
      {
        const int addr = body_i[1];
        acc_out     [addr] = acc_i[1];
        ngb_out     [addr] = ngb_i;
        active_inout[addr] = 1;     
        interactions[addr].x = 0; // apprCount;    to avoid doubling the intearction count
        interactions[addr].y = 0; // direCount ;
      }
    }
  }     //end while
}


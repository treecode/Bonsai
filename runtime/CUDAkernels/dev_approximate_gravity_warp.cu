#include "bonsai.h"
// #include "support_kernels.cu0
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

#if 1
#define _QUADRUPOLE_
#endif

/************************************/
/*********   PREFIX SUM   ***********/
/************************************/

static __device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
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
static __device__ __forceinline__ uint inclusive_scan_warp(int mysum)
{
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

/* inclusive prefix sum for a warp */
static __device__ __forceinline__ int inclusive_scan_warp(int* prefix, int value) 
{
  prefix[laneId] = inclusive_scan_warp<WARP_SIZE2>(value);
  return prefix[WARP_SIZE-1];
}


/* inclusive prefix sum for an array */
/*static __device__ int inclusive_scan_array(int N, int* prefix_in) 
{

  int y = inclusive_scan_warp(prefix_in, prefix_in[laneId]);
  if (N <= WARP_SIZE) return y;

  for (int p = WARP_SIZE; p < N; p += WARP_SIZE) 
  {
    int *prefix = &prefix_in[p];
    const int y1 = inclusive_scan_warp(prefix, prefix[laneId]);
    prefix[laneId] += y;
    y += y1;
  }

  return y;
} */

/**** binary scans ****/

static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}


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

/************************************/
/********* SEGMENTED SCAN ***********/
/************************************/

static __device__ __forceinline__ int ShflSegScanStepB(
            int partial,
            uint distance,
            uint up_offset)
{
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0, %1, %2, 0;"
      "setp.le.u32 p, %2, %3;"
      "@p add.u32 %1, r0, %1;"
      "mov.u32 %0, %1;}"
      : "=r"(partial) : "r"(partial), "r"(up_offset), "r"(distance));
  return partial;
}
  template<const int SIZE2>
static __device__ __forceinline__ int inclusive_segscan_warp_step(int value, const int distance)
{

#if 0
  const int SIZE = 1 << SIZE2; 

  for (int i = 0; i < SIZE2; i++)
    value += __shfl_up(value, 1 << i, SIZE) & BTEST(laneId >= (1<<i)) & BTEST((1<<i) <= distance);
#else
  for (int i = 0; i < SIZE2; i++)
    value = ShflSegScanStepB(value, distance, 1<<i);
#endif

  return value;
}

static __device__ __forceinline__ int lanemask_le()
{
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__ int inclusive_segscan_warp(
    int *shmem, const int packed_value, int &dist_block, int &nseg)
{
  const int  flag = packed_value < 0;
  const int  mask = BTEST(flag);
  const int value = (mask & (-1-packed_value)) + (~mask & 1);

  const int flags = __ballot(flag);

  nseg      += __popc      (flags) ;
  dist_block = __clz(__brev(flags));

  const int distance = __clz(flags & lanemask_le()) + laneId - 31;
  shmem[laneId] = inclusive_segscan_warp_step<WARP_SIZE2>(value, min(distance, laneId));
  const int val = shmem[WARP_SIZE - 1];
  return val;
}

/* does not work if segment size > WARP_SIZE */
static __device__ __forceinline__ int inclusive_segscan_array(int *shmem_in, const int N)
{
  int dist, nseg = 0;
  int y  = inclusive_segscan_warp(shmem_in, shmem_in[laneId], dist, nseg);
  if (N <= WARP_SIZE) return nseg;

  for (int p = WARP_SIZE; p < N; p += WARP_SIZE)
  {
    int *shmem = shmem_in + p;
    int y1  = inclusive_segscan_warp(shmem, shmem[laneId], dist, nseg);
    shmem[laneId] += y & BTEST(laneId < dist);
    y = y1;
  }

  return nseg;
}


/**************************************/
/*************** Tree walk ************/
/**************************************/

  template<int SHIFT>
__forceinline__ static __device__ int ACCS(const int i)
{
  return (i & ((LMEM_STACK_SIZE << SHIFT) - 1))*blockDim.x + threadIdx.x;
}

texture<float4, 1, cudaReadModeElementType> texNodeSize;
texture<float4, 1, cudaReadModeElementType> texNodeCenter;
texture<float4, 1, cudaReadModeElementType> texMultipole;
texture<float4, 1, cudaReadModeElementType> texBody;

/*********** Forces *************/

static __device__ __forceinline__ float4 add_acc(
    float4 acc,  const float4 pos,
    const float massj, const float3 posj,
    const float eps2)
{
#if 1  // to test performance of a tree-walk 
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



/*static __device__ float4 get_D04(float ds2, int selfGrav = 1) {
#if 1
  float ids  = rsqrtf(ds2);  //Does not work with zero-softening
  //   if(isnan(ids)) ids = 0;               //This does work with zero-softening, few percent performance drop
  //float ids  = (1.0f / sqrtf(ds2)) * selfGrav; Slower in Pre CUDA4.1
  ids *= selfGrav;
#else
  const float ids = (ds2 > 0.0f) ? rsqrtf(ds2) : 0.0f;
#endif
  const float ids2 = ids*ids;
  float ids3 = ids *ids2;
  float ids5 = ids3*ids2;
  float ids7 = ids5*ids2;
  return make_float4(ids, -ids3, +3.0f*ids5, -15.0f*ids7);
}  // 9 flops*/
#ifdef _QUADRUPOLE_

static __device__ __forceinline__ float4 add_acc(
    float4 acc, 
    const float4 pos,
    const float mass, const float3 com,
    const float4 Q0,  const float4 Q1, float eps2) 
{
  const float3 dr = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
  const float  r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;

  const float rinv  = rsqrtf(r2);
  const float rinv2 = rinv *rinv;
  const float mrinv  =  mass*rinv;
  const float mrinv3 = rinv2*mrinv;
  const float mrinv5 = rinv2*mrinv3; 
  const float mrinv7 = rinv2*mrinv5;   // 16

#if 0
  float  D0  =  mrinv;
  float  D1  = -mrinv3;
  float  D2  =  mrinv5*( 3.0f);
  float  D3  = -mrinv7*(15.0f);

  float oct_q11 = Q0.x;
  float oct_q22 = Q0.y;
  float oct_q33 = Q0.z;
  float oct_q12 = Q1.x;
  float oct_q13 = Q1.y;
  float oct_q23 = Q1.z;

  float Qii = oct_q11 + oct_q22 + oct_q33;
  float QijRiRj =
         (oct_q11*dr.x*dr.x + oct_q22*dr.y*dr.y + oct_q33*dr.z*dr.z) +
    2.0f*(oct_q12*dr.y*dr.x + oct_q13*dr.z*dr.x + oct_q23*dr.y*dr.z);

  acc.w        -= D0 + 0.5f*D1*Qii + 0.5f*D2*QijRiRj;
  float C01a    = D1 + 0.5f*D2*Qii + 0.5f*D3*QijRiRj;
  acc.x         += C01a*dr.x + D2*(oct_q11*dr.x + oct_q12*dr.y + oct_q13*dr.z);
  acc.y         += C01a*dr.y + D2*(oct_q12*dr.x + oct_q22*dr.y + oct_q23*dr.z);
  acc.z         += C01a*dr.z + D2*(oct_q13*dr.x + oct_q23*dr.y + oct_q33*dr.z);
#else
  float  D0  =  mrinv;
  float  D1  = -mrinv3;
  float  D2  =  mrinv5*( 3.0f);
  float  D3  = -mrinv7*(15.0f); // 3

  const float q11 = Q0.x;
  const float q22 = Q0.y;
  const float q33 = Q0.z;
  const float q12 = Q1.x;
  const float q13 = Q1.y;
  const float q23 = Q1.z;

  const float  q  = q11 + q22 + q33;
  const float3 qR = make_float3(
      q11*dr.x + q12*dr.y + q13*dr.z,
      q12*dr.x + q22*dr.y + q23*dr.z,
      q13*dr.x + q23*dr.y + q33*dr.z);
  const float qRR = qR.x*dr.x + qR.y*dr.y + qR.z*dr.z;  // 22

  acc.w  -= D0 + 0.5f*(D1*q + D2*qRR);
  float C = D1 + 0.5f*(D2*q + D3*qRR);
  acc.x  += C*dr.x + D2*qR.x;
  acc.y  += C*dr.y + D2*qR.y;
  acc.z  += C*dr.z + D2*qR.z;               // 23
#endif  // total: 16 + 3 + 22 + 23 = 64 flops 

  return acc;
}

#endif

/*******************************/
/****** Opening criterion ******/
/*******************************/

//Improved Barnes Hut criterium
static __device__ bool split_node_grav_impbh(
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

//Minimum distance
__device__ bool split_node_grav_md(
    const float4 nodeCenter,
    const float4 nodeSize,
    const float4 groupCenter,
    const float4 groupSize)
{
  //Compute the distance between the group and the cell
  float3 dr = {fabs(groupCenter.x - nodeCenter.x) - (groupSize.x + nodeSize.x),
               fabs(groupCenter.y - nodeCenter.y) - (groupSize.y + nodeSize.y),
               fabs(groupCenter.z - nodeCenter.z) - (groupSize.z + nodeSize.z)};

  dr.x += fabs(dr.x); dr.x *= 0.5f;
  dr.y += fabs(dr.y); dr.y *= 0.5f;
  dr.z += fabs(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  return (ds2 <= fabs(nodeCenter.w));
}




#define TEXTURES

/*******************************/
/******  Force tree-walk  ******/
/*******************************/


template<const int SHIFT, const int BLOCKDIM2, const int NI>
static __device__ 
#if 0 /* __noinline__ crashes the kernel when compled with ABI */
__noinline__
#else
__forceinline__ 
#endif
void approximate_gravity(
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



  /*********** shared memory distribution **********/

  //  begin,    end,   size
  // -----------------------
  const int stack_sz = (LMEM_STACK_SIZE << SHIFT) << BLOCKDIM2;  /* stack allocated per thread-block */
  const int nWarps2 = BLOCKDIM2 - WARP_SIZE2;
  int *approxL = lmem + stack_sz + (LMEM_EXTRA_SIZE >> nWarps2) * warpId;

  int *directS = shmem;                              //  0*DIM,  1*DIM,  1*DIM
  int *nodesS  = directS + WARP_SIZE;                      //  1*DIM, 10*DIM,  9*DIM
  int *prefix  = nodesS  + WARP_SIZE*8;                    //  9*DIM, 10*DIM,  1*DIM

  const int NJMAX = WARP_SIZE*3;
  int    *body_list = (int*   )&nodesS   [WARP_SIZE]; //  2*DIM,   5*DIM,  2*DIM
  float  *sh_mass   = (float* )&body_list[NJMAX]; //  5*DIM,   6*DIM,  1*DIM
  float3 *sh_pos    = (float3*)&sh_mass  [WARP_SIZE]; //  6*DIM,   9*DIM   3*DIM

  int *approxM = approxL;
  int *directM = directS;
  int * nodesM =  nodesS;


  /*********** stack **********/

  int *nstack = lmem;

  /*********** begin tree-walk **********/

  int n_approx = 0;
  int n_direct = 0;


  for (int root_node = node_begend.x; root_node < node_begend.y; root_node += WARP_SIZE) 
  {
    int n_nodes0 = min(node_begend.y - root_node, WARP_SIZE);
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
#if 1
        { prefix[laneId] = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
        const int node  = prefix[min(laneId, n_nodes0 - 1)];
#else  /* eg: seems to work, but I do not remember if that will *always* work */
        int node;
        { node  = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
#endif

#if 0   /* if uncommented, give same results, see below */
        if (blockIdx.x == 0 && warpId == 0)
          printf("laneId = %d  node= %d \n", laneId, node);
#endif


#if 0
        if(n_nodes0 > 0){       //Work around pre 4.1 compiler bug
          n_nodes0 -= WARP_SIZE;
        }
#else
        n_nodes0 -= WARP_SIZE;
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
#ifdef  IMPBH
        //Improved barnes-hut method
        #ifndef TEXTURES
                float4 nodeCOM = multipole_data[node*3];
        #else
                float4 nodeCOM = tex1Dfetch(texMultipole,node*3);
        #endif
        nodeCOM.w      = node_pos.w;
        bool   split   = split_node_grav_impbh(nodeCOM, group_pos, groupSize);
#else
        bool   split   = split_node_grav_md(node_pos, nodeSize, group_pos, groupSize); 
#endif


        bool leaf       = node_pos.w <= 0;  //Small AND equal incase of a 1 particle cell       //Check if it is a leaf
//                  split = true;

        bool flag    = (split && !leaf) && use_node;                        //Flag = use_node + split + not_a_leaf;Use only non_leaf nodes that are to be split
        uint mask    = BTEST(flag);                                       // mask = #FFFFFFFF if use_node+split+not_a_leaf==true, otherwise zero
        int child    =    node_data & 0x0FFFFFFF;                         //Index to the first child of the node
        int nchild   = (((node_data & 0xF0000000) >> 28)) & mask;         //The number of children this node has

        /***
         **** --> calculate prefix
         ***/


        int n_total = inclusive_scan_warp(prefix,  nchild);               // inclusive scan to compute memory offset of each child (return total # of children)
        int offset  = prefix[laneId];
        offset     += n_offset - nchild;                                  // convert inclusive into exclusive scan for referencing purpose

        for (int i = n_offset; i < n_offset + n_total; i += WARP_SIZE)         //nullify part of the array that will be filled with children
          nodesM[laneId + i] = 0;                                          //but do not touch those parts which has already been filled

#if 0  /* the following gives different result than then one in else */
        /* the results become the same if I uncomment printf above */
        if (flag == true)
        {
          nodesM[offset] = child; 
          if (nodesM[offset + 1] == 0) nodesM[offset + 1] = child + 1; 
          if (nodesM[offset + 2] == 0) nodesM[offset + 2] = child + 2;
          if (nodesM[offset + 3] == 0) nodesM[offset + 3] = child + 3;
          if (nodesM[offset + 4] == 0) nodesM[offset + 4] = child + 4;
          if (nodesM[offset + 5] == 0) nodesM[offset + 5] = child + 5;
          if (nodesM[offset + 6] == 0) nodesM[offset + 6] = child + 6;
          if (nodesM[offset + 7] == 0) nodesM[offset + 7] = child + 7;
        }
#elif 0
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
#else
        //This code does not require reading of nodesM before writing thereby preventing
        //possible synchronization , not completed writes , problems
        if(flag)
        {
          for(int i=0; i < nchild; i++)
          {
            nodesM[offset + i] = child + i;
          }
        }
#endif
        n_offset += n_total;    //Increase the offset in the array by the number of newly added nodes

        /***
         **** --> save list of nodes to LMEM
         ***/

        /*** if half of shared memory or more is filled with the the nodes, dump these into slowmem stack ***/
        while(n_offset >= WARP_SIZE) 
        {
          n_offset -= WARP_SIZE;
          const int offs1 = ACCS<SHIFT>(n_stack1);
          nstack[offs1] = nodesM[n_offset + laneId];   n_stack1++;
          n_nodes1 += WARP_SIZE;

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

        /* binary prefix sum */
        flag = !split && use_node;
        n_total = warp_exclusive_scan(flag, offset);
        if (flag) approxM[n_approx + offset] = node;

        n_approx += n_total;

        while (n_approx >= WARP_SIZE) 
        {
          n_approx -= WARP_SIZE;
          const int address      = (approxM[n_approx + laneId] << 1) + approxM[n_approx + laneId];
#ifndef TEXTURES
          const float4 monopole  = multipole_data[address    ];
#else
          const float4 monopole  = tex1Dfetch(texMultipole, address);
#endif

          sh_mass[laneId] = monopole.w;
          sh_pos [laneId] = make_float3(monopole.x,  monopole.y,  monopole.z);

#ifndef _QUADRUPOLE_
          for (int i = 0; i < WARP_SIZE; i++)
            for (int k = 0; k < NI; k++)
              acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i], eps2);
#else
#if 1    /*  a bit faster */
          const float4  Q0  = tex1Dfetch(texMultipole, address + 1);
          const float4  Q1  = tex1Dfetch(texMultipole, address + 2);
          for (int i = 0; i < WARP_SIZE; i++)
          {
            const float4 jQ0 = make_float4(__shfl(Q0.x, i), __shfl(Q0.y, i), __shfl(Q0.z, i), 0.0f);
            const float4 jQ1 = make_float4(__shfl(Q1.x, i), __shfl(Q1.y, i), __shfl(Q1.z, i), 0.0f);
            for (int k = 0; k < NI; k++)
              acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i], jQ0, jQ1, eps2);
          }
#else
          for (int i = 0; i < WARP_SIZE; i++)
          {
            const int address = approxM[n_approx + i] * 3;
            const float4  Q0  = tex1Dfetch(texMultipole, address + 1);
            const float4  Q1  = tex1Dfetch(texMultipole, address + 2);
            for (int k = 0; k < NI; k++)
              acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i], Q0, Q1, eps2);
          }
#endif
#endif /* _QUADRUPOLE_ */
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
        /* binary prefix sum */

        // step 1
        int n_bodies  = inclusive_scan_warp(prefix, nbody);              // inclusive scan to compute memory offset for each body
        offset = prefix[laneId];

        // step 2
        if (flag) prefix[warp_exclusive_scan(flag)] = laneId;   //with tidś whose leaves have to be opened

        directM[laneId]  = offset;                                       //Store a copy of inclusive scan in direct
        offset       -= nbody;                                              //convert inclusive int oexclusive scan
        offset       += 1;                                                  //add unity, since later prefix0[tid] == 0 used to check barrier

        int nl_pre = 0;                                                     //Number of leaves that have already been processed

        while (n_bodies > 0) 
        {
          int nb    = min(n_bodies, NJMAX - n_direct);                    //Make sure number of bides to be extracted does not exceed
          //the amount of allocated shared memory

          // step 0                                                      //nullify part of the body_list that will be filled with bodies
          for (int i = n_direct; i < n_direct + nb; i += WARP_SIZE)            //from the leaves that are being processed
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

          while(n_direct >= WARP_SIZE) 
          {
            n_direct -= WARP_SIZE;


            const float4 posj  = body_pos[body_list[n_direct + laneId]];
#if 0
            const float4 posj  = tex1Dfetch(texBody, body_list[n_direct + tid]);
#endif
            sh_mass[laneId] = posj.w;
            sh_pos [laneId] = make_float3(posj.x, posj.y, posj.z);

            for (int i = 0; i < WARP_SIZE; i++)
              for (int k = 0; k < NI; k++)
                acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i], eps2);
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
#ifndef _QUADRUPOLE_
    for (int i = 0; i < WARP_SIZE; i++)
      for (int k = 0; k < NI; k++)
        acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i],eps2);
#else
    for (int i = 0; i < WARP_SIZE; i++)
    {
      float4 Q0, Q1;
      Q0 = Q1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (i < n_approx)
      {
        const int address = approxM[i] * 3;
        Q0 = tex1Dfetch(texMultipole, address + 1);
        Q1 = tex1Dfetch(texMultipole, address + 2);
      }
      for (int k = 0; k < NI; k++)
        acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i], Q0, Q1, eps2);
    }
#endif
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

    for (int i = 0; i < WARP_SIZE; i++) 
      for (int k = 0; k < NI; k++)
        acc_i[k] = add_acc(acc_i[k], pos_i[k], sh_mass[i], sh_pos[i], eps2);
    direCount += WARP_SIZE*NI;
  }
}


#if 0 /* casues 164 bytes spill to lmem with NTHREAD = 128 */
__launch_bounds__(NTHREAD)
#endif
KERNEL_DECLARE(dev_approximate_gravity)(
      const int n_active_groups,
      int    n_bodies,
      float eps2,
      uint2 node_begend,
      int    *active_groups,
      real4  *body_pos,
      real4  *multipole_data,
      float4 *acc_out,
      real4  *group_body_pos,           //This can be different from body_pos
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

    pos_i[0] = group_body_pos[body_i[0]];
    if(ni > 1) //Only read if we actually have ni == 2
      pos_i[1] = group_body_pos[body_i[1]];

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

#if 1 /* this increase lmem spill count */
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
        approximate_gravity<8, blockDim2, 1>(
            pos_i, group_pos,
            eps2, node_begend,
            multipole_data, body_pos,
            shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
            group_eps, 
            acc_i);
      else
        approximate_gravity<8, blockDim2, 2>(
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
//       ngb_out     [addr] = ngb_i;
      ngb_out     [addr] = addr; //JB Fixed this for demo 
      active_inout[addr] = 1;
      interactions[addr].x = apprCount / ni;
      interactions[addr].y = direCount / ni ;
      if (ni == 2)
      {
        const int addr = body_i[1];
        acc_out     [addr] = acc_i[1];
//         ngb_out     [addr] = ngb_i;
        ngb_out     [addr] = addr; //JB Fixed this for demo 
        active_inout[addr] = 1;     
        interactions[addr].x = apprCount / ni; 
        interactions[addr].y = direCount / ni; 
      }
    }
  }     //end while
}


KERNEL_DECLARE(dev_approximate_gravity_let)(
      const int n_active_groups,
      int    n_bodies,
      float eps2,
      uint2 node_begend,
      int    *active_groups,
      real4  *body_pos,
      real4  *multipole_data,
      float4 *acc_out,
      real4  *group_body_pos,           //This can be different from body_pos
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

    pos_i[0] = group_body_pos[body_i[0]];
    if(ni > 1) //Only read if we actually have ni == 2
      pos_i[1] = group_body_pos[body_i[1]];

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

#if 1 /* this increase lmem spill count */
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
        approximate_gravity<8, blockDim2, 1>(
            pos_i, group_pos,
            eps2, node_begend,
            multipole_data, body_pos,
            shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
            group_eps, 
            acc_i);
      else
        approximate_gravity<8, blockDim2, 2>(
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
      acc_out     [addr].x += acc_i[0].x;
      acc_out     [addr].y += acc_i[0].y;
      acc_out     [addr].z += acc_i[0].z;
      acc_out     [addr].w += acc_i[0].w;
//       ngb_out     [addr] = ngb_i;
      ngb_out     [addr] = addr; //JB Fixed this for demo 
      active_inout[addr] = 1;
      interactions[addr].x = apprCount / ni;
      interactions[addr].y = direCount / ni;
      if (ni == 2)
      {
        const int addr = body_i[1];
        acc_out     [addr].x += acc_i[1].x;
        acc_out     [addr].y += acc_i[1].y;
        acc_out     [addr].z += acc_i[1].z;
        acc_out     [addr].w += acc_i[1].w;
//       ngb_out     [addr] = ngb_i;
        ngb_out     [addr] = addr; //JB Fixed this for demo 
        active_inout[addr] = 1;     
        interactions[addr].x = apprCount / ni;
        interactions[addr].y = direCount / ni;
      }
    }
  }     //end while
}


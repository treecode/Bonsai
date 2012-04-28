  // #include "support_kernels.cu"
#include <stdio.h>

#include "node_specs.h"

#ifdef WIN32
#define M_PI        3.14159265358979323846264338328
#endif


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


#define WARP_SIZE2 5
#define WARP_SIZE  (1 << WARP_SIZE2)

#if 1

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
__device__ __forceinline__ uint inclusive_scan_warp(int mysum, const int idx)
{
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

#else

  template<const int SIZE2>
__device__ __forceinline__ int inclusive_scan_warp(int value, const unsigned int idx) 
{
  const unsigned int laneId = idx & 31;

  const int SIZE = 1 << SIZE2; 
  for (int i = 1; i <= SIZE; i <<= 1) 
  {
    int n = __shfl_up(value, i, SIZE);
    if (laneId >= i)
      value += n;
  }

  return value;
}

#endif

  template<const int BLOCKDIM2>
__device__ __inline__ int inclusive_scan_block(volatile int* shdata, int v_in, const unsigned int idx) 
{
  const unsigned int laneId = idx & (WARP_SIZE - 1);
  const unsigned int warpId = idx >> WARP_SIZE2;

  int val = inclusive_scan_warp<WARP_SIZE2>(v_in, idx);
  if (31 == laneId) shdata[warpId] = val;
  __syncthreads();

  if (0 == warpId) shdata[idx] = inclusive_scan_warp<BLOCKDIM2 - WARP_SIZE2>(shdata[idx], idx);
  __syncthreads();

  if (warpId > 0) val += shdata[warpId - 1];

  return val; 
}

  template<const int BLOCKDIM2>
__device__ __inline__ int2 inclusive_scan_blockS(volatile int* shdata, int v_in, const unsigned int idx) 
{
  const unsigned int laneId = idx & (WARP_SIZE - 1);
  const unsigned int warpId = idx >> WARP_SIZE2;

  int val = inclusive_scan_warp<WARP_SIZE2>(v_in, idx);
  if (31 == laneId) shdata[warpId] = val;
  __syncthreads();

  if (0 == warpId) shdata[idx] = inclusive_scan_warp<BLOCKDIM2 - WARP_SIZE2>(shdata[idx], idx);
  __syncthreads();

  if (warpId > 0) val += shdata[warpId - 1];

  __syncthreads();
  if ((1 << BLOCKDIM2) - 1 == idx) shdata[0] = val;
  __syncthreads();

  return make_int2(val, shdata[0]);
}

template<const int BLOCKDIM2>
__device__ int inclusive_scan_array(volatile int *ptr, volatile int* shdata, const int N, const unsigned int idx) 
{
  int y = 0;

  for (int p = 0; p < N; p += blockDim.x) 
  {
    __syncthreads();
    ptr[p + idx] = inclusive_scan_block<BLOCKDIM2>(shdata, ptr[p+idx], idx);
    ptr[p + idx] += y;

    __syncthreads();

    y = ptr[blockDim.x - 1];
  }

  return y;
}


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
#if 0
#define _ORIG_SHMEM_
#endif


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
#ifdef _ORIG_SHMEM_
  
  int *approxS = (int*)&shmem  [     0];            //  0*DIM,  2*DIM,  2*DIM
  int *directS = (int*)&approxS[ 2*DIM];            //  2*DIM,  3*DIM,  1*DIM
  int *nodesS = (int*)&directS [   DIM];            //  3*DIM, 12*DIM,  9*DIM
  int *prefix = (int*)&nodesS  [9 *DIM];            // 12*DIM, 14*DIM,  2*DIM
  int *sh_body = &approxS[DIM];
  
  int *prefix0 = &prefix[  0];
  int *prefix1 = &prefix[DIM];
  
  const int NJMAX = DIM*2;
  int    *body_list = (int*   )&nodesS   [  DIM]; //  4*DIM,  6*DIM,  2*DIM
  float  *sh_mass   = (float* )&body_list[NJMAX]; //  6*DIM,  7*DIM,  1*DIM
  float3 *sh_pos    = (float3*)&sh_mass  [  DIM]; //  7*DIM, 10*DIM   3*DIM
  
  int *approxM = approxS;
  int *directM = directS;
  int * nodesM =  nodesS;

#else

  const int stack_sz = (LMEM_STACK_SIZE << SHIFT) + 4096;
  int *approxL = lmem + stack_sz; 

  int *directS = shmem;                              //  0*DIM,  1*DIM,  1*DIM
  int *nodesS  = directS + DIM;                      //  1*DIM, 10*DIM,  9*DIM
  int *prefix  = nodesS  + DIM*9;                    // 10*DIM, 12*DIM,  2*DIM
  
  int *prefix0 = &prefix[  0];
  int *prefix1 = &prefix[DIM];
  
  const int NJMAX = DIM*3;
  int    *body_list = (int*   )&nodesS   [  DIM]; //  2*DIM,   5*DIM,  2*DIM
  float  *sh_mass   = (float* )&body_list[NJMAX]; //  5*DIM,   6*DIM,  1*DIM
  float3 *sh_pos    = (float3*)&sh_mass  [  DIM]; //  6*DIM,   9*DIM   3*DIM
  int    *sh_body   = nodesS + DIM*8;             //  9*DIM,  10*DIM,  1*DIM
  
  int *approxM = approxL;
  int *directM = directS;
  int * nodesM =  nodesS;

#endif /* _ORIG_SHMEM_ */

  float  *node_mon0 = sh_mass;
  float3 *node_mon1 = sh_pos; 
  
  float  *sh_pot = sh_mass;
  float3 *sh_acc = sh_pos;

  __shared__ int prefix_shmem[32];

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
        { prefixS[tid] = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
        __syncthreads();
        int node  = prefixS[min(tid, n_nodes0 - 1)];
#else
        int node;
        { node = nstack[ACCS<SHIFT>(c_stack0)];   c_stack0++; }
        __syncthreads();
#endif

        if(n_nodes0 > 0){       //Work around pre 4.1 compiler bug
          n_nodes0 -= DIM;
        }

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


        int2 offset2 = inclusive_scan_blockS<DIM2>(prefix_shmem, nchild, tid);        // inclusive scan to compute memory offset of each child
        int  offset = offset2.x;
        int n_total = offset2.y;
        offset += n_offset - nchild;


        for (int i = n_offset; i < n_offset + n_total; i += DIM)         //nullify part of the array that will be filled with children
          nodesM[tid + i] = 0;                                          //but do not touch those parts which has already been filled
        __syncthreads();                                                 //Thread barrier to make sure all warps finished writing data

        bool flag = (split && !leaf) && use_node;                        //Flag = use_node + split + not_a_leaf;Use only non_leaf nodesM that are to be split
        if (flag) nodesM[offset] = child;                            //Thread with the node that is about to be split
        //writes the first child in the array of nodesM
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

        offset2 = inclusive_scan_blockS<DIM2>(prefix_shmem, 1 - (split || !use_node), tid);
        offset  = offset2.x;
        n_total = offset2.y;
        if (!split && use_node) approxM[n_approx + offset - 1] = node;
        __syncthreads();

        n_approx += n_total;


        while (n_approx >= DIM) 
        {
          n_approx -= DIM;
          const int address      = (approxM[n_approx + tid] << 1) + approxM[n_approx + tid];
#ifndef TEXTURES
          const float4 monopole  = multipole_data[address    ];
#else
          const float4 monopole  = tex1Dfetch(texMultipole, address);
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

        flag         = split && leaf && use_node;                                //flag = split + leaf + use_node
        int  jbody   = node_data & BODYMASK;                                     //the first body in the leaf
        int  nbody   = (((node_data & INVBMASK) >> LEAFBIT)+1) & BTEST(flag);    //number of bodies in the leaf masked with the flag

        body_list[tid] = directM[tid];                                            //copy list of bodies from previous pass to body_list
        sh_body  [tid] = jbody;                                                  //store the leafs first body id into shared memory


        // step 1
        offset = inclusive_scan_block<DIM2>(prefix_shmem, (int)flag, tid);       // inclusive scan on flags to construct array
        if (flag) prefix1[offset - 1] = tid;                                     //with tid's whose leaves have to be opened
        __syncthreads();                                                         //thread barrier, make sure all warps completed the job

        // step2 
        offset2 = inclusive_scan_blockS<DIM2>(prefix_shmem, nbody, tid);        // inclusive scan to compute memory offset for each body
        int offset1  = offset2.x;
        int n_bodies = offset2.y;                                               //Total number of bides extract from the leaves
        __syncthreads();                                                        // thread barrier to make sure that warps completed their jobs

        directM[tid]  = offset1;                                            //Store a copy of inclusive scan in direct
        offset1      -= nbody;                                              //convert inclusive int oexclusive scan
        offset1      += 1;                                                  //add unity, since later prefix0[tid] == 0 used to check barrier
        prefix0[tid] = offset1;
        __syncthreads();

        int nl_pre = 0;                                                     //Number of leaves that have already been processed

        while (n_bodies > 0) 
        {
          int nb    = min(n_bodies, NJMAX - n_direct);                    //Make sure number of bides to be extracted does not exceed
          //the amount of allocated shared memory

          // step 0                                                      //nullify part of the body_list that will be filled with bodies
          for (int i = n_direct; i < n_direct + nb; i += DIM){           //from the leaves that are being processed
            body_list[i + tid] = 0;
          }
          __syncthreads();

          //step 1:
          if (flag && (directM[tid] <= nb) && (prefix0[tid] > 0))        //make sure that the thread indeed carries a leaf
            body_list[n_direct + prefix0[tid] - 1] = 1;                 //whose bodies will be extracted
          __syncthreads();

          //step 2:
          int nl = inclusive_scan_array<DIM2>              // inclusive scan to compute number of leaves to process
            (&body_list[n_direct], prefix_shmem, nb, tid);            // to make sure that there is enough shared memory for bodies
          nb = directM[prefix1[nl_pre + nl - 1]];                        // number of bodies stored in these leaves

          // step 3:
          for (int i = n_direct; i < n_direct + nb; i += DIM) {          //segmented fill of the body_list
            int j = prefix1[nl_pre + body_list[i + tid] - 1];            // compute the first body in shared j-body array
            body_list[i + tid] = (i + tid - n_direct) -                 //add to the index of the first j-body in a child
              (prefix0[j] - 1) + sh_body[j];                            //the index of the first child in body_list array
          }
          __syncthreads();


          /**************************************************
           *  example of what is accomplished in steps 0-4   *
           *       ---------------------------               *
           * step 0: body_list = 000000000000000000000       *
           * step 1: body_list = 100010001000000100100       *
           * step 2: body_list = 111122223333333444555       *
           * step 3: body_list = 012301230123456012012       *
           *         assuming that sh_body[j] = 0            *
           ***************************************************/

          n_bodies     -= nb;                                   //subtract from n_bodies number of bodies that have been extracted
          nl_pre       += nl;                                   //increase the number of leaves that where processed
          directM[tid] -= nb;                                   //subtract the number of extracted bodies in this pass
          prefix0[tid] = max(prefix0[tid] - nb, 0);             //same here, but do not let the number be negative (GT200 bug!?)
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
#ifdef _ORIG_SHMEM_
    __shared__ int shmem[15*(1 << blockDim2)];
#else
    __shared__ int shmem[12*(1 << blockDim2)];
#endif
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
      int *lmem = &MEM_BUF[blockIdx.x*LMEM_STACK_SIZE*blockDim.x];


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

        lmem = &MEM_BUF[gridDim.x*LMEM_STACK_SIZE*blockDim.x];    //Use the extra large buffer
        apprCount = direCount = 0;
        acc_i = approximate_gravity<blockDim2, 8>( DIM2x, DIM2y, tid, tx, ty,
            body_i, pos_i, group_pos,
            eps2, node_begend,
            multipole_data, body_pos,
            shmem, lmem, ngb_i, apprCount, direCount, boxSizeInfo, curGroupSize, boxCenterInfo,
            group_eps, body_vel);

        lmem = &MEM_BUF[blockIdx.x*LMEM_STACK_SIZE*blockDim.x]; //Back to normal location

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



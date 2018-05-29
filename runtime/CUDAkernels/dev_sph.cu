#include "bonsai.h"
#include <stdio.h>
#include <stdarg.h>
#include "node_specs.h"

#include "treewalk_includes.h"

#define PERIODIC_X 1
#define PERIODIC_Y 2
#define PERIODIC_Z 4



__constant__ bodyProps group_body_props;

#if CUDART_VERSION >= 9010
    #include <cuda_fp16.h>
#else

#if defined(__CUDACC_RTC__)
#define __CUDA_FP16_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __CUDA_FP16_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */


typedef struct __align__(4) {
   unsigned int x;
} __half2;
typedef struct __align__(2) {
   unsigned short x;
} __half;

typedef __half2 half2;
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half l, const __half h)
{
   __half2 val;
   asm("{  mov.b32 %0, {%1,%2};}\n"
       : "=r"(val.x) : "h"(l.x), "h"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half(const float f)
{
   __half val;
   asm volatile("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}

__CUDA_FP16_DECL__ float2 __half22float2(const __half2 l)
{
   float hi_float;
   float lo_float;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(l.x));

   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(l.x));

   return make_float2(lo_float, hi_float);
}
#endif



#include "../profiling/bonsai_timing.h"
PROF_MODULE(dev_approximate_gravity);



#define DEVPRINT(fmt, ...) { \
     if (0)                  \
     {                        \
         printf("ONDEV [ %d,%d, %d] line: %d  " fmt,blockIdx.x, threadIdx.x, laneId, __LINE__,  __VA_ARGS__); \
     }   \
 }  \

#include "sph_includes.h"




/**************************************/
/*************** Tree walk ************/
/**************************************/


///* Bonsai SC2014 Kernel */
//
//static __device__ __forceinline__ void computeDensityAndNgb(
//    const float r2, const float hinv2, const float mass,
//    float &density, float &nb)
//{
//#if 0  /* full kernel for reference */
//  const float hinv = 1.0f/h;
//  const float hinv2 = hinv*hinv;
//  const float hinv3 = hinv*hinv2;
//  const float C     = 3465.0f/(512.0f*M_PI)*hinv3;
//  const float q2    = r2*hinv2;
//  const float rho   = fmaxf(0.0f, 1.0f - q2);
//  nb      += ceilf(rho);
//  const float rho2 = rho*rho;
//  density += C * rho2*rho2;
//#elif 0
//  const float rho   = fmaxf(0.0f, 1.0f - r2*hinv2);   /* fma, fmax */
//  const float rho2  = rho*rho;                        /* fmul */
//  density += rho2*rho2;                               /* fma */
//  nb      += ceilf(rho2);                             /* fadd, ceil */
//
//  /*2x fma, 1x fmul, 1x fadd, 1x ceil, 1x fmax */
//  /* total: 6 flops or 8 flops with ceil&fmax */
//#else
//
//  if(r2 < hinv2 && mass > 0)
//  {
//      nb  += 1;
//  }
//
//#endif
//}
//#if 0
//static __device__ __forceinline__ float adjustH(const float h_old, const float nnb)
//{
//	const float nbDesired 	= 7;
//	const float f      	    = 0.5f * (1.0f + cbrtf(nbDesired / nnb));
//	const float fScale 	    = max(min(f, 1.2), 0.8);
//	return (h_old*fScale);
//}
//#endif


#if 0
__device__ bool testParticleMD(const float4 particlePos,
                               const float4 groupCenter,
                               const float4 groupSize,
                               const float  grpH)
{
  //Compute the distance between the group and the cell
  float3 dr = {fabs(groupCenter.x - particlePos.x) - groupSize.x,
               fabs(groupCenter.y - particlePos.y) - groupSize.y,
               fabs(groupCenter.z - particlePos.z) - groupSize.z};

  dr.x += fabs(dr.x); dr.x *= 0.5f;
  dr.y += fabs(dr.y); dr.y *= 0.5f;
  dr.z += fabs(dr.z); dr.z *= 0.5f;

  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  return (ds2 <= grpH);
}
#endif

#if 0
__device__ void useParticleMD(const float4       posi,
                              const float4      *body_jpos,
                              const int          ptclIdx,
                              const float4       distance,
                              float4            *pB,
                              int               &pC)
{
    float iH = posi.w*SPH_KERNEL_SIZE;
    iH      *= iH;

    const float4 M0 = (ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, -1.0f);

    for (int j = 0; j < WARP_SIZE; j++)
    {
      const float4 jM0   = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                       __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
      const float3 dr    = make_float3(jM0.x - posi.x, jM0.y - posi.y, jM0.z - posi.z);
      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      if(r2 <= iH && M0.w >= 0) //Only use valid particles (no negative mass)
      {
          pB[pC] = make_float4(dr.x, dr.y, dr.z, jM0.w);
          pC++;
      }

      //If one of the lists is full we evaluate the currently stored particles
      if(__any(pC == 16))
      {
          for(int z=0; z < pC; z++)
          {

          }
          pC = 0;
      }

    }
}
#endif

/*******************************/
/****** Opening criterion ******/
/*******************************/


/*
 * TODO, we can probably add the cellH/smoothing length to the groupSize to save
 * the cellH variable and instead compare to 0
 */

__device__ bool split_node_sph_md(const float4 nodeSize,
                                  const float4 nodeCenter,
                                  const float4 groupCenter,
                                  const float4 groupSize,
                                  const float  grpH,
                                  const float  cellH)
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
  return (ds2 <= (grpH+cellH));
}

#define SIGN(x) ((x > 0) - (x < 0))
//__device__ bool split_node_sph_md_b(const float4 nodeSize,
//                                  const float4 nodeCenter,
//                                  const float4 groupCenter,
//                                  const float4 groupSize,
//                                  const float  grpH,
//                                  const float  cellH)
//{
//  //Compute the distance between the group and the cell
//  float3 dr = {fabs(groupCenter.x - nodeCenter.x) - (groupSize.x + nodeSize.x),
//               fabs(groupCenter.y - nodeCenter.y) - (groupSize.y + nodeSize.y),
//               fabs(groupCenter.z - nodeCenter.z) - (groupSize.z + nodeSize.z)};
//
////  float3 domainSize = {6.8593750000000000, 0.040594f, 0.038274f};   //Hardcoded for phantom tube
////  const float dxbound = 6.85937500f;
////  const float dybound = 0.040594f;
////  const float dzbound = 0.038274f;
////  if (abs(dr.x) > 0.5*dxbound) dr.x = dr.x - dxbound*SIGN(dr.x);
////  if (abs(dr.y) > 0.5*dybound) dr.y = dr.y - dybound*SIGN(dr.y);
////  if (abs(dr.z) > 0.5*dzbound) dr.z = dr.z - dzbound*SIGN(dr.z);
//
//  dr.x += fabs(dr.x); dr.x *= 0.5f;
//  dr.y += fabs(dr.y); dr.y *= 0.5f;
//  dr.z += fabs(dr.z); dr.z *= 0.5f;
//
//  //Distance squared, no need to do sqrt since opening criteria has been squared
//  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
//  return (ds2 <= 1*(grpH+cellH));
//}



//__device__ bool split_node_sph_md_print(const float4 nodeSize,
//                                  const float4 nodeCenter,
//                                  const float4 groupCenter,
//                                  const float4 groupSize,
//                                  const float  grpH,
//                                  const float  cellH)
//{
//  //Compute the distance between the group and the cell
//  float3 dr = {fabs(groupCenter.x - nodeCenter.x) - (groupSize.x + nodeSize.x),
//               fabs(groupCenter.y - nodeCenter.y) - (groupSize.y + nodeSize.y),
//               fabs(groupCenter.z - nodeCenter.z) - (groupSize.z + nodeSize.z)};
//
//  dr.x += fabs(dr.x); dr.x *= 0.5f;
//  dr.y += fabs(dr.y); dr.y *= 0.5f;
//  dr.z += fabs(dr.z); dr.z *= 0.5f;
//
//  //Distance squared, no need to do sqrt since opening criteria has been squared
//  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
//
//  printf("ON DEV distance: %f %f %f | ds2: %f  grpH: %f cellH: %f \n", dr.x, dr.y, dr.z, ds2, grpH, cellH);
//
//  return (ds2 <= (grpH+cellH));
//}



#define TEXTURES




template<int SHIFT, int BLOCKDIM2, int NI, bool INTCOUNT, template<int NI2, bool FULL> class directOP>
static __device__
uint2 approximate_sph(
                              float4     acc_i   [NI] /* out */,
                        const float4     _pos_i  [NI],
                        const float4     _vel_i  [NI],
                        const float4     hydro_i[NI],
                        const float4     groupPos,
                        const float4    *body_pos_j,
                        const float4    *body_vel_j,
                        const float4    *body_hydro_j,
                        const float2    *body_dens_j,
                        const sphParameters     SPHParams,
                        const uint2      top_cells,
                        int             *shmem,
                        int             *cellList,
                        const float4     groupSize,
                        SPH::density::data      density_i   [NI],
                        SPH::derivative::data   derivative_i[NI],
                        const float4    *nodeSize,
                        const float4    *nodeCenter,
                        const float4    *nodeMultipole,
                        const float      grpH)
{
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);

  /* this helps to unload register pressure */
  float4 pos_i[NI], vel_i[NI];// hydro_i[NI];
#pragma unroll 1
  for (int i = 0; i < NI; i++){
    pos_i[i]   = _pos_i[i];
    vel_i[i]   = _vel_i[i];
//    hydro_i[i] = _hydro_i[i];
  }

  uint2 interactionCounters = {0}; /* # of approximate and exact force evaluations */


  volatile int *tmpList = shmem;

  int directPtclIdx = 0;
  int directCounter = 0;

  for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.y)
      cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;

  int nCells = top_cells.y - top_cells.x;


  int cellListBlock           = 0;
  int nextLevelCellCounter    = 0;
  unsigned int cellListOffset = 0;

  /* process level with n_cells */
#if 1
  while (nCells > 0)
  {
    /* extract cell index from the current level cell list */
    const int cellListIdx = cellListBlock + laneIdx;
    const bool useCell    = cellListIdx < nCells;
    const int cellIdx     = useCell ? cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)] : 0;
    cellListBlock        += min(WARP_SIZE, nCells - cellListBlock);

    /* read from gmem cell's info */
    const float4 cellSize = nodeSize[cellIdx];
    const float4 cellPos  = nodeCenter[cellIdx];
    //const float4 cellSize = tex1Dfetch(ztexNodeSize,   cellIdx);
    //const float4 cellPos  = tex1Dfetch(ztexNodeCenter, cellIdx);


    __half2 openings =  *(__half2*)(&cellPos.w);
    float2 openxy    = __half22float2(openings);    //Tree-code opening is in X, SPH max smoothing is in Y


    float cellH  = 0;

    bool splitCell = false;
    if(directOP<1, true>::type == SPH::HYDROFORCE)
    {
        //For hydro force we have to use mutual forces, so take the maximum smoothing of the cell and the group
        cellH = fabs(openxy.y);
        splitCell         = split_node_sph_md(cellSize, cellPos, groupPos, groupSize, max(cellH, grpH), 0);
    }
    else
    {
        //Should be this splitCell         = split_node_sph_md(cellSize, cellPos, groupPos, groupSize, grpH, 0);
        cellH = fabs(openxy.y);
        splitCell         = split_node_sph_md(cellSize, cellPos, groupPos, groupSize, max(cellH, grpH), 0);
    }


    interactionCounters.x += 1; //Keep track of number of opening tests


    /* compute first child, either a cell if node or a particle if leaf */
    const int cellData   = __float_as_int(cellSize.w);
    const int firstChild =  cellData & 0x0FFFFFFF;
    const int nChildren  = (cellData & 0xF0000000) >> 28;

    if(cellData == 0xFFFFFFFF) splitCell = false;

    /**********************************************/
    /* split cells that satisfy opening condition */
    /**********************************************/
    const bool isNode = openxy.x > 0.0f;

    {
      bool splitNode  = isNode && splitCell && useCell;

      /* use exclusive scan to compute scatter addresses for each of the child cells */
      const int2 childScatter = warpIntExclusiveScan(nChildren & (-splitNode));

      /* make sure we still have available stack space */
      //JB This does not seem to work, I have cells that are being overwritten
      //if (childScatter.y + nCells - cellListBlock > (CELL_LIST_MEM_PER_WARP<<SHIFT))
      //Using this instead which gives the correct results
      if (childScatter.y + nCells - cellListBlock + nextLevelCellCounter > (CELL_LIST_MEM_PER_WARP<<SHIFT))
        return make_uint2(0xFFFFFFFF,0xFFFFFFFF);

      /* if so populate next level stack in gmem */
      if (splitNode)
      {
          const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
          for (int i = 0; i < nChildren; i++)
              cellList[ringAddr<SHIFT>(scatterIdx + i)] = firstChild + i;
      }
      nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */
    }

#if 1
    {
      /***********************************/
      /******       DIRECT          ******/
      /***********************************/

      const bool isLeaf = !isNode;
      bool isDirect     = splitCell && isLeaf && useCell;

      const int firstBody =   cellData & BODYMASK;
      const int     nBody = ((cellData & INVBMASK) >> LEAFBIT)+1;

      const int2 childScatter = warpIntExclusiveScan(nBody & (-isDirect));

      int nParticle  = childScatter.y;
      int nProcessed = 0;
      int2 scanVal   = {0};

      /* conduct segmented scan for all leaves that need to be expanded */
      while (nParticle > 0)
      {
        tmpList[laneIdx] = 1;
        if (isDirect && (childScatter.x - nProcessed < WARP_SIZE))
        {
          isDirect = false;
          tmpList[childScatter.x - nProcessed] = -1-firstBody;
        }

        scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
        const int  ptclIdx = scanVal.x;

        if (nParticle >= WARP_SIZE)
        {
          directOP<NI,true>()(acc_i, pos_i, vel_i, ptclIdx, SPHParams, density_i, derivative_i, hydro_i,
                  body_pos_j, body_vel_j, body_dens_j, body_hydro_j);
          nParticle  -= WARP_SIZE;
          nProcessed += WARP_SIZE;
          if (INTCOUNT)
            interactionCounters.y += WARP_SIZE*NI;
        }
        else
        {
          const int scatterIdx = directCounter + laneIdx;
          tmpList[laneIdx] = directPtclIdx;
          if (scatterIdx < WARP_SIZE)
            tmpList[scatterIdx] = ptclIdx;

          directCounter += nParticle;

          if (directCounter >= WARP_SIZE)
          {
            /* evaluate cells stored in shmem */
            directOP<NI,true>()(acc_i, pos_i, vel_i, tmpList[laneIdx], SPHParams, density_i, derivative_i, hydro_i,
                                body_pos_j, body_vel_j, body_dens_j, body_hydro_j);
            directCounter -= WARP_SIZE;
            const int scatterIdx = directCounter + laneIdx - nParticle;
            if (scatterIdx >= 0)
              tmpList[scatterIdx] = ptclIdx;
            if (INTCOUNT)
              interactionCounters.y += WARP_SIZE*NI;
          }
          directPtclIdx = tmpList[laneIdx];
          nParticle     = 0;
        }
      }
    }
#endif

    /* if the current level is processed, schedule the next level */
    if (cellListBlock >= nCells)
    {
      cellListOffset += nCells;
      nCells          = nextLevelCellCounter;
      cellListBlock   = nextLevelCellCounter = 0;
    }
  }  /* level completed */
#endif


  //Process remaining items
  if (directCounter > 0)
  {
    directOP<NI,false>()(acc_i, pos_i, vel_i, laneIdx < directCounter ? directPtclIdx : -1, SPHParams, density_i, derivative_i, hydro_i,
            body_pos_j, body_vel_j, body_dens_j, body_hydro_j);
    if (INTCOUNT)
      interactionCounters.y += directCounter * NI;
    directCounter = 0;
  }

  return interactionCounters;
}


typedef unsigned long long ullong;


template<int SHIFT2, int BLOCKDIM2, bool ACCUMULATE, template<int NI2, bool FULL> class directOp>
static __device__
bool treewalk_control(
    const int        bid,
    const float      eps2,
    const uint2      node_begend,
    const bool       isFinalLaunch,
    const float4     domainSize,
    const sphParameters     SPHParams,
    const int       *active_groups,
    const bodyProps &group_body,
    const float4    *groupSizeInfo,
    const float4    *groupCenterInfo,
    const float4    *nodeCenter,
    const float4    *nodeSize,
    const float4    *nodeMultipole,
    int             *shmem,
    int             *lmem,
    int2            *interactions,
    int             *active_inout,
    const bodyProps &body_j,
    float4          *body_acc_out,
    float2          *body_dens_out,
    float4          *body_grad_out,
    float4          *body_hydro_out,
    const ullong    *ID)
{

  const real4     *group_body_pos   = group_body.body_pos;
  const real4     *group_body_vel   = group_body.body_vel;
  const float2    *group_body_dens  = group_body.body_dens;
        float4    *group_body_hydro = group_body.body_hydro;

   const real4     *body_pos_j   = body_j.body_pos;
   const real4     *body_vel_j   = body_j.body_vel;
   const float2    *body_dens_j  = body_j.body_dens;
   const real4     *body_hydro_j = body_j.body_hydro;




  /*********** set necessary thread constants **********/
#ifdef DO_BLOCK_TIMESTEP
  real4 group_pos       = groupCenterInfo[active_groups[bid]];
  real4 curGroupSize    = groupSizeInfo[active_groups[bid]];
#else
  real4 group_pos       = groupCenterInfo[bid];
  real4 curGroupSize    = groupSizeInfo[bid];
#endif

  const int   groupData       = __float_as_int(curGroupSize.w);
  const uint body_addr        =   groupData & CRITMASK;
  const uint nb_i             = ((groupData & INVCMASK) >> CRITBIT) + 1;

  uint body_i[2];
  const int ni = nb_i <= WARP_SIZE ? 1 : 2;
  body_i[0]    = body_addr + laneId%nb_i;
  body_i[1]    = body_addr + WARP_SIZE + laneId%(nb_i - WARP_SIZE);

  body_i[0]    = body_addr + laneId%NCRIT; //This ensures that the thread groups work on independent particles.
  if(laneId%NCRIT >= nb_i) body_i[0] = body_addr;

  /*
   * TODO Also consider removing all the [2] sized arrays as 64 particles per group is significantly slower and hence will
   * probably not be used in the SPH kernels
   */

  float4 pos_i [2], vel_i [2], acc_i [2], hydro_i[2];
  SPH::density::data    dens_i[2];
  SPH::derivative::data derivative_i[2];

  acc_i[0]  = acc_i[1]  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  dens_i[0].clear();       dens_i[1].clear();
  derivative_i[0].clear(); derivative_i[1].clear();

  pos_i[0]         = group_body_pos[body_i[0]];
  vel_i[0]         = group_body_vel[body_i[0]];
  hydro_i[0]       = group_body_hydro[body_i[0]];
  dens_i[0].smth   = group_body_dens[body_i[0]].y;

  if(ni > 1){       //Only read if we actually have ni == 2
    pos_i[1]       = group_body_pos[body_i[1]];
    vel_i[1]       = group_body_vel[body_i[1]];
    hydro_i[1]     = group_body_hydro[body_i[1]];
    dens_i[1].smth = group_body_dens[body_i[1]].y;
  }

  //For the hydro force we need the current density value
  if(directOp<1, true>::type == SPH::HYDROFORCE) {
                   dens_i[0].dens = group_body_dens[body_i[0]].x;
        if(ni > 1) dens_i[1].dens = group_body_dens[body_i[1]].x;

        derivative_i[0].x = -10e10; //We are using 'min' on this variable to find minimal time-step
  }

  uint2 counters = {0};


  pos_i[0].w                = dens_i[0].smth;  //y is smoothing range
  if(ni > 1 )   pos_i[1].w  = dens_i[1].smth;  //y is smoothing range


  //Compute the cellH, which is the maximum smoothing value off all particles
  //in a group because as we search for the whole group in parallel we have to
  //be sure to include all possibly required particles
  float cellH = warpAllReduceMax(pos_i[0].w);
  if(ni == 2)
  {
    cellH = max(cellH, warpAllReduceMax(pos_i[1].w));
  }
  cellH *= SPH::kernel_t::supportRadius(); //Multiply with the kernel cut-off radius
  cellH *= cellH; //Squared for distance comparison without sqrt


  //For periodic boundaries, mirror the group and body positions along the periodic axis
  int2 xP = {0,0}, yP = {0,0}, zP = {0,0};
  if(((int)domainSize.w) & PERIODIC_X) { xP = {-1,1}; }
  if(((int)domainSize.w) & PERIODIC_Y) { yP = {-1,1}; }
  if(((int)domainSize.w) & PERIODIC_Z) { zP = {-1,1}; }


  //TODO use templates or parameterize this at some point
  for(int ix=xP.x; ix <= xP.y; ix++)     //Periodic around X
  {
    for(int iy=yP.x; iy <= yP.y; iy++)   //Periodic around Y
    {
      for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
      {
          float4 pGroupPos   = group_pos;
          float4 pBodyPos[2] = {pos_i[0], pos_i[1]};

          pGroupPos.x   += (domainSize.x*ix); pBodyPos[0].x += (domainSize.x*ix); pBodyPos[1].x += (domainSize.x*ix);
          pGroupPos.y   += (domainSize.y*iy); pBodyPos[0].y += (domainSize.y*iy); pBodyPos[1].y += (domainSize.y*iy);
          pGroupPos.z   += (domainSize.z*iz); pBodyPos[0].z += (domainSize.z*iz); pBodyPos[1].z += (domainSize.z*iz);

          uint2 curCounters = {0};

    #if 0
          const bool INTCOUNT = false;
    #else
          const bool INTCOUNT = true;
    #endif
          {
            if (ni == 1)
                curCounters = approximate_sph<SHIFT2, BLOCKDIM2, 1, INTCOUNT, directOp>(
                        acc_i,
                        pBodyPos,
                        vel_i,
                        hydro_i,
                        pGroupPos,
                        body_pos_j,
                        body_vel_j,
                        body_hydro_j,
                        body_dens_j,
                        SPHParams,
                        node_begend,
                        shmem,
                        lmem,
                        curGroupSize,
                        dens_i,
                        derivative_i,
                        nodeSize,
                        nodeCenter,
                        nodeMultipole,
                        cellH);
            else
                curCounters = approximate_sph<SHIFT2, BLOCKDIM2, 2, INTCOUNT, directOp>(
                        acc_i,
                        pBodyPos,
                        vel_i,
                        hydro_i,
                        pGroupPos,
                        body_pos_j,
                        body_vel_j,
                        body_hydro_j,
                        body_dens_j,
                        SPHParams,
                        node_begend,
                        shmem,
                        lmem,
                        curGroupSize,
                        dens_i,
                        derivative_i,
                        nodeSize,
                        nodeCenter,
                        nodeMultipole,
                        cellH);
          }
          if(curCounters.x == 0xFFFFFFFF && curCounters.y == 0xFFFFFFFF)
          {
            return false; //Out of tree-walk memory
          }
          counters.x += curCounters.x;
          counters.y += curCounters.y;
      } //for periodic Z
    } //for periodic Y
  } //for periodic X

  long long int endC = clock64();


  if(directOp<1, true>::type == SPH::DENSITY)
  {
      dens_i[0].dens    = warpGroupReduce(dens_i[0].dens);
      derivative_i[0].x = warpGroupReduce(derivative_i[0].x);
      derivative_i[0].y = warpGroupReduce(derivative_i[0].y);
      derivative_i[0].z = warpGroupReduce(derivative_i[0].z);
      derivative_i[0].w = warpGroupReduce(derivative_i[0].w);

      acc_i[0].x = warpGroupReduce(acc_i[0].x); //For gradient sum
  }
  if(directOp<1, true>::type == SPH::HYDROFORCE)
  {
      acc_i[0].x = warpGroupReduce(acc_i[0].x);
      acc_i[0].y = warpGroupReduce(acc_i[0].y);
      acc_i[0].z = warpGroupReduce(acc_i[0].z);
      acc_i[0].w = warpGroupReduce(acc_i[0].w);

      //Reduce the dt parameter
      derivative_i[0].x = warpGroupReduceMax(derivative_i[0].x);

      //Below is for statistics
      derivative_i[0].z = warpGroupReduce(derivative_i[0].z); //For statistics
      derivative_i[0].y = warpGroupReduce(derivative_i[0].y); //For statistics
  }

  if (laneId < nb_i)
  {
    for(int i=0; i < ni; i++)
    {
        const int addr = body_i[i];
        {
            if(directOp<1, true>::type == SPH::DENSITY)
            {
                //Density requires a bit more work as the smoothing range is based on it

                //Combine current result with previous result and compute the updated smoothing
                //depending on if this was the final call this does not have to be the final dens/smoothing value
                dens_i[i].dens   += body_dens_out[addr].x;

                acc_i[i].x       += body_acc_out[addr].x;   //Used to store gradh

                derivative_i[i].x += body_grad_out[addr].x;
                derivative_i[i].y += body_grad_out[addr].y;
                derivative_i[i].z += body_grad_out[addr].z;
                derivative_i[i].w += body_grad_out[addr].w;


                if(isFinalLaunch)
                {
                    double hi1  = 1.0/group_body_dens[addr].y;
                    double hi21 = hi1*hi1;
                    double hi31 = hi1*hi21;
                    double hi41 = hi21*hi21;

                    const float cnormk = SPH::kernel_t::cnormk;

                    // Scale the density and the gradient, using current smoothing range
                    dens_i[i].dens  *= cnormk*hi31;
                    // Compute the gradient for individual smoothing length based SPH (gradh)
                    acc_i[i].x      *= cnormk*hi41;


                    dens_i[i].finalize(group_body_pos[body_i[i]].w);



                    float omega = dens_i[i].smth / (3*dens_i[i].dens);                     //Compute using new density
                    //float omega = group_body_dens[addr].y / (3*group_body_dens[addr].x); //Compute using prev iteration density

                    float gradh = 1.0f / (1 + omega*acc_i[i].x); //Eq 5 of phantom paper
                    group_body.body_vel[addr].w = gradh;         //Note we store this in the (predicted) velocity array

                    if(ID[addr] >= SPHBOUND)   //Boundary particles do not update their density/smoothing values
                    {
                        body_dens_out[addr] = group_body_dens[addr];
                        dens_i[i].dens = group_body_dens[addr].x;
                        dens_i[i].smth = group_body_dens[addr].y;
                        //TODO(jbedorf) Should we update them within a time-step?
                        //Otherwise the force functions would work with fixed density?
                    }


                    //Need this printf to get non-nan results on GTX1080
                    if(addr == -3830)
                    {
                       printf("TEST: %f %f %f %f \n",  body_grad_out[addr].x, derivative_i[i].x, body_dens_out[addr].x, derivative_i[i].y);
                    }

                    //Compute Balsara switch, Rosswog Eq 63
                    //Absolute gradient |D.v|
                    float temp  = fabs(derivative_i[i].w)*cnormk*hi41;
                    //Cross product: D x V
                    float temp2 = derivative_i[i].x*derivative_i[i].x +
                                  derivative_i[i].y*derivative_i[i].y +
                                  derivative_i[i].z*derivative_i[i].z;
                    temp2 *= cnormk*hi41;

                    //TODO(jbedorf): Should this be the old or the new smoothing?
                    //float temp3 = 1.0e-4 * group_body_hydro[addr].y / dens_i[i].smth;
                    float temp3 = 1.0e-4 * group_body_hydro[addr].y / group_body_dens[addr].y;

                    //Note using group_body_hydro here instead of body_hydro_out to store Balsara Switch
                    group_body_hydro[addr].w = temp / (temp + sqrtf(temp2) + temp3);
                } //is final launch

                body_dens_out[addr]   = make_float2(dens_i[i].dens,dens_i[i].smth);
                body_acc_out[addr].x  = acc_i[i].x;

                body_grad_out[addr].x = derivative_i[i].x;
                body_grad_out[addr].y = derivative_i[i].y;
                body_grad_out[addr].z = derivative_i[i].z;
                body_grad_out[addr].w = derivative_i[i].w;
          }


          if(directOp<1, true>::type == SPH::HYDROFORCE)
          {
              if(ID[addr] >= SPHBOUND) // 0 force for boundary particles
              {
                  acc_i[0].x = 0; acc_i[0].y = 0; acc_i[0].z = 0; ; acc_i[0].w = 0;
              }

              body_acc_out      [addr].x += acc_i[0].x;
              body_acc_out      [addr].y += acc_i[0].y;
              body_acc_out      [addr].z += acc_i[0].z;
              body_acc_out      [addr].w += acc_i[0].w;

              //This stores the time-step/dt
              body_grad_out[addr].x = max(body_grad_out[addr].x,  derivative_i[0].x);


              if(isFinalLaunch)
              {
                  float dt = SPHParams.c_cfl * 2.0 * dens_i[0].smth / body_grad_out[addr].x;
                  body_grad_out[addr].x = dt;
              }

              //TODO remove, this records interaction stats
              body_grad_out[addr].z += derivative_i[i].z;
              body_grad_out[addr].y += derivative_i[i].y;
          }
        }
        active_inout[addr] = 1;
        {
          interactions[addr].x += counters.x / ni;
          interactions[addr].y += counters.y / ni ;
        }
    }

  }

  return true;
}




template<bool ACCUMULATE, int BLOCKDIM2, template<int NI2, bool FULL> class directOp>
static __device__
void approximate_SPH_main(
    const int n_active_groups,
    int       n_bodies,
    float     eps2,
    uint2     node_begend,
    bool      isFinalLaunch,
    const float4 domainSize,
    const sphParameters     SPHParams,
    int      *active_groups,
    bodyProps &group_body,
    int      *active_inout,
    int2     *interactions,
    float4   *boxSizeInfo,
    float4   *groupSizeInfo,
    float4   *boxCenterInfo,
    float4   *groupCenterInfo,
    real4    *multipole_data,
    int      *MEM_BUF,
    bodyProps &body_j,
    float4   *body_acc_out,
    float2   *body_dens_out,
    float4   *body_grad_out,
    float4   *body_hydro_out,
    const unsigned long long *ID)
{
  const int blockDim2 = BLOCKDIM2;
  const int shMemSize = 1 * (1 << blockDim2);
  __shared__ int shmem_pool[shMemSize];

  const int nWarps2 = blockDim2 - WARP_SIZE2;

  const int sh_offs    = (shMemSize >> nWarps2) * warpId;
  int *shmem           = shmem_pool + sh_offs;
  volatile int *shmemv = shmem;


#if 0
#define SHMODE
#endif

#ifdef SHMODE
  const int nWarps  = 1<<nWarps2;
  const int MAXFAILED = 64;
  __shared__ int failedList[MAXFAILED];
  __shared__ unsigned int failed;

  if (threadIdx.x == 0)
    failed = 0;
#endif

  __syncthreads();

  /*********** check if this block is linked to a leaf **********/

  int  bid  = gridDim.x * blockIdx.y + blockIdx.x;

  while(true)
  {
    if(laneId == 0)
    {
      bid         = atomicAdd(&active_inout[n_bodies], 1);
      shmemv[0]    = bid;
    }

    bid   = shmemv[0];
    if (bid >= n_active_groups) return;

    int *lmem = &MEM_BUF[(CELL_LIST_MEM_PER_WARP<<nWarps2)*blockIdx.x + CELL_LIST_MEM_PER_WARP*warpId];
    const bool success = treewalk_control<0,blockDim2,ACCUMULATE, directOp>(
                                    bid,
                                    eps2,
                                    node_begend,
                                    isFinalLaunch,
                                    domainSize,
                                    SPHParams,
                                    active_groups,
                                    group_body,
                                    groupSizeInfo,
                                    groupCenterInfo,
                                    boxCenterInfo,
                                    boxSizeInfo,
                                    multipole_data,
                                    shmem,
                                    lmem,
                                    interactions,
                                    active_inout,
                                    body_j,
                                    body_acc_out,
                                    body_dens_out,
                                    body_grad_out,
                                    body_hydro_out,
                                    ID);

#ifdef SHMODE
    if (!success)
      if (laneId == 0)
        failedList[atomicAdd(&failed,1)] = bid;

    if (failed + nWarps >= MAXFAILED)
    {
      __syncthreads();
      if (warpId == 0)
      {
        int *lmem1 = &MEM_BUF[(CELL_LIST_MEM_PER_WARP<<nWarps2)*blockIdx.x];
        const int n = failed;
        failed = 0;
        for (int it = 0; it < n; it++)
        {
          const bool success = treewalk_control<nWarp2,blockDim2,ACCUMULATE, directOp>(
              failedList[it], 
              eps2,
              node_begend,
              isFinalLaunch,
              active_groups,
              group_body_pos,
              group_body_vel,
              body_pos,
              body_vel,
              groupSizeInfo,
              groupCenterInfo,
              boxCenterInfo,
              boxSizeInfo,
              multipole_data,
              shmem,
              lmem1,
              acc_out,
              interactions,
              active_inout
              body_dens,
              body_grad);
          TODO(jbedorf) Update the above argument list
          assert(success);
        }
      }    
      __syncthreads();
    }

#else

    //Try to get access to the big stack, only one block per time is allowed
    if (!success)
    {
      if(laneId == 0)
      {
        int res = atomicExch(&active_inout[n_bodies+1], 1); //If the old value (res) is 0 we can go otherwise sleep
        int waitCounter  = 0;
        while(res != 0)
        {
          //Sleep
          for(int i=0; i < (1024); i++)
            waitCounter += 1;

          //Test again
          shmem[0] = waitCounter;
          res = atomicExch(&active_inout[n_bodies+1], 1); 
        }
      }

      int *lmem = &MEM_BUF[gridDim.x*(CELL_LIST_MEM_PER_WARP<<nWarps2)];
      const bool success = treewalk_control<8,blockDim2,ACCUMULATE, directOp>(
                                              bid,
                                              eps2,
                                              node_begend,
                                              isFinalLaunch,
                                              domainSize,
                                              SPHParams,
                                              active_groups,
                                              group_body,
                                              groupSizeInfo,
                                              groupCenterInfo,
                                              boxCenterInfo,
                                              boxSizeInfo,
                                              multipole_data,
                                              shmem,
                                              lmem,
                                              interactions,
                                              active_inout,
                                              body_j,
                                              body_acc_out,
                                              body_dens_out,
                                              body_grad_out,
                                              body_hydro_out,
                                              ID);
      assert(success);

      if(laneId == 0)
        atomicExch(&active_inout[n_bodies+1], 0); //Release the lock
    }
#endif /* SHMODE */
  }     //end while
#undef SHMODE
}



  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_density(
          const int n_active_groups,
          int       n_bodies,
          float     eps2,
          uint2     node_begend,
          bool      isFinalLaunch,
          const domainInformation domainInfo,
          const sphParameters     SPHParams,
          int       *active_groups,
          bodyProps group_body,                //The i-particles
          int       *active_inout,
          int2      *interactions,
          float4    *boxSizeInfo,
          float4    *groupSizeInfo,
          float4    *boxCenterInfo,
          float4    *groupCenterInfo,
          real4     *multipole_data,
          int       *MEM_BUF,
          bodyProps body_j,                //The j-particles
          real4     *body_acc_out,
          float2    *body_dens_out,
          float4    *body_hydro_out,
          float4    *body_grad_out,
          const ullong    *ID)
    {
  approximate_SPH_main<false, NTHREAD2, SPH::density::directOperator>(
           n_active_groups,
           n_bodies,
           eps2,
           node_begend,
           isFinalLaunch,
           domainInfo.domainSize,
           SPHParams,
           active_groups,
           group_body,
           active_inout,
           interactions,
           boxSizeInfo,
           groupSizeInfo,
           boxCenterInfo,
           groupCenterInfo,
           multipole_data,
           MEM_BUF,
           body_j,
           body_acc_out,
           body_dens_out,
           body_grad_out,
           body_hydro_out,
           ID);
}


extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_hydro(
          const int n_active_groups,
          int       n_bodies,
          float     eps2,
          uint2     node_begend,
          bool      isFinalLaunch,
          const domainInformation domainInfo,
          const sphParameters     SPHParams,
          int       *active_groups,
          bodyProps group_body,
          int       *active_inout,
          int2      *interactions,
          float4    *boxSizeInfo,
          float4    *groupSizeInfo,
          float4    *boxCenterInfo,
          float4    *groupCenterInfo,
          real4     *multipole_data,
          int       *MEM_BUF,
          bodyProps body_j,
          real4     *body_acc_out,
          float2    *body_dens_out,
          float4    *body_hydro_out,
          float4    *body_grad_out,
          const ullong    *ID)
    {
#if 1
      approximate_SPH_main<false, NTHREAD2, SPH::hydroforce::directOperator>(
          n_active_groups,
          n_bodies,
          eps2,
          node_begend,
          isFinalLaunch,
          domainInfo.domainSize,
          SPHParams,
          active_groups,
          group_body,
          active_inout,
          interactions,
          boxSizeInfo,
          groupSizeInfo,
          boxCenterInfo,
          groupCenterInfo,
          multipole_data,
          MEM_BUF,
          body_j,
          body_acc_out,
          body_dens_out,
          body_grad_out,
          body_hydro_out,
          ID);
#endif
    }


/*
 Function that extracts the identified boundary cells and particles.
 Also modifies their offsets which have been computed during the
 identification function.
*/

extern "C" __global__ void gpu_extractBoundaryTree(
                                const bool       smthOnly,
                                const float4    *nodeSize,
                                const float4    *nodeCenter,
                                const float     *nodeSmooth,
                                const float4    *nodeMulti,
                                const float4    *bodyPos,
                                const float4    *bodyVel,
                                const float2    *bodyDens,
                                const float4    *bodyHydro,
                                int4            *markedNodes2,
                                float4          *boundaryTree)
{
//  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
//  const uint tid = threadIdx.x;
//  const uint id  = bid * blockDim.x + tid;

  int4* markedNodes = &markedNodes2[1];

  const int laneIdx = threadIdx.x & (WARP_SIZE-1);

  const int nNodes = markedNodes2[0].x;
  const int nPart  = markedNodes2[0].y;



  float4 description;
  description.x = int_as_float(nPart);
  description.y = int_as_float(nNodes);
  description.z = int_as_float(markedNodes2[0].z);
  description.w = int_as_float(markedNodes2[0].w);
  if(laneId == 0) boundaryTree[0] = description;


  const int nBodyProps = 4;

  int bPosIdx = 1;
  int bVelIdx = 1 + nPart*1;
  int bRhoIdx = 1 + nPart*2;
  int bHydIdx = 1 + nPart*3;

  //CellProps
  int sizeIdx = 1 + nBodyProps*nPart;
  int cntrIdx = 1 + nBodyProps*nPart + nNodes;
  int smthIdx = 1 + nBodyProps*nPart + nNodes*2;
  int multIdx = 1 + nBodyProps*nPart + nNodes*3;


  for(int i=0; i < nNodes; i+= blockDim.x)
  {
      if(i + laneIdx < nNodes)
      {
        int4 idxInfo = markedNodes[i+laneIdx];
        
        if(smthOnly)
        {
            //Only interested in updated smoothing values
            boundaryTree[smthIdx+i+laneIdx] = make_float4(nodeSmooth[idxInfo.x], 0,0,0);
        }
        else
        {
            float4 size = nodeSize[idxInfo.x];
            if(idxInfo.z == -2 || idxInfo.z >= 0)
            {
                size.w = int_as_float(idxInfo.y); //Normal node and leaf node requires new offset

                if(idxInfo.z >= 0)
                {
                    //This is a single particle leaf that refers to a body, store body
                    boundaryTree[bPosIdx+idxInfo.w] = bodyPos[idxInfo.z];
                    boundaryTree[bVelIdx+idxInfo.w] = bodyVel[idxInfo.z];
                    boundaryTree[bRhoIdx+idxInfo.w] = make_float4(bodyDens[idxInfo.z].x, bodyDens[idxInfo.z].y, 0, 0);
                    boundaryTree[bHydIdx+idxInfo.w] = bodyHydro[idxInfo.z];
                }
            }

            //Copy the tree-cell information
            boundaryTree[sizeIdx+i+laneIdx]         = size;
            boundaryTree[cntrIdx+i+laneIdx]         = nodeCenter[idxInfo.x];
            boundaryTree[smthIdx+i+laneIdx]         = make_float4(nodeSmooth[idxInfo.x], 0,0,0);
            boundaryTree[multIdx+3*(i+laneIdx) + 0] = nodeMulti[idxInfo.x*3 + 0];
            boundaryTree[multIdx+3*(i+laneIdx) + 1] = nodeMulti[idxInfo.x*3 + 1];
            boundaryTree[multIdx+3*(i+laneIdx) + 2] = nodeMulti[idxInfo.x*3 + 2];
        }
      }
  }//for nNodes
}


/*
   Function to identify the boundary cells of a tree
   The tree is walked and boundaries are identified, concurrently
   we compute the offsets to where the children of the cells are written
   for the extraction functions.
*/


extern "C" __global__ void gpu_boundaryTree(
                                            const uint2      top_cells,     //Encode the start and end cells of our initial traverse
                                            const  int       maxDepth,      //Number of levels we will traverse
                                            int             *cellStack,     //Used for tree-traversal
                                            const uint      *nodeBodies,    //Contains the info needed to walk the tree
                                            const uint2     *leafBodies,    //To get indices of the particle information
                                            const uint      *nodeValid,     //Used to determine if a cell is a node or a leaf
                                            int4            *markedNodes2)  //The results
{
  int4 *markedNodes = &markedNodes2[1];


  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int SHIFT   = 0 ;

  //Mark the top nodes
  for (int root_cell = 0; root_cell < top_cells.x; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.x){
      markedNodes[root_cell + laneIdx] = make_int4(root_cell + laneIdx, 0, -1, -1); 
    }
      

  //Fill initial stack
  for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE){
    if (root_cell + laneIdx < top_cells.y)
    {
        cellStack[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;
    }
  }

  int nCells = top_cells.y - top_cells.x;

  int cellListBlock           = 0;
  int nextLevelCellCounter    = 0;
  unsigned int cellListOffset = 0;



  int depth           = 0;
  int nodeWriteOffset = top_cells.x;
  int childOffset     = top_cells.y;
  int partOffset      = 0;

  /* process level with n_cells */
#if 1
  while (nCells > 0)
  {
    /* extract cell index from the current level cell list */
    const int cellListIdx = cellListBlock + laneIdx;
    const bool useCell    = cellListIdx < nCells;
    const int cellIdx     = !useCell ? 0 : cellStack[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
    const int loopSize    = min(WARP_SIZE, nCells - cellListBlock);
    cellListBlock        += loopSize;

    //Get the number of children, firstchild and leaf/node info
    uint firstChild =    nodeBodies[cellIdx] & 0x0FFFFFFF;
    uint nChildren  =  ((nodeBodies[cellIdx] & 0xF0000000) >> 28);
    const bool isNode     =  !(nodeValid [cellIdx] >> 31);    //Leafs have a 1 in bit 32

    if(!isNode)
    {
        //For leafs we require reference and count to the actual particles, which is in a different array
        uint2 bij    = leafBodies[cellIdx];
        firstChild   = bij.x & ILEVELMASK;
        nChildren    = (bij.y - firstChild)-1;  //-1 is historic reasons...
    }

    bool splitCell  = nChildren != 8;
    bool splitNode  = isNode && splitCell && useCell && (depth < maxDepth); //Note the depth test
   
    //We have to do a prefix sum to get new child offsets, do this for nodes and leaves separate
    const int2 childScatter     = warpIntExclusiveScan(nChildren & (-splitNode));
    int input                   = useCell && nChildren == 0 && !isNode;
    const int2 childScatterLeaf = warpIntExclusiveScan(input);


    //Default value:  endpoint, without leaf particles
    int4 output = make_int4(cellIdx, 0xFFFFFFFF, -2, 0);


    if(isNode)
    {
      /* make sure we still have available stack space */
      if (childScatter.y + nCells - cellListBlock + nextLevelCellCounter > (CELL_LIST_MEM_PER_WARP<<SHIFT))
        //return make_uint2(0xFFFFFFFF,0xFFFFFFFF);
         return; //TODO print error

      /* if so populate next level stack in gmem */
      if (splitNode)
      {
          {
              const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
              for (int i = 0; i < nChildren; i++)
                  cellStack[ringAddr<SHIFT>(scatterIdx + i)] = firstChild + i;

             //Compute the new references to the children
             output.y = childOffset+childScatter.x | ((uint)(nChildren) << LEAFBIT);
          }
      }
    }
    else
    {
        //Leaf, if it has 1 child, save the index and new storage location
        if(nChildren == 0)
        {
            output.y = partOffset + childScatterLeaf.x | ((uint)(nChildren) << LEAFBIT);
            output.z = firstChild;
            output.w = partOffset + childScatterLeaf.x; 
        }
    }
      
    childOffset += childScatter.y;
    partOffset  += childScatterLeaf.y;
      
    nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */

    if(useCell)
    {
        markedNodes[nodeWriteOffset+laneIdx] = output;
    }
    
    nodeWriteOffset += loopSize; 


    /* if the current level is processed, schedule the next level */
    if (cellListBlock >= nCells)
    {
      cellListOffset += nCells;
      nCells          = nextLevelCellCounter;
      cellListBlock   = nextLevelCellCounter = 0;
      depth++;
    }
  }  /* level completed */
#endif

  //Store the results
  if(laneIdx == 0)
  {
      markedNodes2[0] = make_int4(childOffset, partOffset, top_cells.x, top_cells.y);
  }

  return;
}


/*
   Function to identify the boundary cells of a tree
   The tree is walked and boundaries are identified, concurrently
   we compute the offsets to where the children of the cells are written
   for the extraction functions.
*/
extern "C" __global__ void gpu_boundaryTree2(
                        const uint2      top_cells,
                        int             *cellList,
                        const float4    *nodeSize,
                        const float4    *nodeCenter,
                        int4            *markedNodes2)
{
  int4 *markedNodes = &markedNodes2[1];


  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int SHIFT  = 0 ;

  //Mark the top nodes
  for (int root_cell = 0; root_cell < top_cells.x; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.x){
      markedNodes[root_cell + laneIdx] = make_int4(root_cell + laneIdx, 0, -1, -1);
    }


  //Fill initial stack
  for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE){
    if (root_cell + laneIdx < top_cells.y)
    {
      cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;
    }
  }

  int nCells = top_cells.y - top_cells.x;

  int cellListBlock           = 0;
  int nextLevelCellCounter    = 0;
  unsigned int cellListOffset = 0;

  int depth          = 0;
  const int maxDepth = 30;

  int nodeWriteOffset = top_cells.x;

  int childOffset = top_cells.y;
  int partOffset  = 0;

  /* process level with n_cells */
#if 1
  while (nCells > 0)
  {
    /* extract cell index from the current level cell list */
    const int cellListIdx = cellListBlock + laneIdx;
    const bool useCell    = cellListIdx < nCells;
    const int cellIdx     = !useCell ? 0 : cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
    const int loopSize    = min(WARP_SIZE, nCells - cellListBlock);
    cellListBlock        += loopSize;

    /* read from gmem cell's info */
    const float4 cellSize = nodeSize[cellIdx];
    const float4 cellPos  = nodeCenter[cellIdx];

    /* compute first child, either a cell if node or a particle if leaf */
    const int cellData   = __float_as_int(cellSize.w);
    const int firstChild =  cellData & 0x0FFFFFFF;
    const int nChildren  = (cellData & 0xF0000000) >> 28;
    const bool isNode    = cellPos.w > 0.0f;


    {
        printf("ON DEV, %d => Use: %d First: %d nChild: %d node: %d depth: %d\t %lg\n", cellIdx, useCell, firstChild, nChildren, isNode, depth, cellPos.w);
    }


    bool splitCell                       =  nChildren != 8;
    if(cellData == 0xFFFFFFFF) splitCell = false;
    bool splitNode  = isNode && splitCell && useCell && (depth < maxDepth);

    //We have to do a prefix sum to get new childoffsets, do this for nodes and leaves seperate
    const int2 childScatter = warpIntExclusiveScan(nChildren & (-splitNode));

    //Compute offset for possible particles
    int input                   = useCell && nChildren == 0 && !isNode;
    const int2 childScatterLeaf = warpIntExclusiveScan(input);


    //Default endpoint, without leaf particles
    int4 output = make_int4(cellIdx, 0xFFFFFFFF, -2, 0);


    if(isNode)
    {
      /* make sure we still have available stack space */
      if (childScatter.y + nCells - cellListBlock + nextLevelCellCounter > (CELL_LIST_MEM_PER_WARP<<SHIFT))
        //return make_uint2(0xFFFFFFFF,0xFFFFFFFF);
         return; //TODO print error

      /* if so populate next level stack in gmem */
      if (splitNode)
      {
          const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
          for (int i = 0; i < nChildren; i++)
              cellList[ringAddr<SHIFT>(scatterIdx + i)] = firstChild + i;

         //Compute the new references to the children
         output.y = childOffset+childScatter.x | ((uint)(nChildren) << LEAFBIT);
      }
    }
    else
    {
        //Leaf, if it has 1 child, save the index and new storage location
        if(nChildren == 0)
        {
            output.y = partOffset + childScatterLeaf.x | ((uint)(nChildren) << LEAFBIT);
            output.z = firstChild;
            output.w = partOffset + childScatterLeaf.x;
        }
    }

    childOffset += childScatter.y;
    partOffset  += childScatterLeaf.y;

    nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */

    if(useCell)
    {
        markedNodes[nodeWriteOffset+laneIdx] = output;
    }

    nodeWriteOffset += loopSize;


    /* if the current level is processed, schedule the next level */
    if (cellListBlock >= nCells)
    {
      cellListOffset += nCells;
      nCells          = nextLevelCellCounter;
      cellListBlock   = nextLevelCellCounter = 0;
      depth++;
    }
  }  /* level completed */
#endif


  //Store the results
  if(laneIdx == 0){
        markedNodes2[0] = make_int4(childOffset, partOffset, top_cells.x, top_cells.y);
    }

    return;
}




#if 0
extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
__global__ void
dev_sph_derivative(
        const int n_active_groups,
        int       n_bodies,
        float     eps2,
        uint2     node_begend,
        bool      isFinalLaunch,
        int       *active_groups,
        const domainInformation domainInfo,

        real4     *group_body_pos,           //This can be different from body_pos
        real4     *group_body_vel,
        float2    *group_body_dens,
        float4    *group_body_grad,
        real4     *group_body_hydro,

        int       *active_inout,
        int2      *interactions,
        float4    *boxSizeInfo,
        float4    *groupSizeInfo,
        float4    *boxCenterInfo,
        float4    *groupCenterInfo,
        real4     *multipole_data,
        int       *MEM_BUF,

        real4     *body_pos_j,
        real4     *body_vel_j,
        float2    *body_dens_j,
        float4    *body_grad_j,
        float4    *body_hydro_j,

        real4     *body_acc_out,
        float2    *body_dens_out,
        float4    *body_hydro_out,
        float4    *body_grad_out,
        const ullong    *ID)
  {
#if 1
    bodyProps group_body, body_j;

    group_body.body_pos   = group_body_pos;
    group_body.body_vel   = group_body_vel;
    group_body.body_dens  = group_body_dens;
    group_body.body_grad  = group_body_grad;
    group_body.body_hydro = group_body_hydro;

    body_j.body_pos   = body_pos_j;
    body_j.body_vel   = body_vel_j;
    body_j.body_dens  = body_dens_j;
    body_j.body_grad  = body_grad_j;
    body_j.body_hydro = body_hydro_j;

approximate_SPH_main<false, NTHREAD2, SPH::derivative::directOperator>(
        n_active_groups,
         n_bodies,
         eps2,
         node_begend,
         isFinalLaunch,
         domainInfo.domainSize,
         active_groups,
         group_body,
         active_inout,
         interactions,
         boxSizeInfo,
         groupSizeInfo,
         boxCenterInfo,
         groupCenterInfo,
         multipole_data,
         MEM_BUF,
         body_j,
         body_acc_out,
         body_dens_out,
         body_grad_out,
         body_hydro_out,
         ID);
#endif
}
#endif


#if 0
extern "C" 
__global__ void gpu_boundaryTree(
                        const uint2      top_cells,
                        int             *cellList,
                        const float4    *nodeSize,
                        const float4    *nodeCenter,
                        uint            *markedNodes,
                        uint            *markedParticles,
                        uint            *levels)
{
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);
  const int SHIFT  = 0 ;

  //Mark the top nodes
  for (int root_cell = 0; root_cell < top_cells.x; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.x)
      markedNodes[root_cell + laneIdx] = (root_cell + laneIdx) | (uint)(1 << 31); 
      

  for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.y)
    {
      markedNodes[root_cell + laneIdx] = (root_cell + laneIdx) | (uint)(1 << 31); 
      cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;
    }

  int nCells = top_cells.y - top_cells.x;


  int cellListBlock           = 0;
  int nextLevelCellCounter    = 0;
  unsigned int cellListOffset = 0;

  int depth          = 0;
  const int maxDepth = 30;

  levels[0] = top_cells.x;
  levels[1] = nCells;
  /* process level with n_cells */
#if 1
  while (nCells > 0)
  {
    /* extract cell index from the current level cell list */
    const int cellListIdx = cellListBlock + laneIdx;
    const bool useCell    = cellListIdx < nCells;
    const int cellIdx     = !useCell ? 0 : cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
    cellListBlock        += min(WARP_SIZE, nCells - cellListBlock);

    /* read from gmem cell's info */
    const float4 cellSize = nodeSize[cellIdx];
    const float4 cellPos  = nodeCenter[cellIdx];

    /* compute first child, either a cell if node or a particle if leaf */
    const int cellData   = __float_as_int(cellSize.w);
    const int firstChild =  cellData & 0x0FFFFFFF;
    const int nChildren  = (cellData & 0xF0000000) >> 28;
    
    //We touched this node so mark it  
    markedNodes[cellIdx] = (cellIdx) | (uint)(1 << 31); 

    bool splitCell         =  nChildren != 8;
    if(cellData == 0xFFFFFFFF) splitCell = false;

    const bool isNode = cellPos.w > 0.0f;
    

    if(isNode)
    {
      bool splitNode  = isNode && splitCell && useCell && (depth < maxDepth);

      /* use exclusive scan to compute scatter addresses for each of the child cells */
      const int2 childScatter = warpIntExclusiveScan(nChildren & (-splitNode));

      /* make sure we still have available stack space */
      if (childScatter.y + nCells - cellListBlock + nextLevelCellCounter > (CELL_LIST_MEM_PER_WARP<<SHIFT))
        //return make_uint2(0xFFFFFFFF,0xFFFFFFFF);
         return; //TODO print error

      /* if so populate next level stack in gmem */
      if (splitNode)
      {
          const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
          for (int i = 0; i < nChildren; i++)
              cellList[ringAddr<SHIFT>(scatterIdx + i)] = firstChild + i;
      }
      nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */
    }
    else
    {
        //Leaf TODO do something with 1 particle leafs
        if(nChildren == 0)
        {
            //1Child
            printf("ON DEV, marking child: %d \n", firstChild);
            markedParticles[firstChild] = firstChild |  (uint)(1 << 31); 
        }
    }


    /* if the current level is processed, schedule the next level */
    if (cellListBlock >= nCells)
    {
      cellListOffset += nCells;
      nCells          = nextLevelCellCounter;
      cellListBlock   = nextLevelCellCounter = 0;
      depth++;
      if(threadIdx.x == 0) levels[depth+1] = nCells;
    }
  }  /* level completed */
#endif

    return;
}

#endif

#if 0
  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_density_let(
          const int n_active_groups,
          int       n_bodies,
          float     eps2,
          uint2     node_begend,
          bool      isFinalLaunch,
          int       *active_groups,

          real4     *group_body_pos,           //This can be different from body_pos
          real4     *group_body_vel,
          float2    *group_body_dens,
          float4    *group_body_grad,
          real4     *group_body_hydro,

          int       *active_inout,
          int2      *interactions,
          float4    *boxSizeInfo,
          float4    *groupSizeInfo,
          float4    *boxCenterInfo,
          float4    *groupCenterInfo,
          real4     *multipole_data,
          int       *MEM_BUF,

          real4     *body_pos_j,
          real4     *body_vel_j,
          float2    *body_dens_j,
          float4    *body_grad_j,
          float4    *body_hydro_j,

          real4     *body_acc_out,
          float2    *body_dens_out,
          float4    *body_hydro_out,
          float4    *body_grad_out)
    {
#if 0
      approximate_SPH_main<true, NTHREAD2, SPH::density::directOperator>(
              n_active_groups,
               n_bodies,
               eps2,
               node_begend,
               isFinalLaunch,
               active_groups,
               group_body_pos,           //This can be different from body_pos
               group_body_vel,           //This can be different from body_vel
               group_body_dens,
               group_body_grad,
               group_body_hydro,
               active_inout,
               interactions,
               boxSizeInfo,
               groupSizeInfo,
               boxCenterInfo,
               groupCenterInfo,
               multipole_data,
               MEM_BUF,
               body_pos_j,
               body_vel_j,
               body_dens_j,
               body_grad_j,
               body_hydro_j,
               body_acc_out,
               body_dens_out,
               body_grad_out,
               body_hydro_out);
#endif
}
#endif

#if 0

  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_hydro_let(
      const int n_active_groups,
      int       n_bodies,
      float     eps2,
      uint2     node_begend,
      int       *active_groups,

      real4     *group_body_pos,           //This can be different from body_pos
      real4     *group_body_vel,
      float2    *group_body_dens,
      float4    *group_body_grad,
      real4     *group_body_hydro,

      int       *active_inout,
      int2      *interactions,
      float4    *boxSizeInfo,
      float4    *groupSizeInfo,
      float4    *boxCenterInfo,
      float4    *groupCenterInfo,
      real4     *multipole_data,
      int       *MEM_BUF,

      real4     *body_pos_j,
      real4     *body_vel_j,
      float2    *body_dens_j,
      float4    *body_grad_j,
      float4    *body_hydro_j,

      real4     *body_acc_out,
      float2    *body_dens_out,
      float4    *body_hydro_out,
      float4    *body_grad_out)
{
#if 0
      approximate_SPH_main<true, NTHREAD2, SPH::hydroforce::directOperator>(
      n_active_groups,
      n_bodies,
      eps2,
      node_begend,
      active_groups,
      group_body_pos,           //This can be different from body_pos
      group_body_vel,           //This can be different from body_vel
      group_body_dens,
      group_body_grad,
      group_body_hydro,
      active_inout,
      interactions,
      boxSizeInfo,
      groupSizeInfo,
      boxCenterInfo,
      groupCenterInfo,
      multipole_data,
      MEM_BUF,
      body_pos_j,
      body_vel_j,
      body_dens_j,
      body_grad_j,
      body_hydro_j,
      body_acc_out,
      body_dens_out,
      body_grad_out,
      body_hydro_out);
#endif
}
#endif
#if 0
  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_derivative_let(
          const int n_active_groups,
          int       n_bodies,
          float     eps2,
          uint2     node_begend,
          int       *active_groups,

          real4     *group_body_pos,           //This can be different from body_pos
          real4     *group_body_vel,
          float2    *group_body_dens,
          float4    *group_body_grad,
          real4     *group_body_hydro,

          int       *active_inout,
          int2      *interactions,
          float4    *boxSizeInfo,
          float4    *groupSizeInfo,
          float4    *boxCenterInfo,
          float4    *groupCenterInfo,
          real4     *multipole_data,
          int       *MEM_BUF,

          real4     *body_pos_j,
          real4     *body_vel_j,
          float2    *body_dens_j,
          float4    *body_grad_j,
          float4    *body_hydro_j,

          real4     *body_acc_out,
          float2    *body_dens_out,
          float4    *body_hydro_out,
          float4    *body_grad_out)
    {
#if 0
      approximate_SPH_main<true, NTHREAD2, SPH::derivative::directOperator>(
              n_active_groups,
               n_bodies,
               eps2,
               node_begend,
               active_groups,
               group_body_pos,           //This can be different from body_pos
               group_body_vel,           //This can be different from body_vel
               group_body_dens,
               group_body_grad,
               group_body_hydro,
               active_inout,
               interactions,
               boxSizeInfo,
               groupSizeInfo,
               boxCenterInfo,
               groupCenterInfo,
               multipole_data,
               MEM_BUF,
               body_pos_j,
               body_vel_j,
               body_dens_j,
               body_grad_j,
               body_hydro_j,
               body_acc_out,
               body_dens_out,
               body_grad_out,
               body_hydro_out);
#endif
}
#endif



#if 0


  template<bool ACCUMULATE, int BLOCKDIM2, template<int NI2, bool FULL> class directOp>
  static __device__
  void approximate_SPH_main(
      const int n_active_groups,
      int       n_bodies,
      float     eps2,
      uint2     node_begend,
      bool      isFinalLaunch,
      int      *active_groups,
      real4    *group_body_pos,
      real4    *group_body_vel,
      float2   *group_body_dens,
      float4   *group_body_grad,
      real4    *group_body_hydro,
      int      *active_inout,
      int2     *interactions,
      float4   *boxSizeInfo,
      float4   *groupSizeInfo,
      float4   *boxCenterInfo,
      float4   *groupCenterInfo,
      real4    *multipole_data,
      int      *MEM_BUF,
      real4    *body_pos_j,
      real4    *body_vel_j,
      float2   *body_dens_j,
      real4    *body_grad_j,
      real4    *body_hydro_j,
      float4   *body_acc_out,
      float2   *body_dens_out,
      float4   *body_grad_out,
      float4   *body_hydro_out,
      const ullong    *ID)
  {
    const int blockDim2 = BLOCKDIM2;
    const int shMemSize = 1 * (1 << blockDim2);
    __shared__ int shmem_pool[shMemSize];

    const int nWarps2 = blockDim2 - WARP_SIZE2;

    const int sh_offs    = (shMemSize >> nWarps2) * warpId;
    int *shmem           = shmem_pool + sh_offs;
    volatile int *shmemv = shmem;


  //  float4 privateInteractionList[64];
    __shared__ float4 sh_privateInteractionList[128];
    float4 *privateInteractionList = sh_privateInteractionList;




  #if 0
  #define SHMODE
  #endif

  #ifdef SHMODE
    const int nWarps  = 1<<nWarps2;
    const int MAXFAILED = 64;
    __shared__ int failedList[MAXFAILED];
    __shared__ unsigned int failed;

    if (threadIdx.x == 0)
      failed = 0;
  #endif

    __syncthreads();

    /*********** check if this block is linked to a leaf **********/

    int  bid  = gridDim.x * blockIdx.y + blockIdx.x;

    while(true)
    {
      if(laneId == 0)
      {
        bid         = atomicAdd(&active_inout[n_bodies], 1);
        shmemv[0]    = bid;
      }

      bid   = shmemv[0];
      if (bid >= n_active_groups) return;

  //    if (bid >= 1) return; //JB TDO REMOVE
  //    if(directOp<1, true>::type == SPH::HYDROFORCE)
  //    if(bid != 59) continue;//JB TODO REMOVE



      int *lmem = &MEM_BUF[(CELL_LIST_MEM_PER_WARP<<nWarps2)*blockIdx.x + CELL_LIST_MEM_PER_WARP*warpId];
      const bool success = treewalk_control<0,blockDim2,ACCUMULATE, directOp>(
                                      bid,
                                      eps2,
                                      node_begend,
                                      isFinalLaunch,
                                      active_groups,
                                      group_body_pos,
                                      group_body_vel,
                                      group_body_dens,
                                      group_body_grad,
                                      group_body_hydro,
                                      groupSizeInfo,
                                      groupCenterInfo,
                                      boxCenterInfo,
                                      boxSizeInfo,
                                      multipole_data,
                                      shmem,
                                      privateInteractionList,
                                      lmem,
                                      interactions,
                                      active_inout,
                                      body_pos_j,
                                      body_vel_j,
                                      body_dens_j,
                                      body_grad_j,
                                      body_hydro_j,
                                      body_acc_out,
                                      body_dens_out,
                                      body_grad_out,
                                      body_hydro_out,
                                      ID);

  #ifdef SHMODE
      if (!success)
        if (laneId == 0)
          failedList[atomicAdd(&failed,1)] = bid;

      if (failed + nWarps >= MAXFAILED)
      {
        __syncthreads();
        if (warpId == 0)
        {
          int *lmem1 = &MEM_BUF[(CELL_LIST_MEM_PER_WARP<<nWarps2)*blockIdx.x];
          const int n = failed;
          failed = 0;
          for (int it = 0; it < n; it++)
          {
            const bool success = treewalk_control<nWarp2,blockDim2,ACCUMULATE, directOp>(
                failedList[it],
                eps2,
                node_begend,
                isFinalLaunch,
                active_groups,
                group_body_pos,
                group_body_vel,
                body_pos,
                body_vel,
                groupSizeInfo,
                groupCenterInfo,
                boxCenterInfo,
                boxSizeInfo,
                multipole_data,
                shmem,
                lmem1,
                acc_out,
                interactions,
                active_inout
                body_dens,
                body_grad);
            TODO(jbedorf) Update the above argument list
            assert(success);
          }
        }
        __syncthreads();
      }

  #else

      //Try to get access to the big stack, only one block per time is allowed
      if (!success)
      {
        if(laneId == 0)
        {
          int res = atomicExch(&active_inout[n_bodies+1], 1); //If the old value (res) is 0 we can go otherwise sleep
          int waitCounter  = 0;
          while(res != 0)
          {
            //Sleep
            for(int i=0; i < (1024); i++)
              waitCounter += 1;

            //Test again
            shmem[0] = waitCounter;
            res = atomicExch(&active_inout[n_bodies+1], 1);
          }
        }

        int *lmem = &MEM_BUF[gridDim.x*(CELL_LIST_MEM_PER_WARP<<nWarps2)];
        const bool success = treewalk_control<8,blockDim2,ACCUMULATE, directOp>(
                                                bid,
                                                eps2,
                                                node_begend,
                                                isFinalLaunch,
                                                active_groups,
                                                group_body_pos,
                                                group_body_vel,
                                                group_body_dens,
                                                group_body_grad,
                                                group_body_hydro,
                                                groupSizeInfo,
                                                groupCenterInfo,
                                                boxCenterInfo,
                                                boxSizeInfo,
                                                multipole_data,
                                                shmem,
                                                privateInteractionList,
                                                lmem,
                                                interactions,
                                                active_inout,
                                                body_pos_j,
                                                body_vel_j,
                                                body_dens_j,
                                                body_grad_j,
                                                body_hydro_j,
                                                body_acc_out,
                                                body_dens_out,
                                                body_grad_out,
                                                body_hydro_out,
                                                ID);
        assert(success);

        if(laneId == 0)
          atomicExch(&active_inout[n_bodies+1], 0); //Release the lock
      }
  #endif /* SHMODE */
    }     //end while
  #undef SHMODE
  }

#endif

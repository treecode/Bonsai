#include "bonsai.h"
#include <stdio.h>
#include <stdarg.h>
#include "node_specs.h"

#include "treewalk_includes.h"



/*
 * TODO
 * - Symmetry tree for symmetric hydro force
 */

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


#if 1
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

#if 1
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
      const float4 jM0   = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
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

#if 1

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

__device__ bool split_node_sph_md_print(const float4 nodeSize,
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

  printf("ON DEV distance: %f %f %f | ds2: %f  grpH: %f cellH: %f \n", dr.x, dr.y, dr.z, ds2, grpH, cellH);

  return (ds2 <= (grpH+cellH));
}


#else
/*
 * TODO When using this pre-compute bHigh and bLow
 * TODO This one does not work
 */
__device__ bool split_node_sph_md(
    const float4 nodeSize,
    const float4 nodeCenter,
    const float4 groupCenter,
    const float4 groupSize,
    const float  grpH,
    const float  cellH)
{
    //TODO NOTE When using this function make sure not to use a squared cellH
    //Test overlap
    const float aHighx = nodeCenter.x+nodeSize.x;
    const float aHighy = nodeCenter.y+nodeSize.y;
    const float aHighz = nodeCenter.z+nodeSize.z;
    const float aLowx  = nodeCenter.x-nodeSize.x;
    const float aLowy  = nodeCenter.y-nodeSize.y;
    const float aLowz  = nodeCenter.z-nodeSize.z;
    const float bHighx = groupCenter.x+groupSize.x+grpH;
    const float bHighy = groupCenter.y+groupSize.y+grpH;
    const float bHighz = groupCenter.z+groupSize.z+grpH;
    const float bLowx  = groupCenter.x-groupSize.x-grpH;
    const float bLowy  = groupCenter.y-groupSize.y-grpH;
    const float bLowz  = groupCenter.z-groupSize.z-grpH;

    bool notOverlap =      (aHighx < bLowx) || (bHighx < aLowx)
                        || (aHighy < bLowy) || (bHighy < aLowy)
                        || (aHighz < bLowz) || (bHighz < aLowz);

    return !notOverlap;
}

#endif


#define TEXTURES




template<int SHIFT, int BLOCKDIM2, int NI, bool INTCOUNT, template<int NI2, bool FULL> class directOP>
static __device__
uint2 approximate_sph(
                              float4     acc_i   [NI] /* out */,
                        const float4     _pos_i  [NI],
                        const float4     _vel_i  [NI],
                        const float4     _hydro_i[NI],
                        const float4     groupPos,
                        const float4    *body_pos_j,
                        const float4    *body_vel_j,
                        const float4    *body_hydro_j,
                        const float2    *body_dens_j,
                        const float      eps2,
                        const uint2      top_cells,
                        int             *shmem,
                        float4          *privateInteractionList,
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
  float4 pos_i[NI], vel_i[NI], hydro_i[NI];
#pragma unroll 1
  for (int i = 0; i < NI; i++){
    pos_i[i]   = _pos_i[i];
    vel_i[i]   = _vel_i[i];
    hydro_i[i] = _hydro_i[i];
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
    const int cellIdx     = !useCell ? 0 : cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
    cellListBlock        += min(WARP_SIZE, nCells - cellListBlock);

    /* read from gmem cell's info */
    const float4 cellSize = nodeSize[cellIdx];
    const float4 cellPos  = nodeCenter[cellIdx];
    //const float4 cellSize = tex1Dfetch(ztexNodeSize,   cellIdx);
    //const float4 cellPos  = tex1Dfetch(ztexNodeCenter, cellIdx);

    //For hydro-force we also compare it with smoothing range of the cell, for other props
    //we just use 0. Let's hope compiler is smart enough to notice it stays 0 in that case
    float cellH  = 0;
    if(directOP<1, true>::type == SPH::HYDROFORCE) cellH = fabs(cellPos.w);


    bool splitCell         = split_node_sph_md(cellSize, cellPos, groupPos, groupSize, grpH, cellH);
    interactionCounters.x += 1; //Keep track of number of opening tests



    /* compute first child, either a cell if node or a particle if leaf */
    const int cellData   = __float_as_int(cellSize.w);
    const int firstChild =  cellData & 0x0FFFFFFF;
    const int nChildren  = (cellData & 0xF0000000) >> 28;
//

//    if(directOP<1, true>::type == SPH::HYDROFORCE)
//    {
//        if(splitCell == 0  && cellIdx == 38)
//        {
//            printf("ON DEV, not opening cell: %d | %f %f %f | %f %f %f | %f  \n", cellIdx,
//                    cellPos.x, cellPos.y, cellPos.z, cellSize.x, cellSize.y, cellSize.z, cellH);
//            printf("ON DEV, not opening group    | %f %f %f | %f %f %f | %f  \n",
//                    groupPos.x, groupPos.y, groupPos.z, groupSize.x, groupSize.y, groupSize.z, cellH);
//            split_node_sph_md_print(cellSize, cellPos, groupPos, groupSize, grpH, cellH);
//        }
//        if(splitCell && cellIdx == 38)
//        {
//            printf("ON DEV, opening cell: %d | %f %f %f | %f %f %f | %f  \n", cellIdx,
//                     cellPos.x, cellPos.y, cellPos.z, cellSize.x, cellSize.y, cellSize.z, cellH);
//            printf("ON DEV,  opening group    | %f %f %f | %f %f %f | %f  \n",
//                    groupPos.x, groupPos.y, groupPos.z, groupSize.x, groupSize.y, groupSize.z, cellH);
//            split_node_sph_md_print(cellSize, cellPos, groupPos, groupSize, grpH, cellH);
//        }
//    }


//    if(directOP<1, true>::type == SPH::HYDROFORCE)
//    if(cellIdx == 492) {
//            printf("ON DEV cell 492, print status: %d | %d %d || %f %f %f  %f %f %f | %f \n",
//                    splitCell, firstChild, nChildren, cellPos.x, cellPos.y, cellPos.z, cellSize.x, cellSize.y, cellSize.z, cellH);
//            {
//                const int firstBody =   cellData & BODYMASK;
//                const int     nBody = ((cellData & INVBMASK) >> LEAFBIT)+1;
//                for(int z=firstBody; z < firstBody+nBody; z++)
//                {
//                    float4 bodyj  = body_pos_j[z];
//                    float2 bodyjd = body_dens_j[z];
//                    {
//                        printf("ON DEVFOUNDX: %f %f %f %.16f %d Leaf: %d  | dens: %f %f \n",
//                                bodyj.x, bodyj.y, bodyj.z, bodyj.w, bodyj.w > 0, cellIdx,
//                                bodyjd.x, bodyjd.y);
//                    }
//                }
//            }
//        }
//    if(directOP<1, true>::type == SPH::HYDROFORCE)
//    if(cellIdx  == 16) {
//            printf("ON DEV cell %d, print status: %d | %d %d  || %f %f %f  %f %f %f | %f \n",
//                    cellIdx, splitCell, firstChild, nChildren,
//                    cellPos.x, cellPos.y, cellPos.z, cellSize.x, cellSize.y, cellSize.z, cellH);
//        }
//    if(directOP<1, true>::type == SPH::HYDROFORCE)
//        if(firstChild  <= 16 && firstChild+nChildren > 16) {
//                printf("ON DEV cell %d, print status: %d | %d %d  || %f %f %f  %f %f %f | %f \n",
//                        cellIdx, splitCell, firstChild, nChildren,
//                        cellPos.x, cellPos.y, cellPos.z, cellSize.x, cellSize.y, cellSize.z, cellH);
//            }



//    if(directOP<1, true>::type == SPH::HYDROFORCE) splitCell = true;


    if(cellData == 0xFFFFFFFF) splitCell = false;

    /**********************************************/
    /* split cells that satisfy opening condition */
    /**********************************************/

    const bool isNode = cellPos.w > 0.0f;

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
      bool isDirect = splitCell && isLeaf && useCell;

      const int firstBody =   cellData & BODYMASK;
      const int     nBody = ((cellData & INVBMASK) >> LEAFBIT)+1;

//      if(directOP<1, true>::type == SPH::HYDROFORCE)
//      {
////          if(((pos_i[0].x > 0.968 && pos_i[0].x < 0.969) &&
////                       (pos_i[0].y > 0.093 && pos_i[0].y < 0.094) &&
////                       (pos_i[0].z > 0.078 && pos_i[0].z < 0.079)))
//          {
//              for(int jb=firstBody; jb < firstBody+nBody; jb++)
//              {
//                  float4 bodyj  = body_pos_j[jb];
//                  float2 bodyjd = body_dens_j[jb];
//                  if(((bodyj.x > 0.98 && bodyj.x < 0.99) &&
//                      (bodyj.y > 0.054 && bodyj.y < 0.055) &&
//                      (bodyj.z > 0.07 && bodyj.z < 0.071)))
//                  {
//                      printf("ON DEVFOUND: %f %f %f %.16f %d Leaf: %d  | dens: %f %f \n",
//                              bodyj.x, bodyj.y, bodyj.z, bodyj.w, bodyj.w > 0, cellIdx,
//                              bodyjd.x, bodyjd.y);
//                  }
//
//              }
//
//          }
//      }


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


        //Test the found particle against the group boundaries to see if this particular particle
        //is actually useful or just a useless by-product of having to use one of it's fellow leaf-particles
        if(0)
        {
            float4 bodyj = body_pos_j[ptclIdx];
            bool    use  = testParticleMD(bodyj,groupPos, groupSize, grpH);
            if(!use) derivative_i[0].z++;
            else     derivative_i[0].w++;
        }


        if (nParticle >= WARP_SIZE)
        {
          directOP<NI,true>()(acc_i, pos_i, vel_i, ptclIdx, eps2, density_i, derivative_i, hydro_i,
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
            directOP<NI,true>()(acc_i, pos_i, vel_i, tmpList[laneIdx], eps2, density_i, derivative_i, hydro_i,
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
    directOP<NI,false>()(acc_i, pos_i, vel_i, laneIdx < directCounter ? directPtclIdx : -1, eps2, density_i, derivative_i, hydro_i,
            body_pos_j, body_vel_j, body_dens_j, body_hydro_j);
    if (INTCOUNT)
      interactionCounters.y += directCounter * NI;
    directCounter = 0;
  }

  return interactionCounters;
}

//Re-compute the box position and size but now with the smoothing length taken into account
__device__ void computeGroupBoundsWithSmoothingLength(const float4 pos[2], const int ni,
                                                      float4 &grpCenter, float4 &grpSize)
{
    //Compute the extends of the 'box' around this particle
    float  smth  = pos[0].w*SPH_KERNEL_SIZE;
    float3 r_min = make_float3(pos[0].x-smth, pos[0].y-smth, pos[0].z-smth);
    float3 r_max = make_float3(pos[0].x+smth, pos[0].y+smth, pos[0].z+smth);

    if(ni > 1)  //Reduce within the thread the 2 possible particles
    {
        smth    = pos[1].w*SPH_KERNEL_SIZE;
        r_min.x = fminf(pos[1].x-smth, r_min.x); r_min.y = fminf(pos[1].y-smth, r_min.y); r_min.z = fminf(pos[1].z-smth, r_min.z);
        r_max.x = fmaxf(pos[1].x+smth, r_max.x); r_max.y = fmaxf(pos[1].y+smth, r_max.y); r_max.z = fmaxf(pos[1].z+smth, r_max.z);
    }

    //Reduce the whole group
    r_min.x = warpAllReduceMin(r_min.x); r_min.y = warpAllReduceMin(r_min.y); r_min.z = warpAllReduceMin(r_min.z);
    r_max.x = warpAllReduceMax(r_max.x); r_max.y = warpAllReduceMax(r_max.y); r_max.z = warpAllReduceMax(r_max.z);

    //Compute the group center and size
    grpCenter.x = 0.5*(r_min.x + r_max.x);
    grpCenter.y = 0.5*(r_min.y + r_max.y);
    grpCenter.z = 0.5*(r_min.z + r_max.z);

    grpSize.x = fmaxf(fabs(grpCenter.x-r_min.x), fabs(grpCenter.x-r_max.x));
    grpSize.y = fmaxf(fabs(grpCenter.y-r_min.y), fabs(grpCenter.y-r_max.y));
    grpSize.z = fmaxf(fabs(grpCenter.z-r_min.z), fabs(grpCenter.z-r_max.z));
}


template<int SHIFT2, int BLOCKDIM2, bool ACCUMULATE, template<int NI2, bool FULL> class directOp>
static __device__
bool treewalk_control(
    const int        bid,
    const float      eps2,
    const uint2      node_begend,
    const bool       isFinalLaunch,
    const int       *active_groups,
    const real4     *group_body_pos,
    const real4     *group_body_vel,
    const float2    *group_body_dens,
    const float4    *group_body_grad,
          float4    *group_body_hydro,
    const float4    *groupSizeInfo,
    const float4    *groupCenterInfo,
    const float4    *nodeCenter,
    const float4    *nodeSize,
    const float4    *nodeMultipole,
    int             *shmem,
    float4          *privateInteractionList,
    int             *lmem,
    int2            *interactions,
    int             *active_inout,
    const real4     *body_pos_j,
    const real4     *body_vel_j,
    const float2    *body_dens_j,
    const real4     *body_grad_j,
    const real4     *body_hydro_j,
    float4          *body_acc_out,
    float2          *body_dens_out,
    float4          *body_grad_out,
    float4          *body_hydro_out)
{
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
  if(laneId%NCRIT > nb_i) body_i[0] = body_addr;

  /*
   * TODO Also consider removing all the [2] sized arrays as 64 particles per group is significantly slower and hence will
   * probably not be used in the SPH kernels
   */
//  if(body_i[0] == 3813)
//  {
//      printf("BOdy is part of grp: %d  group props: %d %d \n", bid,body_addr, nb_i);
//  }

  //3813 BOdy is part of grp: 476
  //20726 BOdy is part of grp: 2590



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

//  if(((pos_i[0].x > 0.968 && pos_i[0].x < 0.969) &&
//            (pos_i[0].y > 0.093 && pos_i[0].y < 0.094) &&
//            (pos_i[0].z > 0.078 && pos_i[0].z < 0.079)))
//          {
//      printf("ON DEV: [ %d %d ] Body is part of grp: %d  addr: %d \t %d %d\n",
//              threadIdx.x, blockIdx.x, bid, body_i[0], body_addr, nb_i);
//          }


  //For the hydro force we need the current density value
  if(directOp<1, true>::type == SPH::HYDROFORCE) {
                   dens_i[0].dens = group_body_dens[body_i[0]].x;
        if(ni > 1) dens_i[1].dens = group_body_dens[body_i[1]].x;

        derivative_i[0].x = -10e10; //We are using 'min' on this variable to find minimal time-step
  }

  uint2 counters = {0};


  pos_i[0].w                = dens_i[0].smth;  //y is smoothing range
  if(ni > 1 )   pos_i[1].w  = dens_i[1].smth;  //y is smoothing range

#if 1
  //Compute the cellH, which is the maximum smoothing value off all particles
  //in a group because as we search for the whole group in parallel we have to
  //be sure to include all possibly required particles
  float cellH = warpAllReduceMax(pos_i[0].w);
  if(ni == 2)
  {
    cellH = max(cellH, warpAllReduceMax(pos_i[1].w));
  }
  cellH *= SPH::kernel_t::supportRadius(); //3.5;   //TODO This (value around 3) is needed to get the results to match those of FDPS, why is that. Looks like there is no hard cutoff in the kernel??
  cellH *= cellH; //Squared for distance comparison without sqrt
#else
  //TODO: This results in more cells being opened. Look into this later, might be related to the 3.5 factor
  computeGroupBoundsWithSmoothingLength(pos_i, ni, group_pos, curGroupSize);
  const float cellH = 0;
#endif

  //For periodic boundaries, mirror the group and body positions along the periodic axis
  //float3 domainSize = {1.0f, 0.125f, 0.125f};   //Hardcoded for our testing IC
//  float3 domainSize = {100.0f, 100.0f, 100.0f};   //Hardcoded for our testing IC
  float3 domainSize = {1.0f, 0.040594f, 0.038274f};   //Hardcoded for phantom tube


  long long int startC = clock64();

  //TODO use templates or parameterize this at some point
//  for(int ix=-1; ix <= 1; ix++)     //Periodic around X
  {
    for(int iy=-1; iy <= 1; iy++)   //Periodic around Y
    {
      for(int iz=-1; iz <= 1; iz++) //Periodic around Z
      {
//          int ix =0; int iz=0; int iy = 0;// iz = 0;
//          if(iy == 1) continue;
//          if(iy == 0 && (directOp<1, true>::type == SPH::DERIVATIVE)) continue;
//          int iy = 1;

          int ix = 0;

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
                        eps2,
                        node_begend,
                        shmem,
                        privateInteractionList,
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
                        eps2,
                        node_begend,
                        shmem,
                        privateInteractionList,
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

  if(directOp<1, true>::type == SPH::DERIVATIVE)
  {
      derivative_i[0].x = warpGroupReduce(derivative_i[0].x);
      derivative_i[0].y = warpGroupReduce(derivative_i[0].y);
      derivative_i[0].z = warpGroupReduce(derivative_i[0].z);
      derivative_i[0].w = warpGroupReduce(derivative_i[0].w);
  }
  if(directOp<1, true>::type == SPH::DENSITY)
  {
      dens_i[0].dens    = warpGroupReduce(dens_i[0].dens);
      derivative_i[0].x = warpGroupReduce(derivative_i[0].x); //For stats
      derivative_i[0].y = warpGroupReduce(derivative_i[0].y); //For stats
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
//      float temp = derivative_i[0].y;
//      derivative_i[0].x = warpGroupReduce(derivative_i[0].x); //For stats
//      derivative_i[0].y = warpGroupReduce(derivative_i[0].y); //For stats

//      if(((pos_i[0].x > 0.968 && pos_i[0].x < 0.969) &&
//                (pos_i[0].y > 0.093 && pos_i[0].y < 0.094) &&
//                (pos_i[0].z > 0.078 && pos_i[0].z < 0.079)))
//              {
//          printf("ON DEV: [ %d %d ] Body is part of grp: %d  addr: %d \tSum: %f %f \n",
//                  threadIdx.x, blockIdx.x, bid, body_i[0], derivative_i[0].y, temp);
//              }

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
                dens_i[i].dens     += body_dens_out[addr].x;
                dens_i[i].finalize(group_body_pos[body_i[i]].w);
                body_dens_out[addr] = make_float2(dens_i[i].dens,dens_i[i].smth);

//                body_grad_out[addr].x = derivative_i[i].x; //For Stats
//                body_grad_out[addr].y = derivative_i[i].y; //For Stats

//                if(addr == 10086)
//                {
//                    printf("ON DEV [ %f %f %f ] RHO OUT : %f %f \n",
//                            pos_i[0].x, pos_i[0].y, pos_i[0].z,
//                            dens_i[i].dens,dens_i[i].smth);
//                }

  //              //TODO remove, this records stats
  //               body_grad_out[addr].x += derivative_i[i].x;
  //               body_grad_out[addr].y += derivative_i[i].y;
  //
  //              if(laneId == -1) body_grad_out[addr].y += endC-startC;
  //              else body_grad_out[addr].y = derivative_i[i].y;
  //              body_grad_out[addr].z = derivative_i[i].z;
  //              body_grad_out[addr].w = derivative_i[i].w;
            }

          if(directOp<1, true>::type == SPH::DERIVATIVE)
          {
            derivative_i[i].x += body_grad_out[addr].x;
            derivative_i[i].y += body_grad_out[addr].y;
            derivative_i[i].z += body_grad_out[addr].z;
            derivative_i[i].w += body_grad_out[addr].w;

            if(isFinalLaunch)
            {
                //Finalize the values as this launch contained the final required data
               derivative_i[i].finalize(body_dens_out[addr].x);

               //Need this printf to get non-nan results on GTX1080
               if(addr == -3830)
               {
                   printf("TEST: %f %f %f %f \n",  body_grad_out[addr].x, derivative_i[i].x, body_dens_out[addr].x, derivative_i[i].y);
               }

               //Compute Balsala switch
               float temp  = fabs(derivative_i[i].w);
               float temp2 = derivative_i[i].x*derivative_i[i].x +
                             derivative_i[i].y*derivative_i[i].y +
                             derivative_i[i].z*derivative_i[i].z;
               float temp3 = 1.0e-4 * group_body_hydro[addr].y / body_dens_out[addr].y;

               //Note using group_body_hydro here instead of body_hydro_out to store Balsala Switch
               group_body_hydro[addr].w = temp / (temp + sqrtf(temp2) + temp3);
            }


            body_grad_out[addr].x = derivative_i[i].x;
            body_grad_out[addr].y = derivative_i[i].y;
            body_grad_out[addr].z = derivative_i[i].z;
            body_grad_out[addr].w = derivative_i[i].w;
          }




          if(directOp<1, true>::type == SPH::HYDROFORCE)
          {
              body_acc_out      [addr].x += acc_i[0].x;
              body_acc_out      [addr].y += acc_i[0].y;
              body_acc_out      [addr].z += acc_i[0].z;
              body_acc_out      [addr].w += acc_i[0].w;

              body_grad_out[addr].x = max(body_grad_out[addr].x,  derivative_i[0].x);


              if(isFinalLaunch)
              {
                  const float C_CFL = 0.3;
                  float dt = C_CFL * 2.0 * dens_i[0].smth / body_grad_out[addr].x;
                  body_grad_out[addr].x = dt;
              }

              //TODO remove, this records interaction stats
              //body_grad_out[addr].x += derivative_i[i].x;
              //body_grad_out[addr].y += derivative_i[i].y;
          }
        }
        active_inout[addr] = 1;
        {
          interactions[addr].x += counters.x / ni;
          interactions[addr].y += counters.y / ni ;
        }
    }
/*
#if 0
    const int addr = body_i[0];
    if (ACCUMULATE)
    {
      acc_out      [addr].x += acc_i[0].x;
      acc_out      [addr].y += acc_i[0].y;
      acc_out      [addr].z += acc_i[0].z;
      acc_out      [addr].w += acc_i[0].w;

      body_grad_out[addr].x += derivative_i[0].x;
      body_grad_out[addr].y += derivative_i[0].y;
      body_grad_out[addr].z += derivative_i[0].z;
      body_grad_out[addr].w += derivative_i[0].w;

      body_dens_out[addr].x += dens_i[0].dens;
      body_dens_out[addr].y += dens_i[0].smth; //TODO can not sum lengths
    }
    else
    {
      acc_out      [addr] =  acc_i[0];
      body_dens_out[addr] = make_float2(dens_i[0].dens, dens_i[0].smth);

      body_grad_out[addr].x = 123; //derivative_i[0].x;
      body_grad_out[addr].y = derivative_i[0].y;
      body_grad_out[addr].z = derivative_i[0].z;
      body_grad_out[addr].w = derivative_i[0].w;
    }
    active_inout[addr] = 1;
    if (ACCUMULATE)
    {
      interactions[addr].x += counters.x / ni;
      interactions[addr].y += counters.y / ni ;
    }
    else
    {
      interactions[addr].x = counters.x / ni;
      interactions[addr].y = counters.y / ni ;
    }
    if (ni == 2)
    {
      const int addr = body_i[1];
      if (ACCUMULATE)
      {
        acc_out      [addr].x += acc_i[1].x;
        acc_out      [addr].y += acc_i[1].y;
        acc_out      [addr].z += acc_i[1].z;
        acc_out      [addr].w += acc_i[1].w;

        body_grad_out[addr].x += derivative_i[1].x;
        body_grad_out[addr].y += derivative_i[1].y;
        body_grad_out[addr].z += derivative_i[1].z;
        body_grad_out[addr].w += derivative_i[1].w;

        body_dens_out[addr].x += dens_i[1].dens;
        body_dens_out[addr].y += dens_i[1].smth; //TODO can not sum lengths
      }
      else
      {
        acc_out      [addr] =  acc_i[1];
        body_dens_out[addr] = make_float2(dens_i[1].dens, dens_i[1].smth);
        body_grad_out[addr].x = derivative_i[1].x;
        body_grad_out[addr].y = derivative_i[1].y;
        body_grad_out[addr].z = derivative_i[1].z;
        body_grad_out[addr].w = derivative_i[1].w;

      }
      active_inout[addr] = 1;
      if (ACCUMULATE)
      {
        interactions[addr].x += counters.x / ni;
        interactions[addr].y += counters.y / ni;
      }
      else
      {
        interactions[addr].x = counters.x / ni;
        interactions[addr].y = counters.y / ni;
      }
    }
#endif
*/
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
    float4   *body_hydro_out)
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
                                    body_hydro_out);

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
                                              body_hydro_out);
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
  approximate_SPH_main<false, NTHREAD2, SPH::density::directOperator>(
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
}




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
#if 1
  approximate_SPH_main<false, NTHREAD2, SPH::derivative::directOperator>(
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




  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_hydro(
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
#if 1
          approximate_SPH_main<false, NTHREAD2, SPH::hydroforce::directOperator>(
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

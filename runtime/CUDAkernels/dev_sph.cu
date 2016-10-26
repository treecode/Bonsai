#include "bonsai.h"
// #include "support_kernels.cu0
#include <stdio.h>
#include "treewalk_includes.h"

#include "../profiling/bonsai_timing.h"
PROF_MODULE(dev_approximate_gravity);


#include "node_specs.h"

#ifdef WIN32
#define M_PI        3.14159265358979323846264338328
#endif



/***********************************/
/***** DENSITY   ******************/

static __device__ __forceinline__ void computeDensityAndNgb(
    const float r2, const float hinv2, const float mass, 
    float &density, float &nb)
{
#if 0  /* full kernel for reference */
  const float hinv = 1.0f/h;
  const float hinv2 = hinv*hinv;
  const float hinv3 = hinv*hinv2;
  const float C     = 3465.0f/(512.0f*M_PI)*hinv3;
  const float q2    = r2*hinv2;
  const float rho   = fmaxf(0.0f, 1.0f - q2);
  nb      += ceilf(rho);
  const float rho2 = rho*rho;
  density += C * rho2*rho2;
#else
  const float rho   = fmaxf(0.0f, 1.0f - r2*hinv2);   /* fma, fmax */
  const float rho2  = rho*rho;                        /* fmul */
  density += rho2*rho2;                               /* fma */
  nb      += ceilf(rho2);                             /* fadd, ceil */

  /*2x fma, 1x fmul, 1x fadd, 1x ceil, 1x fmax */
  /* total: 6 flops or 8 flops with ceil&fmax */
#endif
}

#if 0
static __device__ __forceinline__ float adjustH(const float h_old, const float nnb)
{
	const float nbDesired 	= 42;
	const float f      	= 0.5f * (1.0f + cbrtf(nbDesired / nnb));
	const float fScale 	= max(min(f, 1.2), 0.8);
	return (h_old*fScale);
}
#endif





/**************************************/
/*************** Tree walk ************/
/**************************************/

  template<int SHIFT>
__forceinline__ static __device__ int ringAddr(const int i)
{
  return (i & ((CELL_LIST_MEM_PER_WARP<<SHIFT) - 1));
}



/*********** Forces *************/

static __device__ __forceinline__ float4 add_acc(
          float4  acc,
    const float4  pos,
    const float   massj, const float3 posj,
    const float   eps2,
          float2 &density)
{
#if 1  // to test performance of a tree-walk 
  const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);

  const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  const float r2eps  = r2 + eps2;
  const float rinv   = rsqrtf(r2eps);
  const float rinv2  = rinv*rinv;
  const float mrinv  = massj * rinv;
  const float mrinv3 = mrinv * rinv2;

  acc.w -= mrinv;
  acc.x += mrinv3 * dr.x;
  acc.y += mrinv3 * dr.y;
  acc.z += mrinv3 * dr.z;

  computeDensityAndNgb(r2,pos.w,massj,density.x,density.y);
#endif

  return acc;
}
template<int NI, bool FULL>
static __device__ __forceinline__ void directAcc(
          float4  acc_i[NI], 
    const float4  pos_i[NI],
    const int     ptclIdx,
    const float   eps2,
          float2  density_i[NI],
    const float4 *body)
{
  const float4 M0 = (FULL || ptclIdx >= 0) ? body[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
 //const float4 M0 = (FULL || ptclIdx >= 0) ? tex1Dfetch(texBody, ptclIdx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

//#pragma unroll
  for (int j = 0; j < WARP_SIZE; j++)
  {
    const float4 jM0   = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
    const float  jmass = jM0.w;
    const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
    for (int k = 0; k < NI; k++)
      acc_i[k] = add_acc(acc_i[k], pos_i[k], jmass, jpos, eps2, density_i[k]);
  }
}


/*******************************/
/****** Opening criterion ******/
/*******************************/

#if 1
    
__device__ bool split_node_sph_md(
    const float4 nodeCenter,
    const float4 nodeSize,
    const float4 groupCenter,
    const float4 groupSize,
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

  return (ds2 <= cellH);
}

#endif


#define TEXTURES

template<int SHIFT, int BLOCKDIM2, int NI, bool INTCOUNT>
static __device__ 
uint2 approximate_sph(
    float4 acc_i[NI],
    const float4 _pos_i[NI],
    const float4 groupPos,
    const float4 *body_j,
    const float eps2,
    const uint2 top_cells,
    int *shmem,
    int *cellList,
    const float4 groupSize,
    float2 dens_i[NI],
    const float4 *nodeSize,
    const float4 *nodeCenter,
    const float4 *nodeMultipole,
    const float   cellH
    )
{
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);

  /* this helps to unload register pressure */
  float4 pos_i[NI];
#pragma unroll 1
  for (int i = 0; i < NI; i++)
    pos_i[i] = _pos_i[i];

  uint2 interactionCounters = {0}; /* # of approximate and exact force evaluations */

#pragma unroll 1
  for (int i = 0; i < NI; i++)
    dens_i[i] = make_float2(0,0);


  volatile int *tmpList = shmem;

  int directPtclIdx = 0;
  int directCounter = 0;

  for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.y)
      cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;

  int nCells = top_cells.y - top_cells.x;

  int cellListBlock        = 0;
  int nextLevelCellCounter = 0;

  unsigned int cellListOffset = 0;

  /* process level with n_cells */
#if 1
  while (nCells > 0)
  {
    /* extract cell index from the current level cell list */
    const int cellListIdx = cellListBlock + laneIdx;
    const bool useCell    = cellListIdx < nCells;
    const int cellIdx     = !useCell ? 0 : cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
    cellListBlock += min(WARP_SIZE, nCells - cellListBlock);

    /* read from gmem cell's info */
    const float4 cellSize = nodeSize[cellIdx];
    const float4 cellPos  = nodeCenter[cellIdx];
    //const float4 cellSize = tex1Dfetch(ztexNodeSize,   cellIdx);
    //const float4 cellPos  = tex1Dfetch(ztexNodeCenter, cellIdx);
#if 0
    const float4 cellCOM  = nodeMultipole[cellIdx+cellIdx+cellIdx];
    //const float4 cellCOM  = tex1Dfetch(texMultipole,  cellIdx+cellIdx+cellIdx);

    /* check if cell opening condition is satisfied */
    const float4 cellCOM1 = make_float4(cellCOM.x, cellCOM.y, cellCOM.z, cellPos.w);
    bool splitCell = split_node_grav_impbh2(cellCOM1, groupPos, groupSize);
#else
    bool splitCell = split_node_sph_md(cellSize, cellPos, groupPos, groupSize, cellH); 

#endif

    /* compute first child, either a cell if node or a particle if leaf */
    const int cellData = __float_as_int(cellSize.w);
    const int firstChild =  cellData & 0x0FFFFFFF;
    const int nChildren  = (cellData & 0xF0000000) >> 28;
    
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
      if (childScatter.y + nCells - cellListBlock > (CELL_LIST_MEM_PER_WARP<<SHIFT))
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
          directAcc<NI,true>(acc_i, pos_i, ptclIdx, eps2, dens_i, body_j);
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
            /* evalute cells stored in shmem */
            directAcc<NI,true>(acc_i, pos_i, tmpList[laneIdx], eps2, dens_i, body_j);
            directCounter -= WARP_SIZE;
            const int scatterIdx = directCounter + laneIdx - nParticle;
            if (scatterIdx >= 0)
              tmpList[scatterIdx] = ptclIdx;
            if (INTCOUNT)
              interactionCounters.y += WARP_SIZE*NI;
          }
          directPtclIdx = tmpList[laneIdx];

          nParticle = 0;
        }
      }
    }
#endif

    /* if the current level is processed, schedule the next level */
    if (cellListBlock >= nCells)
    {
      cellListOffset += nCells;
      nCells = nextLevelCellCounter;
      cellListBlock = nextLevelCellCounter = 0;
    }

  }  /* level completed */
#endif


  if (directCounter > 0)
  {
    directAcc<NI,false>(acc_i, pos_i, laneIdx < directCounter ? directPtclIdx : -1, eps2, dens_i, body_j);
    if (INTCOUNT)
      interactionCounters.y += directCounter * NI;
    directCounter = 0;
  }

  return interactionCounters;
}

template<int SHIFT2, int BLOCKDIM2, bool ACCUMULATE>
static __device__ 
bool treewalk(
    const int bid,
    const float eps2,
    const uint2 node_begend,
    const int    *active_groups,
    const real4  *group_body_pos,        
    const real4  *body_j,
    const float4  *groupSizeInfo,
    const float4  *groupCenterInfo,
    const float4  *nodeCenter,
    const float4  *nodeSize,
    const float4  *nodeMultipole,
    int *shmem,
    int *lmem,
    float4 *acc_out,
    int2   *interactions,
    int    *ngb_out,
    int    *active_inout,
    float  *body_h,
    float2 *body_dens_out)
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
  body_i[0] = body_addr + laneId%nb_i;
  body_i[1] = body_addr + WARP_SIZE + laneId%(nb_i - WARP_SIZE);

  float4 pos_i[2];
  float4 acc_i[2];
  float2 dens_i[2];  

  pos_i[0]    = group_body_pos[body_i[0]];
  pos_i[0].w  = 1.0f/body_h[body_i[0]];
  pos_i[0].w *= pos_i[0].w;  /* .w stores 1/h^2 to speed up computations */
  if(ni > 1){       //Only read if we actually have ni == 2
    pos_i[1]    = group_body_pos[body_i[1]];
    pos_i[1].w  = 1.0f/body_h[body_i[1]];  
    pos_i[1].w *= pos_i[1].w;  /* .w stores 1/h^2 to speed up computations */
  }

  acc_i[0]  = acc_i[1]  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  dens_i[0] = dens_i[1] = make_float2(0.0f, 0.0f);



  //Compute the cellH, which is the maximum body_h value
  //as we search for the whole group in parallel we have to 
  //be sure to include all possibly required particles
  float cellH = warpAllReduceMax(body_h[body_i[0]]);
  if(ni == 2)
  {
    cellH = max(cellH, warpAllReduceMax(body_h[body_i[1]]));
  }
  cellH *= cellH; //Squared for distance comparison without sqrt

  if(threadIdx.x == 0)
      printf("ON DEV, cellH for group %d is: %f \n", bid, cellH);


#if 0
  const bool INTCOUNT = false;
#else
  const bool INTCOUNT = true;
#endif
  uint2 counters = {0};
  {
    if (ni == 1)
      counters = approximate_sph<SHIFT2, BLOCKDIM2, 1,INTCOUNT>(
          acc_i,
          pos_i,
          group_pos,
          body_j,
          eps2, 
          node_begend,
          shmem, 
          lmem, 
          curGroupSize,
          dens_i,
          nodeSize,
          nodeCenter,
          nodeMultipole,
          cellH);
    else
      counters = approximate_sph<SHIFT2, BLOCKDIM2, 2,INTCOUNT>(
          acc_i,
          pos_i,
          group_pos,
          body_j,
          eps2, 
          node_begend,
          shmem, 
          lmem, 
          curGroupSize,
          dens_i,
          nodeSize,
          nodeCenter,
          nodeMultipole,
          cellH
          );
  }
  if(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF)
    return false;

#if 0
  /* CUDA 8RC work around */
  if(bid < 0) // bid ==0 && laneId < nb_i && && threadIdx.x == 0)
  {
	  printf("TEST\n");
  	//printf("ON DEV [%d %d : %d %d ] ACC: %f %f %f %f INT: %d %d \n",
  	//		bid, threadIdx.x, nb_i, body_i[0],
  	//		acc_i[0].x,acc_i[0].y,acc_i[0].z,acc_i[0].w,
  	//		counters.x, counters.y);
  }
#endif

  if (laneId < nb_i) 
  {
    const int addr = body_i[0];
    {
      const float hinv = 1.0f/body_h[addr];
      const float C   = 3465.0f/(512.0f*M_PI)*hinv*hinv*hinv;
      dens_i[0].x *= C;  /* scale rho */
    }
    if (ACCUMULATE)
    {
      acc_out     [addr].x += acc_i[0].x;
      acc_out     [addr].y += acc_i[0].y;
      acc_out     [addr].z += acc_i[0].z;
      acc_out     [addr].w += acc_i[0].w;


      body_dens_out[addr].x += dens_i[0].x;
      body_dens_out[addr].y += dens_i[0].y;
    }
    else
    {
      acc_out      [addr] =  acc_i[0];
      body_dens_out[addr] = dens_i[0];
    }
    //       ngb_out     [addr] = ngb_i;
    ngb_out     [addr] = addr; //JB Fixed this for demo 
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
      {
        const float hinv = 1.0f/body_h[addr];
        const float C   = 3465.0f/(512.0f*M_PI)*hinv*hinv*hinv;
        dens_i[1].x *= C;  /* scale rho */
      }
      if (ACCUMULATE)
      {
        acc_out     [addr].x += acc_i[1].x;
        acc_out     [addr].y += acc_i[1].y;
        acc_out     [addr].z += acc_i[1].z;
        acc_out     [addr].w += acc_i[1].w;
      
        body_dens_out[addr].x += dens_i[1].x;
      	body_dens_out[addr].y += dens_i[1].y;
      }
      else
      {
        acc_out      [addr] =  acc_i[1];
        body_dens_out[addr] = dens_i[1];
      
//	body_h[addr] = adjustH(body_h[addr], dens_i[1].y);
      }

      //         ngb_out     [addr] = ngb_i;
      ngb_out     [addr] = addr; //JB Fixed this for demo 
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
  }

  return true;
}

template<bool ACCUMULATE, int BLOCKDIM2>
static __device__
void approximate_gravity_main(
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
    int     *MEM_BUF,
    float   *body_h,
    float2  *body_dens) 
{
  const int blockDim2 = BLOCKDIM2;
  const int shMemSize = 1 * (1 << blockDim2);
  __shared__ int shmem_pool[shMemSize];

  const int nWarps2 = blockDim2 - WARP_SIZE2;

  const int sh_offs = (shMemSize >> nWarps2) * warpId;
  int *shmem = shmem_pool + sh_offs;
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
    const bool success = treewalk<0,blockDim2,ACCUMULATE>(
        bid, 
        eps2,
        node_begend,
        active_groups,
        group_body_pos,
        body_pos,
        groupSizeInfo,
        groupCenterInfo,
        boxCenterInfo,
        boxSizeInfo,
        multipole_data,           
        shmem,
        lmem,
        acc_out,
        interactions,
        ngb_out,
        active_inout,
        body_h,
        body_dens);

#if 0
    if (bid % 10 == 0)
      success = false;
#endif

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
          const bool success = treewalk<nWarp2,blockDim2,ACCUMULATE>(
              failedList[it], 
              eps2,
              node_begend,
              active_groups,
              group_body_pos,
              body_pos,
              groupSizeInfo,
              groupCenterInfo,
        boxCenterInfo,
        boxSizeInfo,
        multipole_data,           
              shmem,
              lmem1,
              acc_out,
              interactions,
              ngb_out,
              active_inout);
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

      int *lmem1 = &MEM_BUF[gridDim.x*(CELL_LIST_MEM_PER_WARP<<nWarps2)];
      const bool success = treewalk<8,blockDim2,ACCUMULATE>(
          bid, 
          eps2,
          node_begend,
          active_groups,
          group_body_pos,
              body_pos,
          groupSizeInfo,
          groupCenterInfo,
        boxCenterInfo,
        boxSizeInfo,
        multipole_data,           
          shmem,
          lmem1,
          acc_out,
          interactions,
          ngb_out,
          active_inout,
          body_h,
          body_dens);
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
      int    n_bodies,
      float eps2,
      uint2 node_begend,
      int    *active_groups,
      real4  *body_pos,
      __restrict__ real4  *multipole_data,
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
      int     *MEM_BUF,
      float   *body_h,
      float2  *body_dens) 
{
  approximate_gravity_main<false, NTHREAD2>(
      n_active_groups,
      n_bodies,
      eps2,
      node_begend,
      active_groups,
      body_pos,
      multipole_data,
      acc_out,
      group_body_pos,           //This can be different from body_pos
      ngb_out,
      active_inout,
      interactions,
      boxSizeInfo,
      groupSizeInfo,
      boxCenterInfo,
      groupCenterInfo,
      body_vel,
      MEM_BUF,
      body_h,
      body_dens);
}


  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_sph_density_let(
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
      int     *MEM_BUF,
      float   *body_h,
      float2  *body_dens) 
{
  approximate_gravity_main<true, NTHREAD2>(
      n_active_groups,
      n_bodies,
      eps2,
      node_begend,
      active_groups,
      body_pos,
      multipole_data,
      acc_out,
      group_body_pos,           //This can be different from body_pos
      ngb_out,
      active_inout,
      interactions,
      boxSizeInfo,
      groupSizeInfo,
      boxCenterInfo,
      groupCenterInfo,
      body_vel,
      MEM_BUF,
      body_h,
      body_dens);
}


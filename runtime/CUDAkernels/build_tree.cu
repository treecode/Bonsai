// //#include "/home/jbedorf/papers/GBPZ2010/codes/jb/build_tree/CUDA/support_kernels.cu"
#include "support_kernels.cu"
#include <stdio.h>
#include "octree.h"

#include "../profiling/bonsai_timing.h"
PROF_MODULE(build_tree);

//////////////////////////////
//////////////////////////////
//////////////////////////////
#define LEVEL_MIN 3

extern "C" __global__ void boundaryReduction(const int n_particles,
                                            real4      *positions,
                                            float3     *output_min,
                                            float3     *output_max)
{
  CUXTIMER("boundaryReduction");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  //const uint idx = bid * blockDim.x + tid;

  volatile __shared__ float3 shmem[512];
  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[256];
  sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
  sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
//   unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  real4 pos;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  //based on reduce6 example
  while (i < n_particles) {
    if (i             < n_particles)
    {
      pos = positions[i];
      r_min.x = fminf(pos.x, r_min.x);
      r_min.y = fminf(pos.y, r_min.y);
      r_min.z = fminf(pos.z, r_min.z);
      r_max.x = fmaxf(pos.x, r_max.x);
      r_max.y = fmaxf(pos.y, r_max.y);
      r_max.z = fmaxf(pos.z, r_max.z);
    }
    if (i + blockSize < n_particles)
    {
      pos = positions[i + blockSize];
      r_min.x = fminf(pos.x, r_min.x);
      r_min.y = fminf(pos.y, r_min.y);
      r_min.z = fminf(pos.z, r_min.z);
      r_max.x = fmaxf(pos.x, r_max.x);
      r_max.y = fmaxf(pos.y, r_max.y);
      r_max.z = fmaxf(pos.z, r_max.z);
    }
    i += gridSize;
  }

  sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
  sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;

  __syncthreads();
  // do reduction in shared mem  
  if(blockDim.x >= 512) if (tid < 256) {sh_MinMax(tid, tid + 256, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 256) if (tid < 128) {sh_MinMax(tid, tid + 128, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 128) if (tid < 64)  {sh_MinMax(tid, tid + 64,  &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();

  if (tid < 32) 
  {
    sh_MinMax(tid, tid + 32, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid + 16, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  8, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  4, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  2, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  1, &r_min, &r_max, sh_rmin,sh_rmax);
  }

  // write result for this block to global mem
  if (tid == 0)
  {
    //Compiler doesnt allow: volatile float3 = float3
    output_min[bid].x = sh_rmin[0].x; output_min[bid].y = sh_rmin[0].y; output_min[bid].z = sh_rmin[0].z;
    output_max[bid].x = sh_rmax[0].x; output_max[bid].y = sh_rmax[0].y; output_max[bid].z = sh_rmax[0].z;
  }

}


//Get the domain size, by taking into account the group size
extern "C" __global__ void boundaryReductionGroups(const int n_groups,
                                                   real4      *positions,
                                                   real4      *sizes,
                                                   float3     *output_min,
                                                   float3     *output_max)
{
  CUXTIMER("boundaryReductionGroups");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  //const uint idx = bid * blockDim.x + tid;

  volatile __shared__ float3 shmem[512];
  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[256];
  sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
  sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
//   unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  real4 pos;
  real4 size;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  //based on reduce6 example
  while (i < n_groups) {
    if (i             < n_groups)
    {
      pos = positions[i];
      size = sizes[i];
      r_min.x = fminf(pos.x-size.x, r_min.x);
      r_min.y = fminf(pos.y-size.y, r_min.y);
      r_min.z = fminf(pos.z-size.z, r_min.z);
      r_max.x = fmaxf(pos.x+size.x, r_max.x);
      r_max.y = fmaxf(pos.y+size.y, r_max.y);
      r_max.z = fmaxf(pos.z+size.z, r_max.z);
    }
    if (i + blockSize < n_groups)
    {
      pos = positions[i + blockSize];
      size = sizes[i + blockSize];
      r_min.x = fminf(pos.x-size.x, r_min.x);
      r_min.y = fminf(pos.y-size.y, r_min.y);
      r_min.z = fminf(pos.z-size.z, r_min.z);
      r_max.x = fmaxf(pos.x+size.x, r_max.x);
      r_max.y = fmaxf(pos.y+size.y, r_max.y);
      r_max.z = fmaxf(pos.z+size.z, r_max.z);
    }
    i += gridSize;
  }

  sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
  sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;

  __syncthreads();
  // do reduction in shared mem  
  if(blockDim.x >= 512) if (tid < 256) {sh_MinMax(tid, tid + 256, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 256) if (tid < 128) {sh_MinMax(tid, tid + 128, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 128) if (tid < 64)  {sh_MinMax(tid, tid + 64,  &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();

  if (tid < 32) 
  {
    sh_MinMax(tid, tid + 32, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid + 16, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  8, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  4, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  2, &r_min, &r_max, sh_rmin,sh_rmax);
    sh_MinMax(tid, tid +  1, &r_min, &r_max, sh_rmin,sh_rmax);
  }

  // write result for this block to global mem
  if (tid == 0)
  {
    //Compiler doesnt allow: volatile float3 = float3
    output_min[bid].x = sh_rmin[0].x; output_min[bid].y = sh_rmin[0].y; output_min[bid].z = sh_rmin[0].z;
    output_max[bid].x = sh_rmax[0].x; output_max[bid].y = sh_rmax[0].y; output_max[bid].z = sh_rmax[0].z;
  }

}

//#define EXACT_KEY

extern "C" __global__ void cl_build_key_list(uint4  *body_key,
                                            real4  *body_pos,
                                            int   n_bodies,
                                            real4  corner) {
  
  CUXTIMER("cl_build_key_list");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;
  
  if (id > n_bodies) return;

  real4 pos = body_pos[id];

  int4 crd;
  
  real domain_fac = corner.w;
  
  #ifndef EXACT_KEY
    crd.x = (int)roundf(__fdividef((pos.x - corner.x), domain_fac));
    crd.y = (int)roundf(__fdividef((pos.y - corner.y) , domain_fac));
    crd.z = (int)roundf(__fdividef((pos.z - corner.z) , domain_fac));
  #else            
    crd.x = (int)((pos.x - corner.x) / domain_fac);
    crd.y = (int)((pos.y - corner.y) / domain_fac);
    crd.z = (int)((pos.z - corner.z) / domain_fac);
  #endif

   uint4 key = get_key(crd);


//   if (id == n_bodies) key = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
  if (id == n_bodies) key = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);

  key.w = id;

  body_key[id] = key;

}

    
  

extern "C" __global__ void cl_build_valid_list(int n_bodies,
                                               int level,
                                               uint4  *body_key,
                                               uint *valid_list,
                                               const uint *workToDo) {
  if (0 == *workToDo) return;
//                                                uint2 *test_key_data) {
  CUXTIMER("cl_build_valid_list");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint id  = bid * blockDim.x + tid;
  const uint4 key_F = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
  
  if (id >= n_bodies) return;   // >=   since the last particle is extra boudnary particle
  
  uint4 mask = get_mask(level);
  mask.x = mask.x | ((uint)1 << 30) | ((uint)1 << 31);

  uint4 key_m;
  uint4 key_c    = body_key[id];
  uint4 key_p;

  if (id == 0)
  {
    key_m = key_F;
  }
  else
  {
    key_m = body_key[id-1];
  }

  if((id+1) <  n_bodies) //The last particle gets a different key to compare with
  {
    key_p = body_key[id+1];
  }
  else
    key_p = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);


  int valid0 = 0;
  int valid1 = 0;

  if (cmp_uint4(key_c, key_F) != 0) {
    key_c.x = key_c.x & mask.x;
    key_c.y = key_c.y & mask.y;
    key_c.z = key_c.z & mask.z;

    key_p.x = key_p.x & mask.x;
    key_p.y = key_p.y & mask.y;
    key_p.z = key_p.z & mask.z;

    key_m.x = key_m.x & mask.x;
    key_m.y = key_m.y & mask.y;
    key_m.z = key_m.z & mask.z;

    valid0 = abs(cmp_uint4(key_c, key_m));
    valid1 = abs(cmp_uint4(key_c, key_p));
  }

   valid_list[id*2]   = id | ((valid0) << 31);
   valid_list[id*2+1] = id | ((valid1) << 31);

}


//////////////////////////////
//////////////////////////////
//////////////////////////////
__device__ uint retirementCountBuildNodes = 0;

extern "C" __global__ void cl_build_nodes(uint level,
                             uint  *compact_list_len,
                             uint  *level_offset,
                             uint  *last_level,
                             uint2 *level_list,
                             uint  *compact_list,
                             uint4 *bodies_key,
                             uint4 *node_key,
                             uint  *n_children,
                             uint2 *node_bodies){

  CUXTIMER("cl_build_nodes");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;

  uint n = (*compact_list_len)/2;
  uint offset = *level_offset;

  for (; id < n; id += gridDim.x * gridDim.y * blockDim.x)
  {
    uint  bi   = compact_list[id*2];
    uint  bj   = compact_list[id*2+1] + 1;
  
    uint4 key  = bodies_key[bi];
    uint4 mask = get_mask(level);
    key = make_uint4(key.x & mask.x, key.y & mask.y, key.z & mask.z, 0); 

    node_bodies[offset+id] = make_uint2(bi | (level << BITLEVELS), bj);
    node_key   [offset+id] = key;
    n_children [offset+id] = 0;
  
    if ((int)level > (int)(LEVEL_MIN - 1)) 
      if (bj - bi <= NLEAF)                            //Leaf can only have NLEAF particles, if its more there will be a split
        for (int i = bi; i < bj; i++)
          bodies_key[i] = make_uint4(0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF); //sets the key to FF to indicate the body is used
  }

  //
  // PHASE 2: Last block updates level list and offset
  //

  int numBlocks = gridDim.x * gridDim.y;
  if (numBlocks > 1)
  {
    __shared__ bool amLast;

    // Thread 0 takes a ticket
    if( tid==0 )
    {
      unsigned int ticket = atomicInc(&retirementCountBuildNodes, numBlocks);
      // If the ticket ID is equal to the number of blocks, we are the last block!
      amLast = (ticket == numBlocks-1);
    }
    __syncthreads();

    // The last block sums the results of all other blocks
    if( amLast && tid == 0)
    {           
      level_list[level] = (n > 0) ? make_uint2(offset, offset + n) : make_uint2(0, 0);
      *level_offset = offset + n;

      if ((level > 0) && (n <= 0) && (level_list[level - 1].x > 0))
        *last_level = level;

      // reset retirement count so that next run succeeds
      retirementCountBuildNodes = 0; 
    }
  }
}

void build_tree_node_levels(octree &tree, 
                            my_dev::dev_mem<uint>  &validList,
                            my_dev::dev_mem<uint>  &compactList,
                            my_dev::dev_mem<uint>  &levelOffset,
                            my_dev::dev_mem<uint>  &maxLevel)
{
   // set devMemCountsx to 1 because it is used to early out when it hits zero
  tree.devMemCountsx[0] = 1;
  tree.devMemCountsx.h2d(1);

  dim3 grid, block;

  //int nodeSum = 0;
  for (uint level = 0; level < MAXLEVELS; level++) {
    // mark bodies to be combined into nodes
    //Calculate dynamic
    int ng = (tree.localTree.n) / 128 + 1;
    grid.x = (int)sqrt((double)ng);
    grid.y = (ng -1)/grid.x +  1; 
    grid.z = 1;
    block.x = 128; block.y = block.z = 1;

    cl_build_valid_list<<<grid, block>>>(tree.localTree.n, 
                                         level, 
                                         tree.localTree.bodies_key.raw_p(),
                                         validList.raw_p(), 
                                         tree.devMemCountsx.raw_p());
      
    //gpuCompact to get number of created nodes    
    tree.gpuCompact(*tree.getDevContext(), validList, compactList, tree.localTree.n*2, 0);
                   
    // assemble nodes   
    grid.x = (120*32)/128; grid.y = 4; grid.z = 1;
    block.x = 128; block.y = 1; block.z = 1;

    cl_build_nodes<<<grid, block>>>(level, 
                                    tree.devMemCountsx.raw_p(), 
                                    levelOffset.raw_p(), 
                                    maxLevel.raw_p(),
                                    tree.localTree.level_list.raw_p(), 
                                    compactList.raw_p(),
                                    tree.localTree.bodies_key.raw_p(),
                                    tree.localTree.node_key.raw_p(),
                                    tree.localTree.n_children.raw_p(),
                                    tree.localTree.node_bodies.raw_p());
  } //end for lvl

  // reset counts to 1 so next compact proceeds...
  tree.devMemCountsx[0] = 1;
  tree.devMemCountsx.h2d(1); 
}

//////////////////////////////
//////////////////////////////
//////////////////////////////


extern "C" __global__ void cl_link_tree(int n_nodes,
                            uint *n_children,
                            uint2 *node_bodies,
                            real4 *bodies_pos,
                            real4 corner,
                            uint2 *level_list,           //TODO could make this constant if it proves usefull
                            uint* valid_list,
                            uint4 *node_keys,
                            uint4 *bodies_key) {

  CUXTIMER("cl_link_tree");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  uint id  = bid * blockDim.x + tid;
  
  if (id >= n_nodes) return;

  uint2 bij  = node_bodies[id];
  uint level = (bij.x &  LEVELMASK) >> BITLEVELS;
  uint bi    =  bij.x & ILEVELMASK;
  uint bj    =  bij.y;

  real4 pos  = bodies_pos[bi];
  int4 crd;
  real domain_fac = corner.w;

  #ifndef EXACT_KEY
    crd.x = (int)roundf(__fdividef((pos.x - corner.x), domain_fac));
    crd.y = (int)roundf(__fdividef((pos.y - corner.y) , domain_fac));
    crd.z = (int)roundf(__fdividef((pos.z - corner.z) , domain_fac));
  #else            
    crd.x = (int)((pos.x - corner.x) / domain_fac);
    crd.y = (int)((pos.y - corner.y) / domain_fac);
    crd.z = (int)((pos.z - corner.z) / domain_fac);
  #endif


  uint4 key = get_key(crd);


  /********* accumulate children *****/
  
  uint4 mask = get_mask(level - 1);
  key = make_uint4(key.x & mask.x, key.y & mask.y,  key.z & mask.z, 0); 

  uint2 cij;

  
  if(id > 0)
    cij = level_list[level-1];

  int ci;
  //Jeroen, modified this since we dont use textures in find_key,
  //the function will fail because out of bound memory access when id==0
  if(id > 0)
    ci = find_key(key, cij, node_keys);
  else
    ci = 0;

  //ci now points to the node that is the parent, was used in previous group method
  //parent_id_list[id] = ci;

  mask = get_imask(mask);
  key = make_uint4(key.x | mask.x, key.y | mask.y, key.z | mask.z, 0);
  if (id > 0)   
    atomicAdd(&n_children[ci], (1 << 28));

  key = get_key(crd);
  mask = get_mask(level);
  key = make_uint4(key.x & mask.x, key.y & mask.y, key.z & mask.z, 0); 

  /********* store the 1st child *****/

  cij = level_list[level+1];
  int cj = -1;

  cj = find_key(key, cij, node_keys);

  atomicOr(&n_children[id], cj); //Atomic since multiple threads can work on this

  uint valid =  id | (uint)(0 << 31); 

  
  if ((int)level > (int)(LEVEL_MIN - 1)) 
    if ((bj - bi) <= NLEAF)    
      valid = id | (uint)(1 << 31);   //Distinguish leaves and nodes

 valid_list[id] = valid; //If valid its a leaf otherwise a node
}

//Determines which level of node starts at which offset
extern "C" __global__ void build_level_list(const int n_nodes,
                                            const int n_leafs,
                                            uint *leafsIdxs,
                                            uint2 *node_bodies,                                      
                                            uint* valid_list)
{
  CUXTIMER("build_level_list");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint id  = bid * blockDim.x + tid;
  
  if (id >= n_nodes-n_leafs) return;

  const int nodeID = leafsIdxs[id+n_leafs];   //Get the idx into the node_bodies array

  int level_c, level_m, level_p;


  uint2 bij   = node_bodies[leafsIdxs[id+n_leafs]];    //current non-leaf
  level_c     = (bij.x &  LEVELMASK) >> BITLEVELS;

  if((id+1) < (n_nodes-n_leafs))        //The last node gets a default lvl
  {
    bij         = node_bodies[leafsIdxs[id+1+n_leafs]]; //next non-leaf
    level_p     = (bij.x &  LEVELMASK) >> BITLEVELS;
  }
  else
    level_p     = MAXLEVELS+5;  //Last is always an end

  //Compare level with the node before and node after
  if(nodeID == 0)
  {
    level_m = -1;    
  }
  else
  {
    bij         = node_bodies[ leafsIdxs[id-1+n_leafs]]; //Get info of previous non-leaf node
    level_m     =  (bij.x &  LEVELMASK) >> BITLEVELS;   
  }

  int valid0 = 0;
  int valid1 = 0;

  valid0 = (level_c != level_m) << 31 | (id+n_leafs);
  valid1 = (level_c != level_p) << 31 | (id+n_leafs);

  valid_list[id*2]   = valid0;
  valid_list[id*2+1] = valid1;

} //end build_level_list


//Finds nodes/leafs that will become groups
//After executions valid_list contains the 
//valid nodes/leafs that form groups
extern "C" __global__ void build_group_list2(int    n_particles,
                                             uint  *validList,
                                             real4  *bodies_pos,
                                             const float DIST,
                                             int   *node_level_list,
                                             int   treeDepth)
{
  CUXTIMER("build_group_list2");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint idx = bid * blockDim.x + tid;

  __shared__ int shmem[128];

  //Compact the node_level_list
  if(bid == 0)
  {
    if(threadIdx.x < (MAXLEVELS*2))
    {
      shmem[threadIdx.x] = node_level_list[threadIdx.x];
    }

    __syncthreads(); //Can most likely do without since its one warp

    //Only selection writes
    if(threadIdx.x < MAXLEVELS)
    {
      node_level_list[threadIdx.x]  = shmem[threadIdx.x*2];
      if(threadIdx.x == treeDepth-1)
          node_level_list[threadIdx.x] = shmem[threadIdx.x*2-1]+1;
    }
  }//if bid == 0
  //end compact node level list

  //Note that we do not include the final particle
  //Since there is no reason to check it
  if (idx >= n_particles) return;

  //Get the current 
  float4 curPos, nexPos, prevPos;

  curPos  =  bodies_pos[idx];

  //Have to check the first and last to prevent out of bound access
  if(idx+1 == n_particles)
    nexPos  =  curPos;
  else
    nexPos = bodies_pos[idx+1];

  if(idx == 0)
    prevPos = curPos;
  else
    prevPos =  bodies_pos[idx-1];

  //Compute geometrical distance
  float dsPlus = ((curPos.x-nexPos.x)*(curPos.x-nexPos.x)) + 
                 ((curPos.y-nexPos.y)*(curPos.y-nexPos.y)) + 
                 ((curPos.z-nexPos.z)*(curPos.z-nexPos.z));

  float dsMin = ((curPos.x-prevPos.x)*(curPos.x-prevPos.x)) + 
                ((curPos.y-prevPos.y)*(curPos.y-prevPos.y)) + 
                ((curPos.z-prevPos.z)*(curPos.z-prevPos.z));

  //Multiples of the preferred group size are _always_ valid
  int validStart = ((idx     % NCRIT) == 0);
  int validEnd   = (((idx+1) % NCRIT) == 0);

  //The extra possible split(s) if the distance between two particles is too large
  if(dsPlus > DIST) validEnd     = 1;
  if(dsMin  > DIST) validStart   = 1;
  
  //Last particle is always the end, n_particles dont have to be a multiple of NCRIT
  //so this is required
  if(idx+1 == n_particles) validEnd = 1;

  //Set valid
  validList[2*idx + 0] = (idx)   | (uint)(validStart << 31);
  validList[2*idx + 1] = (idx+1) | (uint)(validEnd   << 31);    
}

 
//Store per particle the group id it belongs to
//and the start and end particle number of the groups  
extern "C" __global__ void store_group_list(int    n_particles,
                                            int n_groups,
                                            uint  *validList,
                                            uint  *body2group_list,
                                            uint2 *group_list)
{
  CUXTIMER("store_group_list");
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
//   uint idx = bid * blockDim.x + tid;
  
  if(bid >= n_groups) return;

  int start = validList[2*bid];
  int end   = validList[2*bid+1];

  if((start + tid) < end)
  {
    body2group_list[start + tid] = bid;
  }

  if(tid == 0)
  {
     group_list[bid] = make_uint2(start,end);
  }
}

//////////// Functions specific for dust //////////////////

extern "C" __global__ void define_dust_groups(int    n_particles,
					      real4  *dust_pos,
                                              uint  *validList)
{
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
  uint idx = bid * blockDim.x + tid;


  //Note that we do not include the final particle
  //Since there is no reason to check it
  if (idx >= n_particles) return;

 
  //Multiples of the preferred group size are _always_ valid
  int validStart = ((idx     % NCRIT) == 0);
  int validEnd   = (((idx+1) % NCRIT) == 0);


  //Get the current 
  float4 curPos, nexPos, prevPos;

  curPos  =  dust_pos[idx];

  //Have to check the first and last to prevent out of bound access
  if(idx+1 == n_particles)
    nexPos  =  curPos;
  else
    nexPos = dust_pos[idx+1];

  if(idx == 0)
    prevPos = curPos;
  else
    prevPos =  dust_pos[idx-1];

  //Compute geometrical distance
  float dsPlus = ((curPos.x-nexPos.x)*(curPos.x-nexPos.x)) + 
                 ((curPos.y-nexPos.y)*(curPos.y-nexPos.y)) + 
                 ((curPos.z-nexPos.z)*(curPos.z-nexPos.z));

  float dsMin = ((curPos.x-prevPos.x)*(curPos.x-prevPos.x)) + 
                ((curPos.y-prevPos.y)*(curPos.y-prevPos.y)) + 
                ((curPos.z-prevPos.z)*(curPos.z-prevPos.z));


  float DIST = 100;
  //The extra possible split(s) if the distance between two particles is too large
  if(dsPlus > DIST) validEnd     = 1;
  if(dsMin  > DIST) validStart   = 1;


  //Last particle is always the end, n_particles dont have to be a multiple of NCRIT
  //so this is required
  if(idx+1 == n_particles) validEnd = 1;

  //Set valid
  if(validStart)
    validList[2*idx + 0] = (idx)   | (uint)(validStart << 31);
  if(validEnd)
    validList[2*idx + 1] = (idx) | (uint)(validEnd   << 31);    
}

//JB: This one is slightly different from the store_group_list
//since  in my infinite wisdom I decided to make the comparisons
//slightly different when making the new define_dust_groups
extern "C" __global__ void store_dust_groups(int    n_groups,
                                            uint  *validList,
                                            uint  *body2group_list,
                                            uint2 *group_list,
                                            uint  *activeDustGroups)
{
  uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  uint tid = threadIdx.x;
//   uint idx = bid * blockDim.x + tid;
  
  if(bid >= n_groups) return;

  int start = validList[2*bid];
  int end   = validList[2*bid+1];

  if((start + tid) <= end)
  {
     body2group_list[start + tid] = bid;
  }

  if(tid == 0)
  {
     group_list[bid] = (uint2) make_uint2(start,end+1);
     activeDustGroups[bid] = bid;
  }
}

//This function stores the predicted position and velocity
//in the original array. This is used since it reduces
//memory storage and memory reorders after sorting 
//It is slightly less accurate and therefore not used 
//for the real bodies. In the correct function we compute back
extern "C" __global__ void predict_dust_particles(const int n_bodies,
                                                  float tc,
                                                  float tp,
                                                  real4 *pos,
                                                  real4 *vel,
                                                  real4 *acc,
                                                  uint  *body2grouplist,
                                                  uint  *valid_list){                                          
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint idx = bid * blockDim.x + tid;

  if (idx >= n_bodies) return;

  float4 p = pos [idx];
  float4 v = vel [idx];
  float4 a = acc [idx];

  float dt_cb  = tc - tp;

  p.x += v.x*dt_cb + a.x*dt_cb*dt_cb*0.5f;
  p.y += v.y*dt_cb + a.y*dt_cb*dt_cb*0.5f;
  p.z += v.z*dt_cb + a.z*dt_cb*dt_cb*0.5f;
  
  v.x += a.x*dt_cb;
  v.y += a.y*dt_cb;
  v.z += a.z*dt_cb;

  pos[idx] = p;
  vel[idx] = v;

  //This is needed to retain compatability with the original 
  //approximate gravity function
  int grpID = body2grouplist[idx];
  valid_list[grpID] = grpID; 
}



extern "C" __global__ void correct_dust_particles(const int n_bodies,
                                                  float dt_cb,
                                                  uint   *active_list,
                                                  real4 *vel,
                                                  real4 *acc0,
                                                  real4 *acc1) {
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid =  threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;
  if (idx >= n_bodies) return;

  //Check if particle is set to active during approx grav
  #ifdef DO_BLOCK_TIMESTEP
    if (active_list[idx] != 1) return;
  #endif

  float4 a0 = acc0[idx];
  float4 a1 = acc1[idx];
  float4  v = vel[idx];

  //Correct the velocity
  dt_cb *= 0.5f;
  v.x += (a1.x - a0.x)*dt_cb;
  v.y += (a1.y - a0.y)*dt_cb;
  v.z += (a1.z - a0.z)*dt_cb;

  //Store the corrected velocity, accelaration and the new time step info
  vel     [idx] = v;
  acc0    [idx] = a1;
}


#ifndef DEVFUNCTIONDEFINITIONS_H
#define DEVFUNCTIONDEFINITIONS_H


/*******   cudaLaunch workaround *******/

//CUDA 5 forces us to use function pointers, define functions here

//Redefine here since this is included in octree.h before structure is defined
typedef struct setupParams2
{
  int jobs;                     //Minimal number of jobs for each 'processor'
  int blocksWithExtraJobs;      //Some ' processors'  do one extra job all with bid < bWEJ
  int extraElements;            //The elements that didn't fit completely
  int extraOffset;              //Start of the extra elements
}setupParams2;


//Device functions

//Scan, compact, split kernels
extern "C" void  (split_move)( uint2 *valid, uint *output, uint *counts,  const int N, setupParams2 sParam);
extern "C" void  (compact_move)( uint2 *values, uint *output,  uint *counts,  const int N,setupParams2 sParam,const uint *workToDo);
extern "C" void  (compact_count)(volatile uint2 *values,uint *counts, const int N, setupParams2 sParam,const uint *workToDo);
extern "C" void  (exclusive_scan_block)(int *ptr, const int N, int *count);


extern "C" void  (gpu_boundaryReduction)(const int n_particles, real4 *positions, float3 *output_min, float3 *output_max);
extern "C" void  (gpu_boundaryReductionGroups)(const int n_groups, real4      *positions, real4      *sizes, float3     *output_min, float3     *output_max);

//Tree-build kernels
extern "C" void  cl_build_key_list(uint4  *body_key, real4  *body_pos, int   n_bodies, real4  corner);
extern "C" void  cl_build_valid_list(int n_bodies, int level, uint4  *body_key, uint *valid_list, const uint *workToDo);
extern "C" void  cl_build_nodes(uint level, uint  *compact_list_len, uint  *level_offset, uint  *last_level, uint2 *level_list, uint  *compact_list, uint4 *bodies_key, uint4 *node_key, uint  *n_children, uint2 *node_bodies);
extern "C" void  (cl_link_tree)(int n_nodes, uint *n_children, uint2 *node_bodies, real4 *bodies_pos, real4 corner, uint2 *level_list, uint* valid_list, uint4 *node_keys, uint4 *bodies_key,uint  levelMin);
extern "C" void  (store_group_list)(int    n_particles, int n_groups, uint  *validList, uint  *body2group_list, uint2 *group_list);
extern "C" void  (build_group_list2)(const int n_particles, uint *validList, const uint2 startLevelBeginEnd, uint2 *node_bodies, int *node_level_list, int treeDepth);
extern "C" void  (gpu_build_level_list)(const int n_nodes, const int n_leafs, uint *leafsIdxs, uint2 *node_bodies,  uint* valid_list);


//Tree-properties kernels
extern "C" void  (compute_leaf)(const int n_leafs, uint *leafsIdxs, uint2 *node_bodies, real4 *body_pos, double4 *multipole, real4 *nodeLowerBounds, real4 *nodeUpperBounds, real4  *body_vel, uint *body_id, real *body_h, const real h_min);
extern "C" void  (compute_scaling)(const int node_count, double4 *multipole, real4 *nodeLowerBounds, real4 *nodeUpperBounds, uint  *n_children, real4 *multipoleF, float theta, real4 *boxSizeInfo, real4 *boxCenterInfo, uint2 *node_bodies);
extern "C" void  (compute_non_leaf)(const int curLevel, uint  *leafsIdxs, uint  *node_level_list, uint  *n_children, double4 *multipole, real4 *nodeLowerBounds, real4 *nodeUpperBounds);
extern "C" void  (gpu_setPHGroupData)(const int n_groups, const int n_particles,   real4 *bodies_pos, int2  *group_list,real4 *groupCenterInfo, real4 *groupSizeInfo);

//Time integration kernels
extern "C" void  (get_Tnext)(const int n_bodies, float2 *time, float *tnext);
extern "C" void  (predict_particles)(const int n_bodies, float tc, float tp, real4 *pos, real4 *vel, real4 *acc, float2 *time, real4 *pPos, real4 *pVel);
extern "C" void  (get_nactive)(const int n_bodies, uint *valid, uint *tnact);
extern "C" void  (correct_particles)(const int n_bodies, float tc, float2 *time,                                                                              uint   *active_list, real4 *vel, real4 *acc0, real4 *acc1, real4 *pos, real4 *pPos, real4 *pVel, uint  *unsorted, real4 *acc0_new, float2 *time_new);
extern "C" void  (compute_dt)(const int n_bodies, float    tc, float    eta, int      dt_limit, float    eps2, float2   *time, real4    *vel, int      *ngb, real4    *bodies_pos, real4    *bodies_acc, uint     *active_list, float    timeStep);
extern "C" void  (setActiveGroups)(const int n_bodies, float tc, float2 *time,uint  *body2grouplist, uint  *valid_list);
extern "C" void  (compute_energy_double)(const int n_bodies, real4 *pos, real4 *vel, real4 *acc, double2 *energy);
extern "C" void  (dev_approximate_gravity)( const int n_active_groups, int    n_bodies, float eps2, uint2 node_begend, int    *active_groups, real4  *body_pos, real4  *multipole_data, float4 *acc_out, real4  *group_body_pos,         int    *ngb_out, int    *active_inout, int2   *interactions, float4  *boxSizeInfo, float4  *groupSizeInfo, float4  *boxCenterInfo, float4  *groupCenterInfo, real4   *body_vel, int     *MEM_BUF);
extern "C" void  (dev_approximate_gravity_let)( const int n_active_groups, int    n_bodies, float eps2, uint2 node_begend, int    *active_groups, real4  *body_pos, real4  *multipole_data, float4 *acc_out, real4  *group_body_pos,         int    *ngb_out, int    *active_inout, int2   *interactions, float4  *boxSizeInfo, float4  *groupSizeInfo, float4  *boxCenterInfo, float4  *groupCenterInfo, real4   *body_vel, int     *MEM_BUF);

//Parallel.cu kernels
extern "C" void  (gpu_internalMoveSFC2) (int       n_extract, int       n_bodies, uint4  lowBoundary, uint4  highBoundary, int2       *extractList, int       *indexList, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, unsigned long long        *body_id, uint4     *body_key);
extern "C" void  (gpu_extractOutOfDomainParticlesAdvancedSFC2)(int offset, int n_extract, uint2 *extractList, real4 *Ppos, real4 *Pvel, real4 *pos, real4 *vel, real4 *acc0, real4 *acc1, float2 *time, unsigned long long *body_id, uint4 *body_key, bodyStruct *destination);
extern "C" void  (gpu_insertNewParticlesSFC)(int       n_extract, int       n_insert, int       n_oldbodies, int       offset, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, unsigned long long        *body_id, uint4     *body_key, bodyStruct *source);
extern "C" void  (gpu_domainCheckSFCAndAssign)(int    n_bodies, int    nProcs, uint4  lowBoundary, uint4  highBoundary, uint4  *boundaryList,  uint4  *body_key, uint    *validList,  uint   *idList, int procId);

//Other
extern "C" void  (dev_direct_gravity)(float4 *accel, float4 *i_positions, float4 *j_positions, int numBodies_i, int numBodies_j, float eps2);


#endif

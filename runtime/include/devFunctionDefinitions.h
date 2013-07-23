#ifndef DEVFUNCTIONDEFINITIONS_H
#define DEVFUNCTIONDEFINITIONS_H

//#ifndef __CUDACC__

/*******   cudaLaunch workaround *******/

//CUDA 5 forces us to use function pointers, define functions here
//extern "C" void  (build_group_list2)(int    n_particles, uint  *validList, real4  *bodies_pos, const float DIST, int   *node_level_list, int   treeDepth);
//extern "C" void  (cl_link_tree)(int n_nodes, uint *n_children, uint2 *node_bodies, real4 *bodies_pos, real4 corner, uint2 *level_list,  uint4 *node_keys, uint4 *bodies_key);
//Redefine here since this is included in octree.h before structure is defined
typedef struct setupParams2
{
  int jobs;                     //Minimal number of jobs for each 'processor'
  int blocksWithExtraJobs;      //Some ' processors'  do one extra job all with bid < bWEJ
  int extraElements;            //The elements that didn't fit completely
  int extraOffset;              //Start of the extra elements
}setupParams2;


//Device functions
extern "C" void  (gpu_boundaryReduction)(const int n_particles, real4 *positions, float3 *output_min, float3 *output_max);
extern "C" void  (gpu_boundaryReductionGroups)(const int n_groups, real4      *positions, real4      *sizes, float3     *output_min, float3     *output_max);
extern "C" void  cl_build_key_list(uint4  *body_key, real4  *body_pos, int   n_bodies, real4  corner);
extern "C" void  cl_build_valid_list(int n_bodies, int level, uint4  *body_key, uint *valid_list, const uint *workToDo);
extern "C" void  cl_build_nodes(uint level, uint  *compact_list_len, uint  *level_offset, uint  *last_level, uint2 *level_list, uint  *compact_list, uint4 *bodies_key, uint4 *node_key, uint  *n_children, uint2 *node_bodies);
extern "C" void  dataReorderCombined(const int N, uint4 *keyAndPerm, real4 *source1, real4* destination1, real4 *source2, real4* destination2,real4 *source3, real4* destination3);
extern "C" void  dataReorderCombined4(const int N, uint4 *keyAndPerm, real4 *source1,  real4* destination1, int *source2, int* destination2, int *oldOrder);
extern "C" void  dataReorderF2(const int N, uint4 *keyAndPerm, float2 *source1, float2 *destination1, int *source2, int *destination2);

extern "C" void  (sort_count)(volatile uint2 *valid, int *counts, const int N, setupParams sParam, int bitIdx);
extern "C" void  (sort_move_stage_key_value)(uint2 *valid, int *output, uint2 *srcValues, uint *valuesOut, int *counts, const int N, setupParams sParam, int bitIdx);      
extern "C" void  (extractInt_kernel)(uint4 *keys,  uint *simpleKeys, uint *sequence, const int N, int keyIdx);
extern "C" void  (reOrderKeysValues_kernel)(uint4 *keysSrc, uint4 *keysDest, uint *permutation, const int N);

extern "C" void  (gpu_extractKeyAndPerm)(uint4 *newKeys, uint4 *keys, uint *permutation, const int N);
extern "C" void  (gpu_convertKey64to96)(uint4 *keys,  uint4 *newKeys, const int N);
extern "C" void  (dataReorderCombined4)(const int N, uint4 *keyAndPerm, real4 *source1,  real4* destination1, int *source2,    int*   destination2, int *oldOrder);
extern "C" void  (gpu_dataReorderF2)(const int N, uint4 *keyAndPerm, float2 *source1, float2 *destination1, int    *source2, int *destination2);
extern "C" void  (gpu_dataReorderI1)(const int n_particles, int *source, int *destination, uint  *permutation);
extern "C" void  (gpu_dataReorderCombined)(const int N, uint4 *keyAndPerm, real4 *source1, real4* destination1, real4 *source2, real4* destination2, real4 *source3, real4* destination3);

    

extern "C" void  (split_move)( uint2 *valid, uint *output, uint *counts,  const int N, setupParams2 sParam);
extern "C" void  (compact_move)( uint2 *values, uint *output,  uint *counts,  const int N,setupParams2 sParam,const uint *workToDo);
extern "C" void  (compact_count)(volatile uint2 *values,uint *counts, const int N, setupParams2 sParam,const uint *workToDo);
extern "C" void  (exclusive_scan_block)(int *ptr, const int N, int *count);
extern "C" void  (correct_dust_particles)(const int n_bodies, float dt_cb, uint   *active_list, real4 *vel, real4 *acc0, real4 *acc1);
extern "C" void  (predict_dust_particles)(const int n_bodies, float tc, float tp, real4 *pos, real4 *vel, real4 *acc, uint  *body2grouplist, uint  *valid_list);
extern "C" void  (store_dust_groups)(int n_groups, uint  *validList, uint  *body2group_list, uint2 *group_list, uint  *activeDustGroups);
extern "C" void  (define_dust_groups)(int n_particles, real4  *dust_pos, uint  *validList);
extern "C" void  (store_group_list)(int    n_particles, int n_groups, uint  *validList, uint  *body2group_list, uint2 *group_list);
extern "C" void  (build_group_list2)(const int n_particles, uint *validList, const uint2 startLevelBeginEnd, uint2 *node_bodies, int *node_level_list, int treeDepth);
extern "C" void  (gpu_build_level_list)(const int n_nodes, const int n_leafs, uint *leafsIdxs, uint2 *node_bodies,  uint* valid_list);
extern "C" void  (cl_link_tree)(int n_nodes, uint *n_children, uint2 *node_bodies, real4 *bodies_pos, real4 corner, uint2 *level_list, uint* valid_list, uint4 *node_keys, uint4 *bodies_key,uint  levelMin);


extern "C" void  (compute_leaf)(const int n_leafs, uint *leafsIdxs, uint2 *node_bodies, real4 *body_pos, double4 *multipole, real4 *nodeLowerBounds, real4 *nodeUpperBounds, real4  *body_vel, uint *body_id);
extern "C" void  (gpu_setPHGroupData)(const int n_groups, const int n_particles,   real4 *bodies_pos, int2  *group_list,real4 *groupCenterInfo, real4 *groupSizeInfo);
extern "C" void  (compute_scaling)(const int node_count, double4 *multipole, real4 *nodeLowerBounds, real4 *nodeUpperBounds, uint  *n_children, real4 *multipoleF, float theta, real4 *boxSizeInfo, real4 *boxCenterInfo, uint2 *node_bodies);
extern "C" void  (compute_non_leaf)(const int curLevel, uint  *leafsIdxs, uint  *node_level_list, uint  *n_children, double4 *multipole, real4 *nodeLowerBounds, real4 *nodeUpperBounds);
extern "C" void  (compute_energy_double)(const int n_bodies, real4 *pos, real4 *vel, real4 *acc, double2 *energy);
extern "C" void  (correct_particles)(const int n_bodies, float tc, float2 *time,                                                                              uint   *active_list, real4 *vel, real4 *acc0, real4 *acc1, real4 *pos, real4 *pPos, real4 *pVel, uint  *unsorted, real4 *acc0_new, float2 *time_new, int *pIDS, real4 *specialParticles);
extern "C" void  (predict_particles)(const int n_bodies, float tc, float tp, real4 *pos, real4 *vel, real4 *acc, float2 *time, real4 *pPos, real4 *pVel);
extern "C" void  (get_nactive)(const int n_bodies, uint *valid, uint *tnact);
extern "C" void  (get_Tnext)(const int n_bodies, float2 *time, float *tnext);
extern "C" void  (setActiveGroups)(const int n_bodies, float tc, float2 *time,uint  *body2grouplist, uint  *valid_list);
extern "C" void  (dev_approximate_gravity)( const int n_active_groups, int    n_bodies, float eps2, uint2 node_begend, int    *active_groups, real4  *body_pos, real4  *multipole_data, float4 *acc_out, real4  *group_body_pos,         int    *ngb_out, int    *active_inout, int2   *interactions, float4  *boxSizeInfo, float4  *groupSizeInfo, float4  *boxCenterInfo, float4  *groupCenterInfo, real4   *body_vel, int     *MEM_BUF);
extern "C" void  (dev_approximate_gravity_let)( const int n_active_groups, int    n_bodies, float eps2, uint2 node_begend, int    *active_groups, real4  *body_pos, real4  *multipole_data, float4 *acc_out, real4  *group_body_pos,         int    *ngb_out, int    *active_inout, int2   *interactions, float4  *boxSizeInfo, float4  *groupSizeInfo, float4  *boxCenterInfo, float4  *groupCenterInfo, real4   *body_vel, int     *MEM_BUF);
extern "C" void  (dev_determineLET)(const int n_active_groups, int    n_bodies, uint2 node_begend, int    *active_groups, real4  *multipole_data, int    *active_inout, float4  *boxSizeInfo, float4  *groupSizeInfo, float4  *boxCenterInfo, float4  *groupCenterInfo, int     *MEM_BUF, uint2   *markedNodes);

extern "C" void  (gpu_segmentedCoarseGroupBoundary)( const int n_coarse_groups, const int n_groups, uint     *atomicValues,uint     *coarseGroupList,float4   *grpSizes, float4   *grpPositions, float4   *output_min, float4   *output_max);

extern "C" void  (gpu_setPHGroupDataGetKey)(const int n_groups, const int n_particles, real4 *bodies_pos, int2  *group_list, real4 *groupCenterInfo, real4 *groupSizeInfo, uint4  *body_key, float4 corner);
extern "C" void  (gpu_setPHGroupDataGetKey2)(const int n_groups, real4 *bodies_pos, int2  *group_list, uint4  *body_key, float4 corner);


extern "C" void  (compute_dt)(const int n_bodies, float    tc, float    eta, int      dt_limit, float    eps2, float2   *time, real4    *vel, int      *ngb, real4    *bodies_pos, real4    *bodies_acc, uint     *active_list, float    timeStep);

extern "C" void  (dev_direct_gravity)(float4 *accel, float4 *i_positions, float4 *j_positions, int numBodies_i, int numBodies_j, float eps2);


extern "C" void  (doDomainCheck)(int    n_bodies, double4  xlow, double4  xhigh, real4  *body_pos, int    *validList);
extern "C" void  (gpu_domainCheckSFC)(int    n_bodies, uint4  lowBoundary, uint4  highBoundary, uint4  *body_key, int    *validList);                                           
extern "C" void  (gpu_extractSampleParticles)(int    n_bodies, int    sample_freq, real4  *body_pos, real4  *samplePosition);
extern "C" void  (extractOutOfDomainParticlesR4)(int n_extract, int *extractList, real4 *source, real4 *destination);
extern "C" void  (extractOutOfDomainParticlesAdvanced)(int n_extract,int *extractList, real4 *Ppos,real4 *Pvel,real4 *pos,real4 *vel,real4 *acc0,real4 *acc1,float2 *time,int   *body_id,bodyStruct *destination);                                                       
extern "C" void  (gpu_internalMove)(int n_extract, int       n_bodies,double4  xlow, double4  xhigh, int       *extractList, int       *indexList, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, int       *body_id);

extern "C" void  (gpu_insertNewParticles)(int       n_extract, int       n_insert, int       n_oldbodies, int       offset, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, int       *body_id, bodyStruct *source);

extern "C" void  (gpu_internalMoveSFC) (int       n_extract, int       n_bodies, uint4  lowBoundary, uint4  highBoundary, int       *extractList, int       *indexList, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, int       *body_id, uint4     *body_key);
extern "C" void  (gpu_internalMoveSFC2) (int       n_extract, int       n_bodies, uint4  lowBoundary, uint4  highBoundary, int2       *extractList, int       *indexList, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, int       *body_id, uint4     *body_key);

extern "C" void  (gpu_extractOutOfDomainParticlesAdvancedSFC)(int offset, int n_extract, int *extractList, real4 *Ppos, real4 *Pvel, real4 *pos, real4 *vel, real4 *acc0, real4 *acc1, float2 *time, int   *body_id, uint4 *body_key, bodyStruct *destination);
extern "C" void  (gpu_extractOutOfDomainParticlesAdvancedSFC2)(int offset, int n_extract, uint2 *extractList, real4 *Ppos, real4 *Pvel, real4 *pos, real4 *vel, real4 *acc0, real4 *acc1, float2 *time, int   *body_id, uint4 *body_key, bodyStruct *destination);

extern "C" void  (gpu_insertNewParticlesSFC)(int       n_extract, int       n_insert, int       n_oldbodies, int       offset, real4     *Ppos, real4     *Pvel, real4     *pos, real4     *vel, real4     *acc0, real4     *acc1, float2    *time, int       *body_id, uint4     *body_key, bodyStruct *source);
extern "C" void  (gpu_extractSampleParticlesSFC)(int    n_bodies, int    sample_freq, uint4  *body_pos, uint4  *samplePosition);



extern "C" void  (gpu_build_parallel_grps)( uint   compact_list_len, uint   offset, uint  *compact_list, uint4 *bodies_key, uint4 *parGrpBlockKey, uint2 *parGrpBlockInfo, uint  *startBoundary);

extern "C" void  (gpu_segmentedSummaryBasic) (const int n_groups, uint     *validGroups, uint     *atomicValues, uint2    *hashGroupInfo, uint4    *hashGroupKey, uint4    *hashGroupResult, uint4    *sourceData);

extern "C" void  (gpu_domainCheckSFCAndAssign)(int    n_bodies, int    nProcs, uint4  lowBoundary, uint4  highBoundary, uint4  *boundaryList,  uint4  *body_key, uint    *validList,  uint   *idList);

#endif

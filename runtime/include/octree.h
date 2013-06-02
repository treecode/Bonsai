#ifndef _OCTREE_H_
#define _OCTREE_H_

#ifdef USE_MPI
#include "mpi.h"
#endif


#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
#endif

#define USE_CUDA

#ifdef USE_CUDA
//   #include "my_cuda.h"
  #include "my_cuda_rt.h"
#else
  #include "my_ocl.h"
#endif

#include "tipsydefs.h"

#include "node_specs.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>



#include "log.h"

#define PRINT_MPI_DEBUG

using namespace std;

typedef float real;
typedef unsigned int uint;

#define NBLOCK_REDUCE     256
#define NBLOCK_BOUNDARY   120
#define NTHREAD_BOUNDARY  256
#define NBLOCK_PREFIX     512           //At the moment only used during memory alloc

#define NMAXSAMPLE 20000                //Used by first on host domain division

#ifdef USE_B40C
#include "sort.h"
#endif

struct morton_struct {
  uint2 key;
  int   value;
};


typedef struct setupParams {
  int jobs;                     //Minimal number of jobs for each 'processor'
  int blocksWithExtraJobs;      //Some ' processors'  do one extra job all with bid < bWEJ
  int extraElements;            //The elements that didn't fit completely
  int extraOffset;              //Start of the extra elements

} setupParams;

typedef struct nInfoStruct
{
  float x;
  int y,z;
} nInfoStruct;


typedef struct bodyStruct
{
  real4 pos;
  real4 vel;
  real4 acc0;
  real4 acc1;
  real4 Ppos;
  real4 Pvel;
  float2 time;
  int   id;
  int   temp;
  uint4 key;
} bodyStruct;

typedef struct sampleRadInfo
{
  int     nsample;
  double4 rmin;
  double4 rmax;
}sampleRadInfo;


inline int cmp_uint2(const uint2 a, const uint2 b) {
  if      (a.x < b.x) return -1;
  else if (a.x > b.x) return +1;
  else {
    if       (a.y < b.y) return -1;
    else  if (a.y > b.y) return +1;
    return 0;
  }
}

struct cmp_uint2_reverse{
  bool operator()(const uint2 &a, const uint2 &b){
    return (cmp_uint2(a,b) >= 1);
  }
};

inline int cmp_uint4(uint4 a, uint4 b) {
  if      (a.x < b.x) return -1;
  else if (a.x > b.x) return +1;
  else {
    if       (a.y < b.y) return -1;
    else  if (a.y > b.y) return +1;
    else {
      if       (a.z < b.z) return -1;
      else  if (a.z > b.z) return +1;
      return 0;
    } //end z
  }  //end y
} //end x, function


struct cmp_ph_key{
  bool operator () (const uint4 &a, const uint4 &b){
    return ( cmp_uint4( a, b) < 1);
  }
};


//Structure and properties of a tree
class tree_structure
{
  private:
    my_dev::context *devContext;        //Pointer so destructor is only called once  

  public:
    int n;                                //Number of particles in the tree
    int n_leafs;                          //Number of leafs in the tree
    int n_nodes;                          //Total number of nodes in the tree (including leafs)
    int n_groups;                         //Number of groups
    int n_levels;                         //Depth of the tree
    
    uint startLevelMin;                   //The level from which we start the tree-walk
                                          //this is decided by the tree-structure creation

    bool needToReorder;			//Set to true if SetN is called so we know we need to change particle order
    my_dev::dev_mem<real4> bodies_pos;    //The particles positions
    my_dev::dev_mem<uint4> bodies_key;    //The particles keys
    my_dev::dev_mem<real4> bodies_vel;    //Velocities
    my_dev::dev_mem<real4> bodies_acc0;    //Acceleration
    my_dev::dev_mem<real4> bodies_acc1;    //Acceleration
    my_dev::dev_mem<float2> bodies_time;  //The timestep details (.x=tb, .y=te
    my_dev::dev_mem<int>   bodies_ids;
    my_dev::dev_mem<int>   oriParticleOrder;  //Used in the correct function to speedup reorder
    
    my_dev::dev_mem<real4> bodies_Ppos;    //Predicted position
    my_dev::dev_mem<real4> bodies_Pvel;    //Predicted velocity

    my_dev::dev_mem<uint2> level_list;    //List containing the start and end positions of each level

    my_dev::dev_mem<uint>  n_children;
    my_dev::dev_mem<uint2> node_bodies;
    my_dev::dev_mem<uint>  leafNodeIdx;    //First n_leaf items represent indices of leafs
                                           //remaining (n_nodes-n_leafs) are indices of non-leafs
//     my_dev::dev_mem<uint>  group_list;     //The id's of nodes that form a group
    my_dev::dev_mem<uint>  node_level_list; //List containing start and end idxs in (leafNode idx) for each level
    my_dev::dev_mem<uint>  body2group_list; //Contains per particle to which group it belongs

    my_dev::dev_mem<uint2>  group_list;     //The group to particle relation
    my_dev::dev_mem<uint>   coarseGroupCompact; //Holds the ids of the groups that define the let boundaries

    //Variables used for properties
    my_dev::dev_mem<real4>  multipole;      //Array storing the properties for each node (mass, mono, quad pole)

    //Variables used for iteration
    int n_active_groups;
    int n_active_particles;

    my_dev::dev_mem<uint>  activeGrpList;       //Non-compacted list of active grps
    my_dev::dev_mem<uint>  active_group_list;   //Compacted list of active groups
    my_dev::dev_mem<uint>  activePartlist;      //List of active particles
    my_dev::dev_mem<uint>  ngb;                 //List of nearest neighbours

    my_dev::dev_mem<int2>  interactions;        //Counts the number of interactions, mainly for debugging and performance

    //BH Opening criteria
    my_dev::dev_mem<float4> boxSizeInfo;
    my_dev::dev_mem<float4> groupSizeInfo;

    my_dev::dev_mem<float4> boxCenterInfo;
    my_dev::dev_mem<float4> groupCenterInfo;

    my_dev::dev_mem<uint4> parallelHashes;
    my_dev::dev_mem<uint4> parallelBoundaries;

    //Combined buffers:
    /*
    Buffer1 used during: Sorting, Tree-construction, and Tree-traverse:
    Sorting:
      - SrcValues, Output, simpleKeys, permutation, output32b, valuesOutput
    Tree-construction:
      - ValidList, compactList
    Tree-traverse:
      - Interactions, NGB, Active_partliist
    */

    my_dev::dev_mem<uint> generalBuffer1;


    my_dev::dev_mem<float4> fullRemoteTree;

    uint4 remoteTreeStruct;

    real4 corner;                         //Corner of tree-structure
    real  domain_fac;                     //Domain_fac of tree-structure
    
    //Number of dust particles, moved decleration outside to get it to work
    //with renderer
    int n_dust;                           
    #ifdef USE_DUST

      int n_dust_groups;                    //Number of dust groups
      //Dust particle arrays
      my_dev::dev_mem<real4> dust_pos;    //The particles positions
      my_dev::dev_mem<uint4> dust_key;    //The particles keys
      my_dev::dev_mem<real4> dust_vel;    //Velocities
      my_dev::dev_mem<real4> dust_acc0;    //Acceleration
      my_dev::dev_mem<real4> dust_acc1;    //Acceleration
      my_dev::dev_mem<int>   dust_ids;
      
      my_dev::dev_mem<int>   dust2group_list;
      my_dev::dev_mem<int2>   dust_group_list;
      my_dev::dev_mem<int>   active_dust_list;
      my_dev::dev_mem<int>   activeDustGrouplist;      
      my_dev::dev_mem<int2>  dust_interactions;
      
      my_dev::dev_mem<int>   dust_ngb;
      my_dev::dev_mem<real4> dust_groupSizeInfo;    
      my_dev::dev_mem<real4> dust_groupCenterInfo;  
     
      void setNDust(int particles)
      {
        n_dust = particles;
      }  
      
    #endif
    
    
    
    

  tree_structure(){ n = 0;}

  tree_structure(my_dev::context &context)
  {
    n = 0;
    devContext = &context;
    setMemoryContexts();
    needToReorder = true;
  }
  void setContext(my_dev::context &context)
  {
    devContext = &context;
    setMemoryContexts();
    needToReorder = true;
  }

  void setN(int particles)
  {
    n = particles;
    needToReorder = true;
  }


  void setMemoryContexts()
  {
    bodies_pos.setContext(*devContext);
    bodies_key.setContext(*devContext);

    n_children.setContext(*devContext);
    node_bodies.setContext(*devContext);
    leafNodeIdx.setContext(*devContext);

    node_level_list.setContext(*devContext);
    multipole.setContext(*devContext);


    body2group_list.setContext(*devContext);

    bodies_vel.setContext(*devContext);
    bodies_acc0.setContext(*devContext);
    bodies_acc1.setContext(*devContext);
    bodies_time.setContext(*devContext);
    bodies_ids.setContext(*devContext);

    bodies_Ppos.setContext(*devContext);
    bodies_Pvel.setContext(*devContext);

    oriParticleOrder.setContext(*devContext);
    activeGrpList.setContext(*devContext);
    active_group_list.setContext(*devContext);
    level_list.setContext(*devContext);
    activePartlist.setContext(*devContext);
    ngb.setContext(*devContext);
    interactions.setContext(*devContext);

    group_list.setContext(*devContext);
    coarseGroupCompact.setContext(*devContext);

    //BH Opening
    boxSizeInfo.setContext(*devContext);
    groupSizeInfo.setContext(*devContext);
    boxCenterInfo.setContext(*devContext);
    groupCenterInfo.setContext(*devContext);

    parallelHashes.setContext(*devContext);
    parallelBoundaries.setContext(*devContext);

    //General buffers
    generalBuffer1.setContext(*devContext);
   
    fullRemoteTree.setContext(*devContext);
    
    #ifdef USE_DUST
      //Dust buffers
      dust_pos.setContext(*devContext);
      dust_key.setContext(*devContext);
      dust_vel.setContext(*devContext);
      dust_acc0.setContext(*devContext);
      dust_acc1.setContext(*devContext);
      dust_ids.setContext(*devContext);
      
      dust2group_list.setContext(*devContext);
      dust_group_list.setContext(*devContext);
      active_dust_list.setContext(*devContext);
      dust_interactions.setContext(*devContext);
      activeDustGrouplist.setContext(*devContext);
      
      dust_ngb.setContext(*devContext);
      dust_groupSizeInfo.setContext(*devContext);
      dust_groupCenterInfo.setContext(*devContext);
    #endif
    
    
  }

  my_dev::context getContext()
  {
    return *devContext;
  }
};



class octree {
protected:
  int devID;
  int rebuild_tree_rate;

  
  char *execPath;
  char *src_directory;
  
  //Device configuration
  int nMultiProcessors;
  int nBlocksForTreeWalk;

   //Simulation properties
  int           iter;
  float         t_current, t_previous;
  float         snapshotIter;   
  string        snapshotFile;
  float         nextSnapTime;

  int   NTotal, NFirst, NSecond, NThird, snapShotAdd;
  
  float removeDistance;

  float eps2;
  float inv_theta;
  int   dt_limit;
  float eta;
  float timeStep;
  float tEnd;
  int   iterEnd;
  float theta;

  bool  useDirectGravity;

  //Sim stats
  double Ekin, Ekin0, Ekin1;
  double Epot, Epot0, Epot1;
  double Etot, Etot0, Etot1;

  bool   store_energy_flag;
  double tinit;


  // accurate Win32 timing
#ifdef WIN32
  LARGE_INTEGER sysTimerFreq;
  LARGE_INTEGER sysTimerAtStart;
#endif

  // OpenCL context
  my_dev::context devContext;
  bool devContext_flag;
  
  //Streams
  my_dev::dev_stream *gravStream;
  my_dev::dev_stream *execStream;
  my_dev::dev_stream *copyStream;
  my_dev::dev_stream *LETDataToHostStream;
  

  // scan & sort algorithm
  my_dev::kernel  compactCount, exScanBlock, compactMove, splitMove;
  my_dev::kernel  sortCount, sortMove;
  my_dev::kernel  extractInt, reOrderKeysValues;
  my_dev::kernel  convertKey64to96, extractKeyAndPerm;
  my_dev::kernel  dataReorderR4;
  my_dev::kernel  dataReorderF2;
  my_dev::kernel  dataReorderI1;
  my_dev::kernel  dataReorderCombined;

  // tree construction kernels
  my_dev::kernel  build_key_list;
  my_dev::kernel  build_valid_list;
  my_dev::kernel  build_nodes;
  my_dev::kernel  link_tree;
  my_dev::kernel  define_groups;
  my_dev::kernel  build_level_list;
  my_dev::kernel  store_groups;

  my_dev::kernel  boundaryReduction;
  my_dev::kernel  boundaryReductionGroups;
  my_dev::kernel  build_body2group_list;
  my_dev::kernel  segmentedCoarseGroupBoundary;


  // tree properties kernels
  my_dev::kernel  propsNonLeafD, propsLeafD, propsScalingD;

  my_dev::kernel  setPHGroupData;
  my_dev::kernel  setPHGroupDataGetKey;
  my_dev::kernel  setPHGroupDataGetKey2;
  my_dev::kernel  setActiveGrps;

  //Iteraction kernels
  my_dev::kernel getTNext;
  my_dev::kernel predictParticles;
  my_dev::kernel getNActive;
  my_dev::kernel approxGrav;
  my_dev::kernel directGrav;
  my_dev::kernel correctParticles;
  my_dev::kernel computeDt;
  my_dev::kernel computeEnergy;

  my_dev::kernel distanceCheck;
  my_dev::kernel approxGravLET;
  my_dev::kernel determineLET;
  
  //Parallel kernels
  my_dev::kernel domainCheck;
  my_dev::kernel extractSampleParticles;
  my_dev::kernel extractOutOfDomainR4;
  my_dev::kernel extractOutOfDomainBody;
  my_dev::kernel insertNewParticles;
  my_dev::kernel internalMove;

  my_dev::kernel build_parallel_grps;
  my_dev::kernel segmentedSummaryBasic;

  my_dev::kernel domainCheckSFC;
  my_dev::kernel internalMoveSFC;
  my_dev::kernel extractOutOfDomainParticlesAdvancedSFC;
  my_dev::kernel insertNewParticlesSFC;
  my_dev::kernel extractSampleParticlesSFC;

#ifdef USE_B40C
  Sort90 *sorter;
#endif
  
  ///////////////////////

  /////////////////
  void write_dumbp_snapshot(real4 *bodyPositions, real4 *bodyVelocities, int *ids, int n, string fileName);

  void   to_binary(int);
  void   to_binary(uint2);
  uint2  dilate3(int);
  int    undilate3(uint2);

  uint2  get_key(int3);
  int3   get_crd(uint2);
  real4  get_pos(uint2, real, tree_structure&);

  //uint2  get_mask(int);
//  uint4  get_mask(int);
  uint2  get_imask(uint2);

//  int   cmp_uint4(uint4 a, uint4 b);
//  int   find_key(uint4 key, uint2 cij, uint4 *keys);

  int find_key(uint2, vector<uint2>&,         int, int);
  int find_key(uint2, vector<morton_struct>&, int, int);

  ///////////

public:
   double get_time();
   
   void resetEnergy() {store_energy_flag = true;}

   my_dev::context * getDevContext() { return &devContext; };        //Pointer so destructor is only called once  

   void write_dumbp_snapshot_parallel(real4 *bodyPositions, real4 *bodyVelocities, int* bodyIds, int n, string fileName, float time) ;
   void write_dumbp_snapshot_parallel_tipsy(real4 *bodyPositions, real4 *bodyVelocities, int* bodyIds, int n, string fileName,
                                            int NCombTotal, int NCombFirst, int NCombSecond, int NCombThird, float time);
   void write_snapshot_per_process(real4 *bodyPositions, real4 *bodyVelocities, int* bodyIds, int n, string fileName, float time);

   void set_src_directory(string src_dir);

   //Memory used in the whole system, not depending on a certain number of particles
   my_dev::dev_mem<float> tnext;
   my_dev::dev_mem<uint>  nactive;
   //General memory buffers
   my_dev::dev_mem<float3>  devMemRMIN;
   my_dev::dev_mem<float3>  devMemRMAX;

   my_dev::dev_mem<uint> devMemCounts;
   my_dev::dev_mem<uint> devMemCountsx;

   my_dev::dev_mem<real4> specialParticles;//Buffer to store postions of selected particles


   tree_structure localTree;
   tree_structure remoteTree;


   //jb made these functions public for testing
   void set_context(bool disable_timing = false);
   void set_context(std::ofstream &log, bool disable_timing = false);
   void set_context2();

   int getAllignmentOffset(int n);
   int getTextureAllignmentOffset(int n, int size);

   //GPU kernels and functions
   void load_kernels();
   void resetCompact();

   void gpuCompact(my_dev::context&, my_dev::dev_mem<uint> &srcValues,
                   my_dev::dev_mem<uint> &output, int N, int *validCount);
   void gpuSplit(my_dev::context&, my_dev::dev_mem<uint> &srcValues,
                 my_dev::dev_mem<uint> &output, int N, int *validCount);
   void gpuSort(my_dev::context &devContext,
                my_dev::dev_mem<uint4> &srcValues,
                my_dev::dev_mem<uint4> &output,
                my_dev::dev_mem<uint4> &buffer,
                int N, int numberOfBits, int subItems,
                tree_structure &tree);

   void gpuSort_32b(my_dev::context &devContext,
                    my_dev::dev_mem<uint> &srcKeys,     my_dev::dev_mem<uint> &srcValues,
                    my_dev::dev_mem<int>  &keysOutput,  my_dev::dev_mem<uint> &keysAPing,
                    my_dev::dev_mem<uint> &valuesOutput,my_dev::dev_mem<uint> &valuesAPing,
                    int N, int numberOfBits);


    void desort_bodies(tree_structure &tree);
    void sort_bodies(tree_structure &tree, bool doDomainUpdate);
    void getBoundaries(tree_structure &tree, real4 &r_min, real4 &r_max);
    void getBoundariesGroups(tree_structure &tree, real4 &r_min, real4 &r_max);  

    void allocateParticleMemory(tree_structure &tree);
    void allocateTreePropMemory(tree_structure &tree);
    void reallocateParticleMemory(tree_structure &tree);

    void build(tree_structure &tree);
    void compute_properties (tree_structure &tree);
    void compute_properties_double(tree_structure &tree);
    void setActiveGrpsFunc(tree_structure &tree);

    void iterate();

  struct IterationData {
      IterationData() : Nact_since_last_tree_rebuild(0),
          totalGravTime(0), lastGravTime(0), totalBuildTime(0),
          lastBuildTime(0), totalDomTime(0), lastDomTime(0),
          totalWaitTime(0), lastWaitTime(0), startTime(0),
          totalGPUGravTimeLocal(0), totalGPUGravTimeLET(0),
          lastGPUGravTimeLocal(0), lastGPUGravTimeLET(0),
          lastLETCommTime(0), totalLETCommTime(0),
          totalDomUp(0), totalDomEx(0) {}

      int    Nact_since_last_tree_rebuild;
      double totalGravTime; //CPU timers, includes any non-hidden communication cost
      double lastGravTime;
      double totalBuildTime;
      double lastBuildTime;
      double totalDomTime;
      double lastDomTime;
      double totalWaitTime;
      double lastWaitTime;
      double startTime;
      double totalGPUGravTimeLocal; //GPU timers, gravity only
      double totalGPUGravTimeLET;
      double lastGPUGravTimeLocal;
      double lastGPUGravTimeLET;
      double lastLETCommTime; //Time it takes to communicate/build LET structures
      double totalLETCommTime;
      double totalDomUp;
      double totalDomEx;
  };

  void iterate_setup(IterationData &idata); 
  void iterate_teardown(IterationData &idata); 
  bool iterate_once(IterationData &idata); 

  //Subfunctions of iterate, should probally be private 
  void predict(tree_structure &tree);
  void approximate_gravity(tree_structure &tree);
  void direct_gravity(tree_structure &tree);
  void correct(tree_structure &tree);
  double compute_energies(tree_structure &tree);

  int  checkMergingDistance(tree_structure &tree, int iter, double dE);
  void checkRemovalDistance(tree_structure &tree);

  //Parallel version functions
  //Approximate for LET
  void approximate_gravity_let(tree_structure &tree, tree_structure &remoteTree, 
                               int bufferSize, bool doActivePart);

  //Parallel version functions
  int procId, nProcs;   //Process ID in the mpi stack, number of processors in the commm world
  unsigned long long  nTotalFreq_ull;       //Total Number of particles over all processes


  double prevDurStep;   //Duration of gravity time in previous step
  double thisPartLETExTime;     //The time it took to communicate with the neighbours during the last step

  double4 *currentRLow, *currentRHigh;  //Contains the actual domain distribution, to be used
                                        //during the LET-tree generatino

//  real4 *localGrpTreeCntSize;

  real4 *globalGrpTreeCntSize;

  uint *globalGrpTreeCount;
  uint *globalGrpTreeOffsets;

  int  *fullGrpAndLETRequest;
  uint2 *fullGrpAndLETRequestStatistics;

  std::vector<int> infoGrpTreeBuffer;
  std::vector<int> exchangePartBuffer;


  int grpTree_n_nodes;
  int grpTree_n_topNodes;



  real maxLocalEps;               //Contains the maximum local eps/softening value
                                  //will be stored in cur_xlow[i].w after exchange
  real4 rMinLocalTree;            //for particles
  real4 rMaxLocalTree;            //for particles

  real4 rMinGlobal;
  real4 rMaxGlobal;
  
  bool letRunning;
  

  unsigned int totalNumberOfSamples;
  sampleRadInfo *curSysState;

  //Functions
  void mpiInit(int argc,char *argv[], int &procId, int &nProcs);

  //Utility
  void mpiSync();
  int  mpiGetRank();
  int  mpiGetNProcs();
  void AllSum(double &value);
  int  SumOnRootRank(int &value);

  //Main MPI functions

  //Functions for domain division
  void createORB();
  void mpiSumParticleCount(int numberOfParticles);

  void sendCurrentRadiusInfo(real4 &rmin, real4 &rmax);
  void sendCurrentRadiusAndSampleInfo(real4 &rmin, real4 &rmax, int nsample, int *nSamples);
  
  //Function for Device assisted domain division
  void gpu_collect_sample_particles(int nSample, real4 *sampleParticles);
  void gpu_updateDomainDistribution(double timeLocal);
  void gpu_updateDomainOnly();
  int  gpu_exchange_particles_with_overflow_check(tree_structure &tree,
                                                  bodyStruct *particlesToSend, 
                                                  my_dev::dev_mem<uint> &extractList, int nToSend);
  void gpuRedistributeParticles();

//  int  exchange_particles_with_overflow_check(tree_structure &localTree);
  int gpu_exchange_particles_with_overflow_check_SFC(tree_structure &tree,
                                                  bodyStruct *particlesToSend,
                                                  my_dev::dev_mem<uint> &extractList, int nToSend);


   //Local Essential Tree related functions


  void ICRecv(int procId, vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<int> &bodiesIDs);
  void ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities,  int *bodiesIDs, int size);

  void build_NewTopLevels(int n_bodies,
                       uint4 *keys,
                       uint2 *nodes,
                       uint4 *node_keys,
                       uint  *node_levels,
                       int &n_levels,
                       int &n_nodes,
                       int &startGrp,
                       int &endGrp);

  void computeProps_TopLevelTree(
      int topTree_n_nodes,
      int topTree_n_levels,
      uint* node_levels,
      uint2 *nodes,
      real4* topTreeCenters,
      real4* topTreeSizes,
      real4* topTreeMultipole,
      real4* nodeCenters,
      real4* nodeSizes,
      real4* multiPoles,
      double4* tempMultipoleRes);

  void makeLET();

  void parallelDataSummary(tree_structure &tree, float lastExecTime, float lastExecTime2, double &domUpdate, double &domExch);


   void gpuRedistributeParticles_SFC(uint4 *boundaries);

  void build_GroupTree(int n_bodies, uint4 *keys, uint2 *nodes, uint4 *node_keys, uint  *node_levels,
                       int &n_levels, int &n_nodes, int &startGrp, int &endGrp);

  void computeProps_GroupTree(real4 *grpCenter, real4 *grpSize, real4 *treeCnt,
                              real4 *treeSize,  uint2 *nodes,   uint  *node_levels, int    n_levels);

  void sendCurrentInfoGrpTree();

  void computeSampleRateSFC(float lastExecTime, int &nSamples, float &sampleRate);

  void exchangeSamplesAndUpdateBoundarySFC(uint4 *sampleKeys,    int  nSamples,
                                           uint4 *globalSamples, int  *nReceiveCnts, int *nReceiveDpls,
                                           int    totalCount,   uint4 *parallelBoundaries, float lastExectime);

  void essential_tree_exchangeV2(tree_structure &tree,
                                 tree_structure &remote,
                                 nInfoStruct *nodeInfo,
                                 vector<real4> &topLevelTrees,
                                 vector<uint2> &topLevelTreesSizeOffset,
                                 int     nTopLevelTrees);
  void mergeAndLaunchLETStructures(
      tree_structure &tree, tree_structure &remote,
      real4 **treeBuffers,  int* treeBuffersSource, int &topNodeOnTheFlyCount,
      int &recvTree, bool &mergeOwntree, int &procTrees, double &tStart);


#if 0
  template<class T>
  int MP_exchange_particle_with_overflow_check(int ibox,
                                              T *source_buffer,
                                              vector<T> &recv_buffer,
                                              int firstloc,
                                              int nparticles,
                                              int isource,
                                              int &nsend,
                                              unsigned int &recvCount);

  real4* MP_exchange_bhlist(int ibox, int isource,
                                int bufferSize, real4 *letDataBuffer);

  void gpu_collect_hashes(int nHashes, uint4 *hashes, uint4 *boundaries, float lastExecTime, float lastExecTime2);

  void tree_walking_tree_stack_versionC13(
     real4 *multipoleS, nInfoStruct* nodeInfoS, //Local Tree
     real4* grpNodeSizeInfoS, real4* grpNodeCenterInfoS, //remote Tree
     int start, int end, int startGrp, int endGrp,
     int &nAcceptedNodes, int &nParticles,
     uint2 *curLevel, uint2 *nextLevel);

  void stackFill(real4 *LETBuffer, real4 *nodeCenter, real4* nodeSize,
      real4* bodies, real4 *multipole,
      nInfoStruct *nodeInfo,
      int nParticles, int nNodes,
      int start, int end,
      uint *curLevelStack, uint* nextLevelStack);

  int stackBasedTopLEvelsCheck(tree_structure &tree, real4 *topLevelTree, int proc, int topLevels,
                                uint2 *curLevelStack,
                                uint2 *nextLevelStack,
                                int &DistanceCheck);
#endif

  int recursiveBasedTopLEvelsCheckStart(tree_structure &tree,
                                        real4 *treeBuffer,
                                        real4 *grpCenter,
                                        real4 *grpSize,
                                        int startGrp,
                                        int endGrp,
                                        int &DistanceCheck);
  int recursiveTopLevelCheck(uint4 checkNode, real4* treeBoxSizes, real4* treeBoxCenters, real4* treeBoxMoments,
                          real4* grpCenter, real4* grpSize, int &DistanceCheck, int &DistanceCheckPP, int maxLevel);

  //End functions for parallel code
  
      //Functions related to dust
  #ifdef USE_DUST 
    void allocateDustMemory(tree_structure &tree);
    void sort_dust(tree_structure &tree);
    void make_dust_groups(tree_structure &tree);
    void allocateDustGroupBuffers(tree_structure &tree);
    void predictDustStep(tree_structure &tree);
    void correctDustStep(tree_structure &tree);
    void approximate_dust(tree_structure &tree);
    void direct_dust(tree_structure &tree);
    void setDustGroupProperties(tree_structure &tree);
    
    my_dev::kernel define_dust_groups;
    my_dev::kernel store_dust_groups;
    my_dev::kernel predictDust;
    my_dev::kernel correctDust;
  #endif
  
  //
  //Function for setting up the mergers
  bool addGalaxy(int galaxyID);


  //Library interface functions  
  void  setEps(float eps);
  float getEps();
  void  setDt(float dt);
  float getDt();
  void  setTheta(float theta);
  float getTheta();
  void  setTEnd(float tEnd);
  float getTEnd();
  void  setTime(float);
  float getTime();
  float getPot();
  float getKin();

  //End library functions


  void setDataSetProperties(int NTotalT = -1, int NFirstT = -1, int NSecondT = -1, int NThirdT = -1)
  {
    NTotal   = NTotalT;
    NFirst   = NFirstT;
    NSecond  = NSecondT;
    NThird   = NThirdT;
  }

	void set_t_current(const float t)
	{
		t_current = t_previous = t;
	}
	float get_t_current() const
	{
		return t_current;
	}

  octree(char **argv, const int device = 0, const float _theta = 0.75, const float eps = 0.05,
         string snapF = "", float snapI = -1,  float tempTimeStep = 1.0 / 16.0, float tempTend = 1000,
         int _iterEnd = (1<<30),
         int maxDistT = -1, int snapAdd = 0, const int _rebuild = 2,
         bool direct = false)
  : rebuild_tree_rate(_rebuild), procId(0), nProcs(1), thisPartLETExTime(0), useDirectGravity(direct)
  {
#if USE_B40C
    sorter = 0;
#endif

    devContext_flag = false;
    iter            = 0;
    t_current       = t_previous = 0;
    
    src_directory = NULL;

    if(argv != NULL)  execPath = argv[0];
    //First init mpi
    int argc = 0;
    mpiInit(argc, argv, procId, nProcs);

    if(nProcs > 1)
      devID = procId % getNumberOfCUDADevices();
    else
      devID = device;

    char *gpu_prof_log;
    gpu_prof_log=getenv("CUDA_PROFILE_LOG");
    if(gpu_prof_log){
      char tmp[50];
      sprintf(tmp,"process_%d-%d_%s",procId,nProcs, gpu_prof_log);
      #ifdef WIN32
          SetEnvironmentVariable("CUDA_PROFILE_LOG", tmp);
      #else
          setenv("CUDA_PROFILE_LOG",tmp,1);
          LOGF(stderr, "TESTING log on proc: %d val: %s \n", procId, tmp);
      #endif
    }


//    LOGF(stderr, "Settings device : %d\t"  << devID << "\t" << device << "\t" << nProcs <<endl;

    snapshotIter = snapI;
    snapshotFile = snapF;

    timeStep = tempTimeStep;
    tEnd     = tempTend;
    iterEnd  = _iterEnd;

    removeDistance = (float)maxDistT;
    snapShotAdd    = snapAdd;

    //Theta, time-stepping
    inv_theta   = 1.0f/_theta;
    eps2        = eps*eps;
    eta         = 0.02f;
    theta       = _theta;

    nextSnapTime = 0;
    //Calc dt_limit
    const float dt_max = 1.0f / (1 << 4);

    dt_limit = int(-log(dt_max)/log(2.0f));

    store_energy_flag = true;
    
    execStream = NULL;
    gravStream = NULL;
    copyStream = NULL;
    LETDataToHostStream = NULL;
    
    infoGrpTreeBuffer.resize(7*nProcs);
    exchangePartBuffer.resize(8*nProcs);

    globalGrpTreeCntSize = NULL;


    //An initial guess for group broadcasted information
    //We set the statistics for our neighboring processes
    //to 1 and all remote ones to 0 for starters
    fullGrpAndLETRequest           = new int[nProcs];
    fullGrpAndLETRequestStatistics = new uint2[nProcs];
    if(nProcs <= NUMBER_OF_FULL_EXCHANGE)
    {
      //Set all processors to use the full-grp data
      for(int i=0; i < nProcs; i++)
      {
        fullGrpAndLETRequestStatistics[i] = make_uint2(0xFFFFFFF0,i);
      }
      fullGrpAndLETRequestStatistics[procId] = make_uint2(0,procId);
    }
    else
    {
      for(int i=0; i < nProcs; i++) fullGrpAndLETRequestStatistics[i] = make_uint2(0,i);

      //Initially set the neighboring processes to use the full data
      for(int i=1; i <= NUMBER_OF_FULL_EXCHANGE/2; i++)
      {
              int src = (procId-i);
              if(src < 0) src+=nProcs;
//              fprintf(stderr, "[%d] Writing src: %d \n", procId, src);
              fullGrpAndLETRequestStatistics[src] = make_uint2(src+1,src);
      }
      for(int i=1; i <= NUMBER_OF_FULL_EXCHANGE/2; i++)
      {
              int src = (procId+i) % nProcs;
//              fprintf(stderr, "[%d] Writing src: %d \n", procId, src);
              fullGrpAndLETRequestStatistics[src] = make_uint2(src+1,src);
      }
    }


    prevDurStep = -1;   //Set it to negative so we know its the first step

//     my_dev::base_mem::printMemUsage();   
    
    //Init at zero so we can check for n_dust later on
    localTree.n      = 0;
    localTree.n_dust = 0;

#ifdef WIN32
    // initialize windows timer
    QueryPerformanceFrequency(&sysTimerFreq);
    QueryPerformanceCounter(&sysTimerAtStart);
#endif

  }
  ~octree() {    

    delete[] currentRLow;
    delete[] currentRHigh;
    delete[] curSysState;

    if(globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
    if(globalGrpTreeCount) delete[] globalGrpTreeCount;
    if(globalGrpTreeOffsets) delete[] globalGrpTreeOffsets;

    if(fullGrpAndLETRequest)    delete[] fullGrpAndLETRequest;
    if(fullGrpAndLETRequestStatistics) delete[] fullGrpAndLETRequestStatistics;


#if USE_B40C
    delete sorter;
#endif
  };

  void setUseDirectGravity(bool s) { useDirectGravity = s;    }
  bool getUseDirectGravity() const { return useDirectGravity; }
};


#endif // _OCTREE_H_

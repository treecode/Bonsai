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
#include "logFileWriter.h"
#include "SharedMemory.h"
#include "tipsyIO.h"
#include "log.h"
#include "FileIO.h"



#ifdef USE_MPI
  #include "MPIComm.h"
  extern MPIComm *myComm;
#endif

#ifndef WIN32
  #include <unistd.h>
#endif




typedef float              real;
typedef float2             real2;
typedef unsigned int       uint;
typedef unsigned long long ullong; //ulonglong1

#define NBLOCK_REDUCE     256
#define NBLOCK_BOUNDARY   120
#define NTHREAD_BOUNDARY  256
#define NBLOCK_PREFIX     512           //At the moment only used during memory alloc

#define NMAXSAMPLE 20000                //Used by first on host domain division



/*
 * V1 IDs, 32 bit integers
 * >  200000000               => Dark-matter
 * >= 100000000 <  200000000  => Bulge
 * >= 0         <  100000000  => Disk
 *    Possible:
 *      >= 0         <  40000000 => Disk
 *      >= 40000000  <  50000000 => Glowing stars in spiral arms
 *      >= 50000000  <  70000000 => Dust
 *      >= 70000000  < 100000000 => Glow mass less dust particles
 *
 * V2 IDs, 64 bit integers => 9.223.372.036.854.775.807
 *
 * >  3.000.000.000.000.000.000                             => Dark-matter
 * >= 2.000.000.000.000.000.000 < 3.000.000.000.000.000.000 => Bulge
 * >= 0                         < 2.000.000.000.000.000.000 => Disk
 *
 *
 */

#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000

#define PERIODIC_X 1
#define PERIODIC_Y 2
#define PERIODIC_Z 4
#define PERIODIC_XYZ  (PERIODIC_X+PERIODIC_Y+PERIODIC_Z)

#define SELECT_GRAV 1
#define SELECT_SPH  2
#define SELECT_SPHGRAV (SELECT_GRAV+SELECT_SPH)

#define LET_METHOD_GRAV 1
#define LET_METHOD_DENS 2
#define LET_METHOD_DRVT 3
#define LET_METHOD_HYDR 4



typedef struct setupParams {
  int jobs;                     //Minimal number of jobs for each 'processor'
  int blocksWithExtraJobs;      //Some ' processors'  do one extra job all with bid < bWEJ
  int extraElements;            //The elements that didn't fit completely
  int extraOffset;              //Start of the extra elements

} setupParams;


typedef struct sampleRadInfo
{
  int     nsample;
  double4 rmin;
  double4 rmax;
}sampleRadInfo;


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


class particleSet
{
public:
	int n;							 //Number of bodies
    my_dev::dev_mem<real4>  pos;     //The particles positions
    my_dev::dev_mem<uint4>  key;     //The particles keys
    my_dev::dev_mem<real4>  vel;     //Velocities
    my_dev::dev_mem<real4>  acc0;    //Acceleration
    my_dev::dev_mem<real4>  acc1;    //Acceleration
    my_dev::dev_mem<float2> time;    //The timestep details (.x=tb, .y=te
    my_dev::dev_mem<ullong> ids;
    my_dev::dev_mem<real4>  Ppos;    //Predicted position
    my_dev::dev_mem<real4>  Pvel;    //Predicted velocity

    //Density related buffers
    my_dev::dev_mem<real>  h;       //The particles search radius
    my_dev::dev_mem<real2> dens;    //The particles density (x) and number of neighbors (y)

    particleSet(){ n = 0;}

    void setN(int particles) { n = particles; }

    void allocate(int n_bodies = -1)
    {
    	if(n_bodies <= 0) n_bodies = n;
		//Particle properties
		pos.cmalloc(n_bodies+1, true);   //+1 to set end pos, host mapped? TODO mapped not needed right since we use Ppos?
		vel.cmalloc(n_bodies,   false);
		key.cmalloc(n_bodies+1, false);  //+1 to set end key
		ids.cmalloc(n_bodies+1, false);  //+1 to set end key

		Ppos.cmalloc(n_bodies+1, true);  //Memory to store predicted positions, host mapped
		Pvel.cmalloc(n_bodies+1, true);  //Memory to store predicted velocities, host mapped


		acc0.ccalloc(n_bodies, false);   //ccalloc -> init to 0
		acc1.ccalloc(n_bodies, false);   //ccalloc -> init to 0
		time.ccalloc(n_bodies, false);   //ccalloc -> init to 0

		h.cmalloc   (n_bodies, true);
		dens.cmalloc(n_bodies, true);

		//Initialize to -1
		for(int i=0; i < n_bodies; i++) h[i] = -1;
		h.h2d();
    }

    void reallocate(int n_bodies = -1)
    {
    	if(n_bodies <= 0) n_bodies = n;
		//Particle properties
		pos.cresize(n_bodies+1, true);   //+1 to set end pos, host mapped? TODO not needed right since we use Ppos
		vel.cresize(n_bodies, false);
		key.cresize(n_bodies+1, false);  //+1 to set end key
		ids.cresize(n_bodies+1, false);  //+1 to set end key

		Ppos.cresize(n_bodies+1, true);  //Memory to store predicted positions, host mapped
		Pvel.cresize(n_bodies+1, true);  //Memory to store predicted velocities, host mapped


		acc0.cresize(n_bodies, false);   //ccalloc -> init to 0
		acc1.cresize(n_bodies, false);   //ccalloc -> init to 0
		time.cresize(n_bodies, false);   //ccalloc -> init to 0

		h.cresize   (n_bodies, true);
		dens.cresize(n_bodies, true);
    }

};

//Structure and properties of a tree
class tree_structure
{
  public:
    int n;                                //Number of particles in the tree
    int n_leafs;                          //Number of leafs in the tree
    int n_nodes;                          //Total number of nodes in the tree (including leafs)
    int n_groups;                         //Number of groups
    int n_levels;                         //Depth of the tree
    
    uint startLevelMin;                   //The level from which we start the tree-walk
                                          //this is decided by the tree-structure creation

    //Variables used for iteration
    int n_active_groups;
    int n_active_particles;

    real4 corner;                         //Corner of tree-structure
    real  domain_fac;                     //Domain_fac of tree-structure

    particleSet	bodies;

    my_dev::dev_mem<real4>  bodies_pos;     //The particles positions
    my_dev::dev_mem<uint4>  bodies_key;     //The particles keys
    my_dev::dev_mem<real4>  bodies_vel;     //Velocities
    my_dev::dev_mem<real4>  bodies_acc0;    //Acceleration
    my_dev::dev_mem<real4>  bodies_acc1;    //Acceleration
    my_dev::dev_mem<float2> bodies_time;    //The timestep details (.x=tb, .y=te
    my_dev::dev_mem<ullong> bodies_ids;
    my_dev::dev_mem<real4>  bodies_Ppos;    //Predicted position
    my_dev::dev_mem<real4>  bodies_Pvel;    //Predicted velocity
    
    //Density related buffers
    my_dev::dev_mem<real2> bodies_dens;      //The particles density (x) and smoothing length (y)
    my_dev::dev_mem<real2> bodies_dens_out;  //The particles density (x) and smoothing length (y) output computed during tree-walk
    my_dev::dev_mem<real4> bodies_grad;      //The density gradient TODO(jbedorf): This one can probably be removed
    my_dev::dev_mem<real > bodies_h;         //Search radius, keep for compatibility for now
    my_dev::dev_mem<real4> bodies_hydro;     //The hydro properties: x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
    my_dev::dev_mem<real4> bodies_hydro_out; //The hydro result array


    my_dev::dev_mem<uint>   oriParticleOrder;  //Used in the correct function to speedup reorder

    
    my_dev::dev_mem<uint2> level_list;    //List containing the start and end positions of each level
    my_dev::dev_mem<uint>  n_children;
    my_dev::dev_mem<uint2> node_bodies;
    my_dev::dev_mem<uint>  leafNodeIdx;    //First n_leaf items represent indices of leafs
                                           //remaining (n_nodes-n_leafs) are indices of non-leafs

    my_dev::dev_mem<uint>  node_level_list; //List containing start and end idxs in (leafNode idx) for each level
    my_dev::dev_mem<uint>  body2group_list; //Contains per particle to which group it belongs
    my_dev::dev_mem<uint2> group_list;      //The group to particle relation


    //Variables used for properties
    my_dev::dev_mem<real4>  multipole;      	//Array storing the properties for each node (mass, mono, quad pole)

    my_dev::dev_mem<uint>  activeGrpList;       //Non-compacted list of active groups
    my_dev::dev_mem<uint>  active_group_list;   //Compacted list of active groups
    my_dev::dev_mem<uint>  activePartlist;      //List of active particles
    my_dev::dev_mem<uint>  ngb;                 //List of nearest neighbors

    my_dev::dev_mem<int2>  interactions;        //Counts the number of interactions, mainly for debugging and performance

    //Properties of the tree-node boxes
    my_dev::dev_mem<float4> boxSizeInfo;
    my_dev::dev_mem<float4> boxCenterInfo;
    my_dev::dev_mem<float > boxSmoothing;


    my_dev::dev_mem<float4> groupSizeInfo;
    my_dev::dev_mem<float4> groupCenterInfo;

    my_dev::dev_mem<uint4> parallelBoundaries;

    my_dev::dev_mem<int4> smallBoundaryTreeIndices;
    my_dev::dev_mem<int4>  fullBoundaryTreeIndices;

    my_dev::dev_mem<float4> smallBoundaryTree;
    my_dev::dev_mem<float4>  fullBoundaryTree;


    bodyProps group_body;


    //Combined buffers:
    /*
      Buffer1 used during: Sorting, Tree-construction, and Tree-traverse:
      Sorting:
        - SrcValues, Output, simpleKeys, permutation, output32b, valuesOutput
      Tree-construction:
        - ValidList, compactList
      Tree-traverse:
        - Interactions, NGB, Active_partlist
    */

    my_dev::dev_mem<uint> generalBuffer1;


    my_dev::dev_mem<float4> fullRemoteTree;
    uint4 remoteTreeStruct;				  //Properties of the remote tree-structure (particles, nodes, offsets)

    

  tree_structure(){ n = 0;}


  void setN(int particles) { n = particles; }

};



class octree {
protected:
  const MPI_Comm &mpiCommWorld;
  int devID;
  
  char *execPath;
  char *src_directory;
  
  //Device configuration
  int nMultiProcessors;
  int nBlocksForTreeWalk;

   //Simulation properties
  int           iter;
  float         t_current, t_previous;
  float         snapshotIter;   
  float         quickDump, quickRatio;
  bool          quickSync, useMPIIO, mpiRenderMode;
  string        snapshotFile;
  float         nextSnapTime;
  float         nextQuickDump;

  float         statisticsIter;
  float         nextStatsTime;
  int 			rebuild_tree_rate;


  float eps2;
  float inv_theta;
  int   dt_limit;
  float eta;
  float timeStep;
  float tEnd;
  int   iterEnd;
  float theta;

  bool  useDirectGravity;

  //Simulation statistics
  double Ekin, Ekin0, Ekin1;
  double Epot, Epot0, Epot1;
  double Etot, Etot0, Etot1;

  bool   store_energy_flag;
  double tinit;

  LOGFILEWRITER *logFileWriter;



  // Device context
  my_dev::context *devContext;

  
  //Streams
  my_dev::dev_stream *gravStream;
  my_dev::dev_stream *execStream;
  my_dev::dev_stream *copyStream;
  my_dev::dev_stream *LETDataToHostStream;
  

  // scan & split kernels
  my_dev::kernel  compactCount, exScanBlock, compactMove, splitMove;

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


  // tree properties kernels
  my_dev::kernel  propsNonLeafD, propsLeafD, propsScalingD;
  my_dev::kernel  setPHGroupData;
  my_dev::kernel  setActiveGrps;
  my_dev::kernel  gpuBoundaryTree;
  my_dev::kernel  gpuBoundaryTreeExtract;

  //Time integration kernels
  my_dev::kernel getTNext;
  my_dev::kernel predictParticles;
  my_dev::kernel getNActive;
  my_dev::kernel approxGrav;
  my_dev::kernel approxGravLET;
  my_dev::kernel correctParticles;
  my_dev::kernel computeDt;
  my_dev::kernel computeEnergy;

  //SPH kernels
  my_dev::kernel SPHDensity;
  my_dev::kernel SPHDensityLET;
  my_dev::kernel SPHDerivative;
  my_dev::kernel SPHDerivativeLET;
  my_dev::kernel SPHHydro;
  my_dev::kernel SPHHydroLET;
  my_dev::kernel setPressure;

  //Other
  my_dev::kernel directGrav;

  //Parallel kernels
  my_dev::kernel internalMoveSFC2;
  my_dev::kernel extractOutOfDomainParticlesAdvancedSFC2;
  my_dev::kernel insertNewParticlesSFC;
  my_dev::kernel domainCheckSFCAndAssign;

  ///////////////////////

  // accurate Win32 timing
#ifdef WIN32
  LARGE_INTEGER sysTimerFreq;
  LARGE_INTEGER sysTimerAtStart;
#endif

  ///////////

public:
   my_dev::context * getDevContext() { return devContext; };


   //Memory used in the whole system, not depending on a certain number of particles
   my_dev::dev_mem<float> 	tnext;
   my_dev::dev_mem<uint>  	nactive;
   //General memory buffers
   my_dev::dev_mem<float3>  devMemRMIN;
   my_dev::dev_mem<float3>  devMemRMAX;

   my_dev::dev_mem<uint> devMemCounts;
   my_dev::dev_mem<uint> devMemCountsx;

   tree_structure localTree;
   tree_structure remoteTree;

   tipsyIO *fileIO;

   double get_time();
   void resetEnergy() {store_energy_flag = true;}
   void set_src_directory(std::string src_dir);

   void writeLogData(std::string &str){ devContext->writeLogEvent(str.c_str());}
   void writeLogToFile(){ this->logFileWriter->updateLogData(devContext->getLogData());}

   int getAllignmentOffset(int n);
   int getTextureAllignmentOffset(int n, int size);

   //GPU kernels and functions
   void load_kernels();
   void resetCompact();

   void gpuCompact(my_dev::dev_mem<uint> &srcValues,
                   my_dev::dev_mem<uint> &output, int N, int *validCount);
   void gpuSplit(my_dev::dev_mem<uint> &srcValues,
                 my_dev::dev_mem<uint> &output, int N, int *validCount);
   void gpuSort(my_dev::dev_mem<uint4> &srcKeys,
                my_dev::dev_mem<uint>   &permutation, //For 32bit values
                my_dev::dev_mem<uint>   &tempB,       //For 32bit values
                my_dev::dev_mem<uint>   &tempC,       //For 32bit keys
                my_dev::dev_mem<uint>   &tempD,       //For 32bit keys
                my_dev::dev_mem<char>   &tempE,       //For sorting space
                int N);

   template <typename T>
   void dataReorder(const int N, my_dev::dev_mem<uint> &permutation,
                    my_dev::dev_mem<T>  &dIn, my_dev::dev_mem<T>  &scratch,
                    bool overwrite = true,
                    bool devOnly   = false);

   template <typename T>
   void dataReorder2(const int N, my_dev::dev_mem<uint> &permutation,
                     my_dev::dev_mem<T>  &dIn, my_dev::dev_mem<T>  &dOut);

    void sort_bodies(tree_structure &tree, bool doDomainUpdate, bool doFullShuffle = false);
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
          totalDomUp(0), totalDomEx(0), totalDomWait(0),
          totalPredCor(0){}

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
      double totalDomWait;
      double totalPredCor;
  };

  void iterate_setup(IterationData &idata); 
  void iterate_teardown(IterationData &idata); 
  bool iterate_once(IterationData &idata); 

  //Bonsai IO related
  void terminateIO() const;
  template<typename THeader, typename TData>
    void dumpDataCommon(
        SharedMemoryBase<THeader> &header, SharedMemoryBase<TData> &data,
        const std::string &fileNameBase, const float ratio, const bool sync);
  void dumpData();
  void dumpDataMPI();
  void lReadBonsaiFile(std::vector<real4 > &,std::vector<real4 > &, std::vector<ullong> &,
                       std::vector<real2> &bodyDensity, std::vector<real4>       &bodyHydro,
                       std::vector<real4> &bodyDrv,
                       float &tCurrent, const std::string &fileName, const int rank, const int nrank,
                       const MPI_Comm &comm, const bool restart = true, const int reduceFactor = 1);

  //Sub functions of iterate, should probably be private
  void   predict(tree_structure &tree);
  void   approximate_gravity(tree_structure &tree);
  void   direct_gravity(tree_structure &tree);
  void   correct(tree_structure &tree);
  double compute_energies(tree_structure &tree);



  //Parallel version functions

  int procId, nProcs;                   //Process ID in the mpi stack, number of processors in the commm world
  int sharedPID;                        //Shared process ID to be used with the shared memory buffers
  unsigned long long  nTotalFreq_ull;   //Total Number of particles over all processes


  double prevDurStep;   //Duration of gravity time in previous step
  double thisPartLETExTime;     //The time it took to communicate with the neighbours during the last step

  double4 *currentRLow, *currentRHigh;  //Contains the actual domain distribution, to be used
                                        //during the LET-tree generatino

  real4 *globalGrpTreeCntSize;

  uint *globalGrpTreeCount;
  uint *globalGrpTreeOffsets;
  int4 *globalGrpTreeStatistics;    //Used for updating only the smoothing information of the boundary tree

  int2 *fullGrpAndLETRequestStatistics;

  int2 boundaryTreeDimensions; //x info about smallTree, y info about fullTree

  std::vector<int> infoGrpTreeBuffer;
  std::vector<int> exchangePartBuffer;


  float maxExecTimePrevStep;      //Maximum duration of gravity computation over all processes
  float avgExecTimePrevStep;      //Average duration of gravity computation over all processes


  int grpTree_n_nodes;
  int grpTree_n_topNodes;

  real4 rMinLocalTree;            //for particles
  real4 rMaxLocalTree;            //for particles

  real4 rMinGlobal;
  real4 rMaxGlobal;
  
  bool letRunning;
  
  sampleRadInfo *curSysState;

  //Functions
  void mpiSetup();


  //Utility
  void      mpiSync();
  int       mpiGetRank();
  int       mpiGetNProcs();
  void      AllSum(double &value);
  double    AllMin(double value);
  int       SumOnRootRank(int value);
  double    SumOnRootRank(double value);

  void dumpTreeStructureToFile(tree_structure &tree);


  //SPH related

  void   approximate_density    (tree_structure &tree);
  void   approximate_density_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool doActivePart);

  void   approximate_derivative    (tree_structure &tree);
  void   approximate_derivative_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool doActivePart);

  void   approximate_hydro    (tree_structure &tree);
  void   approximate_hydro_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool doActivePart);

  void   distributeBoundaries(bool doOnlyUpdate);
  void   makeDensityLET();
  void   makeDerivativeLET();
  void   makeHydroLET();



  //Main MPI functions

  //Functions for domain division
  void mpiSumParticleCount(int numberOfParticles);

  void ICRecv(int procId, vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<ullong> &bodiesIDs);
  void ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities,  ullong *bodiesIDs, int size);


  void sendCurrentRadiusInfo(real4 &rmin, real4 &rmax);
  
  //Function for Device assisted domain division
  int gpu_exchange_particles_with_overflow_check_SFC2(tree_structure &tree,
                                                    bodyStruct *particlesToSend,
                                                    int *nparticles, int *nsendDispls, int *nreceive,
                                                    int nToSend);
  void approximate_gravity_let(tree_structure &tree, tree_structure &remoteTree,
                                 int bufferSize, bool doActivePart);




   //Local Essential Tree related functions
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

  void parallelDataSummary(tree_structure &tree, float lastExecTime, float lastExecTime2, double &domUpdate, double &domExch, bool initalSetup);


  void gpuRedistributeParticles_SFC(uint4 *boundaries);

  void build_GroupTree(int n_bodies, uint4 *keys, uint2 *nodes, uint4 *node_keys, uint  *node_levels,
                       int &n_levels, int &n_nodes, int &startGrp, int &endGrp);

  void computeProps_GroupTree(real4 *grpCenter, real4 *grpSize, real4 *treeCnt,
                              real4 *treeSize,  uint2 *nodes,   uint  *node_levels, int    n_levels);

  int  gpuDetermineBoundary(tree_structure &tree,
                            const int maxDepth, const uint2 node_begend,
                            my_dev::dev_mem<uint > &validList,
                            my_dev::dev_mem<uint > &stackList,
                            my_dev::dev_mem<int4> &newValid,
                            my_dev::dev_mem<int4> &finalValid,
                            my_dev::dev_mem<float4> &finalBuff);

  int4 getSearchPropertiesBoundaryTrees();
  void sendCurrentInfoGrpTree();
  void updateCurrentInfoGrpTree();

  void exchangeSamplesAndUpdateBoundarySFC(uint4 *sampleKeys,    int  nSamples,
                                           uint4 *globalSamples, int  *nReceiveCnts, int *nReceiveDpls,
                                           int    totalCount,   uint4 *parallelBoundaries, float lastExectime,
                                           bool initialSetup);

  void essential_tree_exchangeV2(tree_structure &tree,
                                 tree_structure &remote,
                                 vector<real4> &topLevelTrees,
                                 vector<uint2> &topLevelTreesSizeOffset,
                                 int     nTopLevelTrees,
                                 const int LETMethod);

  void mergeAndLaunchLETStructures(tree_structure &tree, tree_structure &remote,
                                   real4 **treeBuffers,  int* treeBuffersSource, int &topNodeOnTheFlyCount,
                                   int &recvTree, bool &mergeOwntree, int &procTrees, double &tStart,
                                   const int METHOD);

  void checkGPUAndStartLETComputation(tree_structure &tree,
                                      tree_structure &remote,
                                      int            &topNodeOnTheFlyCount,
                                      int            &nReceived,
                                      int            &procTrees,
                                      double         &tStart,
                                      double         &totalLETExTime,
                                      bool            mergeOwntree,
                                      int            *treeBuffersSource,
                                      real4         **treeBuffers,
                                      const int       LETMethod);


  int recursiveTopLevelCheck(uint4 checkNode, real4* treeBoxSizes, real4* treeBoxCenters, real4* treeBoxMoments,
                          real4* grpCenter, real4* grpSize, int &DistanceCheck, int &DistanceCheckPP, int maxLevel);

  //End functions for parallel code


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

  void set_t_current(const float t) { t_current = t_previous = t; }
  float get_t_current() const       { return t_current; }
  void setUseDirectGravity(bool s)  { useDirectGravity = s;    }
  bool getUseDirectGravity() const  { return useDirectGravity; }

  octree(const MPI_Comm &comm,
         my_dev::context *devContext_,
         char **argv, const int device = 0, const float _theta = 0.75, const float eps = 0.05,
         string snapF = "", float snapI = -1,  
         const float _quickDump       = 0.0,
         const float _quickRatio      = 0.1,
         const bool  _quickSync       = true,
         const bool  _useMPIIO        = false,
         const bool  _mpiRenderMode   = false,
         float tempTimeStep           = 1.0 / 16.0,
         float tempTend               = 1000,
         int _iterEnd                 = (1<<30),
         const int _rebuild           = 2,
         bool direct                  = false,
         const int shrdpid            = 0)
  : devContext(devContext_), mpiCommWorld(comm), rebuild_tree_rate(_rebuild), procId(0), nProcs(1),
    thisPartLETExTime(0), useDirectGravity(direct), quickDump(_quickDump), quickRatio(_quickRatio),
    quickSync(_quickSync), useMPIIO(_useMPIIO), mpiRenderMode(_mpiRenderMode), nextQuickDump(0.0), sharedPID(shrdpid)
  {
    iter            = 0;
    t_current       = t_previous = 0;
    src_directory   = NULL;

    if(argv != NULL)  execPath = argv[0];

    localTree.n = 0;

    devContext = devContext_;

    //Setup the MPI processName and some buffers
    mpiSetup();


    statisticsIter = 0; //0=disabled, 1 = Every N-body unit, 2= every 2nd n-body unit, etc..
    nextStatsTime  = 0;
    nextSnapTime   = 0;

    snapshotIter      = snapI;
    snapshotFile      = snapF;
    store_energy_flag = true;

    timeStep = tempTimeStep;
    tEnd     = tempTend;
    iterEnd  = _iterEnd;

    //Theta, time-stepping
    inv_theta   = 1.0f/_theta;
    eps2        = eps*eps;
    eta         = 0.02f;
    theta       = _theta;

    const float dt_max = 1.0f / (1 << 4); //Calc dt_limit
    dt_limit = int(-log(dt_max)/log(2.0f));

    execStream          = NULL;
    gravStream          = NULL;
    copyStream          = NULL;
    LETDataToHostStream = NULL;
    
    infoGrpTreeBuffer. resize(7*nProcs);
    exchangePartBuffer.resize(8*nProcs);

    globalGrpTreeCntSize = NULL;


    maxExecTimePrevStep = 100;    //Some large values to force updates
    avgExecTimePrevStep = 1;      //Some large values to force updates


    //An initial guess for group broadcasted information
    //We set the statistics for our neighboring processes
    //to 1 and all remote ones to 0 for starters
    fullGrpAndLETRequestStatistics = new int2[nProcs];

    for(int i=0; i < nProcs; i++)
    {
      fullGrpAndLETRequestStatistics[i] = make_int2(0,0);
    }

    prevDurStep = -1;   //Set it to negative so we know its the first step

    
#ifdef USE_MPI
    logFileWriter = new LOGFILEWRITER(nProcs, myComm->MPI_COMM_I, myComm->MPI_COMM_J);
#else
    logFileWriter = new LOGFILEWRITER(nProcs, 0, 0);
#endif


    fileIO = new tipsyIO();


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

    delete logFileWriter;
    delete fileIO;

    if(globalGrpTreeCntSize)    delete[] globalGrpTreeCntSize;
    if(globalGrpTreeCount)      delete[] globalGrpTreeCount;
    if(globalGrpTreeOffsets)    delete[] globalGrpTreeOffsets;
    if(globalGrpTreeStatistics) delete[] globalGrpTreeStatistics;

    if(fullGrpAndLETRequestStatistics) delete[] fullGrpAndLETRequestStatistics;
  };
};



#endif // _OCTREE_H_

#include "octree.h"
#include <xmmintrin.h>
#include "radix.h"
#include <parallel/algorithm>
#include "dd2d.h"


typedef float  _v4sf  __attribute__((vector_size(16)));
typedef int    _v4si  __attribute__((vector_size(16)));

struct v4sf
{
  _v4sf data;
  v4sf() {}
  v4sf(const _v4sf _data) : data(_data) {}
  operator const _v4sf&() const {return data;}
  operator       _v4sf&()       {return data;}

};

#ifdef USE_MPI
#include "mpi.h"
#include <omp.h>
#include "MPIComm.h"
template <> MPI_Datatype MPIComm_datatype<float>() {return MPI_FLOAT; }
MPIComm *myComm;

static MPI_Datatype MPI_V4SF = 0;

  template <>
MPI_Datatype MPIComm_datatype<v4sf>()
{
  if (MPI_V4SF) return MPI_V4SF;
  else {
    int ss = sizeof(v4sf) / sizeof(float);
    assert(0 == sizeof(v4sf) % sizeof(float));
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_V4SF);
    MPI_Type_commit(&MPI_V4SF);
    return MPI_V4SF;
  }
}
void MPIComm_free_type()
{
  if (MPI_V4SF) MPI_Type_free(&MPI_V4SF);
}



#endif

#if ENABLE_LOG
extern bool ENABLE_RUNTIME_LOG;
extern bool PREPEND_RANK;
#endif

//SSE stuff for local tree-walk

inline float __abs(const float x)
{
  return __builtin_fabs(x);
};






#ifdef __AVX__
typedef float  _v8sf  __attribute__((vector_size(32)));
typedef int    _v8si  __attribute__((vector_size(32)));
#endif

static inline _v4sf __abs(const _v4sf x)
{
  const _v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
  return __builtin_ia32_andps(x, (_v4sf)mask);
}

#ifdef __AVX__
static inline _v8sf __abs8(const _v8sf x)
{
  const _v8si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
    0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
  return __builtin_ia32_andps256(x, (_v8sf)mask);
}
#endif











inline int host_float_as_int(float val)
{
  union{float f; int i;} u; //__float_as_int
  u.f           = val;
  return u.i;
}

inline float host_int_as_float(int val)
{
  union{int i; float f;} itof; //__int_as_float
  itof.i           = val;
  return itof.f;
}


inline void _v4sf_transpose(_v4sf &a, _v4sf &b, _v4sf &c, _v4sf &d){
  _v4sf t0 = __builtin_ia32_unpcklps(a, c); // |c1|a1|c0|a0|
  _v4sf t1 = __builtin_ia32_unpckhps(a, c); // |c3|a3|c2|a2|
  _v4sf t2 = __builtin_ia32_unpcklps(b, d); // |d1|b1|d0|b0|
  _v4sf t3 = __builtin_ia32_unpckhps(b, d); // |d3|b3|d2|b2|

  a = __builtin_ia32_unpcklps(t0, t2);
  b = __builtin_ia32_unpckhps(t0, t2);
  c = __builtin_ia32_unpcklps(t1, t3);
  d = __builtin_ia32_unpckhps(t1, t3);
}

#ifdef __AVX__
static inline _v8sf pack_2xmm(const _v4sf a, const _v4sf b){
  // v8sf p;
  _v8sf p = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; // just avoid warning
  p = __builtin_ia32_vinsertf128_ps256(p, a, 0);
  p = __builtin_ia32_vinsertf128_ps256(p, b, 1);
  return p;
}
inline void _v8sf_transpose(_v8sf &a, _v8sf &b, _v8sf &c, _v8sf &d){
  _v8sf t0 = __builtin_ia32_unpcklps256(a, c); // |c1|a1|c0|a0|
  _v8sf t1 = __builtin_ia32_unpckhps256(a, c); // |c3|a3|c2|a2|
  _v8sf t2 = __builtin_ia32_unpcklps256(b, d); // |d1|b1|d0|b0|
  _v8sf t3 = __builtin_ia32_unpckhps256(b, d); // |d3|b3|d2|b2|

  a = __builtin_ia32_unpcklps256(t0, t2);
  b = __builtin_ia32_unpckhps256(t0, t2);
  c = __builtin_ia32_unpcklps256(t1, t3);
  d = __builtin_ia32_unpckhps256(t1, t3);
}
#endif

inline int split_node_grav_impbh_box4( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  nodeCOM,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4])
{
  _v4sf ncx = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x00);
  _v4sf ncy = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x55);
  _v4sf ncz = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xaa);
  _v4sf ncw = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xff);
  _v4sf size = __abs(ncw);

  _v4sf bcx =  (boxCenter[0]);
  _v4sf bcy =  (boxCenter[1]);
  _v4sf bcz =  (boxCenter[2]);
  _v4sf bcw =  (boxCenter[3]);
  _v4sf_transpose(bcx, bcy, bcz, bcw);

  _v4sf bsx =  (boxSize[0]);
  _v4sf bsy =  (boxSize[1]);
  _v4sf bsz =  (boxSize[2]);
  _v4sf bsw =  (boxSize[3]);
  _v4sf_transpose(bsx, bsy, bsz, bsw);

  _v4sf dx = __abs(bcx - ncx) - bsx;
  _v4sf dy = __abs(bcy - ncy) - bsy;
  _v4sf dz = __abs(bcz - ncz) - bsz;

  _v4sf zero = {0.0, 0.0, 0.0, 0.0};
  dx = __builtin_ia32_maxps(dx, zero);
  dy = __builtin_ia32_maxps(dy, zero);
  dz = __builtin_ia32_maxps(dz, zero);

  _v4sf ds2 = dx*dx + dy*dy + dz*dz;
#if 0
  const float c = 10e-4f;
  int ret = __builtin_ia32_movmskps(
      __builtin_ia32_orps(
        __builtin_ia32_cmpleps(ds2,  size),
        __builtin_ia32_cmpltps(ds2 - size, (_v4sf){c,c,c,c})
        )
      );
#else
  int ret = __builtin_ia32_movmskps(
      __builtin_ia32_cmpleps(ds2, size));
#endif
  return ret;
}

inline _v4sf split_node_grav_impbh_box4a( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  nodeCOM,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4])
{
  _v4sf ncx = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x00);
  _v4sf ncy = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x55);
  _v4sf ncz = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xaa);
  _v4sf ncw = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xff);
  _v4sf size = __abs(ncw);

  _v4sf bcx =  (boxCenter[0]);
  _v4sf bcy =  (boxCenter[1]);
  _v4sf bcz =  (boxCenter[2]);
  _v4sf bcw =  (boxCenter[3]);
  _v4sf_transpose(bcx, bcy, bcz, bcw);

  _v4sf bsx =  (boxSize[0]);
  _v4sf bsy =  (boxSize[1]);
  _v4sf bsz =  (boxSize[2]);
  _v4sf bsw =  (boxSize[3]);
  _v4sf_transpose(bsx, bsy, bsz, bsw);

  _v4sf dx = __abs(bcx - ncx) - bsx;
  _v4sf dy = __abs(bcy - ncy) - bsy;
  _v4sf dz = __abs(bcz - ncz) - bsz;

  const _v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
  dx = __builtin_ia32_maxps(dx, zero);
  dy = __builtin_ia32_maxps(dy, zero);
  dz = __builtin_ia32_maxps(dz, zero);

  const _v4sf ds2 = dx*dx + dy*dy + dz*dz;
#if 0
  const float c = 10e-4f;
  _v4sf ret =
    __builtin_ia32_orps(
        __builtin_ia32_cmpleps(ds2,  size),
        __builtin_ia32_cmpltps(ds2 - size, (_v4sf){c,c,c,c})
        );
#else
  _v4sf ret =
    __builtin_ia32_cmpleps(ds2, size);
#endif
#if 0
  const _v4si mask1 = {1,1,1,1};
  const _v4si mask2 = {2,2,2,2};
  ret = __builtin_ia32_andps(ret, (_v4sf)mask1);
  ret = __builtin_ia32_orps (ret,
      __builtin_ia32_andps(
        __builtin_ia32_cmpleps(bcw, (_v4sf){0.0f,0.0f,0.0f,0.0f}),
        (_v4sf)mask2));
#endif
  return ret;
}

#ifdef __AVX__
inline std::pair<v4sf,v4sf> split_node_grav_impbh_box8a( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  nodeCOM,
    const _v4sf  boxCenter[8],
    const _v4sf  boxSize  [8])
{
#if 0
  _v4sf ncx0 = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x00);
  _v4sf ncy0 = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x55);
  _v4sf ncz0 = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xaa);
  _v4sf ncw0 = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xff);
  _v4sf size0 = __abs(ncw0);

  _v8sf ncx = pack_2xmm(ncx0, ncx0);
  _v8sf ncy = pack_2xmm(ncy0, ncy0);
  _v8sf ncz = pack_2xmm(ncz0, ncz0);
  _v8sf size = pack_2xmm(size0, size0);

#else
  _v8sf com = pack_2xmm(nodeCOM, nodeCOM);
  _v8sf ncx = __builtin_ia32_shufps256(com, com, 0x00);
  _v8sf ncy = __builtin_ia32_shufps256(com, com, 0x55);
  _v8sf ncz = __builtin_ia32_shufps256(com, com, 0xaa);
  _v8sf size = __abs8(__builtin_ia32_shufps256(com, com, 0xff));
#endif

  _v8sf bcx = pack_2xmm(boxCenter[0], boxCenter[4]);
  _v8sf bcy = pack_2xmm(boxCenter[1], boxCenter[5]);
  _v8sf bcz = pack_2xmm(boxCenter[2], boxCenter[6]);
  _v8sf bcw = pack_2xmm(boxCenter[3], boxCenter[7]);
  _v8sf_transpose(bcx, bcy, bcz, bcw);

  _v8sf bsx = pack_2xmm(boxSize[0], boxSize[4]);
  _v8sf bsy = pack_2xmm(boxSize[1], boxSize[5]);
  _v8sf bsz = pack_2xmm(boxSize[2], boxSize[6]);
  _v8sf bsw = pack_2xmm(boxSize[3], boxSize[7]);
  _v8sf_transpose(bsx, bsy, bsz, bsw);

  _v8sf dx = __abs8(bcx - ncx) - bsx;
  _v8sf dy = __abs8(bcy - ncy) - bsy;
  _v8sf dz = __abs8(bcz - ncz) - bsz;

  const _v8sf zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f,0.0f};
  dx = __builtin_ia32_maxps256(dx, zero);
  dy = __builtin_ia32_maxps256(dy, zero);
  dz = __builtin_ia32_maxps256(dz, zero);

  const _v8sf ds2 = dx*dx + dy*dy + dz*dz;
#if 0
  const float c = 10e-4f;
  _v8sf ret =
    __builtin_ia32_orps256(
        __builtin_ia32_cmpps256(ds2,  size, 18),  /* le */
        __builtin_ia32_cmpps256(ds2 - size, (_v8sf){c,c,c,c,c,c,c,c}, 17)  /* lt */
        );
#else
  _v8sf ret =
    __builtin_ia32_cmpps256(ds2, size, 18);
#endif
#if 0
  const _v4si mask1 = {1,1,1,1};
  const _v4si mask2 = {2,2,2,2};
  ret = __builtin_ia32_andps(ret, (_v4sf)mask1);
  ret = __builtin_ia32_orps (ret,
      __builtin_ia32_andps(
        __builtin_ia32_cmpleps(bcw, (_v4sf){0.0f,0.0f,0.0f,0.0f}),
        (_v4sf)mask2));
#endif
  const _v4sf ret1 = __builtin_ia32_vextractf128_ps256(ret, 0);
  const _v4sf ret2 = __builtin_ia32_vextractf128_ps256(ret, 1);
  return std::make_pair(ret1,ret2);
}
#endif


template<typename T>
struct Swap
{
  private:
    T &t1;
    T &t2;

  public:

    Swap(T &_t1, T &_t2) : t1(_t1), t2(_t2) {}
    void swap() {t1.swap(t2);}
    const T& first() const { return t1;}
    T& first() { return t1;}
    const T& second() const { return t2;}
    T& second() { return t2;}
};


void extractGroups(
    std::vector<real4> &groupCentre,
    std::vector<real4> &groupSize,
    const real4 *nodeCentre,
    const real4 *nodeSize,
    const int cellBeg,
    const int cellEnd,
    const int nNodes)
{
  groupCentre.clear();
  groupCentre.reserve(nNodes);

  groupSize.clear();
  groupSize.reserve(nNodes);

  const int levelCountMax = nNodes;
  std::vector<int> currLevelVec, nextLevelVec;
  currLevelVec.reserve(levelCountMax);
  nextLevelVec.reserve(levelCountMax);
  Swap<std::vector<int> > levelList(currLevelVec, nextLevelVec);

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);

  int depth = 0;
  while (!levelList.first().empty())
  {
    //LOGF(stderr, " depth= %d \n", depth++);
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint   nodeIdx = levelList.first()[i];
      const float4 centre  = nodeCentre[nodeIdx];
      const float4 size    = nodeSize[nodeIdx];
      const float nodeInfo_x = centre.w;
      const uint  nodeInfo_y = host_float_as_int(size.w);

      const bool lleaf = nodeInfo_x <= 0.0f;
      if (!lleaf)
      {
        const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
        const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has
#if 1
        if (lnchild == 8)
        {
          float4 centre1 = centre;
          centre1.w = -1;
          groupCentre.push_back(centre1);
          groupSize  .push_back(size);
        }
        else
#endif
          for (int i = lchild; i < lchild + lnchild; i++)
            levelList.second().push_back(i);
      }
      else
      {
        float4 centre1 = centre;
        centre1.w = -1;
        groupCentre.push_back(centre1);
        groupSize  .push_back(size);
      }
    }

    levelList.swap();
    levelList.second().clear();
  }
}


void octree::mpiInit(int argc,char *argv[], int &procId, int &nProcs)
{
#ifdef USE_MPI
  int  namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);

  if(!mpiInitialized)
  {
    MPI_Init(&argc,&argv);
    int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    //assert(provided == MPI_THREAD_MULTIPLE);

    //      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    //      assert(provided == MPI_THREAD_FUNNELED);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procId);

  myComm = new MPIComm(procId, nProcs);

  MPI_Get_processor_name(processor_name,&namelen);
#else
  char processor_name[] = "Default";
#endif

#ifdef PRINT_MPI_DEBUG
  LOGF(stderr, "Proc id: %d @ %s , total processes: %d (mpiInit) \n", procId, processor_name, nProcs);
#endif
  fprintf(stderr, "Proc id: %d @ %s , total processes: %d (mpiInit) \n", procId, processor_name, nProcs);

  //Allocate memory for the used buffers
  //    domainRLow  = new double4[nProcs];
  //    domainRHigh = new double4[nProcs];
  //
  //    domHistoryLow   = new int4[nProcs];
  //    domHistoryHigh  = new int4[nProcs];

  //Fill domainRX with constants so we can check if its initialized before
  //    for(int i=0; i < nProcs; i++)
  //    {
  //      domainRLow[i] = domainRHigh[i] = make_double4(1e10, 1e10, 1e10, 1e10);
  //
  //      domHistoryLow[i] = domHistoryHigh[i] = make_int4(0,0,0,0);
  //    }

  currentRLow  = new double4[nProcs];
  currentRHigh = new double4[nProcs];
  //
  //    xlowPrev  = new double4[nProcs];
  //    xhighPrev = new double4[nProcs];
  //

  //    globalCoarseGrpCount     = new uint[nProcs];
  //    globalCoarseGrpOffsets   = new uint[nProcs];

  //    nSampleAndSizeValues    = new int2[nProcs];
  curSysState             = new sampleRadInfo[nProcs];

  globalGrpTreeCount   = new uint[nProcs];
  globalGrpTreeOffsets = new uint[nProcs];
}



//Utility functions
void octree::mpiSync(){
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

int octree::mpiGetRank(){
  return procId;
}

int octree::mpiGetNProcs(){
  return nProcs;
}

void octree::AllSum(double &value)
{
#ifdef USE_MPI
  double tmp = -1;
  MPI_Allreduce(&value,&tmp,1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
  value = tmp;
#endif
}

int octree::SumOnRootRank(int &value)
{
#ifdef USE_MPI
  int temp;
  MPI_Reduce(&value,&temp,1, MPI_INT, MPI_SUM,0, MPI_COMM_WORLD);
  return temp;
#else
  return value;
#endif
}
//end utility



//Main functions


//Functions related to domain decomposition


typedef struct hashInfo
{
  int     nHashes;      //Number of hashes that will be send by the process
  int     nParticles;   //Number of particles the sending process has in total
  double  execTime;     //The time it took this process to compute gravity in the previous step
  double  execTime2;    //A second timing number. We dont want execTime and execTime2 to fluctuate too much
  //balance on the one with the largest difference.
} hashInfo;

int balanceLoad(int *nParticlesOriginal, int *nParticlesNew, float *load,
    int nProcs, int leftIdx, int nTotal, float loadAvg)
{
#ifdef USE_MPI
  //Sum the total load left and right
  int nProcsLeftSide    = nProcs / 2;
  int nProcsRightSide   = nProcs  - nProcsLeftSide;
  int rightIdx          = leftIdx + nProcsLeftSide;

  if(nProcs == 1) {
    LOGF(stderr, "Ready by default \n");
    nParticlesNew[leftIdx] = nTotal;
    return 0;
  }

  LOGF(stderr, "Start balance: nProcs: %d, leftIdx: %d, rightIdx: %d nProcLeft: %d  nProcRight: %d nTotal: %d avg: %f\n",
      nProcs, leftIdx, rightIdx, nProcsLeftSide,nProcsRightSide, nTotal, loadAvg);

  int nPartLeftOriginal = 0, nPartRightOriginal = 0;
  for(int i=leftIdx;  i < rightIdx;                   i++) nPartLeftOriginal  += nParticlesOriginal[i];
  for(int i=rightIdx; i < rightIdx+nProcsRightSide;   i++) nPartRightOriginal += nParticlesOriginal[i];

  //Compute the factor to which to increase by using the received timing numbers
  float loadLeft = 0, loadRight = 0;
  for(int i=leftIdx;  i < rightIdx;                 i++) loadLeft  += load[i];
  for(int i=rightIdx; i < rightIdx+nProcsRightSide; i++) loadRight += load[i];


  float leftTarget  = (loadAvg*nProcsLeftSide)  / (loadLeft);
  float rightTarget = (loadAvg*nProcsRightSide) / (loadRight);
  //Inverse load, for testing LET
  //LET float leftTarget  = 1./((loadAvg*nProcsLeftSide)  / (loadLeft));
  //float rightTarget = 1./((loadAvg*nProcsRightSide) / (loadRight));

  int newLeft = 0, newRight = 0;
  //Check which target we are trying to match, namely the one with minimal work
  if(leftTarget < rightTarget)
    //LET  if(leftTarget > rightTarget)
  {
    //Optimize left
    newLeft  = nPartLeftOriginal*leftTarget;
    newRight = nTotal - newLeft;
  }
  else
  {
    //Optimize right
    newRight  = nPartRightOriginal*rightTarget;
    newLeft   = nTotal - newRight;
  }

  LOGF(stderr, "newLeft: %d , newRight: %d nTotal: %d , leftTarget: %f rightTarget: %f , loadLeft: %f loadRight: %f \n",
      newLeft, newRight, nTotal, leftTarget, rightTarget, loadLeft, loadRight);

  if(nProcs == 2)
  {
    nParticlesNew[leftIdx] = newLeft;
    nParticlesNew[rightIdx] = newRight;
    return 0;
  }

  //Recursive the left and right parts
  balanceLoad(nParticlesOriginal, nParticlesNew, load, nProcsLeftSide, leftIdx,  newLeft, loadAvg);
  balanceLoad(nParticlesOriginal, nParticlesNew, load, nProcsRightSide, rightIdx, newRight, loadAvg);

#endif

  return 0;
}

//Uses one communication by storing data in one buffer and communicate required information,
//such as box-sizes and number of sample particles on this process (Note that the number is only
//used by process-0
void octree::sendCurrentRadiusAndSampleInfo(real4 &rmin, real4 &rmax, int nsample, int *nSamples)
{
  sampleRadInfo curProcState;

  curProcState.nsample      = nsample;
  curProcState.rmin         = make_double4(rmin.x, rmin.y, rmin.z, rmin.w);
  curProcState.rmax         = make_double4(rmax.x, rmax.y, rmax.z, rmax.w);

#ifdef USE_MPI
  //Get the number of sample particles and the domain size information
  MPI_Allgather(&curProcState, sizeof(sampleRadInfo), MPI_BYTE,  curSysState,
      sizeof(sampleRadInfo), MPI_BYTE, MPI_COMM_WORLD);
#else
  curSysState[0] = curProcState;
#endif

  rmin.x                 = (real)(currentRLow[0].x = curSysState[0].rmin.x);
  rmin.y                 = (real)(currentRLow[0].y = curSysState[0].rmin.y);
  rmin.z                 = (real)(currentRLow[0].z = curSysState[0].rmin.z);
  currentRLow[0].w = curSysState[0].rmin.w;

  rmax.x                 = (real)(currentRHigh[0].x = curSysState[0].rmax.x);
  rmax.y                 = (real)(currentRHigh[0].y = curSysState[0].rmax.y);
  rmax.z                 = (real)(currentRHigh[0].z = curSysState[0].rmax.z);
  currentRHigh[0].w = curSysState[0].rmax.w;

  nSamples[0] = curSysState[0].nsample;

  for(int i=1; i < nProcs; i++)
  {
    rmin.x = std::min(rmin.x, (real)curSysState[i].rmin.x);
    rmin.y = std::min(rmin.y, (real)curSysState[i].rmin.y);
    rmin.z = std::min(rmin.z, (real)curSysState[i].rmin.z);

    rmax.x = std::max(rmax.x, (real)curSysState[i].rmax.x);
    rmax.y = std::max(rmax.y, (real)curSysState[i].rmax.y);
    rmax.z = std::max(rmax.z, (real)curSysState[i].rmax.z);

    currentRLow[i].x = curSysState[i].rmin.x;
    currentRLow[i].y = curSysState[i].rmin.y;
    currentRLow[i].z = curSysState[i].rmin.z;
    currentRLow[i].w = curSysState[i].rmin.w;

    currentRHigh[i].x = curSysState[i].rmax.x;
    currentRHigh[i].y = curSysState[i].rmax.y;
    currentRHigh[i].z = curSysState[i].rmax.z;
    currentRHigh[i].w = curSysState[i].rmax.w;

    nSamples[i] = curSysState[i].nsample;
  }
}

void octree::computeSampleRateSFC(float lastExecTime, int &nSamples, float &sampleRate)
{
#ifdef USE_MPI
  double t00 = get_time();
  //Compute the number of particles to sample.
  //Average the previous and current execution time to make everything smoother
  //results in much better load-balance
  static double prevDurStep  = -1;
  static int    prevSampFreq = -1;
  prevDurStep                = (prevDurStep <= 0) ? lastExecTime : prevDurStep;
  double timeLocal           = (lastExecTime + prevDurStep) / 2;

#define LOAD_BALANCE 0
#define LOAD_BALANCE_MEMORY 0

  double nrate = 0;
  if(LOAD_BALANCE) //Base load balancing on the computation time
  {
    double timeSum   = 0.0;

    //Sum the execution times over all processes
    MPI_Allreduce( &timeLocal, &timeSum, 1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    nrate = timeLocal / timeSum;

    if(LOAD_BALANCE_MEMORY)       //Don't fluctuate particles too much
    {
#define SAMPLING_LOWER_LIMIT_FACTOR  (1.9)

      double nrate2 = (double)localTree.n / (double) nTotalFreq_ull;
      nrate2       /= SAMPLING_LOWER_LIMIT_FACTOR;

      if(nrate < nrate2)
      {
        nrate = nrate2;
      }

      double nrate2_sum = 0.0;

      MPI_Allreduce(&nrate, &nrate2_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      nrate /= nrate2_sum;
    }
  }
  else
  {
    nrate = (double)localTree.n / (double)nTotalFreq_ull; //Equal number of particles
  }

  int    nsamp  = (int)(nTotalFreq_ull*0.001f/4) + 1;  //Total number of sample particles, global
  nSamples      = (int)(nsamp*nrate) + 1;
  sampleRate    = localTree.n / (float)nSamples;

  if (procId == 0)
  fprintf(stderr, "NSAMP [%d]: sample: %d nrate: %f final sampleRate: %d localTree.n: %d\tprevious: %d timeLocal: %f prevTimeLocal: %f  Took: %lg\n",
      procId, nSamples, nrate, sampleRate, localTree.n, prevSampFreq,
      timeLocal, prevDurStep,get_time()-t00);
  assert(sampleRate > 1);

  prevDurStep  = timeLocal;
  prevSampFreq = sampleRate;

#endif
}

void octree::exchangeSamplesAndUpdateBoundarySFC(uint4 *sampleKeys,    int  nSamples,
    uint4 *globalSamples, int  *nReceiveCnts, int *nReceiveDpls,
    int    totalCount,   uint4 *parallelBoundaries, float lastExecTime)
{
#ifdef USE_MPI

#if 0 /* evghenii: disable 1D to nable 2D domain decomposition below */
  {
    //Send actual data
    MPI_Gatherv(&sampleKeys[0],    nSamples*sizeof(uint4), MPI_BYTE,
        &globalSamples[0], nReceiveCnts, nReceiveDpls, MPI_BYTE,
        0, MPI_COMM_WORLD);


    if(procId == 0)
    {
      //Sort the keys. Use stable_sort (merge sort) since the separate blocks are already
      //sorted. This is faster than std::sort (quicksort)
      //std::sort(allHashes, allHashes+totalNumberOfHashes, cmp_ph_key());
      double t00 = get_time();

#if 0 /* jb2404 */
      //std::stable_sort(globalSamples, globalSamples+totalCount, cmp_ph_key());
      __gnu_parallel::stable_sort(globalSamples, globalSamples+totalCount, cmp_ph_key());
#else
#if 0
      {
        const int BITS = 32*2;  /*  32*1 = 32 bit sort, 32*2 = 64 bit sort, 32*3 = 96 bit sort */
        typedef RadixSort<BITS> Radix;
        LOGF(stderr,"Boundary :: using %d-bit RadixSort\n", BITS);

        Radix radix(totalCount);
#if 0
        typedef typename Radix::key_t key_t;
#endif

        Radix::key_t *keys;
        posix_memalign((void**)&keys, 64, totalCount*sizeof(Radix::key_t));

#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
          keys[i] = Radix::key_t(globalSamples[i]);

        radix.sort(keys);

#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
          globalSamples[i] = keys[i].get_uint4();

        free(keys);

      }
#else
      {
        LOGF(stderr,"Boundary :: using %d-bit RadixSort\n", 64);
        unsigned long long *keys;
        posix_memalign((void**)&keys, 64, totalCount*sizeof(unsigned long long));

#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
        {
          const uint4 key = globalSamples[i];
          keys[i] = 
            static_cast<unsigned long long>(key.y) | (static_cast<unsigned long long>(key.x) << 32);
        }

#if 0
        RadixSort64 r(totalCount);
        r.sort(keys);
#else
        __gnu_parallel::sort(keys, keys+totalCount);
#endif
#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
        {
          const unsigned long long key = keys[i];
          globalSamples[i] = (uint4){
            (uint)((key >> 32) & 0x00000000FFFFFFFF),
              (uint)((key      ) & 0x00000000FFFFFFFF),
              0,0};
        }
        free(keys);
      }
#endif

#endif
      LOGF(stderr,"Boundary took: %lg  Items: %d\n", get_time()-t00, totalCount);


      //Split the samples in equal parts to get the boundaries


      int procIdx   = 1;

      globalSamples[totalCount] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
      parallelBoundaries[0]     = make_uint4(0x0, 0x0, 0x0, 0x0);
      //Chop in equal sized parts
      for(int i=1; i < nProcs; i++)
      {
        int idx = (size_t(i)*size_t(totalCount))/size_t(nProcs);

        //jb2404
        if(iter == 0){
          if((i%1000) == 0) fprintf(stderr, " Boundary %d taken from : %d \n" ,i, idx);
          if(i >= nProcs-10) fprintf(stderr, " Boundary %d taken from : %d \n" ,i, idx);
        }

        parallelBoundaries[procIdx++] = globalSamples[idx];
      }
#if 0
      int perProc = totalCount / nProcs;
      int tempSum   = 0;
      for(int i=0; i < totalCount; i++)
      {
        tempSum += 1;
        if(tempSum >= perProc)
        {
          //LOGF(stderr, "Boundary at: %d\t%d %d %d %d \t %d \n",
          //              i, globalSamples[i+1].x,globalSamples[i+1].y,globalSamples[i+1].z,globalSamples[i+1].w, tempSum);
          tempSum = 0;
          parallelBoundaries[procIdx++] = globalSamples[i+1];
        }
      }//for totalNumberOfHashes
#endif


      //Force final boundary to be the highest possible key value
      parallelBoundaries[nProcs]  = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

      delete[] globalSamples;
    }

    //Send the boundaries to all processes
    MPI_Bcast(&parallelBoundaries[0], sizeof(uint4)*(nProcs+1), MPI_BYTE, 0, MPI_COMM_WORLD);
  }

#else
  /* evghenii: 2d sampling comes here, 
   * make sure that locakTree.bodies_key.d2h in src/build.cpp.
   * if you don't see my comment there, don't use this version. it will be
   * blow up :)
   */
  if (procId == 0)
    delete[] globalSamples;

  {
    const double t0 = get_time();

    const int nkeys_loc = localTree.n;
    assert(nkeys_loc > 0);

    std::vector<DD2D::Key> key_sample;
    key_sample.reserve(nkeys_loc);

    /**** sample keys ****/

    const int npx = myComm->n_proc_i;  /* number of procs doing domain decomposition */

#if 0  /* evghenii: somehow, this choice increases imballance. don't get why ... */
    const int nmean = nTotalFreq_ull/nProcs;
    const int nsamples_glb = nmean / 10;
#else
    const int nsamples_glb = nkeys_loc / 100;
#endif

    /*** sample keys ***/

    /* LB step */
#if 1
    const double f = 1.0;
#else
    static double prevDurStep = -1;
    static int prevSampFreq = -1;
    prevDurStep = (prevDurStep <= 0) ? lastExecTime : prevDurStep;

    double timeLocal = (lastExecTime + prevDurStep) / 2;
    double timeSum = 0.0;

    MPI_Allreduce( &timeLocal, &timeSum, 1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const  double f = timeLocal / timeSum * nProcs;
#endif

    const int nsamples_loc = static_cast<int>(nsamples_glb / npx * f);
    const double stride = std::max((double)nkeys_loc/(double)nsamples_loc, 1.0);
    for (double i = 0; i < (double)nkeys_loc; i += stride)
    {
      const uint4 key = localTree.bodies_key[(int)i];
      key_sample.push_back(DD2D::Key(
            (static_cast<unsigned long long>(key.y) ) | 
            (static_cast<unsigned long long>(key.x) << 32) 
            ));
    }

#if 0  /* diagonostic, sanity check */
    {
      int nkeys_loc = key_sample.size();
      int nkeys_glb = 0;
      MPI_Allreduce(&nkeys_loc, &nkeys_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      assert(nkeys_glb <= nsamples_glb*npx);
    }
#endif

    const DD2D dd(procId, npx, nProcs, key_sample, MPI_COMM_WORLD);

    /* distribute keys */
    for (int p = 0; p < nProcs; p++)
    {
      const DD2D::Key key = dd.keybeg(p);
      parallelBoundaries[p] = (uint4){
        (uint)((key.key >> 32) & 0x00000000FFFFFFFF),
          (uint)((key.key      ) & 0x00000000FFFFFFFF),
          0,0};
    }
    parallelBoundaries[nProcs] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

    const double dt = get_time() - t0;
    if (procId == 0)
      fprintf(stderr, " it took %g sec to complete 2D domain decomposition\n", dt);
  }
#endif

  if(procId == -1)
  {
    for(int i=0; i < nProcs; i++)
    {
      fprintf(stderr, "Proc: %d Going from: >= %u %u %u  to < %u %u %u \n",i,
          parallelBoundaries[i].x,   parallelBoundaries[i].y,   parallelBoundaries[i].z,
          parallelBoundaries[i+1].x, parallelBoundaries[i+1].y, parallelBoundaries[i+1].z);
    }
  }
#endif
}
//End Domain decomposition based on Sample particle related functions

//Domain decomposition based on particle hashes related functions


//Uses one communication by storing data in one buffer and communicate required information,
//such as box-sizes and number of sample particles on this process. Nsample is set to 0
//since it is not used in this function/hash-method
void octree::sendCurrentRadiusInfo(real4 &rmin, real4 &rmax)
{
  sampleRadInfo curProcState;

  int nsample               = 0; //Place holder to just use same datastructure
  curProcState.nsample      = nsample;
  curProcState.rmin         = make_double4(rmin.x, rmin.y, rmin.z, rmin.w);
  curProcState.rmax         = make_double4(rmax.x, rmax.y, rmax.z, rmax.w);

#ifdef USE_MPI
  //Get the number of sample particles and the domain size information
  MPI_Allgather(&curProcState, sizeof(sampleRadInfo), MPI_BYTE,  curSysState,
      sizeof(sampleRadInfo), MPI_BYTE, MPI_COMM_WORLD);
#else
  curSysState[0] = curProcState;
#endif

  rmin.x                 = (real)(currentRLow[0].x = curSysState[0].rmin.x);
  rmin.y                 = (real)(currentRLow[0].y = curSysState[0].rmin.y);
  rmin.z                 = (real)(currentRLow[0].z = curSysState[0].rmin.z);
  currentRLow[0].w = curSysState[0].rmin.w;

  rmax.x                 = (real)(currentRHigh[0].x = curSysState[0].rmax.x);
  rmax.y                 = (real)(currentRHigh[0].y = curSysState[0].rmax.y);
  rmax.z                 = (real)(currentRHigh[0].z = curSysState[0].rmax.z);
  currentRHigh[0].w = curSysState[0].rmax.w;

  for(int i=1; i < nProcs; i++)
  {
    rmin.x = std::min(rmin.x, (real)curSysState[i].rmin.x);
    rmin.y = std::min(rmin.y, (real)curSysState[i].rmin.y);
    rmin.z = std::min(rmin.z, (real)curSysState[i].rmin.z);

    rmax.x = std::max(rmax.x, (real)curSysState[i].rmax.x);
    rmax.y = std::max(rmax.y, (real)curSysState[i].rmax.y);
    rmax.z = std::max(rmax.z, (real)curSysState[i].rmax.z);

    currentRLow[i].x = curSysState[i].rmin.x;
    currentRLow[i].y = curSysState[i].rmin.y;
    currentRLow[i].z = curSysState[i].rmin.z;
    currentRLow[i].w = curSysState[i].rmin.w;

    currentRHigh[i].x = curSysState[i].rmax.x;
    currentRHigh[i].y = curSysState[i].rmax.y;
    currentRHigh[i].z = curSysState[i].rmax.z;
    currentRHigh[i].w = curSysState[i].rmax.w;
  }
}


void octree::gpu_collect_hashes(int nHashes, uint4 *hashes, uint4 *boundaries, float lastExecTime, float lastExecTime2)
{
#ifdef USE_MPI
  double t0 = get_time();

  hashInfo hInfo;
  hInfo.nHashes     = nHashes;
  hInfo.nParticles  = this->localTree.n;
  hInfo.execTime    = lastExecTime;
  hInfo.execTime2   = lastExecTime2;

  LOGF(stderr, "Exectime: Proc: %d -> %f \n", procId, hInfo.execTime);

  int       *nReceiveCnts  = NULL;
  int       *nReceiveDpls  = NULL;
  float     *execTimes     = NULL;
  float     *execTimes2    = NULL;
  hashInfo  *recvHashInfo  = new hashInfo[nProcs];

  //First receive the number of hashes
  MPI_Gather(&hInfo, sizeof(hashInfo), MPI_BYTE, recvHashInfo, sizeof(hashInfo), MPI_BYTE, 0, MPI_COMM_WORLD);

  int    totalNumberOfHashes = 0;
  float  timeSum, timeSum2   = 0;
  int    nTotal              = 0;
  uint4  *allHashes          = NULL;

  //Compute receive offsets (only process 0), total number of particles, total execution time.
  if(procId == 0)
  {
    nReceiveCnts  = new int  [nProcs];
    nReceiveDpls  = new int  [nProcs];
    execTimes     = new float[nProcs];
    execTimes2    = new float[nProcs];

    nReceiveCnts[0]      = recvHashInfo[0].nHashes;
    execTimes[0]         = recvHashInfo[0].execTime;
    execTimes2[0]        = recvHashInfo[0].execTime2;

    //Receive counts and displacements
    totalNumberOfHashes += nReceiveCnts[0];
    nReceiveCnts[0]      = nReceiveCnts[0]*sizeof(uint4); //Convert to correct data size
    nReceiveDpls[0]      = 0;
    nTotal               = recvHashInfo[0].nParticles;
    timeSum              = recvHashInfo[0].execTime;
    timeSum2             = recvHashInfo[0].execTime2;


    for(int i=1; i < nProcs; i++)
    {
      nReceiveCnts[i]      = recvHashInfo[i].nHashes;
      execTimes[i]         = recvHashInfo[i].execTime;
      execTimes2[i]        = recvHashInfo[i].execTime2;

      totalNumberOfHashes += nReceiveCnts[i];
      nReceiveCnts[i]      = nReceiveCnts[i]*sizeof(uint4);
      nReceiveDpls[i]      = nReceiveDpls[i-1] + nReceiveCnts[i-1];

      nTotal   += recvHashInfo[i].nParticles;
      timeSum  += recvHashInfo[i].execTime;
      timeSum2 += recvHashInfo[i].execTime2;
    }
    allHashes                      = new uint4[totalNumberOfHashes+1];
    allHashes[totalNumberOfHashes] = make_uint4(0,0,0,0); //end boundary



    //Loop so we can decide on which number to balance
    float avgLoadTime1 = timeSum  / nProcs;
    float avgLoadTime2 = timeSum2 / nProcs;

    float maxTime1Diff = 0, maxTime2Diff = 0;
    for(int i=0; i < nProcs; i++)
    {
      float temp1 = abs((avgLoadTime1/recvHashInfo[i].execTime)-1);
      maxTime1Diff = max(temp1, maxTime1Diff);
      float temp2 = abs((avgLoadTime2/recvHashInfo[i].execTime2)-1);
      maxTime2Diff = max(temp2, maxTime2Diff);
    }

    if(0){
      if(maxTime2Diff > maxTime1Diff)
      {
        for(int i=0; i < nProcs; i++)
        {
          execTimes[i] = execTimes[2];
        }
        timeSum = timeSum2;
      }}

    LOGF(stderr, "Max diff  Time1: %f\tTime2: %f Proc0: %f \t %f \n",
        maxTime1Diff, maxTime2Diff, recvHashInfo[0].execTime, recvHashInfo[0].execTime2);
  } //if procId == 0

  //Collect hashes on process 0
  MPI_Gatherv(&hashes[0],    nHashes*sizeof(uint4), MPI_BYTE,
      &allHashes[0], nReceiveCnts,          nReceiveDpls, MPI_BYTE,
      0, MPI_COMM_WORLD);

  //  MPI_Gatherv((procId ? &sampleArray[0] : MPI_IN_PLACE), nsample*sizeof(real4), MPI_BYTE,
  //              &sampleArray[0], nReceiveCnts, nReceiveDpls, MPI_BYTE,
  //              0, MPI_COMM_WORLD);

  if(procId == 0)
  {
    delete[] nReceiveCnts;
    delete[] nReceiveDpls;

    int       *nPartPerProc  = new int[nProcs];

    //Sort the keys. Use stable_sort (merge sort) since the seperate blocks are already
    //sorted. This is faster than std::sort (quicksort)
    //std::sort(allHashes, allHashes+totalNumberOfHashes, cmp_ph_key());
    std::stable_sort(allHashes, allHashes+totalNumberOfHashes, cmp_ph_key());


#define LOAD_BALANCE 0
#define LOAD_BALANCE_MEMORY 0

#if LOAD_BALANCE
    //Load balancing version, based on gravity approximation execution times.

    LOGF(stderr, "Time sum: %f \n", timeSum);

    //Normalize, fractions : timeSum / gravTime -> gives a relative time number
    float normSum = 0;
    for(int i=0; i < nProcs; i++){
      // execTimes[i] = timeSum / execTimes[i];
      // normSum     += execTimes[i];
      LOGF(stderr, "Exec after norm: %d\t %f \tn: %d \n",i, execTimes[i],recvHashInfo[i].nParticles );
    }
    LOGF(stderr, "Normalized sum:%f  \n",normSum);


    int *npartPerProcOld = new int[nProcs];
    float *loadPerProc   = new float[nProcs];

    for(int i=0; i < nProcs; i++)
    {
      npartPerProcOld[i] = recvHashInfo[i].nParticles;
      loadPerProc[i] = recvHashInfo[i].execTime;
    }
    float loadAvg = timeSum / nProcs;

    //Adjust the boundary locations
    balanceLoad(npartPerProcOld,nPartPerProc, loadPerProc,
        nProcs,0,nTotal,loadAvg);

    delete[] npartPerProcOld;
    delete[] loadPerProc;
    //End adjusting



    //Compute the number of particles to be assign per process
    for(int i=0; i < nProcs; i++){
      //        nPartPerProc[i] = (execTimes[i] / normSum) * nTotal;

      //  nPartPerProc[i] = recvHashInfo[i].nParticles*(2*(execTimes[i] / normSum));
      //fprintf(stderr, "Npart per proc: %d\t %d \n",i, nPartPerProc[i]);
    }

    //Average with number of particles of previous step
    //TODO
    //float fac = 0.25; //25% old, 75% new
    float fac = 0.50; //Average
    for(int i=0; i < nProcs; i++){
      LOGF(stderr, "Npart per proc: new %d\told %d (avg final: %d)\n",
          nPartPerProc[i], recvHashInfo[i].nParticles,
          ((int)((recvHashInfo[i].nParticles*fac) + (nPartPerProc[i] *(1-fac)))));
      nPartPerProc[i] = (int)((recvHashInfo[i].nParticles*fac) + (nPartPerProc[i] *(1-fac)));

    }


    delete[] execTimes;
    delete[] execTimes2;
    //Now try to adjust this with respect to memory load-balance

    bool doPrint          = true;
    bool doMemLoadBalance = (LOAD_BALANCE_MEMORY) ? true : false;
    if(doMemLoadBalance)
    {
      const int maxTries          = std::max(10,nProcs);
      const double maxDiff        = 1.9; //Each process has to be within a factor 1.9 of the maximum load
      const int allowedDifference = 5;

      for(int tries = maxTries; tries > 0; tries--)
      {
        int maxNumber       = 0;
        int minNumber       = nPartPerProc[0];
        int countOutsideMin = 0;
        int sum             = 0;

        if(doPrint){
          LOGF(stderr,"Before memory load adjustment: \n");
          for(int i=0; i < nProcs; i++){
            LOGF(stderr, "%d \t %d \n", i, nPartPerProc[i]);
            sum += nPartPerProc[i];
          }
          LOGF(stderr, "Sum: %d \n", sum);
        }

        //First find the max and min process load and compute the minimum number of
        //particles a process should have according to the defined balance
        maxNumber = countOutsideMin = 0;

        for(int i=0; i < nProcs; i++){
          maxNumber = std::max(maxNumber, nPartPerProc[i]);
          minNumber = std::min(minNumber, nPartPerProc[i]);
        }

        double requiredNumber = maxNumber / maxDiff;

        if(doPrint){
          LOGF(stderr, "Max: %d  Min: %d , maxDiff factor: %f  required: %f \n",
              maxNumber, minNumber, maxDiff, requiredNumber);
        }

        if((abs(minNumber-requiredNumber)) <= allowedDifference)
        {
          break; //Difference between required and achieved is within what we allow, accept this
        }

        //Count the number of procs below the minNumber and compute number within the limits
        for(int i=0; i < nProcs; i++) if(requiredNumber > nPartPerProc[i]) countOutsideMin++;
        int countInsideMin = nProcs - countOutsideMin;

        if(countOutsideMin == 0)
          break; //Success, all within the range minimum required...maximum assigned

        //Compute particles to be added  to the processes outside the range, we take
        //(0.5*(Particles Required - minLoad)) / (# processes outside the min range) .
        //To evenly add particles , to prevent large jumps when # outside is small we use
        //the factor 0.5. For the particles to be removed we do the same but now use
        //number of processes inside the min range.
        int addNumberOfParticles = (0.5*(requiredNumber - minNumber)) / countOutsideMin;
        addNumberOfParticles     = std::max(1, addNumberOfParticles);

        int removeNumberOfParticles = (0.5*(requiredNumber - minNumber)) / countInsideMin;
        removeNumberOfParticles     = std::max(1, removeNumberOfParticles);

        if(doPrint){
          LOGF(stderr, "#Outside: %d , #Inside: %d Adding total: %f per proc: %d  Removing total: %f per proc: %d \n",
              countOutsideMin, countInsideMin, requiredNumber-minNumber,
              addNumberOfParticles, requiredNumber - minNumber, removeNumberOfParticles);
        }

        //Finally modify the particle counts :-)
        for(int i=0; i < nProcs; i++)
        {
          if(nPartPerProc[i] < requiredNumber)
            nPartPerProc[i] += addNumberOfParticles;
          else if(nPartPerProc[i] > requiredNumber)
            nPartPerProc[i] -= removeNumberOfParticles;
        }//end modify

        if(doPrint){
          LOGF(stderr,"After memory load adjustment: \n");
          for(int i=0; i < nProcs; i++) fprintf(stderr, "%d \t %d \n", i, nPartPerProc[i]);
          LOGF(stderr, "Tries left: %d \n\n\n", tries);
        }
      }//for tries
    } //if doMemLoadBalance
#else //#if LOAD_BALANCE
    //Per process, equal number of particles
    int nPerProc = nTotal / nProcs;
    for(int i=0; i < nProcs; i++) nPartPerProc[i] = nPerProc;
    LOGF(stderr, "Number of particles per process: %d \t %d \n", nTotal, nPerProc);
#endif


    //All set and done, get the boundaries
    int tempSum   = 0;
    int procIdx   = 1;
    boundaries[0] = make_uint4(0x0, 0x0, 0x0, 0x0);
    for(int i=0; i < totalNumberOfHashes; i++)
    {
      tempSum += allHashes[i].w;
      if(tempSum >= nPartPerProc[procIdx-1])
      {
        LOGF(stderr, "Boundary at: %d\t%d %d %d %d \t %d \n",
            i, allHashes[i+1].x,allHashes[i+1].y,allHashes[i+1].z,allHashes[i+1].w, tempSum);
        tempSum = 0;
        boundaries[procIdx++] = allHashes[i+1];
      }
    }//for totalNumberOfHashes


    //Force final boundary to be the highest possible key value
    boundaries[nProcs]  = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

    delete[] nPartPerProc;
    delete[] allHashes;
  }//if procId == 0
  delete[] recvHashInfo;


  //Send the boundaries to all processes
  MPI_Bcast(boundaries,  sizeof(uint4)*(nProcs+1),MPI_BYTE,0,MPI_COMM_WORLD);

  if(procId == 0){
    for(int i=0; i < nProcs; i++)
    {
      LOGF(stderr, "Proc: %d Going from: >= %u %u %u  to < %u %u %u \n",i,
          boundaries[i].x,boundaries[i].y,boundaries[i].z,boundaries[i+1].x,boundaries[i+1].y,boundaries[i+1].z);
    }
    LOGF(stderr, "Exchanging and sorting of hashes took: %f \n", get_time()-t0);
  }




#endif //if USE_MPI
  //exit(0);
}


  template<class T>
int octree::MP_exchange_particle_with_overflow_check(int ibox,
    T *source_buffer,
    vector<T> &recv_buffer,
    int firstloc,
    int nparticles,
    int isource,
    int &nsend,
    unsigned int &recvCount)
{
#ifdef USE_MPI
  MPI_Status status;
  int local_proc_id = procId;
  nsend = nparticles;

  //first send&get the number of particles to send&get
  unsigned int nreceive;

  //Send and get the number of particles that are exchanged
  MPI_Sendrecv(&nsend,1,MPI_INT,ibox,local_proc_id*10,
      &nreceive,1,MPI_INT,isource,isource*10,MPI_COMM_WORLD,
      &status);

  int ss         = sizeof(T);
  int sendoffset = nparticles-nsend;

  //Resize the receive buffer
  if((nreceive + recvCount) > recv_buffer.size())
  {
    recv_buffer.resize(nreceive + recvCount);
  }


  //Send the actual particles
  MPI_Sendrecv(&source_buffer[firstloc+sendoffset],ss*nsend,MPI_BYTE,ibox,local_proc_id*10+1,
      &recv_buffer[recvCount],ss*nreceive,MPI_BYTE,isource,isource*10+1,
      MPI_COMM_WORLD,&status);

  recvCount += nreceive;

  //     int iret = 0;
  //     int giret;
  //     MPI_Allreduce(&iret, &giret,1, MPI_INT, MPI_MAX,MPI_COMM_WORLD);
  //     return giret;
#endif
  return 0;
}


//Function that uses the GPU to get a set of particles that have to be
//send to other processes
void octree::gpuRedistributeParticles_SFC(uint4 *boundaries)
{
  //Memory buffers to hold the extracted particle information
  my_dev::dev_mem<uint>  validList(devContext);
  my_dev::dev_mem<uint>  compactList(devContext);

  int memOffset1 = compactList.cmalloc_copy(localTree.generalBuffer1,
      localTree.n, 0);
  int memOffset2 = validList.cmalloc_copy(localTree.generalBuffer1,
      localTree.n, memOffset1);

  //https://github.com/egaburov/fvmhd3d/blob/master/MPI/myMPI.h

  uint4 lowerBoundary = boundaries[this->procId];
  uint4 upperBoundary = boundaries[this->procId+1];

  validList.zeroMem();
  domainCheckSFC.set_arg<int>(0,     &localTree.n);
  domainCheckSFC.set_arg<uint4>(1,   &lowerBoundary);
  domainCheckSFC.set_arg<uint4>(2,   &upperBoundary);
  domainCheckSFC.set_arg<cl_mem>(3,  localTree.bodies_key.p());
  domainCheckSFC.set_arg<cl_mem>(4,  validList.p());
  domainCheckSFC.setWork(localTree.n, 128);
  domainCheckSFC.execute(execStream->s());
  execStream->sync();

  //Create a list of valid and invalid particles
  this->resetCompact();  //Make sure compact has been reset
  int validCount;
  gpuSplit(devContext, validList, compactList, localTree.n, &validCount);

  LOGF(stderr, "Found %d particles outside my domain, inside: %d \n", validCount, localTree.n-validCount);


  //Check if the memory size, of the generalBuffer is large enough to store the exported particles
  //if not allocate more but make sure that the copy of compactList survives
  int tempSize = localTree.generalBuffer1.get_size() - localTree.n;
  int needSize = (int)(1.01f*(validCount*(sizeof(bodyStruct)/sizeof(int))));

#if 0
  if(tempSize < needSize)
  {
    int itemsNeeded = needSize + localTree.n + 4096; //Slightly larger as before for offset space

    compactList.d2h();  //Copy the compact list to the host we need this list intact
    int *tempBuf = new int[localTree.n];
    memcpy(tempBuf, &compactList[0], localTree.n*sizeof(int));

    //Resize the general buffer
    localTree.generalBuffer1.cresize(itemsNeeded, false);
    //Reset memory pointers
    memOffset1 = compactList.cmalloc_copy(localTree.generalBuffer1,
        localTree.n, 0);

    //Restore the compactList
    memcpy(&compactList[0], tempBuf, localTree.n*sizeof(int));
    compactList.h2d();

    delete[] tempBuf;
  }

  my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);

  memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1,
      validCount, memOffset1);


  extractOutOfDomainParticlesAdvancedSFC.set_arg<int>(0,    &validCount);
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(1, compactList.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(2, localTree.bodies_Ppos.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(3, localTree.bodies_Pvel.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(4, localTree.bodies_pos.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(5, localTree.bodies_vel.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(6, localTree.bodies_acc0.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(7, localTree.bodies_acc1.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(8, localTree.bodies_time.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(9, localTree.bodies_ids.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(10, localTree.bodies_key.p());
  extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(11, bodyBuffer.p());
  extractOutOfDomainParticlesAdvancedSFC.setWork(validCount, 128);
  extractOutOfDomainParticlesAdvancedSFC.execute(execStream->s());

  bodyBuffer.d2h(validCount);

#else

  int stepSize          = (tempSize / (sizeof(bodyStruct) / sizeof(int)))-512;

  bool doInOneGo = true;

  bodyStruct *extraBodyBuffer = NULL;

  if(stepSize > needSize)
  {
    //We can do it in one go
    doInOneGo = true;
  }
  else
  {
    //We need an extra CPU buffer
    doInOneGo       = false;
    extraBodyBuffer = new bodyStruct[validCount];
    assert(extraBodyBuffer != NULL);
  }

  my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);

  memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1,
      stepSize, memOffset1);

  int extractOffset = 0;
  for(unsigned int i=0; i < validCount; i+= stepSize)
  {
    int items = min(stepSize, (int)(validCount-i));

    if(items > 0)
    {
      extractOutOfDomainParticlesAdvancedSFC.set_arg<int>(0,    &extractOffset);
      extractOutOfDomainParticlesAdvancedSFC.set_arg<int>(1,    &items);
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(2, compactList.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(3, localTree.bodies_Ppos.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(4, localTree.bodies_Pvel.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(5, localTree.bodies_pos.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(6, localTree.bodies_vel.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(7, localTree.bodies_acc0.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(8, localTree.bodies_acc1.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(9, localTree.bodies_time.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(10, localTree.bodies_ids.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(11, localTree.bodies_key.p());
      extractOutOfDomainParticlesAdvancedSFC.set_arg<cl_mem>(12, bodyBuffer.p());
      extractOutOfDomainParticlesAdvancedSFC.setWork(items, 128);
      extractOutOfDomainParticlesAdvancedSFC.execute(execStream->s());

      bodyBuffer.d2h(items); // validCount);
      if(!doInOneGo)
      {
        //Do the extra memory copy to the CPU buffer
        memcpy(&extraBodyBuffer[extractOffset], &bodyBuffer[0], sizeof(bodyStruct)*items);
        extractOffset += items;
      }
    }
  }//end for

  if(doInOneGo)
  {
    extraBodyBuffer = &bodyBuffer[0]; //Assign correct pointer
  }

#endif


  //Now we have to move particles from the back of the array to the invalid spots
  //this can be done in parallel with exchange operation to hide some time

  //One integer for counting, true-> initialize to zero so counting starts at 0
  my_dev::dev_mem<uint>  atomicBuff(devContext);
  memOffset1 = atomicBuff.cmalloc_copy(localTree.generalBuffer1,1, memOffset1);
  atomicBuff.zeroMem();


  //Internal particle movement
  internalMoveSFC.set_arg<int>(0,     &validCount);
  internalMoveSFC.set_arg<int>(1,     &localTree.n);
  internalMoveSFC.set_arg<uint4>(2,   &lowerBoundary);
  internalMoveSFC.set_arg<uint4>(3,   &upperBoundary);
  internalMoveSFC.set_arg<cl_mem>(4,  compactList.p());
  internalMoveSFC.set_arg<cl_mem>(5,  atomicBuff.p());
  internalMoveSFC.set_arg<cl_mem>(6,  localTree.bodies_Ppos.p());
  internalMoveSFC.set_arg<cl_mem>(7,  localTree.bodies_Pvel.p());
  internalMoveSFC.set_arg<cl_mem>(8,  localTree.bodies_pos.p());
  internalMoveSFC.set_arg<cl_mem>(9,  localTree.bodies_vel.p());
  internalMoveSFC.set_arg<cl_mem>(10, localTree.bodies_acc0.p());
  internalMoveSFC.set_arg<cl_mem>(11, localTree.bodies_acc1.p());
  internalMoveSFC.set_arg<cl_mem>(12, localTree.bodies_time.p());
  internalMoveSFC.set_arg<cl_mem>(13, localTree.bodies_ids.p());
  internalMoveSFC.set_arg<cl_mem>(14, localTree.bodies_key.p());
  internalMoveSFC.setWork(validCount, 128);
  internalMoveSFC.execute(execStream->s());

  //this->gpu_exchange_particles_with_overflow_check_SFC(localTree, &bodyBuffer[0], compactList, validCount);
  this->gpu_exchange_particles_with_overflow_check_SFC(localTree, &extraBodyBuffer[0], compactList, validCount);

  if(!doInOneGo) delete[] extraBodyBuffer;

} //End gpuRedistributeParticles

//Exchange particles with other processes
int octree::gpu_exchange_particles_with_overflow_check_SFC(tree_structure &tree,
    bodyStruct *particlesToSend,
    my_dev::dev_mem<uint> &extractList,
    int nToSend)
{
#ifdef USE_MPI

  int myid      = procId;
  int nproc     = nProcs;
  int iloc      = 0;
  int nbody     = nToSend;


  bodyStruct  tmpp;

  //  int *firstloc   = new int[nProcs+1];
  //  int *nparticles = new int[nProcs+1];
  //  int *nreceive   = new int[nProcs];
  //  int *nsendbytes = new int[nProcs];
  //  int *nrecvbytes = new int[nProcs];
  //
  //  int *nsendDispls = new int [nProcs+1];
  //  int *nrecvDispls = new int [nProcs+1];

  int *firstloc    = &exchangePartBuffer[0*nProcs];                //Size nProcs+1
  int *nparticles  = &exchangePartBuffer[1*(nProcs+1)];            //Size nProcs+1
  int *nsendDispls = &exchangePartBuffer[2*(nProcs+1)];            //Size nProcs+1
  int *nrecvDispls = &exchangePartBuffer[3*(nProcs+1)];            //Size nProcs+1
  int *nreceive    = &exchangePartBuffer[4*(nProcs+1)];            //Size nProcs
  int *nsendbytes  = &exchangePartBuffer[4*(nProcs+1) + 1*nProcs]; //Size nProcs
  int *nrecvbytes  = &exchangePartBuffer[4*(nProcs+1) + 2*nProcs]; //Size nProcs



  memset(nparticles,  0, sizeof(int)*(nProcs+1));
  memset(nreceive,    0, sizeof(int)*(nProcs));
  memset(nsendbytes,  0, sizeof(int)*(nProcs));
  memset(nsendDispls, 0, sizeof(int)*(nProcs));

  // Loop over particles and determine which particle needs to go where
  // reorder the bodies in such a way that bodies that have to be send
  // away are stored after each other in the array
  double t1 = get_time();

  //Array reserve some memory before hand , 1%
  vector<bodyStruct> array2Send;
  array2Send.reserve((int)(nToSend*1.5));



  static int firstSort = 1;
  if(firstSort)
  {
    std::vector<int> offsets(nProcs+2);
    std::vector<int> items(nProcs+2);

    int location       = 0;
    offsets[location]  = 0;

    for(int i=0; i < nToSend; i++)
    {
      uint4 key  = particlesToSend[i].key;

      bool assigned = false;
      while(!assigned)
      {
        uint4 lowerBoundary = tree.parallelBoundaries[location];
        uint4 upperBoundary = tree.parallelBoundaries[location+1];

        int bottom = cmp_uint4(key, lowerBoundary);
        int top    = cmp_uint4(key, upperBoundary);

        assert(bottom >= 0);

        if(top < 0)
        {
          //is in box
          assigned = true;
        }
        else
        {
          //outside box
          offsets[++location] = i;
          assert(location < nProcs);
        }
      }//while
    }//for

    //Fill remaining processes
    while(location <= nProcs)
      offsets[++location] = nToSend;

    //Fill items
    for(int ib=0; ib < nProcs; ib++)
    {
      items[ib] = offsets[ib+1]-offsets[ib];
    }

    assert(items[procId] == 0);

    for(int ib =0; ib < nProcs; ib++)
    {
      nparticles[ib] = items[ib];
      nsendbytes[ib] = nparticles[ib]*sizeof(bodyStruct);
      nsendDispls[ib] = offsets[ib]*sizeof(bodyStruct);
    }

    //  char buff[512];
    //  sprintf(buff, "Proc: ");
    //  for(int i=0; i < nProcs; i++)
    //  {
    //    sprintf(buff, "%s [%d, %d] ", buff, nparticles[i], nsendbytes[i]);
    //  }
    //  LOGF(stderr,"%s \n", buff);

    array2Send.clear();
    array2Send.insert(array2Send.end(), particlesToSend, particlesToSend+nToSend);

#if 1
    for(int ib=0; ib < nProcs; ib++)
    {
      uint4 lowerBoundary = tree.parallelBoundaries[ib];
      uint4 upperBoundary = tree.parallelBoundaries[ib+1];

      for(int i= offsets[ib]; i <  offsets[ib+1]; i++)
      {
        uint4 key  = particlesToSend[i].key;
        int bottom = cmp_uint4(key, lowerBoundary);
        int top    = cmp_uint4(key, upperBoundary);

        if(bottom >= 0 && top < 0)
        {
          //Inside
        }
        else
        {
          assert(!"Particle not in the box");
        }
      }//for i
    }//for ib
#endif


    //LOGF(stderr,  "EXCHANGE reorder iter: %d  took: %lg \tItems: %d Total-n: %d\n",
    if(iter == 0 && procId == 0)
      fprintf(stderr,  "EXCHANGE reorder iter: %d  took: %lg \tItems: %d Total-n: %d\n",
          iter, get_time()-t1, nToSend, tree.n);


    firstSort = 0;
  }
  else


  {
    //Sort the statistics, causing the processes with which we interact most to be on top
    std::sort(fullGrpAndLETRequestStatistics, fullGrpAndLETRequestStatistics+nProcs, cmp_uint2_reverse());


    for(int ib=0;ib<nproc;ib++)
    {
      //    int ibox          = (ib+myid)%nproc;
      int ibox = fullGrpAndLETRequestStatistics[ib].y;

      firstloc[ibox]    = iloc;      //Index of the first particle send to proc: ibox
      nsendDispls[ibox] = iloc*sizeof(bodyStruct);

      for(int i=iloc; i<nbody;i++)
      {
        uint4 lowerBoundary = tree.parallelBoundaries[ibox];
        uint4 upperBoundary = tree.parallelBoundaries[ibox+1];

        uint4 key  = particlesToSend[i].key;
        int bottom = cmp_uint4(key, lowerBoundary);
        int top    = cmp_uint4(key, upperBoundary);


        if(bottom >= 0 && top < 0)
        {
          //Reorder the particle information
          tmpp                  = particlesToSend[iloc];
          particlesToSend[iloc] = particlesToSend[i];
          particlesToSend[i]    = tmpp;

          //Put the particle in the array of to send particles
          array2Send.push_back(particlesToSend[iloc]);

          iloc++;
        }// end if
      }//for i=iloc
      nparticles[ibox] = iloc-firstloc[ibox];//Number of particles that has to be send to proc: ibox
      nsendbytes[ibox] = nparticles[ibox]*sizeof(bodyStruct);
    } // for(int ib=0;ib<nproc;ib++)


    //   printf("Required search time: %lg ,proc: %d found in our own box: %d n: %d  to others: %ld \n",
    //          get_time()-t1, myid, nparticles[myid], tree.n, array2Send.size());
    if(iloc < nbody)
    {
      LOGF(stderr, "Exchange_particle error: A particle could not be assigned to a box: iloc: %d total: %d \n", iloc,nbody);
      exit(0);
    }
    LOGF(stderr,  "EXCHANGE reorder iter: %d  took: %lg \tItems: %d Total-n: %d\n",
        iter, get_time()-t1, nToSend, tree.n);

  }
  t1 = get_time();
#if 0
  MPI_Alltoall(nparticles, 1, MPI_INT, nreceive, 1, MPI_INT, MPI_COMM_WORLD);


  //Compute how much we will receive and the offsets and displacements
  nrecvDispls[0]   = 0;
  nrecvbytes [0]   = nreceive[0]*sizeof(bodyStruct);
  unsigned int recvCount  = nreceive[0];
  for(int i=1; i < nProcs; i++)
  {
    nrecvbytes [i] = nreceive[i]*sizeof(bodyStruct);
    nrecvDispls[i] = nrecvDispls[i-1] + nrecvbytes [i-1];
    recvCount     += nreceive[i];
  }

  //  LOGF(stderr,"Going to receive: %d || %d %d : %d %d\n",
  //      recvCount, nrecvbytes[0], nrecvDispls[0],
  //      nrecvbytes[1], nrecvDispls[1]);
  //  LOGF(stderr,"Going to send: %d || %d %d : %d %d\n",
  //      nToSend, nsendbytes[0], nsendDispls[0],
  //        nsendbytes[1], nsendDispls[1]);

  vector<bodyStruct> recv_buffer3(recvCount);

  MPI_Alltoallv(&array2Send[0],   nsendbytes, nsendDispls, MPI_BYTE,
      &recv_buffer3[0], nrecvbytes, nrecvDispls, MPI_BYTE,
      MPI_COMM_WORLD);

  //  delete[] nsendbytes;
  //  delete[] nrecvbytes;
  //  delete[] nreceive;
  //  delete[] nsendDispls;
  //  delete[] nrecvDispls;

#elif 0
  MPI_Alltoall(nparticles, 1, MPI_INT, nreceive, 1, MPI_INT, MPI_COMM_WORLD);
  double t92 = get_time();
  unsigned int recvCount  = nreceive[0];
  for (int i = 1; i < nproc; i++)
  {
    recvCount     += nreceive[i];
  }
  vector<bodyStruct> recv_buffer3(recvCount);
  double t93=get_time();

  int recvOffset = 0;

  static MPI_Status stat;
  for (int dist = 1; dist < nproc; dist++)
  {
    const int src = (nproc + myid - dist) % nproc;
    const int dst = (nproc + myid + dist) % nproc;
    const int scount = nsendbytes[dst];
    const int rcount = nreceive[src]*sizeof(bodyStruct);
    if ((myid/dist) & 1)
    {

      if (scount > 0) MPI_Send(&array2Send[nsendDispls[dst]/sizeof(bodyStruct)], scount, MPI_BYTE, dst, 1, MPI_COMM_WORLD);
      if (rcount > 0) MPI_Recv(&recv_buffer3[recvOffset], rcount, MPI_BYTE   , src, 1, MPI_COMM_WORLD, &stat);

      recvOffset +=  nreceive[src];
    }
    else
    {
      if (rcount > 0) MPI_Recv(&recv_buffer3[recvOffset], rcount, MPI_BYTE   , src, 1, MPI_COMM_WORLD, &stat);
      if (scount > 0) MPI_Send(&array2Send[nsendDispls[dst]/sizeof(bodyStruct)], scount, MPI_BYTE, dst, 1, MPI_COMM_WORLD);

      recvOffset +=  nreceive[src];
    }
  }


  //  delete[] nsendbytes;
  //  delete[] nrecvbytes;
  //  delete[] nreceive;
  //  delete[] nsendDispls;
  //  delete[] nrecvDispls;
  double t94 = get_time();

#elif 1

  double t91 = get_time();
  MPI_Alltoall(nparticles, 1, MPI_INT, nreceive, 1, MPI_INT, MPI_COMM_WORLD);



  double t92 = get_time();
  unsigned int recvCount  = nreceive[0];

  for (int i = 1; i < nproc; i++)
  {
    recvCount     += nreceive[i];
  }
  vector<bodyStruct> recv_buffer3(recvCount);
  double t93=get_time();

  int recvOffset = 0;

#define NMAXPROC 32768
  static MPI_Status stat[NMAXPROC];
  static MPI_Request req[NMAXPROC*2];

  int nreq = 0;
  for (int dist = 1; dist < nproc; dist++)
  {
    const int src    = (nproc + myid - dist) % nproc;
    const int dst    = (nproc + myid + dist) % nproc;
    const int scount = nsendbytes[dst];
    const int rcount = nreceive[src]*sizeof(bodyStruct);

#if 1
    if (scount > 0) MPI_Isend(&array2Send[nsendDispls[dst]/sizeof(bodyStruct)], scount, MPI_BYTE, dst, 1, MPI_COMM_WORLD, &req[nreq++]);
    if(rcount > 0)
    {
      MPI_Irecv(&recv_buffer3[recvOffset], rcount, MPI_BYTE, src, 1, MPI_COMM_WORLD, &req[nreq++]);
      recvOffset += nreceive[src];
    }
#else

    MPI_Status stat;
    MPI_Sendrecv(&array2Send[nsendDispls[dst]/sizeof(bodyStruct)],
        scount, MPI_BYTE, dst, 1,
        &recv_buffer3[recvOffset], rcount, MPI_BYTE, src, 1, MPI_COMM_WORLD, &stat);
    recvOffset += nreceive[src];
#endif	
  }

  double t94 = get_time();
  MPI_Waitall(nreq, req, stat);

  LOGF(stderr, "EXCHANGE comm iter: %d  a2asize: %lg alloc: %lg data-start: %lg data-wait: %lg \n",
      iter, t92-t91, t93-t92, t94-t93, get_time() -t94);

#else

  //Allocate two times the amount of memory of that which we send
  vector<bodyStruct> recv_buffer3(nbody*2);
  unsigned int recvCount = 0;

  //Exchange the data with the other processors
  int ibend = -1;
  int nsend;
  int isource = 0;
  for(int ib=nproc-1;ib>0;ib--)
  {
    int ibox = (ib+myid)%nproc; //index to send...

    if (ib == nproc-1)
    {
      isource= (myid+1)%nproc;
    }
    else
    {
      isource = (isource+1)%nproc;
      if (isource == myid)isource = (isource+1)%nproc;
    }


    if(MP_exchange_particle_with_overflow_check<bodyStruct>(ibox, &array2Send[0],
          recv_buffer3, firstloc[ibox] - nparticles[myid],
          nparticles[ibox], isource,
          nsend, recvCount))
    {
      ibend = ibox; //Here we get if exchange failed
      ib = 0;
    }//end if mp exchang
  }//end for all boxes


  if(ibend == -1){

  }else{
    //Something went wrong
    cerr << "ERROR in exchange_particles_with_overflow_check! \n"; exit(0);
  }
#endif

  LOGF(stderr,"Required inter-process communication time: %lg ,proc: %d\n", get_time()-t1, myid);

  //Compute the new number of particles:
  int newN = tree.n + recvCount - nToSend;

#if 1
  if(iter < 1){ /* jb2404 */
    fprintf(stderr, "Proc: %d Exchange, received %d \tSend: %d newN: %d\n",
        procId, recvCount, nToSend, newN); 
  }
#endif

  execStream->sync();   //make certain that the particle movement on the device
  //is complete before we resize

  //Allocate 10% extra if we have to allocate, to reduce the total number of
  //memory allocations
  int memSize = newN;
  if(tree.bodies_acc0.get_size() < newN)
    memSize = newN * MULTI_GPU_MEM_INCREASE;

  LOGF(stderr,"Going to allocate memory for %d particles \n", newN);

  //Have to resize the bodies vector to keep the numbering correct
  //but do not reduce the size since we need to preserve the particles
  //in the over sized memory
  tree.bodies_pos. cresize(memSize + 1, false);
  tree.bodies_acc0.cresize(memSize,     false);
  tree.bodies_acc1.cresize(memSize,     false);
  tree.bodies_vel. cresize(memSize,     false);
  tree.bodies_time.cresize(memSize,     false);
  tree.bodies_ids. cresize(memSize + 1, false);
  tree.bodies_Ppos.cresize(memSize + 1, false);
  tree.bodies_Pvel.cresize(memSize + 1, false);
  tree.bodies_key. cresize(memSize + 1, false);

  //This one has to be at least the same size as the number of particles in order to
  //have enough space to store the other buffers
  //Can only be resized after we are done since we still have
  //parts of memory pointing to that buffer (extractList)
  //Note that we allocate some extra memory to make everything texture/memory aligned
  tree.generalBuffer1.cresize_nocpy(3*(memSize)*4 + 4096, false);


  //Now we have to copy the data in batches in case the generalBuffer1 is not large enough
  //Amount we can store:
  int spaceInIntSize    = 3*(memSize)*4;
  int stepSize          = spaceInIntSize / (sizeof(bodyStruct) / sizeof(int));

  my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);

  int memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1,
      stepSize, 0);

  //  fprintf(stderr, "Exchange, received %d \tSend: %d newN: %d\tItems that can be insert in one step: %d\n",
  //      recvCount, nToSend, newN, stepSize);
  LOGF(stderr, "Exchange, received %d \tSend: %d newN: %d\tItems that can be insert in one step: %d\n",
      recvCount, nToSend, newN, stepSize);

  int insertOffset = 0;
  for(unsigned int i=0; i < recvCount; i+= stepSize)
  {
    int items = min(stepSize, (int)(recvCount-i));

    if(items > 0)
    {
      //Copy the data from the MPI receive buffers into the GPU-send buffer
      memcpy(&bodyBuffer[0], &recv_buffer3[insertOffset], sizeof(bodyStruct)*items);

      bodyBuffer.h2d(items);

      //Start the kernel that puts everything in place
      insertNewParticlesSFC.set_arg<int>(0,    &nToSend);
      insertNewParticlesSFC.set_arg<int>(1,    &items);
      insertNewParticlesSFC.set_arg<int>(2,    &tree.n);
      insertNewParticlesSFC.set_arg<int>(3,    &insertOffset);
      insertNewParticlesSFC.set_arg<cl_mem>(4, localTree.bodies_Ppos.p());
      insertNewParticlesSFC.set_arg<cl_mem>(5, localTree.bodies_Pvel.p());
      insertNewParticlesSFC.set_arg<cl_mem>(6, localTree.bodies_pos.p());
      insertNewParticlesSFC.set_arg<cl_mem>(7, localTree.bodies_vel.p());
      insertNewParticlesSFC.set_arg<cl_mem>(8, localTree.bodies_acc0.p());
      insertNewParticlesSFC.set_arg<cl_mem>(9, localTree.bodies_acc1.p());
      insertNewParticlesSFC.set_arg<cl_mem>(10, localTree.bodies_time.p());
      insertNewParticlesSFC.set_arg<cl_mem>(11, localTree.bodies_ids.p());
      insertNewParticlesSFC.set_arg<cl_mem>(12, localTree.bodies_key.p());
      insertNewParticlesSFC.set_arg<cl_mem>(13, bodyBuffer.p());
      insertNewParticlesSFC.setWork(items, 128);
      insertNewParticlesSFC.execute(execStream->s());
    }

    insertOffset += items;
  }

  //   printf("Required gpu malloc time step 1: %lg \t Size: %d \tRank: %d \t Size: %d \n",
  //          get_time()-t1, newN, mpiGetRank(), tree.bodies_Ppos.get_size());
  //   t1 = get_time();
  tree.setN(newN);

  //Resize the arrays of the tree
  reallocateParticleMemory(tree);

  //   printf("Required gpu malloc time step 2: %lg \n", get_time()-t1);
  //   printf("Total GPU interaction time: %lg \n", get_time()-t2);

#endif
  int retValue = 0;

  //  delete[] firstloc;
  //  delete[] nparticles;

  return retValue;
}



//Functions related to the LET Creation and Exchange

//Broadcast the group-tree structure (used during the LET creation)
//First we gather the size, so we can create/allocate memory
//and then we broad-cast the final structure
//This basically is a sync-operation and therefore can be quite costly
//Maybe we want to do this in a separate thread
/****** EGABUROV ****/
void octree::sendCurrentInfoGrpTree()
{
#ifdef USE_MPI

#if 1  /* new group code, EGABUROV *****/
  localTree.boxSizeInfo.waitForCopyEvent();
  localTree.boxCenterInfo.waitForCopyEvent();

  std::vector<real4> groupCentre, groupSize;
  extractGroups(
      groupCentre, groupSize,
      &localTree.boxCenterInfo[0],
      &localTree.boxSizeInfo[0],
      localTree.level_list[localTree.startLevelMin].x,
      localTree.level_list[localTree.startLevelMin].y,
      localTree.n_nodes);

  groupCentre.insert(groupCentre.end(), groupSize.begin(), groupSize.end());

  int nGroups = groupCentre.size();
  LOGF(stderr, "ExtractGroups n: %d [%d]  size= %f %f %f  cnt= %f %f %f \n",
      nGroups, (int)groupSize.size(),
      groupSize[0].x, groupSize[0].y, groupSize[0].z,
      groupCentre[0].x, groupCentre[0].y, groupCentre[0].z);


  std::vector<int> globalSizeArray(nProcs), displacement(nProcs,0);
  MPI_Allgather(&nGroups,  sizeof(int), MPI_BYTE,
      &globalSizeArray[0], sizeof(int), MPI_BYTE, MPI_COMM_WORLD); /* to globalSize Array */

  int runningOffset = 0;
  for (int i = 0; i < nProcs; i++)
  {
    this->globalGrpTreeCount[i] = globalSizeArray[i];
    this->globalGrpTreeOffsets[i] = runningOffset;
    fullGrpAndLETRequest[i] = 0;

    displacement[i] = runningOffset*sizeof(real4);

    runningOffset += globalSizeArray[i];
    globalSizeArray[i] *= sizeof(real4);
  }



  if (globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
  const int totalNumberOfGroups = runningOffset;  /*check if defined */
  globalGrpTreeCntSize = new real4[totalNumberOfGroups]; /* totalNumberOfGroups = 2*nGroups_recvd */

  /* compute displacements for allgatherv */
  MPI_Allgatherv(
      &groupCentre[0], sizeof(real4)*nGroups, MPI_BYTE,
      globalGrpTreeCntSize, &globalSizeArray[0], &displacement[0], MPI_BYTE,
      MPI_COMM_WORLD);



#elif 1
  /*
     als ontvangst van alltoall positief, sturen we de size van de kleine tree
     als negatief is het ook de size en geven we aan dat we ook de
     volledige tree willen (vergelijk baar met een sendReq van 1 in de vorige code)

     daarna all gather met de psotieve size en offsets om alles te ontvange

     daarna voor de negatieve doen we de volledige tree, maar daarvoor hebben we size
     voor nodig. Oke werkt niet

     gebruikt uint2 voor sendeq

     uint2.x = size van de top tree. Als positief, dan heben we alleen dit nodig. Als negatief willen we volledige tree
     uint2.y = size van de volledige tree.

     na de all2all doen we een all_gatherv, voor de kleine tree die iedereen krijgt
     daarna isend/irecv zoals in domain exchange voor de volledige tree.
     Hopelijk is dat sneller dan de huidige methode
     */

  if(nProcs > NUMBER_OF_FULL_EXCHANGE)
  {
    //Sort the data and take the top NUMBER_OF_FULL_EXCHANGE to be used
    std::partial_sort(fullGrpAndLETRequestStatistics,
        fullGrpAndLETRequestStatistics+NUMBER_OF_FULL_EXCHANGE, //Top items
        fullGrpAndLETRequestStatistics+nProcs,
        cmp_uint2_reverse());

    //Set the top NUMBER_OF_FULL_EXCHANGE to be active
    memset(fullGrpAndLETRequest, 0, sizeof(int)*nProcs);
    for(int i=0; i < NUMBER_OF_FULL_EXCHANGE; i++)
    {
      fullGrpAndLETRequest[fullGrpAndLETRequestStatistics[i].y] = 1;
    }
  }
  else
  {
    //Set everything active, except ourself
    for(int i=0; i < nProcs; i++)
      fullGrpAndLETRequest[i] = 1;
    fullGrpAndLETRequest[procId] = 0;
  }


  std::vector<int2> sendReq2(nProcs);
  std::vector<int2> recvReq2(nProcs);

  //Set the sizes and indicate what we need and don't need
  //if .x is negative, we need the full tree
  for(int i=0; i < nProcs; i++)
  {
    sendReq2[i].x = grpTree_n_topNodes*2;
    sendReq2[i].y = grpTree_n_nodes*2;
    if(fullGrpAndLETRequest[i] == 1) sendReq2[i].x = -1*sendReq2[i].x;
  }
  //Do the all2all


  double t00 = get_time();
  //gather the requests
  MPI_Alltoall(&sendReq2[0], 1*sizeof(int2), MPI_BYTE,
      &recvReq2[0], 1*sizeof(int2), MPI_BYTE,
      MPI_COMM_WORLD);
  double t10 = get_time();

  //Compute offsets, sizes, displacements, etc
  unsigned int allGatherRecvOffset = 0;

  unsigned int nWantFullTreeCount = 0;
  unsigned int nWantFullTreeList[nProcs];

  int *allGatherRecvSizeBytes = &infoGrpTreeBuffer[0*nProcs];
  int *allGatherRecvDispBytes = &infoGrpTreeBuffer[1*nProcs];

  for(int i=0; i < nProcs; i++)
  {
    if(recvReq2[i].x < 0)
    {
      nWantFullTreeList[nWantFullTreeCount++] = i;
    }

    //The size of the all gather is always the same
    allGatherRecvSizeBytes[i] = abs(recvReq2[i].x)   * sizeof(real4);

    /* egaburov */
    if(fullGrpAndLETRequest[i] == 1)
    {
      //Fill in the full info for later, eventhough we first receive
      //the small tree
      this->globalGrpTreeCount[i]   = recvReq2[i].y;  /* egaburov, this stores , it is used by getLETopt */
      this->globalGrpTreeOffsets[i] = allGatherRecvOffset;

      allGatherRecvDispBytes[i]     = allGatherRecvOffset * sizeof(real4);
      allGatherRecvOffset          += recvReq2[i].y;
    }
    /* egaburov */
    else
    {
      //Fill in the small-tree info, which is enough for us
      this->globalGrpTreeCount[i]   = abs(recvReq2[i].x);
      this->globalGrpTreeOffsets[i] = allGatherRecvOffset;

      allGatherRecvDispBytes[i]     = allGatherRecvOffset * sizeof(real4);
      allGatherRecvOffset          += abs(recvReq2[i].x);
    }
  } //for i < nProcs


  //Allocate memory
  if(globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
  globalGrpTreeCntSize = new real4[allGatherRecvOffset];

  double t30 = get_time();


  //Do the all gather
  MPI_Allgatherv(&localGrpTreeCntSize[0],                     //Begin of array
      sizeof(real4)*(2*grpTree_n_topNodes),        //Number of top-nodes
      MPI_BYTE,
      globalGrpTreeCntSize,                        //Receive buffer
      allGatherRecvSizeBytes,                      //Array with size per node
      allGatherRecvDispBytes,                      //Array with offset per node
      MPI_BYTE, MPI_COMM_WORLD);

  double t40 = get_time();

  //Next do the non-blocking send and receives
#define NMAXPROC 32768
  static MPI_Status stat[NMAXPROC];
  static MPI_Request req[NMAXPROC*2];

  int nreq = 0;

  //The sends
  for(int i=0; i < nWantFullTreeCount; i++)
  {
    int dst  = nWantFullTreeList[i];
    int size = sizeof(real4)*2*grpTree_n_nodes; //Times two it is size and center in one
    //LOGF(stderr, "Sending full to: %d size: %d \n", dst, size);
    MPI_Isend(&localGrpTreeCntSize[2*grpTree_n_topNodes], size,
        MPI_BYTE, dst, 42, MPI_COMM_WORLD, &req[nreq++]);
  }

  //The receives
  for(int i=0; i < min(NUMBER_OF_FULL_EXCHANGE, nProcs); i++)
  {
    if(fullGrpAndLETRequestStatistics[i].y == procId) continue;
    int src    = fullGrpAndLETRequestStatistics[i].y;
    int size   = this->globalGrpTreeCount[src]   * sizeof(real4);
    int offset = this->globalGrpTreeOffsets[src];

    //LOGF(stderr, "Receiving full %d from: %d size: %d  Offset: %d\n", i, src, size, offset);
    MPI_Irecv(&globalGrpTreeCntSize[offset], size, MPI_BYTE,
        src, 42, MPI_COMM_WORLD, &req[nreq++]);
  }

  double t50 = get_time();
  MPI_Waitall(nreq, req, stat);
  double t60 = get_time();

  LOGF(stderr, "Gathering Grp-Tree timings, request: %lg allocs: %lg AllGather: %lg Sends: %lg Wait: %lg Total: %lg NGroups: %d\n",
      t10-t00, t30-t10, t40-t30, t50-t40, t60-t50, t60-t00, allGatherRecvOffset / 2);

#elif 1

  //Per process a request for a full node or only the topnode

  int *sendReq           = &infoGrpTreeBuffer[0*nProcs];
  int *recvReq           = &infoGrpTreeBuffer[1*nProcs];
  int *incomingDataSizes = &infoGrpTreeBuffer[2*nProcs];
  int *sendDisplacement  = &infoGrpTreeBuffer[3*nProcs];
  int *sendCount         = &infoGrpTreeBuffer[4*nProcs];
  int *recvDisplacement  = &infoGrpTreeBuffer[5*nProcs];
  int *recvSizeBytes     = &infoGrpTreeBuffer[6*nProcs];

  if(nProcs > NUMBER_OF_FULL_EXCHANGE)
  {
    //Sort the data and take the top NUMBER_OF_FULL_EXCHANGE to be used
    std::partial_sort(fullGrpAndLETRequestStatistics,
        fullGrpAndLETRequestStatistics+NUMBER_OF_FULL_EXCHANGE, //Top items
        fullGrpAndLETRequestStatistics+nProcs,
        cmp_uint2_reverse());

    //Set the top NUMBER_OF_FULL_EXCHANGE to be active
    memset(fullGrpAndLETRequest, 0, sizeof(int)*nProcs);
    for(int i=0; i < NUMBER_OF_FULL_EXCHANGE; i++)
    {
      fullGrpAndLETRequest[fullGrpAndLETRequestStatistics[i].y] = 1;
    }
  }
  else
  {
    //Set everything active, except ourself
    for(int i=0; i < nProcs; i++)
      fullGrpAndLETRequest[i] = 1;
    fullGrpAndLETRequest[procId] = 0;
  }

  memcpy(sendReq, fullGrpAndLETRequest, sizeof(int)*nProcs);


  double t00 = get_time();
  //gather the requests
  MPI_Alltoall(sendReq, 1, MPI_INT, recvReq, 1, MPI_INT, MPI_COMM_WORLD);
  double t10 = get_time();


  //Debug print
  //    char buff[512];
  //    sprintf(buff, "%d A:\t", procId);
  //
  //    {
  //      for(int i=0; i < nProcs; i++)
  //      {
  //        sprintf(buff, "%s%d\t",buff, recvReq[i]);
  //      }
  //      LOGF(stderr, "%s\n", buff);
  //    }

  //We now know which process requires the full-tree (recvReq[process] == 1)
  //and which process can get away with only the top-node (recvReq[process] == 0)

  int outGoingFullRequests = 0;
  //Set the memory sizes, reuse sendReq, also directly set memory displacements, we need those anyway
  for(int i=0; i < nProcs; i++)
  {
    if(recvReq[i] == 1)
    {
      sendReq[i]          = 2*grpTree_n_nodes; //Times two since we send size and center in one array
      sendDisplacement[i] = 2*sizeof(real4)*grpTree_n_topNodes;  //Jump 2 ahead, first 2 is top-node only
      outGoingFullRequests++;                 //Count how many full-trees we send. That is how many
      //direct receives we expect in LET phase
    }
    else
    {
      if(procId != i)
      {
        sendReq[i]          = 2*grpTree_n_topNodes;   //2 times a float4
        sendDisplacement[i] = 0;
      }
      else
      {
        //Make sure to not include ourself
        sendReq[i] = 0; sendDisplacement[i] = 0;
      }
    }
  }
  //Send the memory sizes
  MPI_Alltoall(sendReq, 1, MPI_INT, incomingDataSizes, 1, MPI_INT, MPI_COMM_WORLD);
  double t20 = get_time();
  //Debug print
  //    sprintf(buff, "%d B:\t", procId);
  //    {
  //      for(int i=0; i < nProcs; i++)
  //      {
  //        sprintf(buff, "%s%d\t",buff, incomingDataSizes[i]);
  //      }
  //      LOGF(stderr, "%s\n", buff);
  //    }

  //Compute memory offsets, for the receive
  unsigned int recvCount  = incomingDataSizes[0];
  recvDisplacement[0]     = 0;
  sendReq[0]              = sendReq[0]*sizeof(real4);
  recvSizeBytes [0]       = incomingDataSizes[0]*sizeof(real4);

  this->globalGrpTreeCount[0]   = incomingDataSizes[0];
  this->globalGrpTreeOffsets[0] = 0;

  for(int i=1; i < nProcs; i++)
  {
    sendReq[i]          = sendReq[i]*sizeof(real4);
    recvSizeBytes [i]   = incomingDataSizes[i]*sizeof(real4);
    recvDisplacement[i] = recvDisplacement[i-1] + recvSizeBytes [i-1];
    recvCount          += incomingDataSizes[i];

    this->globalGrpTreeCount[i]   = incomingDataSizes[i];
    this->globalGrpTreeOffsets[i] = this->globalGrpTreeOffsets[i-1] + this->globalGrpTreeCount[i-1];
  }

  //Debug print
  //    sprintf(buff, "%d C:\t", procId);
  //    {
  //      for(int i=0; i < nProcs; i++)
  //      {
  //        sprintf(buff, "%s[%d,%d]\t",buff, recvSizeBytes[i],recvDisplacement[i]);
  //      }
  //      LOGF(stderr, "%s\n", buff);
  //    }

  //Allocate memory
  if(globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
  globalGrpTreeCntSize = new real4[recvCount];

  double t30 = get_time();

  //Compute how much we will receive and the offsets and displacements

  assert(0);
#if 0
  MPI_Alltoallv(&localGrpTreeCntSize[0], sendReq, sendDisplacement, MPI_BYTE,
      &globalGrpTreeCntSize[0], recvSizeBytes, recvDisplacement, MPI_BYTE,
      MPI_COMM_WORLD);
#else
  myComm->ugly_all2allv_char((float*)&localGrpTreeCntSize[0], sendReq, (float*)&globalGrpTreeCntSize[0], recvSizeBytes);
#endif
  double t40 = get_time();

  LOGF(stderr, "Gathering Grp-Tree timings, request: %lg size: %lg MemOffset: %lg data: %lg Total: %lg NGroups: %d\n",
      t10-t00, t20-t10, t30-t20, t40-t30, t40-t00, recvCount);

  //    exit(0);
#elif 1
  //Send the full groups to all processes
  int *treeGrpCountBytes   = new int[nProcs];
  int *receiveOffsetsBytes = new int[nProcs];

  double t0 = get_time();
  //Send the number of group-tree-nodes that belongs to this process, and gather
  //that information from the other processors
  int temp = 2*grpTree_n_nodes; //Times two since we send size and center in one array
  if(grpTree_n_nodes == 0) temp = 1;
  MPI_Allgather(&temp,                    sizeof(int),  MPI_BYTE,
      this->globalGrpTreeCount, sizeof(uint), MPI_BYTE, MPI_COMM_WORLD);

  double tSize = get_time()-t0;


  //Compute offsets using prefix sum and total number of groups we will receive
  this->globalGrpTreeOffsets[0]   = 0;
  treeGrpCountBytes[0]            = this->globalGrpTreeCount[0]*sizeof(real4);
  receiveOffsetsBytes[0]          = 0;
  for(int i=1; i < nProcs; i++)
  {
    this->globalGrpTreeOffsets[i]  = this->globalGrpTreeOffsets[i-1] + this->globalGrpTreeCount[i-1];

    treeGrpCountBytes[i]   = this->globalGrpTreeCount[i]  *sizeof(real4);
    receiveOffsetsBytes[i] = this->globalGrpTreeOffsets[i]*sizeof(real4);

    //      LOGF(stderr,"Proc: %d Received on idx: %d\t%d prefix: %d \n", procId, i, globalGrpTreeCount[i], globalGrpTreeOffsets[i]);
  }

  int totalNumberOfGroups = this->globalGrpTreeOffsets[nProcs-1]+this->globalGrpTreeCount[nProcs-1];

  //Allocate memory
  if(globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
  globalGrpTreeCntSize = new real4[totalNumberOfGroups];

  double t2 = get_time();
  //Exchange the coarse group boundaries
  MPI_Allgatherv(&localGrpTreeCntSize[2],  temp*sizeof(real4), MPI_BYTE,
      globalGrpTreeCntSize, treeGrpCountBytes,
      receiveOffsetsBytes,  MPI_BYTE, MPI_COMM_WORLD);

  LOGF(stderr, "Gathering Grp-Tree timings, size: %lg data: %lg Total: %lg NGroups: %d\n",
      tSize, get_time()-t2, get_time()-t0, totalNumberOfGroups);

  delete[] treeGrpCountBytes;
  delete[] receiveOffsetsBytes;


#else
  //Send the full groups to all processes
  int *treeGrpCountBytes   = new int[nProcs];
  int *receiveOffsetsBytes = new int[nProcs];

  double t0 = get_time();
  //Send the number of group-tree-nodes that belongs to this process, and gather
  //that information from the other processors
  int temp = 2*grpTree_n_nodes; //Times two since we send size and center in one array
  if(grpTree_n_nodes == 0) temp = 1;
  MPI_Allgather(&temp,                    sizeof(int),  MPI_BYTE,
      this->globalGrpTreeCount, sizeof(uint), MPI_BYTE, MPI_COMM_WORLD);

  double tSize = get_time()-t0;


  //Compute offsets using prefix sum and total number of groups we will receive
  this->globalGrpTreeOffsets[0]   = 0;
  treeGrpCountBytes[0]            = this->globalGrpTreeCount[0]*sizeof(real4);
  receiveOffsetsBytes[0]          = 0;
  for(int i=1; i < nProcs; i++)
  {
    this->globalGrpTreeOffsets[i]  = this->globalGrpTreeOffsets[i-1] + this->globalGrpTreeCount[i-1];

    treeGrpCountBytes[i]   = this->globalGrpTreeCount[i]  *sizeof(real4);
    receiveOffsetsBytes[i] = this->globalGrpTreeOffsets[i]*sizeof(real4);

    //      LOGF(stderr,"Proc: %d Received on idx: %d\t%d prefix: %d \n", procId, i, globalGrpTreeCount[i], globalGrpTreeOffsets[i]);
  }

  int totalNumberOfGroups = this->globalGrpTreeOffsets[nProcs-1]+this->globalGrpTreeCount[nProcs-1];

  //Allocate memory
  if(globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
  globalGrpTreeCntSize = new real4[totalNumberOfGroups];

  double t2 = get_time();
  //Exchange the coarse group boundaries
  MPI_Allgatherv(localGrpTreeCntSize,  temp*sizeof(real4), MPI_BYTE,
      globalGrpTreeCntSize, treeGrpCountBytes,
      receiveOffsetsBytes,  MPI_BYTE, MPI_COMM_WORLD);

  LOGF(stderr, "Gathering Grp-Tree timings, size: %lg data: %lg Total: %lg NGroups: %d\n",
      tSize, get_time()-t2, get_time()-t0, totalNumberOfGroups);

  delete[] treeGrpCountBytes;
  delete[] receiveOffsetsBytes;
#endif


#else
  //TODO check if we need something here
  //  curSysState[0] = curProcState;
#endif
}



//////////////////////////////////////////////////////
// ***** Local essential tree functions ************//
//////////////////////////////////////////////////////

inline int split_node_grav_impbh(
    const _v4sf nodeCOM1,
    const _v4sf boxCenter1,
    const _v4sf boxSize1)
{
  const _v4si mask = {0xffffffff, 0xffffffff, 0xffffffff, 0x0};
  const _v4sf size = __abs(__builtin_ia32_shufps(nodeCOM1, nodeCOM1, 0xFF));

  //mask to prevent NaN signalling / Overflow ? Required to get good pre-SB performance
  const _v4sf nodeCOM   = __builtin_ia32_andps(nodeCOM1,   (_v4sf)mask);
  const _v4sf boxCenter = __builtin_ia32_andps(boxCenter1, (_v4sf)mask);
  const _v4sf boxSize   = __builtin_ia32_andps(boxSize1,   (_v4sf)mask);


  const _v4sf dr   = __abs(boxCenter - nodeCOM) - boxSize;
  const _v4sf ds   = dr + __abs(dr);
  const _v4sf dsq  = ds*ds;
  const _v4sf t1   = __builtin_ia32_haddps(dsq, dsq);
  const _v4sf t2   = __builtin_ia32_haddps(t1, t1);
  const _v4sf ds2  = __builtin_ia32_shufps(t2, t2, 0x00)*(_v4sf){0.25f, 0.25f, 0.25f, 0.25f};


#if 1
  const float c = 10e-4f;
  const int res = __builtin_ia32_movmskps(
      __builtin_ia32_orps(
        __builtin_ia32_cmpleps(ds2,  size),
        __builtin_ia32_cmpltps(ds2 - size, (_v4sf){c,c,c,c})
        )
      );
#else
  const int res = __builtin_ia32_movmskps(
      __builtin_ia32_cmpleps(ds2,  size));
#endif

  return res;
}


template<typename T, int STRIDE>
void shuffle2vec(
    std::vector<T> &data1,
    std::vector<T> &data2)
{
  const int n = data1.size();
  assert(n%STRIDE == 0);
  std::vector<int> keys(n/STRIDE);
  for (int i = 0, idx=0; i < n; i += STRIDE, idx++)
    keys[idx] = i;
  std::random_shuffle(keys.begin(), keys.end());

  std::vector<T> rdata1(n), rdata2(n);
  for (int i = 0, idx=0; i < n; i += STRIDE, idx++)
  {
    const int key = keys[idx];
    for (int j = 0; j < STRIDE; j++)
    {
      rdata1[i+j] = data1[key+j];
      rdata2[i+j] = data2[key+j];
    }
  }

  data1.swap(rdata1);
  data2.swap(rdata2);
}
template<bool TRANSPOSE>
inline int split_node_grav_impbh_box4simd1( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  ncx,
    const _v4sf  ncy,
    const _v4sf  ncz,
    const _v4sf  size,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4])
{

  _v4sf bcx =  (boxCenter[0]);
  _v4sf bcy =  (boxCenter[1]);
  _v4sf bcz =  (boxCenter[2]);
  _v4sf bcw =  (boxCenter[3]);

  _v4sf bsx =  (boxSize[0]);
  _v4sf bsy =  (boxSize[1]);
  _v4sf bsz =  (boxSize[2]);
  _v4sf bsw =  (boxSize[3]);

  if (TRANSPOSE)
  {
    _v4sf_transpose(bcx, bcy, bcz, bcw);
    _v4sf_transpose(bsx, bsy, bsz, bsw);
  }

  const _v4sf zero = {0.0, 0.0, 0.0, 0.0};

  _v4sf dx = __abs(bcx - ncx) - bsx;
  _v4sf dy = __abs(bcy - ncy) - bsy;
  _v4sf dz = __abs(bcz - ncz) - bsz;

  dx = __builtin_ia32_maxps(dx, zero);
  dy = __builtin_ia32_maxps(dy, zero);
  dz = __builtin_ia32_maxps(dz, zero);

  const _v4sf ds2 = dx*dx + dy*dy + dz*dz;
#if 0
  const float c = 10e-4;
  const int ret = __builtin_ia32_movmskps(
      __builtin_ia32_orps(
        __builtin_ia32_cmpleps(ds2,  size),
        __builtin_ia32_cmpltps(ds2 - size, (_v4sf){c,c,c,c})
        )
      );
#else
  const int ret = __builtin_ia32_movmskps(
      __builtin_ia32_cmpleps(ds2, size));
#endif
  return ret;
}
int2 getLET1(
    real4 **LETBuffer_ptr,
    const real4 *nodeCentre,
    const real4 *nodeSize,
    const real4 *multipole,
    const int cellBeg,
    const int cellEnd,
    const real4 *bodies,
    const int nParticles,
    const real4 *groupSizeInfo,
    const real4 *groupCentreInfo,
    const int nGroups,
    const int nNodes,
    unsigned long long &nflops)
{
  std::vector<int2> LETBuffer_node;
  std::vector<int > LETBuffer_ptcl;
  LETBuffer_node.reserve(nNodes);
  LETBuffer_ptcl.reserve(nParticles);

  nflops = 0;

  int nExportPtcl = 0;
  int nExportCell = 0;
  int nExportCellOffset = cellEnd;

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});


  const _v4sf*            bodiesV = (const _v4sf*)bodies;
  const _v4sf*          nodeSizeV = (const _v4sf*)nodeSize;
  const _v4sf*        nodeCentreV = (const _v4sf*)nodeCentre;
  const _v4sf*         multipoleV = (const _v4sf*)multipole;
  const _v4sf*   groupSizeV = (const _v4sf*)groupSizeInfo;
  const _v4sf* groupCenterV = (const _v4sf*)groupCentreInfo;



  const int levelCountMax = nNodes;
  std::vector<int> currLevelVec, nextLevelVec;
  currLevelVec.reserve(levelCountMax);
  nextLevelVec.reserve(levelCountMax);
  Swap<std::vector<int> > levelList(currLevelVec, nextLevelVec);

  const int SIMDW  = 4;

  const int nGroups4 = ((nGroups-1)/SIMDW + 1)*SIMDW;
  std::vector<v4sf> groupCentreSIMD(nGroups4), groupSizeSIMD(nGroups4);
#if 1
  const bool TRANSPOSE_SPLIT = false;
#else
  const bool TRANSPOSE_SPLIT = true;
#endif
  for (int ib = 0; ib < nGroups4; ib += SIMDW)
  {
    _v4sf bcx = groupCenterV[std::min(ib+0,nGroups-1)];
    _v4sf bcy = groupCenterV[std::min(ib+1,nGroups-1)];
    _v4sf bcz = groupCenterV[std::min(ib+2,nGroups-1)];
    _v4sf bcw = groupCenterV[std::min(ib+3,nGroups-1)];

    _v4sf bsx = groupSizeV[std::min(ib+0,nGroups-1)];
    _v4sf bsy = groupSizeV[std::min(ib+1,nGroups-1)];
    _v4sf bsz = groupSizeV[std::min(ib+2,nGroups-1)];
    _v4sf bsw = groupSizeV[std::min(ib+3,nGroups-1)];

    if (!TRANSPOSE_SPLIT)
    {
      _v4sf_transpose(bcx, bcy, bcz, bcw);
      _v4sf_transpose(bsx, bsy, bsz, bsw);
    }

    groupCentreSIMD[ib+0] = bcx;
    groupCentreSIMD[ib+1] = bcy;
    groupCentreSIMD[ib+2] = bcz;
    groupCentreSIMD[ib+3] = bcw;

    groupSizeSIMD[ib+0] = bsx;
    groupSizeSIMD[ib+1] = bsy;
    groupSizeSIMD[ib+2] = bsz;
    groupSizeSIMD[ib+3] = bsw;
  }

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);

  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
#if 1
    if (nGroups > 128)   /* randomizes algo, can give substantial speed-up */
      shuffle2vec<v4sf,SIMDW>(groupCentreSIMD, groupSizeSIMD);
#endif
    for (int i = 0; i < csize; i++)
    {
      const uint        nodeIdx  = levelList.first()[i];
      const float nodeInfo_x = nodeCentre[nodeIdx].w;
      const uint  nodeInfo_y = host_float_as_int(nodeSize[nodeIdx].w);

      _v4sf nodeCOM = multipoleV[nodeIdx*3];
      nodeCOM       = __builtin_ia32_vec_set_v4sf (nodeCOM, nodeInfo_x, 3);

      int split = false;

      /**************/


      const _v4sf vncx = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x00);
      const _v4sf vncy = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x55);
      const _v4sf vncz = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xaa);
      const _v4sf vncw = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xff);
      const _v4sf vsize = __abs(vncw);

      nflops += nGroups*20;  /* effective flops, can be less */
      for (int ib = 0; ib < nGroups4 && !split; ib += SIMDW)
        split |= split_node_grav_impbh_box4simd1<TRANSPOSE_SPLIT>(vncx,vncy,vncz,vsize, (_v4sf*)&groupCentreSIMD[ib], (_v4sf*)&groupSizeSIMD[ib]);

      /**************/

      real4 size  = nodeSize[nodeIdx];
      int sizew = 0xFFFFFFFF;

      if (split)
      {
        const bool lleaf = nodeInfo_x <= 0.0f;
        if (!lleaf)
        {
          const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
          const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has
          sizew = (nExportCellOffset | (lnchild << LEAFBIT));
          nExportCellOffset += lnchild;
          for (int i = lchild; i < lchild + lnchild; i++)
            levelList.second().push_back(i);
        }
        else
        {
          const int pfirst =    nodeInfo_y & BODYMASK;
          const int np     = (((nodeInfo_y & INVBMASK) >> LEAFBIT)+1);
          sizew = (nExportPtcl | ((np-1) << LEAFBIT));
          for (int i = pfirst; i < pfirst+np; i++)
            LETBuffer_ptcl.push_back(i);
          nExportPtcl += np;
        }
      }

      LETBuffer_node.push_back((int2){nodeIdx, sizew});
      nExportCell++;
    }

    levelList.swap();
    levelList.second().clear();
  }

  assert((int)LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)LETBuffer_node.size() == nExportCell);

  /* now copy data into LETBuffer */
  {
    //LETBuffer.resize(nExportPtcl + 5*nExportCell);
#pragma omp critical //Malloc seems to be not so thread safe..
    *LETBuffer_ptr = (real4*)malloc(sizeof(real4)*(1+ nExportPtcl + 5*nExportCell));
    real4 *LETBuffer = *LETBuffer_ptr;
    _v4sf *vLETBuffer      = (_v4sf*)(&LETBuffer[1]);
    //_v4sf *vLETBuffer      = (_v4sf*)&LETBuffer     [0];

    int nStoreIdx = nExportPtcl;
    int multiStoreIdx = nStoreIdx + 2*nExportCell;
    for (int i = 0; i < nExportPtcl; i++)
    {
      const int idx = LETBuffer_ptcl[i];
      vLETBuffer[i] = bodiesV[idx];
    }
    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = LETBuffer_node[i];
      const int idx = packed_idx.x;
      const float sizew = host_int_as_float(packed_idx.y);
      const _v4sf size = __builtin_ia32_vec_set_v4sf(nodeSizeV[idx], sizew, 3);

      //       vLETBuffer[nStoreIdx            ] = nodeCentreV[idx];     /* centre */
      //       vLETBuffer[nStoreIdx+nExportCell] = size;                 /*  size  */

      vLETBuffer[nStoreIdx+nExportCell] = nodeCentreV[idx];     /* centre */
      vLETBuffer[nStoreIdx            ] = size;                 /*  size  */

      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+0];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+1];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+2];  /* multipole.x */
      nStoreIdx++;
    }
  }

  return (int2){nExportCell, nExportPtcl};
}
int2 getLETopt(
    //std::vector<real4> &LETBuffer,
    real4 **LETBuffer_ptr,
    const real4 *nodeCentre,
    const real4 *nodeSize,
    const real4 *multipole,
    const int cellBeg,
    const int cellEnd,
    const real4 *bodies,
    const int nParticles,
    const real4 *groupSizeInfo,
    const real4 *groupCentreInfo,
    const int groupBeg,
    const int groupEnd,
    const int nNodes,
    unsigned long long &nflops)
{
  std::vector<int2> LETBuffer_node;
  std::vector<int > LETBuffer_ptcl;
  LETBuffer_node.reserve(nNodes);
  LETBuffer_ptcl.reserve(nParticles);

  nflops = 0;

  int nExportCell = 0;
  int nExportPtcl = 0;
  int nExportCellOffset = cellEnd;

  const _v4sf*            bodiesV = (const _v4sf*)bodies;
  const _v4sf*          nodeSizeV = (const _v4sf*)nodeSize;
  const _v4sf*        nodeCentreV = (const _v4sf*)nodeCentre;
  const _v4sf*         multipoleV = (const _v4sf*)multipole;
  const _v4sf*   grpNodeSizeInfoV = (const _v4sf*)groupSizeInfo;
  const _v4sf* grpNodeCenterInfoV = (const _v4sf*)groupCentreInfo;

  const int levelCountMax = nNodes;
  std::vector<uint4> currLevelVec, nextLevelVec;
  currLevelVec.reserve(levelCountMax);
  nextLevelVec.reserve(levelCountMax);
  Swap<std::vector<uint4> > levelList(currLevelVec, nextLevelVec);

  std::vector<int> currGroupLevelVec, nextGroupLevelVec;
  currGroupLevelVec.reserve(levelCountMax);
  nextGroupLevelVec.reserve(levelCountMax);
  Swap<std::vector<int> > levelGroups(currGroupLevelVec, nextGroupLevelVec);

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});


#if 0 /* AVX */
#ifndef __AVX__
#error "AVX is not defined"
#endif
  const int SIMDW  = 8;
  std::vector< std::pair<v4sf,v4sf> > groupSplitFlag;
#define AVXIMBH
#else
  const int SIMDW  = 4;
  std::vector<v4sf> groupSplitFlag;
#define SSEIMBH
#endif
  groupSplitFlag.reserve(levelCountMax);

  /* copy group info into current level buffer */
  for (int group = groupBeg; group < groupEnd; group++)
    levelGroups.first().push_back(group);

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back((uint4){cell, 0, (int)levelGroups.first().size(),0});

  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint4       nodePacked = levelList.first()[i];

      const uint  nodeIdx  = nodePacked.x;
      const float nodeInfo_x = nodeCentre[nodeIdx].w;
      const uint  nodeInfo_y = host_float_as_int(nodeSize[nodeIdx].w);

      const _v4sf nodeCOM  = __builtin_ia32_vec_set_v4sf(multipoleV[nodeIdx*3], nodeInfo_x, 3);
      const bool lleaf = nodeInfo_x <= 0.0f;

      const int groupBeg = nodePacked.y;
      const int groupEnd = nodePacked.z;
      nflops += 20*((groupEnd - groupBeg-1)/SIMDW+1)*SIMDW;

      groupSplitFlag.clear();
      for (int ib = groupBeg; ib < groupEnd; ib += SIMDW)
      {
        _v4sf centre[SIMDW], size[SIMDW];
        for (int laneIdx = 0; laneIdx < SIMDW; laneIdx++)
        {
          const int group = levelGroups.first()[std::min(ib+laneIdx, groupEnd-1)];
          centre[laneIdx] = grpNodeCenterInfoV[group];
          size  [laneIdx] =   grpNodeSizeInfoV[group];
        }
#ifdef AVXIMBH
        groupSplitFlag.push_back(split_node_grav_impbh_box8a(nodeCOM, centre, size));
#else
        groupSplitFlag.push_back(split_node_grav_impbh_box4a(nodeCOM, centre, size));
#endif
      }

      const int groupNextBeg = levelGroups.second().size();
      int split = false;
      for (int idx = groupBeg; idx < groupEnd; idx++)
      {
        const bool gsplit = ((uint*)&groupSplitFlag[0])[idx - groupBeg];
        if (gsplit)
        {
          split = true;
          const int group = levelGroups.first()[idx];
          if (!lleaf)
          {
            const bool gleaf = groupCentreInfo[group].w <= 0.0f;
            if (!gleaf)
            {
              const int childinfoGrp  = ((uint4*)groupSizeInfo)[group].w;
              const int gchild  =   childinfoGrp & 0x0FFFFFFF;
              const int gnchild = ((childinfoGrp & 0xF0000000) >> 28) ;
              for (int i = gchild; i <= gchild+gnchild; i++)
                levelGroups.second().push_back(i);
            }
            else
              levelGroups.second().push_back(group);
          }
          else
            break;
        }
      }

      real4 size  = nodeSize[nodeIdx];
      int sizew = 0xFFFFFFFF;

      if (split)
      {
        if (!lleaf)
        {
          const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
          const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has
          sizew = (nExportCellOffset | (lnchild << LEAFBIT));
          nExportCellOffset += lnchild;
          for (int i = lchild; i < lchild + lnchild; i++)
            levelList.second().push_back((uint4){i,groupNextBeg,(int)levelGroups.second().size()});
        }
        else
        {
          const int pfirst =    nodeInfo_y & BODYMASK;
          const int np     = (((nodeInfo_y & INVBMASK) >> LEAFBIT)+1);
          sizew = (nExportPtcl | ((np-1) << LEAFBIT));
          for (int i = pfirst; i < pfirst+np; i++)
            LETBuffer_ptcl.push_back(i);
          nExportPtcl += np;
        }
      }

      LETBuffer_node.push_back((int2){nodeIdx, sizew});
      nExportCell++;

    }
    levelList.swap();
    levelList.second().clear();

    levelGroups.swap();
    levelGroups.second().clear();
  }

  assert((int)LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)LETBuffer_node.size() == nExportCell);

  /* now copy data into LETBuffer */
  {
    //LETBuffer.resize(nExportPtcl + 5*nExportCell);
#pragma omp critical //Malloc seems to be not so thread safe..
    *LETBuffer_ptr = (real4*)malloc(sizeof(real4)*(1+ nExportPtcl + 5*nExportCell));
    real4 *LETBuffer = *LETBuffer_ptr;
    _v4sf *vLETBuffer      = (_v4sf*)(&LETBuffer[1]);
    //_v4sf *vLETBuffer      = (_v4sf*)&LETBuffer     [0];

    int nStoreIdx = nExportPtcl;
    int multiStoreIdx = nStoreIdx + 2*nExportCell;
    for (int i = 0; i < nExportPtcl; i++)
    {
      const int idx = LETBuffer_ptcl[i];
      vLETBuffer[i] = bodiesV[idx];
    }
    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = LETBuffer_node[i];
      const int idx = packed_idx.x;
      const float sizew = host_int_as_float(packed_idx.y);
      const _v4sf size = __builtin_ia32_vec_set_v4sf(nodeSizeV[idx], sizew, 3);

      //       vLETBuffer[nStoreIdx            ] = nodeCentreV[idx];     /* centre */
      //       vLETBuffer[nStoreIdx+nExportCell] = size;                 /*  size  */

      vLETBuffer[nStoreIdx+nExportCell] = nodeCentreV[idx];     /* centre */
      vLETBuffer[nStoreIdx            ] = size;                 /*  size  */

      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+0];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+1];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+2];  /* multipole.x */
      nStoreIdx++;
    }
  }

  return (int2){nExportCell, nExportPtcl};
}



//Compute PH key, same function as on device
static uint4 host_get_key(int4 crd)
{
  const int bits = 30;  //20 to make it same number as morton order
  int i,xi, yi, zi;
  int mask;
  int key;

  //0= 000, 1=001, 2=011, 3=010, 4=110, 5=111, 6=101, 7=100
  //000=0=0, 001=1=1, 011=3=2, 010=2=3, 110=6=4, 111=7=5, 101=5=6, 100=4=7
  const int C[8] = {0, 1, 7, 6, 3, 2, 4, 5};

  int temp;

  mask = 1 << (bits - 1);
  key  = 0;

  uint4 key_new;

  for(i = 0; i < bits; i++, mask >>= 1)
  {
    xi = (crd.x & mask) ? 1 : 0;
    yi = (crd.y & mask) ? 1 : 0;
    zi = (crd.z & mask) ? 1 : 0;

    int index = (xi << 2) + (yi << 1) + zi;

    if(index == 0)
    {
      temp = crd.z; crd.z = crd.y; crd.y = temp;
    }
    else  if(index == 1 || index == 5)
    {
      temp = crd.x; crd.x = crd.y; crd.y = temp;
    }
    else  if(index == 4 || index == 6)
    {
      crd.x = (crd.x) ^ (-1);
      crd.z = (crd.z) ^ (-1);
    }
    else  if(index == 7 || index == 3)
    {
      temp = (crd.x) ^ (-1);
      crd.x = (crd.y) ^ (-1);
      crd.y = temp;
    }
    else
    {
      temp = (crd.z) ^ (-1);
      crd.z = (crd.y) ^ (-1);
      crd.y = temp;
    }

    key = (key << 3) + C[index];

    if(i == 19)
    {
      key_new.y = key;
      key = 0;
    }
    if(i == 9)
    {
      key_new.x = key;
      key = 0;
    }
  } //end for

  key_new.z = key;

  return key_new;
}

typedef struct letObject
{
  real4       *buffer;
  int          size;
  int          destination;
#ifdef USE_MPI
  MPI_Request  req;
#endif
} letObject;



int octree::recursiveTopLevelCheck(uint4 checkNode, real4* treeBoxSizes, real4* treeBoxCenters, real4* treeBoxMoments,
    real4* grpCenter, real4* grpSize, int &DistanceCheck, int &DistanceCheckPP, int maxLevel)
{
  int nodeID = checkNode.x;
  int grpID  = checkNode.y;
  int endGrp = checkNode.z;

  real4 nodeCOM  = treeBoxMoments[nodeID*3];
  real4 nodeSize = treeBoxSizes  [nodeID];
  real4 nodeCntr = treeBoxCenters[nodeID];

  //     LOGF(stderr,"Checking node: %d grpID: %d endGrp: %d depth: %d \n",nodeID, grpID, endGrp, maxLevel);

  int res = maxLevel;

  nodeCOM.w = nodeCntr.w;
  for(int grp=grpID; grp < endGrp; grp++)
  {
    real4 grpcntr = grpCenter[grp];
    real4 grpsize = grpSize[grp];

    bool split = false;
    {
      //         DistanceCheck++;
      //         DistanceCheckPP++;
      //Compute the distance between the group and the cell
      float3 dr = make_float3(fabs((float)grpcntr.x - nodeCOM.x) - (float)grpsize.x,
          fabs((float)grpcntr.y - nodeCOM.y) - (float)grpsize.y,
          fabs((float)grpcntr.z - nodeCOM.z) - (float)grpsize.z);

      dr.x += fabs(dr.x); dr.x *= 0.5f;
      dr.y += fabs(dr.y); dr.y *= 0.5f;
      dr.z += fabs(dr.z); dr.z *= 0.5f;

      //Distance squared, no need to do sqrt since opening criteria has been squared
      float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

      if (ds2     <= fabs(nodeCOM.w))           split = true;
      if (fabs(ds2 - fabs(nodeCOM.w)) < 10e-04) split = true; //Limited precision can result in round of errors. Use this as extra safe guard

      //         LOGF(stderr,"Node: %d grp: %d  split: %d || %f %f\n", nodeID, grp, split, ds2, nodeCOM.w);
    }
    //       LOGF(stderr,"Node: %d grp: %d  split: %d Leaf: %d \t %d %f \n",nodeID, grp, split, nodeCntr.w <= 0,(host_float_as_int(nodeSize.w) == 0xFFFFFFFF), nodeCntr.w);

    if(split)
    {
      if(host_float_as_int(nodeSize.w) == 0xFFFFFFFF)
      {
        //We want to split, but then we go to deep. So we need a full tree-walk
        return -1;
      }

      int child, nchild;
      int childinfo = host_float_as_int(nodeSize.w);
      bool leaf = nodeCntr.w <= 0;

      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;           //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ; //The number of children this node has

        //Process node children
        for(int y=child; y < child+nchild; y++)
        {
          //Go one level deeper into the tree
          //uint4 nextCheck = make_uint4(y, grp, endGrp, 0);
          uint4 nextCheck = make_uint4(y, grpID, endGrp, 0);
          int res2 = recursiveTopLevelCheck(nextCheck, treeBoxSizes, treeBoxCenters, treeBoxMoments,
              grpCenter, grpSize, DistanceCheck, DistanceCheckPP, maxLevel+1);
          if(res2 < 0) return -1;

          res = max(res,res2); //Return max level reached
        }
      }//!leaf
      else
      {
        //It is a leaf, no need to check any other groups
        return res;
      }
      //No need to check further groups since this one already succeeded
      grp = endGrp;
    }//Split
  }
  return res;
}

int octree::recursiveBasedTopLEvelsCheckStart(tree_structure &tree,
    real4 *treeBuffer,
    real4 *grpCenter,
    real4 *grpSize,
    int startGrp,
    int endGrp,
    int &DistanceCheck)
{
  //Tree info
  const int nParticles = host_float_as_int(treeBuffer[0].x);
  const int nNodes     = host_float_as_int(treeBuffer[0].y);

  //  LOGF(stderr,"Working with %d and %d || %d %d\n", nParticles, nNodes, 1+nParticles+nNodes,nTopLevelTrees );

  real4* treeBoxSizes   = &treeBuffer[1+nParticles];
  real4* treeBoxCenters = &treeBuffer[1+nParticles+nNodes];
  real4* treeBoxMoments = &treeBuffer[1+nParticles+2*nNodes];

  const int nodeID = 0;
  uint4 checkNode = make_uint4(nodeID, startGrp, endGrp, 0);

  int DistanceCheckPP = 0;
  int maxLevel = recursiveTopLevelCheck(checkNode, treeBoxSizes, treeBoxCenters, treeBoxMoments,
      grpCenter, grpSize, DistanceCheck, DistanceCheckPP, 0);


  //  LOGF(stderr, "Finally Max level found: %d Process : %d \n", maxLevel, ibox)
  return maxLevel;
}



void octree::essential_tree_exchangeV2(tree_structure &tree,
    tree_structure &remote,
    nInfoStruct *nodeInfo,
    vector<real4> &topLevelTrees,
    vector<uint2> &topLevelTreesSizeOffset,
    int     nTopLevelTrees)
{
#ifdef USE_MPI
  double t0         = get_time();

  bool mergeOwntree = false;              //Default do not include our own tree-structure, thats mainly used for testing
  int level_start   = tree.startLevelMin; //Depth of where to start the tree-walk
  int procTrees     = 0;                  //Number of trees that we've received and processed

  real4  *bodies              = &tree.bodies_Ppos[0];
  real4  *velocities          = &tree.bodies_Pvel[0];
  real4  *multipole           = &tree.multipole[0];
  real4  *nodeSizeInfo        = &tree.boxSizeInfo[0];
  real4  *nodeCenterInfo      = &tree.boxCenterInfo[0];

  real4 **treeBuffers;

  //creates a new array of pointers to int objects, with space for the local tree
  treeBuffers  = new real4*[mpiGetNProcs()];
  int *treeBuffersSource = new int[nProcs];

  real4 *recvAllToAllBuffer = NULL;
  real4 *recvAllGatherVBuffer = NULL;


  //Timers for the LET Exchange
  static double totalLETExTime    = 0;
  thisPartLETExTime               = 0;
  double tStart                   = get_time();


  int topNodeOnTheFlyCount = 0;

  this->fullGrpAndLETRequestStatistics[procId] = make_uint2(0, procId); //Reset our box

  uint2 node_begend;
  node_begend.x   = tree.level_list[level_start].x;
  node_begend.y   = tree.level_list[level_start].y;

  int resultOfQuickCheck[nProcs];

  int quickCheckSendSizes [nProcs];
  int quickCheckSendOffset[nProcs];

  int quickCheckRecvSizes [nProcs];
  int quickCheckRecvOffset[nProcs];


  int nCompletedQuickCheck = 0;

  resultOfQuickCheck[procId]    = 99; //Mark ourself
  quickCheckSendSizes[procId]   =  0;
  quickCheckSendOffset[procId]  =  0;

  int nQuickCheckSends          = 0;


  omp_set_num_threads(16);

  letObject *computedLETs = new letObject[nProcs-1];

  int omp_ticket      = 0;
  int nComputedLETs   = 0;
  int nReceived       = 0;
  int nSendOut        = 0;


  //Use multiple OpenMP threads in parallel to build and exchange LETs
#pragma omp parallel
  {
    int tid      = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    if(tid != 1) //Thread 0, does LET creation and GPU control, Thread == 1 does MPI communication, all others do LET creation
    {
      nInfoStruct  *nodeInfo_private;
      uint2        *curLevelStack;
      uint2        *nextLevelStack;



#pragma omp critical
      {
        nodeInfo_private = new nInfoStruct[localTree.n_nodes];

        //Disabled wit the new getLETopt function
        //const int LETCreateStackSize = 2*512*1024; //TODO make this dynamic in some sense
        //  curLevelStack    = new uint2[LETCreateStackSize];
        //  nextLevelStack   = new uint2[LETCreateStackSize];
      }
      //Each thread requires it's own copy since we modify some values during checking
      memcpy(&nodeInfo_private[0], &nodeInfo[0], sizeof(nInfoStruct)*localTree.n_nodes);


      int DistanceCheck = 0;
      double tGrpTest = get_time();

      while(true) //Continue until everything is computed
      {
        int currentTicket = 0;

#pragma omp critical
        currentTicket = omp_ticket++; //Get a unique ticket to determine which process to build the LET for

        if(currentTicket == (nProcs-1)) //Skip ourself
        {
          // LOGF(stderr,"Quick test was done, thread: %d checks: %d Since start: %lg\n", tid, DistanceCheck, get_time()-tGrpTest);

          /*    char buff[5120];
                sprintf(buff, "GrpTesting tookB: %lg Checks: %d Res: ", get_time()-tGrpTest, DistanceCheck);
                for(int i=0; i < nProcs; i++)
                {
                sprintf(buff, "%s%d\t",buff, resultOfQuickCheck[i]);
                }
                LOGF(stderr, "%s\n", buff);
                */
          continue;
        }

        if(currentTicket >= (2*(nProcs) -1)) //Break out if everything is processed
          break;

        bool doQuickLETCheck = (currentTicket < (nProcs - 1));
        int ib               = (nProcs-1)-(currentTicket%nProcs);
        int ibox             = (ib+procId)%nProcs; //index to send...

        //Above could be replaced by a priority list, based on previous
        //loops (eg nearest neighbours first)

        double t1 = get_time();

        int doFullGrp = fullGrpAndLETRequest[ibox];

        //Group info for this process
        int idx          =   globalGrpTreeOffsets[ibox];
        real4 *grpCenter =  &globalGrpTreeCntSize[idx];
        idx             += this->globalGrpTreeCount[ibox] / 2; //Divide by two to get halfway
        real4 *grpSize   =  &globalGrpTreeCntSize[idx];

        //Retrieve required for the tree-walk from the top node
        union{int i; float f;} itof; //float as int

        itof.f       = grpCenter[0].x;
        int startGrp = itof.i;
        itof.f       = grpCenter[0].y;
        int endGrp   = itof.i;

        if(!doFullGrp)
        {
          //This is a topNode only
          startGrp = 0;
          endGrp   = this->globalGrpTreeCount[ibox] / 2;
        }


        if(doQuickLETCheck)
        {

          //Use this to 'disable' the Quick LET checks, with this disabled all
          //communication will be done as point to point
          //#define DO_NOT_DO_QUICK_LET_CHECK

#ifdef DO_NOT_DO_QUICK_LET_CHECK

          resultOfQuickCheck[ibox]   = -1;
          quickCheckSendSizes[ibox]  = 0;
          quickCheckSendOffset[ibox] = 0;
#pragma omp critical
          nCompletedQuickCheck++;
          continue;
#else


          if(doFullGrp)
          {
            //can skip this one beforehand, otherwise we would not have received the full-grp info
            resultOfQuickCheck[ibox]   = -1;
            quickCheckSendSizes[ibox]  = 0;
            quickCheckSendOffset[ibox] = 0;
          }
          else
          {
            //Determine if we do the quick-check, or the full check
            int maxLevel = recursiveBasedTopLEvelsCheckStart(tree,
                &topLevelTrees[topLevelTreesSizeOffset[nTopLevelTrees].y],
                grpCenter, grpSize, startGrp, endGrp, DistanceCheck);
            resultOfQuickCheck[ibox] = maxLevel;

            //#define MAXLEVELSIZE_ALLGATHER 2048
#define MAXLEVELSIZE_ALLGATHER -1
#if 0
            if(maxLevel >= 0)
            {
              if((topLevelTreesSizeOffset[maxLevel].x * sizeof(real4)) > MAXLEVELSIZE_ALLGATHER)
              {
                LOGF(stderr, "NOT using : %d  %d \t size: %d \n", ibox, maxLevel, topLevelTreesSizeOffset[maxLevel].x * sizeof(real4));
                resultOfQuickCheck[ibox] = -1;
                maxLevel = -1;
              }
            }
#endif

            if(maxLevel >= 0)
            {
              quickCheckSendSizes[ibox]  = topLevelTreesSizeOffset[maxLevel].x; //Size
              quickCheckSendOffset[ibox] = topLevelTreesSizeOffset[maxLevel].y; //Offset
#pragma omp critical
              nQuickCheckSends++;

              //Store the statistics
              this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(maxLevel, ibox);
            }
            else
            {
              quickCheckSendSizes[ibox]   = 0;
              quickCheckSendOffset[ibox]  = 0;
            }

            //Also set the size and offset for the alltoall call. It will be two calls
            //first with integer size -> alltoall
            //second with integer size and offsets and displacements -> alltoallV
          }

#pragma omp critical
          nCompletedQuickCheck++;

          continue;
#endif
        }
        //Only continue if 'nCompletedQuickCheck' is done, otherwise some thread might still be
        //executing the quick check!
        while(1)
        {
          if(nCompletedQuickCheck == nProcs-1)
            break;
          usleep(10);
        }

        //If we arrive here, we did the quick tests, so now go checking if we need to do the full test
        if(resultOfQuickCheck[ibox] >= 0)
        {
          //We can skip this process, its been taken care of during the quick check
          continue;
        }


        int countNodes = 0, countParticles = 0;
#if 0
        double tz = get_time();
        if(tree.n > 0)
        {
          tree_walking_tree_stack_versionC13(
              &localTree.multipole[0], &nodeInfo_private[0], //Local Tree
              grpSize, grpCenter, //remote Tree
              node_begend.x, node_begend.y, startGrp, endGrp-1,
              countNodes, countParticles,
              curLevelStack, nextLevelStack);
        }
        double tCount = get_time()-tz;

        //Record the number of particles required
        //this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles, ibox);

        //Test, use particles and node stats, but let particles count more heavy
        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);

        //Buffer that will contain all the data:
        //|real4| 2*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
        //1 + 1*particleCount + nodeCount + nodeCount + 3*nodeCount

        //Increase the number of particles and the number of nodes by the texture-offset
        //such that these are correctly aligned in memory
        //countParticles += getTextureAllignmentOffset(countParticles, sizeof(real4));
        //countNodes     += getTextureAllignmentOffset(countNodes    , sizeof(real4));

        //0-1 )                               Info about #particles, #nodes, start and end of tree-walk
        //1- Npart)                           The particle positions
        //1+1*Npart-Nnode )                   The nodeSizeData
        //1+1*Npart+Nnode   - Npart+2*Nnode ) The nodeCenterData
        //1+1*Npart+2*Nnode - Npart+5*Nnode ) The multipole data, is 3x number of nodes (mono and quadrupole data)
        int bufferSize = 1 + 1*countParticles + 5*countNodes;


        //We could try to reuse an existing buffer. TODO

        real4 *LETDataBuffer;
#pragma omp critical //Malloc seems to be not so thread safe..
        LETDataBuffer = (real4*)malloc(sizeof(real4)*bufferSize);

        double ty = get_time();
        stackFill(&LETDataBuffer[1],
            &localTree.boxCenterInfo[0],  &localTree.boxSizeInfo[0], &localTree.bodies_Ppos[0],
            &localTree.multipole[0],        nodeInfo_private, countParticles, countNodes,
            node_begend.x, node_begend.y, (uint*)curLevelStack, (uint*)nextLevelStack);

        if (ENABLE_RUNTIME_LOG)
        {
          fprintf(stderr,"Proc: %d LET count&fill [%d,%d]: Dest: %d Count: %lg Fill; %lg, Total : %lg (#P: %d \t#N: %d) \tsince start: %lg \n",
              procId, procId, tid, ibox, doFullGrp, tCount, get_time() - ty, get_time()-t1, countParticles, countNodes, get_time()-t0);
        }

#else
        double tz = get_time();
        real4 *LETDataBuffer;
        unsigned long long int nflops = 0;

#if 0
        if (ENABLE_RUNTIME_LOG)
          fprintf(stderr,"Proc: %d starting getLetOp  Dest: %d \n", procId, ibox);
        int2  nExport = getLETopt(
            &LETDataBuffer,
            &nodeCenterInfo[0],
            &nodeSizeInfo[0],
            &multipole[0],
            node_begend.x,
            node_begend.y,
            &bodies[0],
            tree.n,
            grpSize,
            grpCenter,
            startGrp,
            endGrp,
            tree.n_nodes, nflops);
#else
        if (ENABLE_RUNTIME_LOG)
          fprintf(stderr,"Proc: %d starting getLet1  Dest: %d \n", procId, ibox);
        assert(startGrp == 0);
        int2  nExport = getLET1(
            &LETDataBuffer,
            &nodeCenterInfo[0],
            &nodeSizeInfo[0],
            &multipole[0],
            node_begend.x,
            node_begend.y,
            &bodies[0],
            tree.n,
            grpSize,
            grpCenter,
            endGrp,
            tree.n_nodes, nflops);
#endif

        countParticles  = nExport.y;
        countNodes      = nExport.x;
        int bufferSize  = 1 + 1*countParticles + 5*countNodes;
        //Test, use particles and node stats, but let particles count more heavy
        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);
        if (ENABLE_RUNTIME_LOG)
        {
          fprintf(stderr,"Proc: %d LET getLetOp count&fill [%d,%d]: Full: %d Dest: %d Total : %lg (#P: %d \t#N: %d) nNodes= %d  nGroups= %d \tsince start: %lg \n", procId, procId, tid, doFullGrp, ibox, get_time()-tz,countParticles, countNodes,
              tree.n_nodes, endGrp, get_time()-t0);
        }


#endif

        //Set the tree properties, before we exchange the data
        LETDataBuffer[0].x = host_int_as_float(countParticles);    //Number of particles in the LET
        LETDataBuffer[0].y = host_int_as_float(countNodes);        //Number of nodes     in the LET
        LETDataBuffer[0].z = host_int_as_float(node_begend.x);     //First node on the level that indicates the start of the tree walk
        LETDataBuffer[0].w = host_int_as_float(node_begend.y);     //last node on the level that indicates the start of the tree walk

        //In a critical section to prevent multiple threads writing to the same location
#pragma omp critical
        {
          computedLETs[nComputedLETs].buffer      = LETDataBuffer;
          computedLETs[nComputedLETs].destination = ibox;
          computedLETs[nComputedLETs].size        = sizeof(real4)*bufferSize;
          nComputedLETs++;
        }


        if(tid == 0)
        {
          //This determines if we interrupt the computation by starting a gravity kernel on the GPU
          if(gravStream->isFinished())
          {
            LOGF(stderr,"GRAVFINISHED %d recvTree: %d  Time: %lg Since start: %lg\n",
                procId, nReceived, get_time()-t1, get_time()-t0);

            //Only start if there actually is new data
            if((nReceived - procTrees) > 0)
            {
              int recvTree      = 0;
              int topNodeCount  = 0;
              int oriTopCount   = 0;
#pragma omp critical(updateReceivedProcessed)
              {
                recvTree             = nReceived;
                topNodeCount         = topNodeOnTheFlyCount;
                oriTopCount          = topNodeOnTheFlyCount;
                topNodeOnTheFlyCount = 0;
              }

              double t000 = get_time();
              mergeAndLaunchLETStructures(tree, remote, treeBuffers, treeBuffersSource,
                  topNodeCount, recvTree, mergeOwntree, procTrees, tStart);
              LOGF(stderr, "Merging and launching iter: %d took: %lg \n", iter, get_time()-t000);


              //Correct the topNodeOnTheFlyCounter
#pragma omp critical(updateReceivedProcessed)
              {
                //Compute how many are left, and add these back to the globalCounter
                int nTopNodesLeft     = oriTopCount-topNodeCount;
                topNodeOnTheFlyCount += nTopNodesLeft;
              }

              totalLETExTime += thisPartLETExTime;
            }// (nReceived - procTrees) > 0)
          }// isFinished
        }//tid == 0

      }//end while

      //All data that has to be send out is computed
      if(tid == 0)
      {
        //Thread 0 starts the GPU work so it stays alive until that is complete
        while(procTrees != nProcs-1) //Exit when everything is processed
        {
          bool startGrav = false;
          if(nReceived == nProcs-1) //Indicates that we have received all there is to receive
          {
            startGrav = true;
          }

          //This determines if we interrupt the computation/waiting by starting a gravity kernel on the GPU
          //Since the GPU is being idle
          if(gravStream->isFinished())
          {
            //Only start if there actually is new data
            if((nReceived - procTrees) > 0) startGrav = true;
          }

          if(startGrav)
          {
            int recvTree      = 0;
            int topNodeCount  = 0;
            int oriTopCount   = 0;
#pragma omp critical(updateReceivedProcessed)
            {
              recvTree             = nReceived;
              topNodeCount         = topNodeOnTheFlyCount;
              oriTopCount          = topNodeOnTheFlyCount;
              topNodeOnTheFlyCount = 0;
            }

            double t000 = get_time();
            mergeAndLaunchLETStructures(tree, remote, treeBuffers, treeBuffersSource,
                topNodeCount,recvTree, mergeOwntree, procTrees, tStart);
            LOGF(stderr, "Merging and launching iter: %d took: %lg \n", iter, get_time()-t000);

            //Correct the topNodeOnTheFlyCounter
#pragma omp critical(updateReceivedProcessed)
            {
              //Compute how many are left, and add these back to the globalCounter
              int nTopNodesLeft     = oriTopCount-topNodeCount;
              topNodeOnTheFlyCount += nTopNodesLeft;
            }

            totalLETExTime += thisPartLETExTime;
          }
          else
          {
            usleep(10);
          }//if startGrav
        }//while 1
      }//if tid==0


      //delete[] curLevelStack;
      //delete[] nextLevelStack;
      delete[] nodeInfo_private;
    }
    else if(tid == 1)
    {

      //All to all part

#ifndef DO_NOT_DO_QUICK_LET_CHECK
      while(1)
      {
        if(nCompletedQuickCheck == nProcs-1)
          break;
        usleep(10);
      }

      //Send the sizes

#if MAXLEVELSIZE_ALLGATHER > 0
      //Combined all_gatherv and alltoall
      //First determine the maximum level

      int allGatherLevel = 0;
      for(int level=0; level < nTopLevelTrees; level++)
      {
        if((topLevelTreesSizeOffset[level].x * sizeof(real4)) > MAXLEVELSIZE_ALLGATHER)
          break;
        allGatherLevel++;
      }

      int allGatherLevelSize   = topLevelTreesSizeOffset[allGatherLevel].x*sizeof(real4);
      int allGatherLevelOffset = topLevelTreesSizeOffset[allGatherLevel].y*sizeof(real4);

      //Build up the data we are going to send/receive
      //To each process we send a copy of the tree at level 'allGatherLevel'
      //using all_gatherv
      //We indicate if that is sufficient, if not we indicate if we do an alltoallv send
      //or an getLET send.

      std::vector<int4> summaryOfDataToSend(nProcs);
      std::vector<int4> summaryOfDataToReceive(nProcs);

      // std::vector<int4> all2allVSizes  (nProcs);
      // std::vector<int4> all2allVOffsets(nProcs);

      resultOfQuickCheck[procId]    = -1; //Mark ourself
      quickCheckSendSizes[procId]   =  0;
      quickCheckSendOffset[procId]  =  0;

      for(int proc = 0; proc < nProcs; proc++)
      {
        if(resultOfQuickCheck[proc] >= 0 && resultOfQuickCheck[proc] <= allGatherLevel)
        {//Send using allGather
          summaryOfDataToSend[proc].x = 1; //It needs the top level
          summaryOfDataToSend[proc].y = allGatherLevelSize; //Size of the top level
          summaryOfDataToSend[proc].z = 0; //Not used

          //Set quickCheckSize to zero since it will not do an all2all
          quickCheckSendSizes[proc]   =  0;
          quickCheckSendOffset[proc]  =  0;
        }
        else if(resultOfQuickCheck[proc] >= 0)
        {//Send using alltoall
          summaryOfDataToSend[proc].x = 2; //It needs the top level
          summaryOfDataToSend[proc].y = allGatherLevelSize; //Size of the top level
          summaryOfDataToSend[proc].z = sizeof(real4)*topLevelTreesSizeOffset[resultOfQuickCheck[proc]].x; //alltoall size


          quickCheckSendSizes[proc]   =  sizeof(real4)*topLevelTreesSizeOffset[resultOfQuickCheck[proc]].x;
          quickCheckSendOffset[proc]  =  sizeof(real4)*topLevelTreesSizeOffset[resultOfQuickCheck[proc]].y;        
        }
        else
        { //Send with getLET
          summaryOfDataToSend[proc].x = 3; //It needs the top level
          summaryOfDataToSend[proc].y = allGatherLevelSize; //Size of the top level

          quickCheckSendSizes[proc]   =  0;
          quickCheckSendOffset[proc]  =  0;        
        }
      }//for each process

      LOGF(stderr, "Going to do the all to all size communication! Iter: %d Since begin: %lg \n", iter, get_time()-tStart);
      double t100 = get_time();
      MPI_Alltoall(
          &summaryOfDataToSend[0],    sizeof(int4), MPI_BYTE,
          &summaryOfDataToReceive[0], sizeof(int4), MPI_BYTE,
          MPI_COMM_WORLD);

      LOGF(stderr, "Completed_alltoall size comm! Iter: %d Took: %lg ( %lg )\n", iter, get_time()-t100, get_time()-t0);

      //Count the number of incomming data items
      //First the allgather, size and offsets
      int allGatherOffset = 0;
      int allToAllOffset  = 0;
      std::vector<int> allGatherSizes(nProcs);  //Sizes to receive
      std::vector<int> allGatherOffsets(nProcs); //offsets
      std::vector<int> allGatherUses   (nProcs);  //Indicate if we use this (1) or not (-1)

      //We could reduce memory use by putting the non-used data in a seperate buffer, for now just
      //put it in a lineair array
      for(int proc = 0; proc < nProcs; proc++)
      {
        allGatherSizes[proc]   = summaryOfDataToReceive[proc].y;
        allGatherOffsets[proc] = allGatherOffset;
        allGatherOffset       += summaryOfDataToReceive[proc].y;

        if(summaryOfDataToReceive[proc].x == 1)
          allGatherUses[proc] = 1;
        else
          allGatherUses[proc] = -1;

        //Sum the alltoall size to be able to alloc
        if(summaryOfDataToReceive[proc].x == 2)
          allToAllOffset += summaryOfDataToReceive[proc].z;
      }

      recvAllGatherVBuffer =  new real4[allGatherOffset / sizeof(real4)];
      recvAllToAllBuffer   =  new real4[allToAllOffset  / sizeof(real4)];

      double tGatherStart = get_time();

      MPI_Allgatherv(&(topLevelTrees[allGatherLevelOffset / sizeof(real4)]),    //Begin of array
          allGatherLevelSize,                      //Number of top-nodes
          MPI_BYTE,
          recvAllGatherVBuffer,               //Receive buffer
          &allGatherSizes[0],      //Array with size per node
          &allGatherOffsets[0],    //Array with offset per node
          MPI_BYTE, MPI_COMM_WORLD);

      double tGatherEnd = get_time();

      LOGF(stderr, "All_GatherV took: %lg ( %lg ) Size: %lg  \n", 
          tGatherEnd-tGatherStart, get_time()-t0, (allGatherOffset  / sizeof(real4)*sizeof(real4))/(double)(1024*1024));

      for(int i=0; i < nProcs; i++)
      {
        int offset   = allGatherOffsets[i] / sizeof(real4);
        int p        = host_float_as_int(recvAllGatherVBuffer[offset].x);
        int n        = host_float_as_int(recvAllGatherVBuffer[offset].y);
        int topStart = host_float_as_int(recvAllGatherVBuffer[offset].z);
        int topEnd   = host_float_as_int(recvAllGatherVBuffer[offset].w);

        LOGF(stderr, "I received from %d  the following [%d %d \t | %d  %d \n",
            i, p, n, topStart, topEnd);
      }

#pragma omp critical(updateReceivedProcessed)
      {
        //This is in a critical section since topNodeOnTheFlyCount is reset
        //by the GPU worker thread (thread == 0)
        int offset = 0;
        for(int i=0;  i < nProcs; i++)
        {
          if(allGatherUses[i] > 0)
          {
            int offset             = allGatherOffsets[i] / sizeof(real4);
            treeBuffers[nReceived] = &recvAllGatherVBuffer[offset];

            //Increase the top-node count
            int p        = host_float_as_int(treeBuffers[nReceived][0].x);
            int n        = host_float_as_int(treeBuffers[nReceived][0].y);
            int topStart = host_float_as_int(treeBuffers[nReceived][0].z);
            int topEnd   = host_float_as_int(treeBuffers[nReceived][0].w);

            //          LOGF(stderr, "Received from: %d  start: %d end: %d P: %d N: %d\n",
            //                       i, topStart, topEnd, p, n);

            topNodeOnTheFlyCount        += (topEnd-topStart);
            treeBuffersSource[nReceived] = 1; //1 indicate quick check source
            nReceived++;
          }
        }
        LOGF(stderr, "Received trees using quickcheck-agv: %d top-nodes: %d \n", nReceived, topNodeOnTheFlyCount);
      }

      //Receive data using alltoall
      allToAllOffset  = 0;
      std::vector<int> allToAllRecvSizes  (nProcs);  //Sizes to receive
      std::vector<int> allToAllRecvOffsets(nProcs);  //offsets
      std::vector<int> allToAllUses   (nProcs);  //Indicate if we use this (1) or not (-1)    
      for(int proc = 0; proc < nProcs; proc++)
      {
        if(summaryOfDataToReceive[proc].x == 2)
        {
          allGatherUses[proc] = 1;
          allToAllRecvSizes[proc]    = summaryOfDataToReceive[proc].z;
          allToAllRecvOffsets[proc]  = allToAllOffset;
          allToAllOffset            += summaryOfDataToReceive[proc].z;
        }       
        else
        {
          allGatherUses[proc]        = -1;
          allToAllRecvSizes[proc]    = 0;
          allToAllRecvOffsets[proc]  = 0;      
        }
      }    

      LOGF(stderr, "Starting 1D alltoall \n");
      double t110 = get_time();
      MPI_Alltoallv(&topLevelTrees[0],   
          quickCheckSendSizes, quickCheckSendOffset, MPI_BYTE,
          &recvAllToAllBuffer[0],
          &allToAllRecvSizes[0], &allToAllRecvOffsets[0], MPI_BYTE,
          MPI_COMM_WORLD);
      LOGF(stderr, "[%d] Completed_alltoall 1D data communication! Iter: %d Took: %lg ( %lg )\tSize: %lg MB \n",
          procId, iter, get_time()-t110,  get_time()-t0, (allToAllOffset  / sizeof(real4)*sizeof(real4))/(double)(1024*1024));    

#pragma omp critical(updateReceivedProcessed)
      {
        //This is in a critical section since topNodeOnTheFlyCount is reset
        //by the GPU worker thread (thread == 0)
        int offset    = 0;
        int na2acount = 0;
        int na2atopnode = 0;
        for(int i=0;  i < nProcs; i++)
        {
          if(allGatherUses[i] > 0)
          {
            int items              = allToAllRecvSizes[i]  / sizeof(real4);
            treeBuffers[nReceived] = &recvAllToAllBuffer[offset];
            offset                += items;

            //Increase the top-node count
            int p        = host_float_as_int(treeBuffers[nReceived][0].x);
            int n        = host_float_as_int(treeBuffers[nReceived][0].y);
            int topStart = host_float_as_int(treeBuffers[nReceived][0].z);
            int topEnd   = host_float_as_int(treeBuffers[nReceived][0].w);

            //       LOGF(stderr, "Received from: %d  start: %d end: %d P: %d N: %d\n",
            //                     i, topStart, topEnd, p, n);

            na2atopnode += (topEnd-topStart);                            

            topNodeOnTheFlyCount        += (topEnd-topStart);
            treeBuffersSource[nReceived] = 1; //1 indicate quick check source
            nReceived++;
            na2acount++;
          }
        }
        if (ENABLE_RUNTIME_LOG)
        {
          fprintf(stderr,"Proc: %d Received trees using quickcheck-a2a: %d top-nodes: %d \n", procId, na2acount, na2atopnode);
        }
      }


#else



      LOGF(stderr, "Going to do the alltoall size communication! Iter: %d Since begin: %lg \n", iter, get_time()-tStart);
      double t100 = get_time();
      MPI_Alltoall(quickCheckSendSizes, 1, MPI_INT, quickCheckRecvSizes, 1, MPI_INT, MPI_COMM_WORLD);
      LOGF(stderr, "Completed_alltoall size communication! Iter: %d Took: %lg ( %lg )\n", iter, get_time()-t100, get_time()-t0);

      //Compute offsets, allocate memory
      int recvCountItems      = quickCheckRecvSizes[0];
      quickCheckRecvSizes[0]  = quickCheckRecvSizes[0]*sizeof(real4);
      quickCheckRecvOffset[0] = 0;
      for(int i=1; i < nProcs; i++)
      {
        recvCountItems         += quickCheckRecvSizes[i];
        quickCheckRecvSizes[i]  = quickCheckRecvSizes[i]*sizeof(real4);
        quickCheckRecvOffset[i] = quickCheckRecvSizes[i-1] +  quickCheckRecvOffset[i-1];
      }
      //int totalSize = quickCheckRecvOffset[nProcs-1] + quickCheckRecvSizes[nProcs-1];
      /*          char buff[5120];
                  sprintf(buff, "AlltoAllSend: ");
                  for(int i=0; i < nProcs; i++)
                  {
                  sprintf(buff, "%s[%d,%d]\t",buff, quickCheckSendSizes[i], quickCheckSendOffset[i]);
                  }
                  LOGF(stderr, "%s\n", buff);

                  sprintf(buff, "AlltoAllRecv: ");
                  for(int i=0; i < nProcs; i++)
                  {
                  sprintf(buff, "%s[%d,%d]\t",buff, quickCheckRecvSizes[i], quickCheckRecvOffset[i]);
                  }
                  LOGF(stderr, "%s\n", buff);

                  int test2 = quickCheckRecvSizes[nProcs-1] + quickCheckRecvOffset[nProcs-1];
                  LOGF(stderr, "Allocating %ld size: %f MB \n", (recvCountItems*sizeof(real4)),(recvCountItems*sizeof(real4))/((float)(1024*1024)));
                  LOGF(stderr, "Allocating2 %d |  %d | %d  \n", test2, sizeof(real4), test2 / sizeof(real4));
                  */
      //      quickCheckSendOffset , quickCheckSendSizes
      //
      double tmem = get_time();
      recvAllToAllBuffer =  new real4[recvCountItems];
      LOGF(stderr, "Completed_alltoall mem alloc! Iter: %d Took: %lg \n", iter, get_time()-tmem);

      //Convert the values to bytes to get correct offsets and sizes
      for(int i=0; i < nProcs; i++)
      {
        quickCheckSendSizes[i]  *= sizeof(real4);
        quickCheckSendOffset[i] *= sizeof(real4);

      }

      double t110 = get_time();

#if 0
      LOGF(stderr, "Starting 1D alltoall \n");
      MPI_Alltoallv(&topLevelTrees[0],       quickCheckSendSizes, quickCheckSendOffset, MPI_BYTE,
          &recvAllToAllBuffer[0],  quickCheckRecvSizes, quickCheckRecvOffset, MPI_BYTE,
          MPI_COMM_WORLD);
      LOGF(stderr, "[%d] Completed_alltoall 1D data communication! Iter: %d Took: %lg ( %lg )\tSize: %ld MB \n",
          procId, iter, get_time()-t110,  get_time()-t0, (recvCountItems*sizeof(real4))/(1024*1024));
#else
      {

#if 1  /* use this if data is aligned */
        typedef v4sf vec4;
#else  /* otherwise use this to avoid segfault on the unaligned data */
        typedef float4 vec4;
#endif

        std::vector<v4sf> topLevel;
        int nsendtotal = 0;
        for (int i= 0; i < nProcs; i++)
          nsendtotal += quickCheckSendSizes[i]/sizeof(v4sf);

        topLevel.resize(nsendtotal);
        int cntr = 0;
        for (int i= 0; i < nProcs; i++)
        {
          assert(quickCheckSendSizes[i] % sizeof(vec4) == 0);
          assert(quickCheckSendOffset[i] % sizeof(vec4) == 0);
          const int nsend = quickCheckSendSizes [i]/sizeof(v4sf);
          const int displ = quickCheckSendOffset[i]/sizeof(v4sf);
          for (int j = 0; j < nsend; j++)
            topLevel[cntr++] = ((v4sf*)&topLevelTrees[0])[displ + j];
        }

        LOGF(stderr, "Starting 2D alltoall copy took: %lg \n", get_time()-t110);
        mpiSync();
#if 0
        myComm->ugly_all2allv_char((float*)&topLevel[0],
            quickCheckSendSizes,
            (float*)&recvAllToAllBuffer[0]);
#else
        double tbla = get_time();
        std::vector<int> scount_topLevel(nProcs);
        for (int i= 0; i < nProcs; i++)
          scount_topLevel[i] = quickCheckSendSizes[i]/sizeof(v4sf);
        myComm->all2allv_2D(topLevel, &scount_topLevel[0]);
        for (size_t i = 0; i < topLevel.size(); i++)
          ((v4sf*)recvAllToAllBuffer)[i] = topLevel[i];
        double tbla2  = get_time();
        LOGF(stderr,"Prep took: %lg  Items: %d \n", tbla2-tbla, (int)topLevel.size());
#endif


#if 1
        int temp[nProcs];
        int newIdx = 0;
        for(int x=0; x<myComm->n_proc_i; x++)
        {
          for(int y=0; y<myComm->n_proc_j; y++)
          {
            //int k        = x*myComm->n_proc_j + y*myComm->n_proc_i;
            int k = myComm->n_proc_i*y + x;
            temp[newIdx] = quickCheckRecvSizes[k];
            //LOGF(stderr, "Recv moving %d -> %d value: %d \n", k, newIdx, temp[newIdx]);
            newIdx++;
          }
        }
        memcpy(quickCheckRecvSizes, temp, sizeof(int)*nProcs);
#endif
      }

      LOGF(stderr, "[%d] Completed_alltoall 2D data communication! Iter: %d Took: %lg ( %lg )\tSize: %f MB \n",
          procId, iter, get_time()-t110,  get_time()-t0, (recvCountItems*sizeof(real4))/(double)(1024*1024));
#endif


#pragma omp critical(updateReceivedProcessed)
      {
        //This is in a critical section since topNodeOnTheFlyCount is reset
        //by the GPU worker thread (thread == 0)
        //
        //	char buff[4096];
        //	sprintf(buff, "Proc: %d Recv Ori: ", procId);
        //
        int offset = 0;
        for(int i=0;  i < nProcs; i++)
        {

          //	  sprintf(buff,"%s [%d, %d ], ", buff,quickCheckRecvSizes[i], quickCheckRecvOffset[i]);

          int items  = quickCheckRecvSizes[i]  / sizeof(real4);
          if(items > 0)
          {
            //int offset = quickCheckRecvOffset[i] / sizeof(real4);

            treeBuffers[nReceived] = &recvAllToAllBuffer[offset];

            offset += items;

            //Increase the top-node count
            int topStart = host_float_as_int(treeBuffers[nReceived][0].z);
            int topEnd   = host_float_as_int(treeBuffers[nReceived][0].w);

            LOGF(stderr, "Received from: %d  start: %d end: %d  offset: %d  offset old: %lu\n",
                i, topStart, topEnd, offset, quickCheckRecvOffset[i] / sizeof(real4));

            topNodeOnTheFlyCount += (topEnd-topStart);
            treeBuffersSource[nReceived] = 1; //1 indicate quick check source
            nReceived++;
          }
        }
        //   LOGF(stderr,"%s\n", buff);
      }


      LOGF(stderr, "Received trees using quickcheck: %d top-nodes: %d \n", nReceived, topNodeOnTheFlyCount);

#endif //the alltoall only code
#endif //#ifndef DO_NOT_DO_QUICK_LET_CHECK


      while(1)
      {
        //Sending part
        int tempComputed = nComputedLETs;

        if(tempComputed > nSendOut)
        {
          for(int i=nSendOut; i < tempComputed; i++)
          {
            //fprintf(stderr,"[%d] Sending out data to: %d \n", procId, computedLETs[i].destination);
            MPI_Isend(&(computedLETs[i].buffer)[0],computedLETs[i].size,
                MPI_BYTE, computedLETs[i].destination, 999,
                MPI_COMM_WORLD, &(computedLETs[i].req));
          }
          nSendOut = tempComputed;
        }

        //Receiving
        MPI_Status probeStatus;
        MPI_Status recvStatus;
        int flag  = 0;

        do
        {
          MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &probeStatus);

          if(flag)
          {
            int count;
            MPI_Get_count(&probeStatus, MPI_BYTE, &count);
            //fprintf(stderr,"%d\tThere is a message of size: %d %ld From: %d tag: %d\n",tid, count, count / sizeof(real4), probeStatus.MPI_SOURCE, probeStatus.MPI_TAG);

            double tY = get_time();
            real4 *recvDataBuffer = new real4[count / sizeof(real4)];
            double tZ = get_time();
            MPI_Recv(&recvDataBuffer[0], count, MPI_BYTE, probeStatus.MPI_SOURCE, probeStatus.MPI_TAG, MPI_COMM_WORLD,&recvStatus);

            LOGF(stderr, "Receive complete from: %d  || recvTree: %d since start: %lg ( %lg ) alloc: %lg Recv: %lg Size: %d\n",
                recvStatus.MPI_SOURCE, 0, get_time()-tStart,get_time()-t0,tZ-tY, get_time()-tZ, count);

            treeBuffers[nReceived] = recvDataBuffer;
            treeBuffersSource[nReceived] = 0; //0 indicates point to point source

            //Increase the top-node count
            int topStart = host_float_as_int(treeBuffers[nReceived][0].z);
            int topEnd   = host_float_as_int(treeBuffers[nReceived][0].w);


#pragma omp critical(updateReceivedProcessed)
            {
              //This is in a critical section since topNodeOnTheFlyCount is reset
              //by the GPU worker thread (thread == 0)
              topNodeOnTheFlyCount += (topEnd-topStart);
              nReceived++;
            }

            flag = 0;
          }//if flag

          //TODO we could at an other probe here to keep, receiving data
          //untill there is nothing more

        }while(flag);

        //Exit if we have send and received all there is
        if(nReceived == nProcs-1)
          if((nSendOut+nQuickCheckSends) == nProcs-1)
            break;

        //Check if we can clean up some sends in between the receive/send process
        MPI_Status waitStatus;
        int testFlag = 0;
        for(int i=0; i  < nSendOut; i++)
        {
          if(computedLETs[i].buffer != NULL) MPI_Test(&(computedLETs[i].req), &testFlag, &waitStatus);
          if (testFlag)
          {
            free(computedLETs[i].buffer);
            computedLETs[i].buffer = NULL;
            testFlag               = 0;
          }
        }//end for nSendOut

        //TODO only do this sleep if we did not send/receive something

        usleep(10);

      } //while (1) surrounding the thread-id==1 code

      //Wait till all outgoing sends have completed
      MPI_Status waitStatus;
      for(int i=0; i < nSendOut; i++)
      {
        if(computedLETs[i].buffer)
        {
          MPI_Wait(&(computedLETs[i].req), &waitStatus);
          free(computedLETs[i].buffer);
          computedLETs[i].buffer = NULL;
        }
      }//for i < nSendOut

    }//if tid = 1
  }//end OMP section

#if 1 //Moved freeing of memory to here for ha-pacs workaround
  for(int i=0; i < nProcs-1; i++)
  {
    if(treeBuffersSource[i] == 0) //Check if its a point to point source
    {
      delete[] treeBuffers[i];    //Free the memory of this part of the LET
      treeBuffers[i] = NULL;
    }
  }
#endif

  if(recvAllToAllBuffer) delete[] recvAllToAllBuffer;
  delete[] treeBuffersSource;
  delete[] computedLETs;
  delete[] treeBuffers;
  LOGF(stderr,"LET Creation and Exchanging time [%d] curStep: %g\t   Total: %g  Full-step: %lg  since last start: %lg\n", procId, thisPartLETExTime, totalLETExTime, get_time()-t0, get_time()-tStart);

#endif
}//essential tree-exchange


void octree::mergeAndLaunchLETStructures(
    tree_structure &tree, tree_structure &remote,
    real4 **treeBuffers, int *treeBuffersSource,
    int &topNodeOnTheFlyCount,
    int &recvTree, bool &mergeOwntree, int &procTrees, double &tStart)
{
  //Now we have to merge the separate tree-structures into one big-tree

  int PROCS  = recvTree-procTrees;


#if 0 //This is no longer safe now that we use OpenMP and overlapping communication/computation
  //to use this (only in debug/test case) make sure GPU work is only launched AFTER ALL data
  //is received
  if(mergeOwntree)
  {
    real4  *bodies              = &tree.bodies_Ppos[0];
    real4  *velocities          = &tree.bodies_Pvel[0];
    real4  *multipole           = &tree.multipole[0];
    real4  *nodeSizeInfo        = &tree.boxSizeInfo[0];
    real4  *nodeCenterInfo      = &tree.boxCenterInfo[0];
    int     level_start         = tree.startLevelMin;
    //Add the processors own tree to the LET tree
    int particleCount   = tree.n;
    int nodeCount       = tree.n_nodes;

    int realParticleCount = tree.n;
    int realNodeCount     = tree.n_nodes;

    particleCount += getTextureAllignmentOffset(particleCount, sizeof(real4));
    nodeCount     += getTextureAllignmentOffset(nodeCount    , sizeof(real4));

    int bufferSizeLocal = 1 + 1*particleCount + 5*nodeCount;

    treeBuffers[PROCS]  = new real4[bufferSizeLocal];

    //Note that we use the real*Counts otherwise we read out of the array boundaries!!
    int idx = 1;
    memcpy(&treeBuffers[PROCS][idx], &bodies[0],         sizeof(real4)*realParticleCount);
    idx += particleCount;
    //      memcpy(&treeBuffers[PROCS][idx], &velocities[0],     sizeof(real4)*realParticleCount);
    //      idx += particleCount;
    memcpy(&treeBuffers[PROCS][idx], &nodeSizeInfo[0],   sizeof(real4)*realNodeCount);
    idx += nodeCount;
    memcpy(&treeBuffers[PROCS][idx], &nodeCenterInfo[0], sizeof(real4)*realNodeCount);
    idx += nodeCount;
    memcpy(&treeBuffers[PROCS][idx], &multipole[0],      sizeof(real4)*realNodeCount*3);

    treeBuffers[PROCS][0].x = host_int_as_float(particleCount);
    treeBuffers[PROCS][0].y = host_int_as_float(nodeCount);
    treeBuffers[PROCS][0].z = host_int_as_float(tree.level_list[level_start].x);
    treeBuffers[PROCS][0].w = host_int_as_float(tree.level_list[level_start].y);

    topNodeOnTheFlyCount += (tree.level_list[level_start].y-tree.level_list[level_start].x);

    PROCS                   = PROCS + 1; //Signal that we added one more tree-structure
    mergeOwntree            = false;     //Set it to false in case we do not merge all trees at once, we only include our own once
  }
#endif

  //Arrays to store and compute the offsets
  int *particleSumOffsets  = new int[mpiGetNProcs()+1];
  int *nodeSumOffsets      = new int[mpiGetNProcs()+1];
  int *startNodeSumOffsets = new int[mpiGetNProcs()+1];
  uint2 *nodesBegEnd       = new uint2[mpiGetNProcs()+1];

  //Offsets start at 0 and then are increased by the number of nodes of each LET tree
  particleSumOffsets[0]           = 0;
  nodeSumOffsets[0]               = 0;
  startNodeSumOffsets[0]          = 0;
  nodesBegEnd[mpiGetNProcs()].x   = nodesBegEnd[mpiGetNProcs()].y = 0; //Make valgrind happy
  int totalTopNodes               = 0;

  //#define DO_NOT_USE_TOP_TREE //If this is defined there is no tree-build on top of the start nodes
  vector<real4> topBoxCenters(1*topNodeOnTheFlyCount);
  vector<real4> topBoxSizes  (1*topNodeOnTheFlyCount);
  vector<real4> topMultiPoles(3*topNodeOnTheFlyCount);
  vector<real4> topTempBuffer(3*topNodeOnTheFlyCount);
  vector<int  > topSourceProc; //Do not assign size since we use 'insert'


  int nParticlesCounted   = 0;
  int nNodesCounted       = 0;
  int nProcsProcessed     = 0;
  bool continueProcessing = true;

  //Calculate the offsets
  for(int i=0; i < PROCS ; i++)
  {
    int particles = host_float_as_int(treeBuffers[procTrees+i][0].x);
    int nodes     = host_float_as_int(treeBuffers[procTrees+i][0].y);

    nParticlesCounted += particles;
    nNodesCounted     += nodes;

    //Check if we go over the limit, if so, we have two options:
    // - Ignore this last one, if we have processed nodes before (nProcsProcessed > 0)
    // - Process this one anyway and hope we have enough memory, do this if nProcsProcessed == 0
    //   otherwise we would make no progress

    int localLimit   =  tree.n            + 5*tree.n_nodes;
    int currentCount =  nParticlesCounted + 5*nNodesCounted;

    if(currentCount > localLimit)
    {
      LOGF(stderr, "Processing breaches memory limit. Limits local: %d, current: %d processed: %d \n",
          localLimit, currentCount, nProcsProcessed);

      if(nProcsProcessed > 0)
      {
        break; //Ignore this process, will be used next loop
      }

      //Stop after this process
      continueProcessing = false;
    }
    nProcsProcessed++;

    //Continue processing this domain

    nodesBegEnd[i].x = host_float_as_int(treeBuffers[procTrees+i][0].z);
    nodesBegEnd[i].y = host_float_as_int(treeBuffers[procTrees+i][0].w);

    particleSumOffsets[i+1]     = particleSumOffsets[i]  + particles;
    nodeSumOffsets[i+1]         = nodeSumOffsets[i]      + nodes - nodesBegEnd[i].y;    //Without the top-nodes
    startNodeSumOffsets[i+1]    = startNodeSumOffsets[i] + nodesBegEnd[i].y-nodesBegEnd[i].x;

    //Copy the properties for the top-nodes
    int nTop = nodesBegEnd[i].y-nodesBegEnd[i].x;
    memcpy(&topBoxSizes[totalTopNodes],
        &treeBuffers[procTrees+i][1+1*particles+nodesBegEnd[i].x],             sizeof(real4)*nTop);
    memcpy(&topBoxCenters[totalTopNodes],
        &treeBuffers[procTrees+i][1+1*particles+nodes+nodesBegEnd[i].x],       sizeof(real4)*nTop);
    memcpy(&topMultiPoles[3*totalTopNodes],
        &treeBuffers[procTrees+i][1+1*particles+2*nodes+3*nodesBegEnd[i].x], 3*sizeof(real4)*nTop);
    topSourceProc.insert(topSourceProc.end(), nTop, i ); //Assign source process id

    totalTopNodes += nodesBegEnd[i].y-nodesBegEnd[i].x;

    if(continueProcessing == false)
      break;
  }

  //Modify NPROCS, to set it to what we actually processed. Same for the
  //number of top-nodes, which is later passed back to the calling function
  //to update the overall number of top-nodes that is left to be processed
  PROCS                = nProcsProcessed;
  topNodeOnTheFlyCount = totalTopNodes;




#ifndef DO_NOT_USE_TOP_TREE
  uint4 *keys          = new uint4[topNodeOnTheFlyCount];
  //Compute the keys for the top nodes based on their centers
  for(int i=0; i < topNodeOnTheFlyCount; i++)
  {
    real4 nodeCenter = topBoxCenters[i];
    int4 crd;
    crd.x = (int)((nodeCenter.x - tree.corner.x) / tree.corner.w);
    crd.y = (int)((nodeCenter.y - tree.corner.y) / tree.corner.w);
    crd.z = (int)((nodeCenter.z - tree.corner.z) / tree.corner.w);

    keys[i]   = host_get_key(crd);
    keys[i].w = i;
  }//for i,

  //Sort the cells by their keys
  std::sort(keys, keys+topNodeOnTheFlyCount, cmp_ph_key());

  int *topSourceTempBuffer = (int*)&topTempBuffer[2*topNodeOnTheFlyCount]; //Allocated after sizes and centers

  //Shuffle the top-nodes after sorting
  for(int i=0; i < topNodeOnTheFlyCount; i++)
  {
    topTempBuffer[i]                      = topBoxSizes[i];
    topTempBuffer[i+topNodeOnTheFlyCount] = topBoxCenters[i];
    topSourceTempBuffer[i]                = topSourceProc[i];
  }
  for(int i=0; i < topNodeOnTheFlyCount; i++)
  {
    topBoxSizes[i]   = topTempBuffer[                       keys[i].w];
    topBoxCenters[i] = topTempBuffer[topNodeOnTheFlyCount + keys[i].w];
    topSourceProc[i] = topSourceTempBuffer[                 keys[i].w];
  }
  for(int i=0; i < topNodeOnTheFlyCount; i++)
  {
    topTempBuffer[3*i+0]                  = topMultiPoles[3*i+0];
    topTempBuffer[3*i+1]                  = topMultiPoles[3*i+1];
    topTempBuffer[3*i+2]                  = topMultiPoles[3*i+2];
  }
  for(int i=0; i < topNodeOnTheFlyCount; i++)
  {
    topMultiPoles[3*i+0]                  = topTempBuffer[3*keys[i].w+0];
    topMultiPoles[3*i+1]                  = topTempBuffer[3*keys[i].w+1];
    topMultiPoles[3*i+2]                  = topTempBuffer[3*keys[i].w+2];
  }

  //Build the tree
  //Assume we do not need more than 4 times number of top nodes.
  //but use a minimum of 2048 to be save
  uint2 *nodes    = new uint2[max(4*topNodeOnTheFlyCount, 2048)];
  uint4 *nodeKeys = new uint4[max(4*topNodeOnTheFlyCount, 2048)];

  //Build the tree
  uint node_levels[MAXLEVELS];
  int topTree_n_levels;
  int topTree_startNode;
  int topTree_endNode;
  int topTree_n_nodes;
  build_NewTopLevels(topNodeOnTheFlyCount,   &keys[0],          nodes,
      nodeKeys,        node_levels,       topTree_n_levels,
      topTree_n_nodes, topTree_startNode, topTree_endNode);

  LOGF(stderr, "Start %d end: %d Number of Original nodes: %d \n", topTree_startNode, topTree_endNode, topNodeOnTheFlyCount);

  //Next compute the properties
  float4  *topTreeCenters    = new float4 [  topTree_n_nodes];
  float4  *topTreeSizes      = new float4 [  topTree_n_nodes];
  float4  *topTreeMultipole  = new float4 [3*topTree_n_nodes];
  double4 *tempMultipoleRes  = new double4[3*topTree_n_nodes];

  computeProps_TopLevelTree(topTree_n_nodes,
      topTree_n_levels,
      node_levels,
      nodes,
      topTreeCenters,
      topTreeSizes,
      topTreeMultipole,
      &topBoxCenters[0],
      &topBoxSizes[0],
      &topMultiPoles[0],
      tempMultipoleRes);

  //Tree properties computed, now do some magic to put everything in one array

#else
  int topTree_n_nodes = 0;
#endif //DO_NOT_USE_TOP_TREE

  //Modify the offsets of the children to fix the index references to their childs
  for(int i=0; i < topNodeOnTheFlyCount; i++)
  {
    real4 center  = topBoxCenters[i];
    real4 size    = topBoxSizes  [i];
    int   srcProc = topSourceProc[i];

    bool leaf        = center.w <= 0;

    int childinfo    = host_float_as_int(size.w);
    int child, nchild;

    if(childinfo == 0xFFFFFFFF)
    {
      //End point, do not modify it should not be split
      child = childinfo;
    }
    else
    {
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                  //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;        //The number of children this node has

        //Calculate the new start for non-leaf nodes.
        child = child - nodesBegEnd[srcProc].y + topTree_n_nodes + totalTopNodes + nodeSumOffsets[srcProc];
        child = child | (nchild << 28);                        //Merging back in one integer

        if(nchild == 0) child = 0;                             //To prevent incorrect negative values
      }//if !leaf
      else
      { //Leaf
        child   =   childinfo & BODYMASK;                      //the first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag

        child   =  child + particleSumOffsets[srcProc];        //Increasing offset
        child   = child | ((nchild-1) << LEAFBIT);             //Merging back to one integer
      }//end !leaf
    }//if endpoint

    topBoxSizes[i].w =  host_int_as_float(child);      //store the modified offset
  }//For topNodeOnTheFly


  //Compute total particles and total nodes, totalNodes is WITHOUT topNodes
  int totalParticles    = particleSumOffsets[PROCS];
  int totalNodes        = nodeSumOffsets[PROCS];

  //To bind parts of the memory to different textures, the memory start address
  //has to be aligned with XXX bytes, so nodeInformation*sizeof(real4) has to be
  //increased by an offset, so that the node data starts at a XXX byte boundary
  //this is already done on the sending process, but since we modify the structure
  //it has to be done again
  int nodeTextOffset = getTextureAllignmentOffset(totalNodes+totalTopNodes+topTree_n_nodes, sizeof(real4));
  int partTextOffset = getTextureAllignmentOffset(totalParticles                          , sizeof(real4));

  totalParticles    += partTextOffset;

  //Compute the total size of the buffer
  int bufferSize     = 1*(totalParticles) + 5*(totalNodes+totalTopNodes+topTree_n_nodes + nodeTextOffset);

  thisPartLETExTime += get_time() - tStart;
  //Allocate memory on host and device to store the merged tree-structure
  if(bufferSize > remote.fullRemoteTree.get_size())
  {
    //Can only resize if we are sure the LET is not running
    if(letRunning)
    {
      gravStream->sync(); //Wait till the LET run is finished
    }
    remote.fullRemoteTree.cresize_nocpy(bufferSize, false);  //Change the size but ONLY if we need more memory
  }
  tStart = get_time();

  real4 *combinedRemoteTree = &remote.fullRemoteTree[0];

  //First copy the properties of the top_tree nodes and the original top-nodes

#ifndef DO_NOT_USE_TOP_TREE
  //The top-tree node properties
  //Sizes
  memcpy(&combinedRemoteTree[1*(totalParticles)],
      topTreeSizes, sizeof(real4)*topTree_n_nodes);
  //Centers
  memcpy(&combinedRemoteTree[1*(totalParticles) + (totalNodes + totalTopNodes + topTree_n_nodes + nodeTextOffset)],
      topTreeCenters, sizeof(real4)*topTree_n_nodes);
  //Multipoles
  memcpy(&combinedRemoteTree[1*(totalParticles) +
      2*(totalNodes+totalTopNodes+topTree_n_nodes+nodeTextOffset)],
      topTreeMultipole, sizeof(real4)*topTree_n_nodes*3);

  //Cleanup
  delete[] keys;
  delete[] nodes;
  delete[] nodeKeys;
  delete[] topTreeCenters;
  delete[] topTreeSizes;
  delete[] topTreeMultipole;
  delete[] tempMultipoleRes;
#endif

  //The top-boxes properties
  //sizes
  memcpy(&combinedRemoteTree[1*(totalParticles) + topTree_n_nodes],
      &topBoxSizes[0], sizeof(real4)*topNodeOnTheFlyCount);
  //Node center information
  memcpy(&combinedRemoteTree[1*(totalParticles) + (totalNodes + totalTopNodes + topTree_n_nodes + nodeTextOffset) + topTree_n_nodes],
      &topBoxCenters[0], sizeof(real4)*topNodeOnTheFlyCount);
  //Multipole information
  memcpy(&combinedRemoteTree[1*(totalParticles) +
      2*(totalNodes+totalTopNodes+topTree_n_nodes+nodeTextOffset)+3*topTree_n_nodes],
      &topMultiPoles[0], sizeof(real4)*topNodeOnTheFlyCount*3);

  //Copy all the 'normal' pieces of the different trees at the correct memory offsets
  for(int i=0; i < PROCS; i++)
  {
    //Get the properties of the LET, TODO this should be changed in int_as_float instead of casts
    int remoteP      = host_float_as_int(treeBuffers[i+procTrees][0].x);    //Number of particles
    int remoteN      = host_float_as_int(treeBuffers[i+procTrees][0].y);    //Number of nodes
    int remoteB      = host_float_as_int(treeBuffers[i+procTrees][0].z);    //Begin id of top nodes
    int remoteE      = host_float_as_int(treeBuffers[i+procTrees][0].w);    //End   id of top nodes
    int remoteNstart = remoteE-remoteB;

    //Particles
    memcpy(&combinedRemoteTree[particleSumOffsets[i]],   &treeBuffers[i+procTrees][1], sizeof(real4)*remoteP);

    //Non start nodes, nodeSizeInfo
    memcpy(&combinedRemoteTree[1*(totalParticles) +  totalTopNodes + topTree_n_nodes + nodeSumOffsets[i]],
        &treeBuffers[i+procTrees][1+1*remoteP+remoteE], //From the last start node onwards
        sizeof(real4)*(remoteN-remoteE));

    //Non start nodes, nodeCenterInfo
    memcpy(&combinedRemoteTree[1*(totalParticles) + totalTopNodes + topTree_n_nodes + nodeSumOffsets[i] +
        (totalNodes + totalTopNodes + topTree_n_nodes + nodeTextOffset)],
        &treeBuffers[i+procTrees][1+1*remoteP+remoteE + remoteN], //From the last start node onwards
        sizeof(real4)*(remoteN-remoteE));

    //Non start nodes, multipole
    memcpy(&combinedRemoteTree[1*(totalParticles) +  3*(totalTopNodes+topTree_n_nodes) +
        3*nodeSumOffsets[i] + 2*(totalNodes+totalTopNodes+topTree_n_nodes+nodeTextOffset)],
        &treeBuffers[i+procTrees][1+1*remoteP+remoteE*3 + 2*remoteN], //From the last start node onwards
        sizeof(real4)*(remoteN-remoteE)*3);

    /*
       |real4| 1*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
       1 + 1*particleCount + nodeCount + nodeCount + 3*nodeCount

       Info about #particles, #nodes, start and end of tree-walk
       The particle positions
       The nodeSizeData
       The nodeCenterData
       The multipole data, is 3x number of nodes (mono and quadrupole data)

       Now that the data is copied, modify the offsets of the tree so that everything works
       with the new correct locations and references. This takes place in two steps:
       First  the top nodes
       Second the normal nodes
       Has to be done in two steps since they are not continuous in memory if NPROCS > 2
       */

    //Modify the non-top nodes for this process
    int modStart =  totalTopNodes + topTree_n_nodes + nodeSumOffsets[i] + 1*(totalParticles);
    int modEnd   =  modStart      + remoteN-remoteE;

    for(int j=modStart; j < modEnd; j++)
    {
      real4 nodeCenter = combinedRemoteTree[j+totalTopNodes+topTree_n_nodes+totalNodes+nodeTextOffset];
      real4 nodeSize   = combinedRemoteTree[j];
      bool leaf        = nodeCenter.w <= 0;

      int childinfo = host_float_as_int(nodeSize.w);
      int child, nchild;

      if(childinfo == 0xFFFFFFFF)
      { //End point
        child = childinfo;
      }
      else
      {
        if(!leaf)
        {
          //Node
          child    =    childinfo & 0x0FFFFFFF;                   //Index to the first child of the node
          nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has

          //Calculate the new start (non-leaf)
          child = child - nodesBegEnd[i].y + totalTopNodes + topTree_n_nodes + nodeSumOffsets[i];

          child = child | (nchild << 28); //Combine and store

          if(nchild == 0) child = 0;                              //To prevent incorrect negative values
        }else{ //Leaf
          child   =   childinfo & BODYMASK;                       //the first body in the leaf
          nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);      //number of bodies in the leaf masked with the flag

          child = child + particleSumOffsets[i];                 //Modify the particle offsets
          child = child | ((nchild-1) << LEAFBIT);               //Merging the data back into one integer
        }//end !leaf
      }
      combinedRemoteTree[j].w =  host_int_as_float(child);      //Store the modified value
    }//for non-top nodes

#if 0 //Ha-pacs fix
    if(treeBuffersSource[i+procTrees] == 0) //Check if its a point to point source
    {
      delete[] treeBuffers[i+procTrees];    //Free the memory of this part of the LET
      treeBuffers[i+procTrees] = NULL;
    }
#endif


  } //for PROCS

  /*
     The final tree structure looks as follows:
     particlesT1, particlesT2,...mparticlesTn |,
     topNodeSizeT1, topNodeSizeT2,..., topNodeSizeT2 | nodeSizeT1, nodeSizeT2, ...nodeSizeT3 |,
     topNodeCentT1, topNodeCentT2,..., topNodeCentT2 | nodeCentT1, nodeCentT2, ...nodeCentT3 |,
     topNodeMultT1, topNodeMultT2,..., topNodeMultT2 | nodeMultT1, nodeMultT2, ...nodeMultT3

     NOTE that the Multi-pole data consists of 3 float4 values per node
     */
  //     fprintf(stderr,"Modifying the LET took: %g \n", get_time()-t1);

  LOGF(stderr,"Number of local bodies: %d number LET bodies: %d number LET nodes: %d top nodes: %d Processed trees: %d (%d) \n",
      tree.n, totalParticles, totalNodes, totalTopNodes, PROCS, procTrees);

  //Store the tree properties (number of particles, number of nodes, start and end topnode)
  remote.remoteTreeStruct.x = totalParticles;
  remote.remoteTreeStruct.y = totalNodes+totalTopNodes+topTree_n_nodes;
  remote.remoteTreeStruct.z = nodeTextOffset;

#ifndef DO_NOT_USE_TOP_TREE
  //Using this we use our newly build tree as starting point
  totalTopNodes             = topTree_startNode << 16 | topTree_endNode;

  //Using this we get back our original start-points and do not use the extra tree.
  //totalTopNodes             = (topTree_n_nodes << 16) | (topTree_n_nodes+topNodeOnTheFlyCount);
#else
  totalTopNodes             = (0 << 16) | (topNodeOnTheFlyCount);  //If its a merged tree we start at 0
#endif

  remote.remoteTreeStruct.w = totalTopNodes;
  topNodeOnTheFlyCount      = 0; //Reset counters

  delete[] particleSumOffsets;
  delete[] nodeSumOffsets;
  delete[] startNodeSumOffsets;
  delete[] nodesBegEnd;



  thisPartLETExTime += get_time() - tStart;

  //procTrees = recvTree;
  procTrees += PROCS; //Changed since PROCS can be smaller than total number that can be processed


#if 0
  if(iter == 20)
  {
    char fileName[256];
    sprintf(fileName, "letParticles-%d.bin", mpiGetRank());
    ofstream nodeFile;
    //nodeFile.open(nodeFileName.c_str());
    nodeFile.open(fileName, ios::out | ios::binary | ios::app);
    if(nodeFile.is_open())
    {
      for(int i=0; i < totalParticles; i++)
      {
        nodeFile.write((char*)&combinedRemoteTree[i], sizeof(real4));
      }
      nodeFile.close();
    }
  }
#endif

  //Check if we need to summarize which particles are active,
  //only done during the last approximate_gravity_let call
  bool doActivePart = (procTrees == mpiGetNProcs() -1);

  approximate_gravity_let(this->localTree, this->remoteTree, bufferSize, doActivePart);
}


//SSE optimized MAC check
inline int split_node_grav_impbh_sse(
    const _v4sf nodeCOM1,
    const _v4sf boxCenter1,
    const _v4sf boxSize1)
{
  const _v4si mask = {0xffffffff, 0xffffffff, 0xffffffff, 0x0};
  const _v4sf size = __abs(__builtin_ia32_shufps(nodeCOM1, nodeCOM1, 0xFF));

  //mask to prevent NaN signalling / Overflow ? Required to get good pre-SB performance
  const _v4sf nodeCOM   = __builtin_ia32_andps(nodeCOM1,   (_v4sf)mask);
  const _v4sf boxCenter = __builtin_ia32_andps(boxCenter1, (_v4sf)mask);
  const _v4sf boxSize   = __builtin_ia32_andps(boxSize1,   (_v4sf)mask);


  const _v4sf dr   = __abs(boxCenter - nodeCOM) - boxSize;
  const _v4sf ds   = dr + __abs(dr);
  const _v4sf dsq  = ds*ds;
  const _v4sf t1   = __builtin_ia32_haddps(dsq, dsq);
  const _v4sf t2   = __builtin_ia32_haddps(t1, t1);
  const _v4sf ds2  = __builtin_ia32_shufps(t2, t2, 0x00)*(_v4sf){0.25f, 0.25f, 0.25f, 0.25f};


  const float c = 10e-4f;
  const int res = __builtin_ia32_movmskps(
      __builtin_ia32_orps(
        __builtin_ia32_cmpleps(ds2,  size),
        __builtin_ia32_cmpltps(ds2 - size, (_v4sf){c,c,c,c})
        )
      );

  //return 1;
  return res;

}

//Walk the group tree over the local-data tree.
//Counts the number of particles and nodes that will be selected
void octree::tree_walking_tree_stack_versionC13(
    real4 *multipoleS, nInfoStruct* nodeInfoS, //Local Tree
    real4* grpNodeSizeInfoS, real4* grpNodeCenterInfoS, //remote Tree
    int start, int end, int startGrp, int endGrp,
    int &nAcceptedNodes, int &nParticles,
    uint2 *curLevel, uint2 *nextLevel)
{
#ifdef USE_MPI

  //nodeInfo.z bit values:
  //  bit 0 : Node has been visit (1)
  //  bit 1 : Node is split  (2)
  //  bit 2 : Node is a leaf which particles have been added (4)

  const _v4sf*         multipoleV = (const _v4sf*)        multipoleS;
  const _v4sf*   grpNodeSizeInfoV = (const _v4sf*)  grpNodeSizeInfoS;
  const _v4sf* grpNodeCenterInfoV = (const _v4sf*)grpNodeCenterInfoS;

  for(int i=0; i < start; i++) nodeInfoS[i].z = 3;

  nAcceptedNodes = start;
  nParticles     = 0;

  int curLevelCount  = 0;
  int nextLevelCount = 0;

  for(int k = start; k < end; k++)
  {
    uint2 stackItem;
    stackItem.x = k;
    curLevel[curLevelCount++] = stackItem;
  }

  bool overRuleBegin = true;

  while(curLevelCount > 0)
  {
    nextLevelCount = 0;
    for(int idx = 0; idx < curLevelCount; idx++)
      //for(int idx = curLevelCount-1; idx >= 0; idx--) //std::stack order
    {
      const uint2 stackItem  = curLevel[idx];
      const uint nodeID      = stackItem.x;

      //Tree-node information
      const nInfoStruct nodeInfoX = nodeInfoS[nodeID];

      //Early out if this is an accepted leaf node
      if(nodeInfoX.z  & 4) continue;

      //Only mark the first time, saves writes --> Always writing turns out to be faster
      if(nodeInfoX.z == 0)
      {
        nAcceptedNodes++;
        nodeInfoS[nodeID].z = 1;
      }

      //Read the COM and combine it with opening angle criteria from nodeInfoX.x
      _v4sf nodeCOM = multipoleV[nodeID*3];
      nodeCOM       = __builtin_ia32_vec_set_v4sf (nodeCOM, nodeInfoX.x, 3);

      int begin, end;

      //I need this since I can't guarantee that I can encode the start-grp info
      //in the available bytes.
      if(overRuleBegin)
      {
        begin = startGrp; end   = endGrp;
      }
      else
      {
        begin = stackItem.y & 0x0FFFFFFF;
        end   = begin +  ((stackItem.y & 0xF0000000) >> 28) ;
      }

      for(int grpId=begin; grpId <= end; grpId++)
      {
        //Group information
        const _v4sf grpCenter = grpNodeCenterInfoV[grpId];
        const _v4sf grpSize   = grpNodeSizeInfoV[grpId];

        const int split = split_node_grav_impbh_sse(nodeCOM, grpCenter, grpSize);
        //        const int split = 1;

        if(split)
        {
          const bool leaf        = nodeInfoX.x <= 0;

          if(!leaf)
          {
            //nodeInfoS[nodeID].z = nodeInfoS[nodeID].z |  3; //Mark this node as being split, usefull for
            nodeInfoS[nodeID].z = 3; //Mark this node as being split, useful for
            //when creating LET tree, we can mark end-points
            //Sets the split, and visit bits

            const int child    =    nodeInfoX.y & 0x0FFFFFFF;            //Index to the first child of the node
            const int nchild   = (((nodeInfoX.y & 0xF0000000) >> 28)) ;  //The number of children this node has
#if 0
            int childinfoGrp;
            if( __builtin_ia32_vec_ext_v4sf(grpCenter, 3) <= 0)
            { //Its a leaf so we stay with this group
              //              childinfoGrp = grpId | (1) << 28;
              childinfoGrp = grpId;
            }
            else
              childinfoGrp    = __builtin_ia32_vec_ext_v4si((_v4si)grpSize,3);
#else
            int childinfoGrp = grpId;
            //If its not a leaf so we continue down the group-tree
            if( __builtin_ia32_vec_ext_v4sf(grpCenter, 3) > 0)
              childinfoGrp    = __builtin_ia32_vec_ext_v4si((_v4si)grpSize,3);
#endif

            //Go check the child nodes and child grps
            for(int i=child; i < child+nchild; i++)
            { //Add the nodes to the stack
              nextLevel[nextLevelCount++] = make_uint2(i, childinfoGrp);
            }
          }//if !leaf
          else if(nodeInfoS[nodeID].z != 7)
          {
            int nchild  = (((nodeInfoX.y & INVBMASK) >> LEAFBIT)+1);

            //Mark this leaf as completed, no further checks required
            nodeInfoS[nodeID].z = 7;
            nParticles        += nchild;
            break; //We can jump out of these groups
          }//if !leaf
        }//if !split
      }//for grps
    }// end inner while

    //Swap stacks
    uint2 *temp = nextLevel;
    nextLevel   = curLevel;
    curLevel    = temp;

    curLevelCount  = nextLevelCount;
    nextLevelCount = 0;
    overRuleBegin = false;
  } //end inner while
#endif //if USE_MPI
} //end function


//Function that walks over the pre-processed data (processed/generated by the tree-tree walk)
//and fills the LET buffers with the data from the particles and the nodes
void octree::stackFill(real4 *LETBuffer, real4 *nodeCenter, real4* nodeSize,
    real4* bodies, real4 *multipole,
    nInfoStruct *nodeInfo,
    int nParticles, int nNodes,
    int start, int end,
    uint *curLevelStack, uint* nextLevelStack)
{
#ifdef USE_MPI

  int curLeveCount   = 0;
  int nextLevelCount = 0;

  int nStoreIdx = nParticles;

  nParticles = 0;

  int multiStoreIdx = nStoreIdx+2*nNodes; //multipole starts after particles and nodeSize, nodeCenter

  //Copy top nodes, directly after the bodies
  for(int node=0; node < start; node++)
  {
    LETBuffer[nStoreIdx]              = nodeSize[node];
    LETBuffer[nStoreIdx+nNodes]       = nodeCenter[node];
    memcpy(&LETBuffer[multiStoreIdx], &multipole[3*node], sizeof(float4)*(3));
    multiStoreIdx += 3;
    nStoreIdx++;
  }

  int childNodeOffset     = end;

  for(int node=start; node < end; node++){
    curLevelStack[curLeveCount++] = node;
  }

  while(curLeveCount > 0)
  {
    for(int i=0; i < curLeveCount; i++)
    {
      const uint node               = curLevelStack[i];
      const nInfoStruct curNodeInfo = nodeInfo[node];
      nodeInfo[node].z = 0; //Reset the node for next round/tree

      const int childinfo = curNodeInfo.y;
      uint newChildInfo   = 0xFFFFFFFF; //Mark node as not split

      uint child, nchild;

      if(curNodeInfo.z & 2) //Split
      {
        if(curNodeInfo.z & 4)
        { //Leaf that is split
          child   =   childinfo & BODYMASK;
          nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);
          newChildInfo = nParticles; //Set the start of the particle buffer

          memcpy(&LETBuffer[nParticles], &bodies[child], sizeof(float4)*(nchild));
          nParticles += nchild;

          nchild  =  nchild-1; //Minus 1 to get it in right format when combining
          //with newChildInfo
        }
        else
        { //Normal node that is split
          child    =    childinfo & BODYMASK;           //Index to the first child of the node
          nchild   = (((childinfo & INVBMASK) >> LEAFBIT)) ; //The number of children this node has

          newChildInfo     = childNodeOffset;
          childNodeOffset += nchild;

          for(int j=child; j < child+nchild; j++)
          {
            nextLevelStack[nextLevelCount++] = j;
          }//for
        }//if leaf
      }//if split


      //Copy this node info
      LETBuffer[nStoreIdx]            = nodeSize[node];
      LETBuffer[nStoreIdx].w          = host_int_as_float(newChildInfo | (nchild << LEAFBIT));
      LETBuffer[nStoreIdx+nNodes]     = nodeCenter[node];
      memcpy(&LETBuffer[multiStoreIdx], &multipole[3*node], sizeof(float4)*(3));
      multiStoreIdx += 3;
      nStoreIdx++;
      if(procId < 0)
      {
        if(node < 200)
          LOGF(stderr, "Node-normal: %d\tMultipole: %f\n", node, multipole[3*node].x);
      }
    }//end for

    //Swap stacks
    uint *temp          = curLevelStack;
    curLevelStack       = nextLevelStack;
    nextLevelStack      = temp;
    curLeveCount        = nextLevelCount;
    nextLevelCount      = 0;
  }

#endif //if USE_MPI
}//stackFill


//Exchange the LET structure, this is a point to point communication operation
real4* octree::MP_exchange_bhlist(int ibox, int isource,
    int bufferSize, real4 *letDataBuffer)
{
#ifdef USE_MPI
  MPI_Status status;
  int nrecvlist;
  int nlist = bufferSize;

  double t0 = get_time();
  //first send&get the number of particles to send&get
  MPI_Sendrecv(&nlist,1,MPI_INT,ibox,procId*10, &nrecvlist,
      1,MPI_INT,isource,isource*10,MPI_COMM_WORLD, &status);

  double t1= get_time();
  //Resize the buffer so it has the correct size and then exchange the tree
  real4 *recvDataBuffer = new real4[nrecvlist];
  /*
http://books.google.nl/books?id=x79puJ2YkroC&lpg=PA90&ots=54LRnnXOH4&dq=mpi_irecv%20producer%20consumer&pg=PA83#v=onepage&q=mpi_irecv%20producer%20consumer&f=false
http://www.mpi-forum.org/docs/mpi-11-html/node47.html
http://supercomputingblog.com/mpi/mpi-tutorial-5-asynchronous-communication/
http://cs.ucsb.edu/~hnielsen/cs140/mpi-deadlocks.html
http://www.mpi-forum.org/docs/mpi-11-html/node50.html
MPI_test
MPI_wait*/


  double t2=get_time();
  //Particles
  MPI_Sendrecv(&letDataBuffer[0], nlist*sizeof(real4), MPI_BYTE, ibox, procId*10+1,
      &recvDataBuffer[0], nrecvlist*sizeof(real4), MPI_BYTE, isource, isource*10+1,
      MPI_COMM_WORLD, &status);

  LOG("LET Data Exchange: %d <-> %d  sync-size: %f  alloc: %f  data: %f Total: %lg MB : %f \n",
      ibox, isource, t1-t0, t2-t1, get_time()-t2, get_time()-t0, (nlist*sizeof(real4)/(double)(1024*1024)));

  return recvDataBuffer;
#else
  return NULL;
#endif
}



void octree::ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities,  int *bodiesIDs, int toSend)
{
#ifdef USE_MPI
  //First send the number of particles, then the actual sample data
  MPI_Send(&toSend, 1, MPI_INT, destination, destination*2 , MPI_COMM_WORLD);

  //Send the positions, velocities and ids
  MPI_Send( bodyPositions,  toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+1, MPI_COMM_WORLD);
  MPI_Send( bodyVelocities, toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+2, MPI_COMM_WORLD);
  MPI_Send( bodiesIDs,      toSend*sizeof(int),    MPI_BYTE, destination, destination*2+3, MPI_COMM_WORLD);

  /*    MPI_Send( (real*)&bodyPositions[0],  toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+1, MPI_COMM_WORLD);
        MPI_Send( (real*)&bodyVelocities[0], toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+2, MPI_COMM_WORLD);
        MPI_Send( (int *)&bodiesIDs[0],      toSend*sizeof(int),    MPI_BYTE, destination, destination*2+3, MPI_COMM_WORLD);*/
#endif
}

void octree::ICRecv(int recvFrom, vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<int> &bodiesIDs)
{
#ifdef USE_MPI
  MPI_Status status;
  int nreceive;
  int procId = mpiGetRank();

  //First send the number of particles, then the actual sample data
  MPI_Recv(&nreceive, 1, MPI_INT, recvFrom, procId*2, MPI_COMM_WORLD,&status);

  bodyPositions.resize(nreceive);
  bodyVelocities.resize(nreceive);
  bodiesIDs.resize(nreceive);

  //Recv the positions, velocities and ids
  MPI_Recv( (real*)&bodyPositions[0],  nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+1, MPI_COMM_WORLD,&status);
  MPI_Recv( (real*)&bodyVelocities[0], nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+2, MPI_COMM_WORLD,&status);
  MPI_Recv( (int *)&bodiesIDs[0],      nreceive*sizeof(int),    MPI_BYTE, recvFrom, procId*2+3, MPI_COMM_WORLD,&status);
#endif
}

void octree::determine_sample_freq(int numberOfParticles)
{
  //Sum the number of particles on all processes
#ifdef USE_MPI
  //int tmp;
  //MPI_Allreduce(&numberOfParticles,&tmp,1, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
  //nTotalFreq = tmp;

  unsigned long long tmp;
  unsigned long long tmp2 = numberOfParticles;
  MPI_Allreduce(&tmp2,&tmp,1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,MPI_COMM_WORLD);
  nTotalFreq_ull = tmp;
#else
  nTotalFreq = numberOfParticles;
#endif


#ifdef PRINT_MPI_DEBUG
  if(procId == 0)
    LOG("Total number of particles: %llu\n", nTotalFreq_ull);
#endif

  int maxsample = (int)(NMAXSAMPLE*0.8); // 0.8 is safety factor
  sampleFreq = (nTotalFreq_ull+(unsigned long long)maxsample-1)/ (unsigned long long)maxsample;

  if(procId == 0)  LOGF(stderr,"Sample Frequency: %d \n", sampleFreq);

  prevSampFreq = sampleFreq;

}

#if 0
/*************************************************************************
 *                                                                        *
 *                          NON-Used  / Old functions                     *
 *                                                                        *
/*************************************************************************/





//Sort function based on Makinos function
//Sorts (a part) of the coordinate array
//containing the sample particles
//Either sorts the x,y or z direction
//lo is the lower bound of the to sorted part
//up is the upper bound of the to sorted part
//cid is the index/axes to sort
//cid=0=x, cid=1=y and cid=2=z
void octree::sortCoordinates(real4 *r, int lo, int up, int cid )
{
  int i, j;
  real4 tempr;
  while ( up>lo ) {
    i = lo;
    j = up;
    tempr = r[lo];
    /*** Split file in two ***/
    while ( i<j )
    {
      if(cid==0)
        for ( ; r[j].x > tempr.x; j-- );
      else if(cid==1)
        for ( ; r[j].y > tempr.y; j-- );
      else
        for ( ; r[j].z > tempr.z; j-- );

      if(cid==0)
        for ( r[i]=r[j]; i<j && r[i].x <= tempr.x; i++ );
      else if(cid==1)
        for ( r[i]=r[j]; i<j && r[i].y <= tempr.y; i++ );
      else
        for ( r[i]=r[j]; i<j && r[i].z <= tempr.z; i++ );

      r[j] = r[i];
    }
    r[i] = tempr;
    /*** Sort recursively, the smallest first ***/
    if ( i-lo < up-i )
    {
      sortCoordinates(r,lo,i-1,cid);
      lo = i+1;
    }
    else
    {
      sortCoordinates(r,i+1,up,cid);
      up = i-1;
    }
  }
}

bool sortByX (real4 i,real4 j) { return (i.x<j.x); }
bool sortByY (real4 i,real4 j) { return (i.y<j.y); }
bool sortByZ (real4 i,real4 j) { return (i.z<j.z); }


void octree::sortCoordinates2(real4 *r, int lo, int up, int cid )
{
  up += 1;
  if(cid == 0)
    std::sort(&r[lo], &r[up], sortByX);
  else if(cid == 1)
    std::sort(&r[lo], &r[up], sortByY);
  else
    std::sort(&r[lo], &r[up], sortByZ);

}


//Copied from Makino code
void octree::createORB()
{
  int n0, n1;
  n0 = (int)pow(nProcs+0.1,0.33333333333333333333);
  while(nProcs % n0)
    n0--;

  nx = n0;
  n1 = nProcs/nx;
  n0 = (int)sqrt(n1+0.1);
  while(n1 % n0)
    n0++;

  ny = n0; nz = n1/n0;
  int ntmp;
  if (nz > ny){
    ntmp = nz; nz = ny; ny = ntmp;
  }
  if (ny > nx){
    ntmp = nx; nx = ny; ny = ntmp;
  }
  if (nz > ny){
    ntmp = nz; nz = ny; ny = ntmp;
  }
  if (nx*ny*nz != nProcs){
    cerr << "create_division: Intenal Error " << nProcs << " " << nx
      << " " << ny << " " << nz <<endl;
  }

#ifdef PRINT_MPI_DEBUG
  if(procId == 0) LOG("Division: nx: %d ny: %d nz: %d \n", nx, ny, nz);
#endif
}






void octree::sendCurrentRadiusInfoCoarse(real4 *rmin, real4 *rmax, int n_coarseGroups)
{
#ifdef USE_MPI
  int *coarseGrpCountBytes = new int[nProcs];
  int *receiveOffsetsBytes = new int[nProcs];
  //Send the number of coarseGroups that belongs to this process, and gather
  //That information from the other processors
  MPI_Allgather(&n_coarseGroups,            sizeof(int),  MPI_BYTE,
      this->globalCoarseGrpCount, sizeof(uint), MPI_BYTE, MPI_COMM_WORLD);


  //Compute offsets using prefix sum and total number of groups we will receive
  this->globalCoarseGrpOffsets[0] = 0;
  coarseGrpCountBytes[0]          = this->globalCoarseGrpCount[0]*sizeof(real4);
  receiveOffsetsBytes[0]          = 0;
  for(int i=1; i < nProcs; i++)
  {
    this->globalCoarseGrpOffsets[i]  = this->globalCoarseGrpOffsets[i-1] + this->globalCoarseGrpCount[i-1];

    coarseGrpCountBytes[i] = this->globalCoarseGrpCount[i]  *sizeof(real4);
    receiveOffsetsBytes[i] = this->globalCoarseGrpOffsets[i]*sizeof(real4);

    LOGF(stderr,"Proc: %d Received on idx: %d\t%d prefix: %d \n",
        procId, i, globalCoarseGrpCount[i], globalCoarseGrpOffsets[i]);
  }

  int totalNumberOfGroups = this->globalCoarseGrpOffsets[nProcs-1]+this->globalCoarseGrpCount[nProcs-1];

  //Allocate memory
  if(coarseGroupBoundMin) delete[]  coarseGroupBoundMin;
  if(coarseGroupBoundMax) delete[]  coarseGroupBoundMax;

  if(coarseGroupBoxCenter) delete[] coarseGroupBoxCenter;
  if(coarseGroupBoxSize)   delete[] coarseGroupBoxSize;

  coarseGroupBoundMax = new real4[totalNumberOfGroups];
  coarseGroupBoundMin = new real4[totalNumberOfGroups];

  coarseGroupBoxCenter = new double4[totalNumberOfGroups];
  coarseGroupBoxSize   = new double4[totalNumberOfGroups];

  //Exchange the coarse group boundaries
  MPI_Allgatherv(rmin, n_coarseGroups*sizeof(real4), MPI_BYTE,
      coarseGroupBoundMin, coarseGrpCountBytes,
      receiveOffsetsBytes, MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgatherv(rmax, n_coarseGroups*sizeof(real4), MPI_BYTE,
      coarseGroupBoundMax, coarseGrpCountBytes,
      receiveOffsetsBytes, MPI_BYTE, MPI_COMM_WORLD);

  //Compute center and size
  for(int i= 0; i < totalNumberOfGroups; i++)
  {

    double4 boxCenter = {     0.5*(coarseGroupBoundMin[i].x  + coarseGroupBoundMax[i].x),
      0.5*(coarseGroupBoundMin[i].y  + coarseGroupBoundMax[i].y),
      0.5*(coarseGroupBoundMin[i].z  + coarseGroupBoundMax[i].z), 0};
    double4 boxSize   = {fabs(0.5*(coarseGroupBoundMax[i].x - coarseGroupBoundMin[i].x)),
      fabs(0.5*(coarseGroupBoundMax[i].y - coarseGroupBoundMin[i].y)),
      fabs(0.5*(coarseGroupBoundMax[i].z - coarseGroupBoundMin[i].z)), 0};

    coarseGroupBoxCenter[i] = boxCenter;
    coarseGroupBoxSize[i]   = boxSize;
  }

  delete[] coarseGrpCountBytes;
  delete[] receiveOffsetsBytes;
#else
  //TODO check if we need something here
  //  curSysState[0] = curProcState;
#endif
}



//Uses one communication by storing data in one buffer
//nsample can be set to zero if this call is only used
//to get updated domain information
void octree::sendSampleAndRadiusInfo(int nsample, real4 &rmin, real4 &rmax)
{
  sampleRadInfo curProcState;

  curProcState.nsample      = nsample;
  curProcState.rmin         = make_double4(rmin.x, rmin.y, rmin.z, rmin.w);
  curProcState.rmax         = make_double4(rmax.x, rmax.y, rmax.z, rmax.w);

  globalRmax            = 0;
  totalNumberOfSamples  = 0;

#ifdef USE_MPI
  //Get the number of sample particles and the domain size information
  MPI_Allgather(&curProcState, sizeof(sampleRadInfo), MPI_BYTE,  curSysState,
      sizeof(sampleRadInfo), MPI_BYTE, MPI_COMM_WORLD);
#else
  curSysState[0] = curProcState;
#endif

  rmin.x                 =  (real)curSysState[0].rmin.x;
  rmin.y                 =  (real)curSysState[0].rmin.y;
  rmin.z                 =  (real)curSysState[0].rmin.z;

  rmax.x                 =  (real)curSysState[0].rmax.x;
  rmax.y                 =  (real)curSysState[0].rmax.y;
  rmax.z                 =  (real)curSysState[0].rmax.z;

  totalNumberOfSamples   = curSysState[0].nsample;


  for(int i=1; i < nProcs; i++)
  {
    rmin.x = std::min(rmin.x, (real)curSysState[i].rmin.x);
    rmin.y = std::min(rmin.y, (real)curSysState[i].rmin.y);
    rmin.z = std::min(rmin.z, (real)curSysState[i].rmin.z);

    rmax.x = std::max(rmax.x, (real)curSysState[i].rmax.x);
    rmax.y = std::max(rmax.y, (real)curSysState[i].rmax.y);
    rmax.z = std::max(rmax.z, (real)curSysState[i].rmax.z);


    totalNumberOfSamples   += curSysState[i].nsample;
  }

  if(procId == 0)
  {
    if(fabs(rmin.x)>globalRmax)  globalRmax=fabs(rmin.x);
    if(fabs(rmin.y)>globalRmax)  globalRmax=fabs(rmin.y);
    if(fabs(rmin.z)>globalRmax)  globalRmax=fabs(rmin.z);
    if(fabs(rmax.x)>globalRmax)  globalRmax=fabs(rmax.x);
    if(fabs(rmax.y)>globalRmax)  globalRmax=fabs(rmax.y);
    if(fabs(rmax.z)>globalRmax)  globalRmax=fabs(rmax.z);

    if(totalNumberOfSamples > sampleArray.size())
    {
      sampleArray.resize(totalNumberOfSamples);
    }
  }
}

void octree::gpu_collect_sample_particles(int nSample, real4 *sampleParticles)
{
  int *nReceiveCnts  = new int[nProcs];
  int *nReceiveDpls  = new int[nProcs];
  nReceiveCnts[0] = nSample*sizeof(real4);
  nReceiveDpls[0] = 0;

  if(procId == 0)
  {
    for(int i=1; i < nProcs; i++)
    {
      nReceiveCnts[i] = curSysState[i].nsample*sizeof(real4);
      nReceiveDpls[i] = nReceiveDpls[i-1] + nReceiveCnts[i-1];
    }
  }

  //Collect sample particles
#ifdef USE_MPI
  MPI_Gatherv(&sampleParticles[0], nSample*sizeof(real4), MPI_BYTE,
      &sampleArray[0], nReceiveCnts, nReceiveDpls, MPI_BYTE,
      0, MPI_COMM_WORLD);
#else
  std::copy(sampleParticles, sampleParticles + nSample, sampleArray.begin());
#endif

  delete[] nReceiveCnts;
  delete[] nReceiveDpls;
}


void octree::collect_sample_particles(real4 *bodies,
    int nbody,
    int sample_freq,
    vector<real4> &sampleArray,
    int &nsample,
    double &rmax)
{
  //Select the sample particles
  int ii, i;
  for(i = ii= 0;ii<nbody; i++,ii+=sample_freq)
  {
    sampleArray.push_back(bodies[ii]);
  }
  nsample = i;

  //Now gather the particles at process 0
  //NOTE: Im using my own implementation instead of makino's which
  //can grow out of the memory array (I think...)

  //Instead of using mpi-reduce but broadcast or something we can receive the
  //individual values,so we dont haveto send them in the part below, saves
  //communication time!!!  This function is only used once so no problem
  int *nSampleValues = new int[nProcs];
  int *nReceiveCnts  = new int[nProcs];
  int *nReceiveDpls  = new int[nProcs];

#ifdef USE_MPI
  MPI_Gather(&nsample, 1, MPI_INT, nSampleValues, 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
  nSampleValues[0] = nsample;
#endif

  //Increase the size of the result buffer if needed
  if(procId == 0)
  {
    //Sum the total amount of sample particles
    unsigned int totalNumberOfSamples = 0;

    for(int i=0; i < nProcs; i++)
    {
      totalNumberOfSamples += nSampleValues[i];
    }

    if(totalNumberOfSamples > sampleArray.size())
    {
      sampleArray.resize(totalNumberOfSamples);
    }
  }

  //Compute buffer and displacements for MPI_Gatherv
  nReceiveCnts[0] = nsample*sizeof(real4);
  nReceiveDpls[0] = 0;

  if(procId == 0)
  {
    for(int i=1; i < nProcs; i++)
    {
      nReceiveCnts[i] = nSampleValues[i]*sizeof(real4);
      nReceiveDpls[i] = nReceiveDpls[i-1] + nReceiveCnts[i-1];
    }
  }

  //Collect sample particles, note the MPI_IN_PLACE to prevent MPI errors
#ifdef USE_MPI
  MPI_Gatherv((procId ? &sampleArray[0] : MPI_IN_PLACE), nsample*sizeof(real4), MPI_BYTE,
      &sampleArray[0], nReceiveCnts, nReceiveDpls, MPI_BYTE,
      0, MPI_COMM_WORLD);
#endif

  nsample = (nReceiveCnts[mpiGetNProcs()-1] +  nReceiveDpls[mpiGetNProcs()-1]) / sizeof(real4);

  //Find the maximum particle position
  double tmp = 0;
  for(i = 0;i<nbody; i++)
  {
    real4 r = bodies[i];
    //check x,y and z
    if(fabs(r.x)>tmp)  tmp=fabs(r.x);
    if(fabs(r.y)>tmp)  tmp=fabs(r.y);
    if(fabs(r.z)>tmp)  tmp=fabs(r.z);
  }

  //Find the global maximum
#ifdef USE_MPI
  MPI_Allreduce(&tmp, &rmax,1, MPI_DOUBLE, MPI_MAX,MPI_COMM_WORLD);
#else
  rmax = tmp;
#endif

  delete[] nSampleValues;
  delete[] nReceiveCnts;
  delete[] nReceiveDpls;
}

void octree::createDistribution(real4 *bodies, int n_bodies)
{
  determine_sample_freq(n_bodies);

  vector<real4> sampleArray;
  sampleArray.reserve(NMAXSAMPLE);

  int     nsample;  //Number of samples for this process
  double  rmax;     //maximum coordinate used to create a box

  //Get the sample particles from the other processes
  collect_sample_particles(bodies, n_bodies, sampleFreq, sampleArray, nsample, rmax);

  //Now that we have a sample from all proces we setup the space division
  //Processor 0 determines the division
  if(procId == 0)
    determine_division(nsample, sampleArray,nx, ny, nz, rmax,domainRLow, domainRHigh);

#ifdef USE_MPI
  //Now broadcast the results to all other processes
  MPI_Bcast(domainRLow,  sizeof(double4)*nProcs,MPI_BYTE,0,MPI_COMM_WORLD);
  MPI_Bcast(domainRHigh, sizeof(double4)*nProcs,MPI_BYTE,0,MPI_COMM_WORLD);
#endif

  return;
}


/*
   Only update the box-sizes, box-boundaries
   of the different processes, do not do
   anything related to sample particles
   */
void octree::gpu_updateDomainOnly()
{
  real4 r_min, r_max;
  //Get the current system/particle boundaries
  this->getBoundaries(localTree, r_min, r_max);

  int nSamples = 0;
  this->sendSampleAndRadiusInfo(nSamples, r_min, r_max);
  rMinGlobal = r_min;
  rMaxGlobal = r_max;
}



//Calculates the dimension of the box
//np number of sample particles
//pos the sample particle positions
//cid the coordinate index, 0=x, 1=y, 2=z
//istart/iend the start and end position of the sorted array
//rmax the maximum coordinate
//xlow/xhigh the box coordinates
void octree::calculate_boxdim(int np, real4 pos[], int cid, int istart, int iend,
    double rmax, double & xlow, double & xhigh)
{
  if(istart == 0)
  {
    xlow = -rmax;
  }
  else
  {
    if(cid==0)
      xlow = (pos[istart].x + pos[istart-1].x)/2;
    else if(cid==1)
      xlow = (pos[istart].y + pos[istart-1].y)/2;
    else
      xlow = (pos[istart].z + pos[istart-1].z)/2;
  }

  if(iend == np-1)
  {
    xhigh = rmax;
  }
  else
  {
    if(cid==0)
      xhigh = (pos[iend].x + pos[iend+1].x)/2;
    else if(cid==1)
      xhigh = (pos[iend].y + pos[iend+1].y)/2;
    else
      xhigh = (pos[iend].z + pos[iend+1].z)/2;
  }
}

inline double computeNewCoordinate(double old, double newc, int prevChange)
{

  //return newc;

  int curChange = (fabs(old) > fabs(newc));

  double factor1 = 1, factor2 = 2, factor3 = 1;

  if(prevChange != curChange)
  {
    //Different direction take half a step
    factor1 = 1; factor2 = 2;
  }
  else
  {
    //Same direction, take some bigger step (3/4th)
    factor1 = 3; factor2 = 4;

    //Same direction, take full step
    //    factor3 = 0; factor1 = 1; factor2 = 1;
  }

  double temp = (factor3*old + factor1*newc) / factor2;

  return temp;

  //Default
  //return newc;
  //avg
  //  return (old+newc)/2;
  //3/4:
  //  return (old + 3*newc) / 4;

}

void octree::determine_division(int np,         // number of particles
    vector<real4> &pos,     // positions of particles
    int nx,
    int ny,
    int nz,
    double rmax,
    double4 xlow[],         // left-bottom coordinate of divisions
    double4 xhigh[])        // size of divisions
{
  int numberOfProcs = nProcs;
  int *istart  = new int[numberOfProcs+1];
  int *iend    = new int[numberOfProcs+1];
  int n = nx*ny*nz;

  //     fprintf(stderr, "TIME4 TEST:  %d  %d \n", np-1, pos.size());
  //
  //     double t1 = get_time();
  sortCoordinates(&pos[0], 0, np-1, 0);
  //     sortCoordinates2(&pos[0], 0, np-1, 0);

  //     double t2 = get_time();
  //     fprintf(stderr, "TIME4 TEST: %g  %d \n", t2-t1, np-1);

  //Split the array in more or less equal parts
  for(int i = 0;i<n;i++)
  {
    istart[i] = (i*np)/n;
    //NOTE: It was i>= 0, changed this otherwise it writes before begin array...
    if(i > 0 )
      iend[i-1]=istart[i]-1;
  }
  iend[n-1] = np-1;

  //     borderCnt++;

  //Split the x-axis
  for(int ix = 0;ix<nx;ix++)
  {
    double x0, x1;
    int ix0 = ix*ny*nz;
    int ix1 = (ix+1)*ny*nz;
    calculate_boxdim(np, &pos[0], 0,istart[ix0],iend[ix1-1],rmax,x0,x1);
    for(int i=ix0; i<ix1; i++)
    {
      //Check the domain borders and set a constant
      if(istart[ix0] == 0)
      {
        xlowPrev[i].x = 10e10;
      }
      if(iend[ix1-1] == np-1)
      {
        xhighPrev[i].x = 10e10;
      }

      xlow[i].x   = x0;
      xhigh[i].x  = x1;
    }
  }

  //For each x split the various y parts
  for(int ix = 0;ix<nx;ix++)
  {
    int ix0 = ix*ny*nz;
    int ix1 = (ix+1)*ny*nz;
    int npy = iend[ix1-1] - istart[ix0] + 1;
    sortCoordinates(&pos[0], istart[ix0],iend[ix1-1], 1);
    for(int iy = 0;iy<ny;iy++){
      double y0, y1;
      int iy0 = ix0+iy*nz;
      int iy1 = ix0+(iy+1)*nz;
      calculate_boxdim(npy, &pos[istart[ix0]], 1,istart[iy0]-istart[ix0],
          iend[iy1-1]-istart[ix0], rmax, y0,y1);
      for(int i=iy0; i<iy1; i++)
      {
        //Check domain borders and set a constant
        if(istart[iy0]-istart[ix0] == 0)
        {
          xlowPrev[i].y = 10e10;
        }
        if( iend[iy1-1]-istart[ix0] == npy-1)
        {
          xhighPrev[i].y = 10e10;
        }

        xlow[i].y  = y0;
        xhigh[i].y = y1;
      }
    }
  }

  //For each x and for each y split the z axis
  for(int ix = 0;ix<nx;ix++){
    int ix0 = ix*ny*nz;
    for(int iy = 0;iy<ny;iy++){
      int iy0 = ix0+iy*nz;
      int iy1 = ix0+(iy+1)*nz;
      int npz = iend[iy1-1] - istart[iy0] + 1;
      sortCoordinates(&pos[0], istart[iy0],iend[iy1-1], 2);
      for(int iz = 0;iz<nz;iz++){
        double z0, z1;
        int iz0 = iy0+iz;
        calculate_boxdim(npz, &pos[istart[iy0]], 2,istart[iz0]-istart[iy0],
            iend[iz0]-istart[iy0], rmax, z0,z1);

        //Check the domain borders
        if(istart[iz0]-istart[iy0] == 0)
        {
          xlowPrev[iz0].z = 10e10;
        }
        if(iend[iz0]-istart[iy0] == npz-1)
        {
          xhighPrev[iz0].z = 10e10;
        }

        xlow[iz0].z   = z0;
        xhigh[iz0].z  = z1;
      }
    }
  }


  //Do the magic to get a better load balance by changing the decompositon
  //slightly
  static bool isFirstStep  = true;

  if(!isFirstStep)
  {
    double temp;
    for(int i=0; i < nProcs; i++)
    {
      //       fprintf(stderr,"DOMAIN LOW  %d || CUR: %f %f %f \tPREV: %f %f %f\n", i,
      //               xlow[i].x, xlow[i].y, xlow[i].z,
      //               xlowPrev[i].x, xlowPrev[i].y, xlowPrev[i].z);
      //       fprintf(stderr,"DOMAIN HIGH %d || CUR: %f %f %f \tPREV: %f %f %f\n", i,
      //               xhigh[i].x, xhigh[i].y, xhigh[i].z,
      //               xhighPrev[i].x, xhighPrev[i].y, xhighPrev[i].z);
      //
      //Do magic !
      if(xlowPrev[i].x != 10e10)
      {
        temp               = computeNewCoordinate(xlowPrev[i].x, xlow[i].x, domHistoryLow[i].x);
        domHistoryLow[i].x = (abs(xlowPrev[i].x) > fabs(xlow[i].x));
        xlow[i].x          = temp;
      }
      if(xhighPrev[i].x != 10e10)
      {
        temp                 = computeNewCoordinate(xhighPrev[i].x, xhigh[i].x, domHistoryHigh[i].x);
        domHistoryHigh[i].x  = (abs(xhighPrev[i].x) > fabs(xhigh[i].x));
        xhigh[i].x           = temp;
      }

      if(xlowPrev[i].y != 10e10)
      {
        temp               = computeNewCoordinate(xlowPrev[i].y, xlow[i].y, domHistoryLow[i].y);
        domHistoryLow[i].y = (abs(xlowPrev[i].y) > fabs(xlow[i].y));
        xlow[i].y          = temp;
      }
      if(xhighPrev[i].y != 10e10)
      {
        temp                 = computeNewCoordinate(xhighPrev[i].y, xhigh[i].y, domHistoryHigh[i].y);
        domHistoryHigh[i].y  = (abs(xhighPrev[i].y) > fabs(xhigh[i].y));
        xhigh[i].y           = temp;
      }

      if(xlowPrev[i].z != 10e10)
      {
        temp               = computeNewCoordinate(xlowPrev[i].z, xlow[i].z, domHistoryLow[i].z);
        domHistoryLow[i].z = (abs(xlowPrev[i].z) > fabs(xlow[i].z));
        xlow[i].z          = temp;
      }
      if(xhighPrev[i].z != 10e10)
      {
        temp                 = computeNewCoordinate(xhighPrev[i].z, xhigh[i].z, domHistoryHigh[i].z);
        domHistoryHigh[i].z  = (abs(xhighPrev[i].z) > fabs(xhigh[i].z));
        xhigh[i].z           = temp;
      }

    }
  }


  //Copy the current decomposition to the previous for next step
  for(int i=0; i < nProcs; i++)
  {
    xlowPrev[i]   = xlow[i];
    xhighPrev[i]  = xhigh[i];
  }

  isFirstStep = false;

  //Free up memory
  delete[] istart;
  delete[] iend;

  return;
}



/*
   Get the domain boundaries
   Get the sample particles
   Send them to process 0
   Process 0 computes the new domain decomposition and broadcasts this
   */
void octree::gpu_updateDomainDistribution(double timeLocal)
{
  my_dev::dev_stream aSyncStream;

  real4 r_min, r_max;
  //    double t1 = get_time();
  //Get the current system/particle boundaries
  this->getBoundaries(localTree, r_min, r_max);

  int finalNRate;

  //Average the previous and current execution time to make everything smoother
  //results in much better load-balance
  prevDurStep = (prevDurStep <= 0) ? timeLocal : prevDurStep;
  timeLocal   = (timeLocal + prevDurStep) / 2;

  double nrate = 0;
#if 1
  //Only base load balancing on the computation time
  double timeSum   = 0.0;

  //Sum the execution times
#ifdef USE_MPI
  MPI_Allreduce( &timeLocal, &timeSum, 1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  timeSum = timeLocal;
#endif

  nrate = timeLocal / timeSum;

  if(1)       //Don't fluctuate particles too much
  {
#define SAMPLING_LOWER_LIMIT_FACTOR  (1.9)

    double nrate2 = (double)localTree.n / (double) nTotalFreq;
    nrate2       /= SAMPLING_LOWER_LIMIT_FACTOR;

    if(nrate < nrate2)
    {
      nrate = nrate2;
    }

    double nrate2_sum = 0.0;
#ifdef USE_MPI
    MPI_Allreduce( &nrate, &nrate2_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    nrate2_sum = nrate;
#endif

    nrate /= nrate2_sum;
  }
#else
  //Equal number of particles
  nrate = (double)localTree.n / (double)nTotalFreq;
#endif

  int    nsamp    = (int)(nTotalFreq *0.001f/4.0) + 1;  //Total number of sample particles, global
  int nsamp_local = (int)(nsamp*nrate) + 1;
  int nSamples    = nsamp_local;

  finalNRate      = localTree.n / nsamp_local;

  LOGF(stderr, "NSAMP [%d]: sample: %d nrate: %f finalrate: %d localTree.n: %d  \
      previous: %d timeLocal: %f prevTimeLocal: %f \n",
      procId, nsamp_local, nrate, finalNRate, localTree.n, prevSampFreq,
      timeLocal, prevDurStep);

  prevDurStep  = timeLocal;
  prevSampFreq = finalNRate;


  my_dev::dev_mem<real4>  samplePositions(devContext);

  samplePositions.cmalloc_copy(localTree.generalBuffer1, nSamples, 0);


  //   double t2 = get_time();
  //   fprintf(stderr, "TIME1 (boundaries) %g \n", t2 - t1);

  //Get the sample particles from the device and only copy
  //the number of particles that is used
  //Action overlaps with the communication of domain boundary
  extractSampleParticles.set_arg<int>(0,     &localTree.n);
  extractSampleParticles.set_arg<int>(1,     &finalNRate);
  extractSampleParticles.set_arg<cl_mem>(2,  localTree.bodies_Ppos.p());
  extractSampleParticles.set_arg<cl_mem>(3,  samplePositions.p());
  extractSampleParticles.setWork(nSamples, 256);
  extractSampleParticles.execute(aSyncStream.s());


  //JB since Fermi had problems with pinned memory
  //we cant do this async
  if (this->getDevContext()->getComputeCapability() < 350)
  {
    aSyncStream.sync();
    samplePositions.d2h(nSamples);
  }
  else
  {
    samplePositions.d2h(nSamples, false, aSyncStream.s());
  }


  //Get number of sample particles per process and domain size information
  this->sendSampleAndRadiusInfo(nSamples, r_min, r_max);
  rMinGlobal = r_min;
  rMaxGlobal = r_max;

  //    double t3 = get_time();
  //   fprintf(stderr, "TIME2 (get and send sample info) %g \t %g \n", t3 - t2, t3-t1);
  aSyncStream.sync();
  gpu_collect_sample_particles(nSamples, &samplePositions[0]);

  //double t4 = get_time();
  //fprintf(stderr, "TIME3 (get and send sample particles) %g \t %g \n", t4 - t3, t4-t1);

  //Processor 0 determines the division
  if(procId == 0)
    determine_division(totalNumberOfSamples, sampleArray,nx, ny, nz, globalRmax, domainRLow, domainRHigh);

  //   double t5 = get_time();
  //   fprintf(stderr, "TIME4 (determ div ) %g \t %g \n", t5 - t4, t5-t1);

  //Now broadcast the results to all other processes
#ifdef USE_MPI
  MPI_Bcast(domainRLow,  sizeof(double4)*nProcs,MPI_BYTE,0,MPI_COMM_WORLD);
  MPI_Bcast(domainRHigh, sizeof(double4)*nProcs,MPI_BYTE,0,MPI_COMM_WORLD);
#endif

  //   double t5 = get_time();
  //   fprintf(stderr, "TIME4 (determ div and bcast) %g \t %g \n", t5 - t4, t5-t1);
  //   fprintf(stderr, "TIME4 (Total sample part)  %g \n", t5-t1);

  //   if(this->nProcs > 1)
  //   {
  //     if(this->procId == 0)
  //       for(int i = 0;i< this->nProcs;i++)
  //       {
  //         cerr << "Domain: " << i << " " << this->domainRLow[i].x << " " << this->domainRLow[i].y << " " << this->domainRLow[i].z << " "
  //                                        << this->domainRHigh[i].x << " " << this->domainRHigh[i].y << " " << this->domainRHigh[i].z <<endl;
  //       }
  //   }


  return;
}



//Checks if the position falls within the specified box
inline int isinbox(real4 pos, double4 xlow, double4 xhigh)
{
  if((pos.x < xlow.x)||(pos.x > xhigh.x))
    return 0;
  if((pos.y < xlow.y)||(pos.y > xhigh.y))
    return 0;
  if((pos.z < xlow.z)||(pos.z > xhigh.z))
    return 0;

  return 1;
}




//Send particles to the appropriate processors
int octree::exchange_particles_with_overflow_check(tree_structure &tree)
{
  int myid      = procId;
  int nproc     = nProcs;
  int iloc      = 0;
  int totalsent = 0;
  int nbody     = tree.n;


  real4  *bodiesPositions = &tree.bodies_pos[0];
  real4  *velocities      = &tree.bodies_vel[0];
  real4  *bodiesAcc0      = &tree.bodies_acc0[0];
  real4  *bodiesAcc1      = &tree.bodies_acc1[0];
  float2 *bodiesTime      = &tree.bodies_time[0];
  int    *bodiesIds       = &tree.bodies_ids[0];
  real4  *predictedBodiesPositions = &tree.bodies_Ppos[0];
  real4  *predictedVelocities      = &tree.bodies_Pvel[0];

  real4  tmpp;
  float2 tmpp2;
  int    tmpp3;
  int *firstloc   = new int[nProcs+1];
  int *nparticles = new int[nProcs+1];

  // Loop over particles and determine which particle needs to go where
  // reorder the bodies in such a way that bodies that have to be send
  // away are stored after each other in the array
  double t1 = get_time();

  //Array reserve some memory at forehand , 1%
  vector<bodyStruct> array2Send;
  //vector<bodyStruct> array2Send(((int)(tree.n * 0.01)));

  for(int ib=0;ib<nproc;ib++)
  {
    int ibox       = (ib+myid)%nproc;
    firstloc[ibox] = iloc;      //Index of the first particle send to proc: ibox

    for(int i=iloc; i<nbody;i++)
    {
      //      if(myid == 0){PRC(i); PRC(pb[i].get_pos());}
      if(isinbox(predictedBodiesPositions[i], domainRLow[ibox], domainRHigh[ibox]))
      {
        //Position
        tmpp                  = bodiesPositions[iloc];
        bodiesPositions[iloc] = bodiesPositions[i];
        bodiesPositions[i]    = tmpp;
        //Velocity
        tmpp             = velocities[iloc];
        velocities[iloc] = velocities[i];
        velocities[i]    = tmpp;
        //Acc0
        tmpp             = bodiesAcc0[iloc];
        bodiesAcc0[iloc] = bodiesAcc0[i];
        bodiesAcc0[i]    = tmpp;
        //Acc1
        tmpp             = bodiesAcc1[iloc];
        bodiesAcc1[iloc] = bodiesAcc1[i];
        bodiesAcc1[i]    = tmpp;
        //Predicted position
        tmpp                           = predictedBodiesPositions[iloc];
        predictedBodiesPositions[iloc] = predictedBodiesPositions[i];
        predictedBodiesPositions[i]    = tmpp;
        //Predicted velocity
        tmpp                  = predictedVelocities[iloc];
        predictedVelocities[iloc] = predictedVelocities[i];
        predictedVelocities[i]    = tmpp;
        //Time-step
        tmpp2            = bodiesTime[iloc];
        bodiesTime[iloc] = bodiesTime[i];
        bodiesTime[i]    = tmpp2;
        //IDs
        tmpp3            = bodiesIds[iloc];
        bodiesIds[iloc]  = bodiesIds[i];
        bodiesIds[i]     = tmpp3;

        //Put the particle in the array of to send particles
        if(ibox != myid)
        {
          bodyStruct body;
          body.pos  = bodiesPositions[iloc];
          body.vel  = velocities[iloc];
          body.acc0 = bodiesAcc0[iloc];
          body.acc1 = bodiesAcc1[iloc];
          body.time = bodiesTime[iloc];
          body.id   = bodiesIds[iloc];
          body.Ppos  = predictedBodiesPositions[iloc];
          body.Pvel  = predictedVelocities[iloc];
          array2Send.push_back(body);
        }

        iloc++;
      }// end if
    }//for i=iloc
    nparticles[ibox] = iloc-firstloc[ibox];//Number of particles that has to be send to proc: ibox
  } // for(int ib=0;ib<nproc;ib++)

  LOG("Required search time: %lg ,proc: %d found in our own box: %d n: %d  send to others: %ld \n",
      get_time()-t1, myid, nparticles[myid], tree.n, array2Send.size());

  t1 = get_time();

  totalsent = nbody - nparticles[myid];

  int tmp;
#ifdef USE_MPI
  MPI_Reduce(&totalsent,&tmp,1, MPI_INT, MPI_SUM,0,MPI_COMM_WORLD);
#else
  tmp = totalsent;
#endif

  if(procId == 0)
  {
    totalsent = tmp;
    LOG("Exchanged particles = %d \n", totalsent);
  }

  if(iloc < nbody)
  {
    cerr << procId <<" exchange_particle error: particle in no box...iloc: " << iloc
      << " and nbody: " << nbody << "\n";
  }

  vector<bodyStruct> recv_buffer3(nbody- nparticles[myid]);

  int tempidFirst, tempRecvCount;
  unsigned int recvCount = 0;

  //Exchange the data with the other processors
  int ibend = -1;
  int nsend = 0;
  int isource = 0;
  for(int ib=nproc-1;ib>0;ib--)
  {
    int ibox = (ib+myid)%nproc; //index to send...

    if (ib == nproc-1)
    {
      isource= (myid+1)%nproc;
    }
    else
    {
      isource = (isource+1)%nproc;
      if (isource == myid)isource = (isource+1)%nproc;
    }

    if(MP_exchange_particle_with_overflow_check<bodyStruct>(ibox, &array2Send[0],
          recv_buffer3, firstloc[ibox] - nparticles[myid],
          nparticles[ibox], isource,
          nsend, recvCount))
    {
      ibend = ibox; //Here we get if exchange failed
      ib = 0;
    }//end if mp exchang
  }//end for all boxes

  LOG("Required inter-process communication time: %lg ,proc: %d\n", get_time()-t1, myid);
  t1 = get_time();
  double t2= t1;

  //    ... should do something different for nsend...
  int idfirst;
  if(ibend >= 0)
  {
    idfirst = firstloc[ibend]+nparticles[ibend]-nsend;
  }
  else
  {
    idfirst = nparticles[myid];
  }

  //Have to resize the bodies vector to keep the numbering correct
  tree.setN(idfirst+recvCount);
  tree.bodies_pos.cresize (idfirst+recvCount + 1, false);
  tree.bodies_acc0.cresize(idfirst+recvCount,     false);
  tree.bodies_acc1.cresize(idfirst+recvCount,     false);
  tree.bodies_vel.cresize (idfirst+recvCount,     false);
  tree.bodies_time.cresize(idfirst+recvCount,     false);
  tree.bodies_ids.cresize (idfirst+recvCount + 1, false);
  tree.bodies_Ppos.cresize(idfirst+recvCount + 1, false);
  tree.bodies_Pvel.cresize(idfirst+recvCount + 1, false);

  //This one has to be at least the same size as the number of particles inorder to
  //have enough space to store the other buffers
  tree.generalBuffer1.cresize(3*(idfirst+recvCount)*4, false);

  LOG("Benodigde gpu malloc tijd stap 1: %lg \t Size: %d \tRank: %d \t Size: %d \n",
      get_time()-t1, idfirst+recvCount, mpiGetRank(), tree.bodies_Ppos.get_size());
  t1 = get_time();

  tempidFirst = idfirst; tempRecvCount = recvCount;

  //Copy data from struct into the main arrays
  for(unsigned int P=0; P < recvCount; P++)
  {
    tree.bodies_pos[idfirst+P]  = recv_buffer3[P].pos;        tree.bodies_vel[idfirst+P]      = recv_buffer3[P].vel;
    tree.bodies_acc0[idfirst+P] = recv_buffer3[P].acc0;       tree.bodies_acc1[idfirst+P]     = recv_buffer3[P].acc1;
    tree.bodies_time[idfirst+P] = recv_buffer3[P].time;       tree.bodies_ids[idfirst+P]      = recv_buffer3[P].id;
    tree.bodies_Ppos[idfirst+P] = recv_buffer3[P].Ppos;       tree.bodies_Pvel[idfirst+P]     = recv_buffer3[P].Pvel;
  }

  LOG("Required DATA in struct copy time: %lg \n", get_time()-t1); t1 = get_time();


  if(ibend == -1){

  }else{
    //Something went wrong
    cerr << "ERROR in exchange_particles_with_overflow_check! \n"; exit(0);
  }


  //Resize the arrays of the tree
  reallocateParticleMemory(tree);

  LOG("Required gpu malloc time step 2: %lg \n", get_time()-t1);
  LOG("Total GPU interaction time: %lg \n", get_time()-t2);

  int retValue = 0;


  delete[] firstloc;
  delete[] nparticles;

  return retValue;
}




//Function that uses the GPU to get a set of particles that have to be
//send to other processes
void octree::gpuRedistributeParticles()
{
  //Memory buffers to hold the extracted particle information
  my_dev::dev_mem<uint>  validList(devContext);
  my_dev::dev_mem<uint>  compactList(devContext);

  int memOffset1 = compactList.cmalloc_copy(localTree.generalBuffer1,
      localTree.n, 0);
  int memOffset2 = validList.cmalloc_copy(localTree.generalBuffer1,
      localTree.n, memOffset1);

  double4 thisXlow  = domainRLow [this->procId];
  double4 thisXhigh = domainRHigh[this->procId];

  domainCheck.set_arg<int>(0,     &localTree.n);
  domainCheck.set_arg<double4>(1, &thisXlow);
  domainCheck.set_arg<double4>(2, &thisXhigh);
  domainCheck.set_arg<cl_mem>(3,  localTree.bodies_Ppos.p());
  domainCheck.set_arg<cl_mem>(4,  validList.p());
  domainCheck.setWork(localTree.n, 128);
  domainCheck.execute(execStream->s());

  //Create a list of valid and invalid particles
  this->resetCompact();  //Make sure compact has been reset
  int validCount;
  gpuSplit(devContext, validList, compactList, localTree.n, &validCount);


  //Check if the memory size, of the generalBuffer is large enough to store the exported particles
  int tempSize = localTree.generalBuffer1.get_size() - localTree.n;
  int needSize = (int)(1.01f*(validCount*(sizeof(bodyStruct)/sizeof(int))));

  if(tempSize < needSize)
  {
    int itemsNeeded = needSize + localTree.n + 4092; //Slightly larger as before for offset space

    //Copy the compact list to the host we need this list intact
    compactList.d2h();
    int *tempBuf = new int[localTree.n];
    memcpy(tempBuf, &compactList[0], localTree.n*sizeof(int));

    //Resize the general buffer
    localTree.generalBuffer1.cresize(itemsNeeded, false);
    //Reset memory pointers
    memOffset1 = compactList.cmalloc_copy(localTree.generalBuffer1,
        localTree.n, 0);

    //Restore the compactList
    memcpy(&compactList[0], tempBuf, localTree.n*sizeof(int));
    compactList.h2d();

    delete[] tempBuf;
  }

  my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);

  memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1,
      localTree.n, memOffset1);

  extractOutOfDomainBody.set_arg<int>(0,    &validCount);
  extractOutOfDomainBody.set_arg<cl_mem>(1, compactList.p());
  extractOutOfDomainBody.set_arg<cl_mem>(2, localTree.bodies_Ppos.p());
  extractOutOfDomainBody.set_arg<cl_mem>(3, localTree.bodies_Pvel.p());
  extractOutOfDomainBody.set_arg<cl_mem>(4, localTree.bodies_pos.p());
  extractOutOfDomainBody.set_arg<cl_mem>(5, localTree.bodies_vel.p());
  extractOutOfDomainBody.set_arg<cl_mem>(6, localTree.bodies_acc0.p());
  extractOutOfDomainBody.set_arg<cl_mem>(7, localTree.bodies_acc1.p());
  extractOutOfDomainBody.set_arg<cl_mem>(8, localTree.bodies_time.p());
  extractOutOfDomainBody.set_arg<cl_mem>(9, localTree.bodies_ids.p());
  extractOutOfDomainBody.set_arg<cl_mem>(10, bodyBuffer.p());
  extractOutOfDomainBody.setWork(validCount, 128);
  extractOutOfDomainBody.execute(execStream->s());

  bodyBuffer.d2h(validCount);

  //Now we have to move particles from the back of the array to the invalid spots
  //this can be done in parallel with exchange operation to hide some time

  //One integer for counting, true-> initialize to zero so counting starts at 0
  my_dev::dev_mem<uint>  atomicBuff(devContext, 1, true);

  //Internal particle movement
  internalMove.set_arg<int>(0,    &validCount);
  internalMove.set_arg<int>(1,    &localTree.n);
  internalMove.set_arg<double4>(2,    &thisXlow);
  internalMove.set_arg<double4>(3,    &thisXhigh);
  internalMove.set_arg<cl_mem>(4, compactList.p());
  internalMove.set_arg<cl_mem>(5, atomicBuff.p());
  internalMove.set_arg<cl_mem>(6, localTree.bodies_Ppos.p());
  internalMove.set_arg<cl_mem>(7, localTree.bodies_Pvel.p());
  internalMove.set_arg<cl_mem>(8, localTree.bodies_pos.p());
  internalMove.set_arg<cl_mem>(9, localTree.bodies_vel.p());
  internalMove.set_arg<cl_mem>(10, localTree.bodies_acc0.p());
  internalMove.set_arg<cl_mem>(11, localTree.bodies_acc1.p());
  internalMove.set_arg<cl_mem>(12, localTree.bodies_time.p());
  internalMove.set_arg<cl_mem>(13, localTree.bodies_ids.p());
  internalMove.setWork(validCount, 128);
  internalMove.execute(execStream->s());

  this->gpu_exchange_particles_with_overflow_check(localTree, &bodyBuffer[0], compactList, validCount);

} //End gpuRedistributeParticles




void octree::essential_tree_exchange(vector<real4> &treeStructure, tree_structure &tree, tree_structure &remote)
{
  int myid    = procId;
  int nproc   = nProcs;
  int isource = 0;

  double t0 = get_time();

  bool mergeOwntree = false;          //Default do not include our own tree-structre, thats mainly used for testing
  int step          = nProcs - 1;     //Default merge all remote trees into one structure
  //   step              = 1;
  int level_start   = tree.startLevelMin; //Depth of where to start the tree-walk
  int procTrees     = 0;              //Number of trees that we've received and processed

  real4  *bodies              = &tree.bodies_Ppos[0];
  real4  *velocities          = &tree.bodies_Pvel[0];
  real4  *multipole           = &tree.multipole[0];
  real4  *nodeSizeInfo        = &tree.boxSizeInfo[0];
  real4  *nodeCenterInfo      = &tree.boxCenterInfo[0];

  vector<real4> recv_particles;
  vector<real4> recv_multipoleData;
  vector<real4> recv_nodeSizeData;
  vector<real4> recv_nodeCenterData;






  if(procId == -1)
  {
    uint2 node_begend;
    node_begend.x   = tree.level_list[level_start].x;
    node_begend.y   = tree.level_list[level_start].y;

    int remoteId = 1;

    char buffFilename[256];
    sprintf(buffFilename, "grpAndTreeDump_%d_%d.bin", procId, remoteId);

    ofstream outFile(buffFilename, ios::out|ios::binary);
    if(outFile.is_open())
    {
      //Write the properties
      outFile.write((char*)&globalCoarseGrpCount[remoteId], sizeof(int));
      outFile.write((char*)&node_begend.x, sizeof(int));
      outFile.write((char*)&node_begend.y, sizeof(int));
      outFile.write((char*)&tree.n_nodes, sizeof(int));
      outFile.write((char*)&tree.n, sizeof(int));

      //Write the groups
      for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      {
        int idx = globalCoarseGrpOffsets[remoteId] + i;
        double4 boxCenter = coarseGroupBoxCenter[idx];
        double4 boxSize   = coarseGroupBoxSize  [idx];
        outFile.write((char*)&boxCenter, sizeof(double4));
        outFile.write((char*)&boxSize, sizeof(double4));
      }

      //Write the particles
      for(int i=0; i < tree.n; i++)
      {
        outFile.write((char*)&bodies[i], sizeof(real4));
      }

      //Write the multipole
      for(int i=0; i < tree.n_nodes; i++)
      {
        outFile.write((char*)&multipole[i*3 + 0], sizeof(real4));
        outFile.write((char*)&multipole[i*3 + 1], sizeof(real4));
        outFile.write((char*)&multipole[i*3 + 2], sizeof(real4));
      }

      //Write the nodeSizeInfo
      for(int i=0; i < tree.n_nodes; i++)
      {
        outFile.write((char*)&nodeSizeInfo[i], sizeof(real4));
      }

      //Write the nodeCenterInfo
      for(int i=0; i < tree.n_nodes; i++)
      {
        outFile.write((char*)&nodeCenterInfo[i], sizeof(real4));
      }

      outFile.close();

    }

  }





  real4 **treeBuffers;

  //creates a new array of pointers to int objects, with space for the local tree
  treeBuffers  = new real4*[mpiGetNProcs()];

  //Timers for the LET Exchange
  static double totalLETExTime    = 0;
  //   double thisPartLETExTime = 0;
  thisPartLETExTime = 0;
  double tStart = 0;

  //   for(int z=nproc-1; z > 0; z-=step)
  for(int z=nproc-1; z > 0; )
  {
    tStart = get_time();

    step = min(step, z);

    int recvTree = 0;
    //For each process
    for(int ib = z; recvTree < step; recvTree++, ib--)
    {
      int ibox = (ib+myid)%nproc; //index to send...
      if (ib == nproc-1){
        isource= (myid+1)%nproc;
      }else{
        isource = (isource+1)%nproc;
        if (isource == myid)isource = (isource+1)%nproc;
      }

      /*
         cerr << "\nibox: " << ibox << endl;
         cerr << "Other proc has box: low: " << let_xlow[ibox].x << "\t" <<  let_xlow[ibox].y  << "\t" <<  let_xlow[ibox].z
         << "\thigh: "  << let_xhigh[ibox].x << "\t" <<  let_xhigh[ibox].y  << "\t" <<  let_xhigh[ibox].z << endl;*/

      double4 boxCenter = {     0.5*(currentRLow[ibox].x  + currentRHigh[ibox].x),
        0.5*(currentRLow[ibox].y  + currentRHigh[ibox].y),
        0.5*(currentRLow[ibox].z  + currentRHigh[ibox].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[ibox].x - currentRLow[ibox].x)),
        fabs(0.5*(currentRHigh[ibox].y - currentRLow[ibox].y)),
        fabs(0.5*(currentRHigh[ibox].z - currentRLow[ibox].z)), 0};


      //       printf("Other proc min and max: [%f %f %f] \t [%f %f %f] \n", currentRLow[ibox].x, currentRLow[ibox].y, currentRLow[ibox].z,
      //               currentRHigh[ibox].x, currentRHigh[ibox].y, currentRHigh[ibox].z);

      //   printf("Other proc center and size: [%f %f %f] \t [%f %f %f] \n", boxCenter.x, boxCenter.y, boxCenter.z,
      //          boxSize.x, boxSize.y, boxSize.z);

      uint2 node_begend;
      node_begend.x   = tree.level_list[level_start].x;
      node_begend.y   = tree.level_list[level_start].y;

      int particleCount, nodeCount;

      double t1 = get_time();
      //      create_local_essential_tree_count(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      //                                  boxCenter, boxSize, (float)currentRLow[ibox].w, node_begend.x, node_begend.y,
      //                                  particleCount, nodeCount);
      create_local_essential_tree_count(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
          ibox, (float)currentRLow[ibox].w, node_begend.x, node_begend.y,
          particleCount, nodeCount);

      LOG("LET count (ibox: %d): %lg \t Coarse groups %d Since start: %lg\n", ibox, get_time()-t1,  globalCoarseGrpCount[ibox],get_time()-t0);
      LOG("LET count:  Particle count %d Node count: %d\n", particleCount, nodeCount);
      //Buffer that will contain all the data:
      //|real4| 2*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
      //1 + 2*particleCount + nodeCount + nodeCount + 3*nodeCount

      //Increase the number of particles and the number of nodes by the texture-offset such that these are correctly
      //aligned in memory
      particleCount += getTextureAllignmentOffset(particleCount, sizeof(real4));
      nodeCount     += getTextureAllignmentOffset(nodeCount    , sizeof(real4));

      //0-1 )                               Info about #particles, #nodes, start and end of tree-walk
      //1- Npart)                           The particle positions
      //1+Npart-Npart )                     The particle velocities
      //1+2*Npart-Nnode )                   The nodeSizeData
      //1+*2Npart+Nnode - Npart+2*Nnode )   The nodeCenterData
      //1+2*Npart+2*Nnode - Npart+5*Nnode ) The multipole data, is 3x number of nodes (mono and quadrupole data)
      int bufferSize = 1 + 2*particleCount + 5*nodeCount;
      real4 *letDataBuffer = new real4[bufferSize];

      //      create_local_essential_tree_fill(bodies, velocities, multipole, nodeSizeInfo, nodeCenterInfo,
      //                                  boxCenter, boxSize, (float)currentRLow[ibox].w, node_begend.x, node_begend.y,
      //                                  particleCount, nodeCount, letDataBuffer);
      create_local_essential_tree_fill(bodies, velocities, multipole, nodeSizeInfo, nodeCenterInfo,
          ibox, (float)currentRLow[ibox].w, node_begend.x, node_begend.y,
          particleCount, nodeCount, letDataBuffer);
      LOG("LET count&fill: %lg  since start: %lg \n", get_time()-t1, get_time()-t0);


      /*
         printf("LET count&fill: %lg\n", get_time()-t1);
         t1 = get_time();
         printf("Speciaal: %lg\n", get_time()-t1);
         t1 = get_time();
         create_local_essential_tree(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
         boxCenter, boxSize, node_begend.x, node_begend.y,
         particles, multipoleData, nodeSizeData, nodeCenterData);
         printf("Gewoon: %lg\n", get_time()-t1); */

      //Set the tree properties, before we exchange the data
      letDataBuffer[0].x = (float)particleCount;         //Number of particles in the LET
      letDataBuffer[0].y = (float)nodeCount;             //Number of nodes     in the LET
      letDataBuffer[0].z = (float)node_begend.x;         //First node on the level that indicates the start of the tree walk
      letDataBuffer[0].w = (float)node_begend.y;         //last node on the level that indicates the start of the tree walk

      double t9 = get_time();
      //Exchange the data of the tree structures  between the processes
      treeBuffers[recvTree] = MP_exchange_bhlist(ibox, isource, bufferSize, letDataBuffer);
      LOG("LET exchange trees: %d <-> %d  took: %lg  since start: %lg \n", ibox, isource,get_time()-t9, get_time()-t0);

      delete[] letDataBuffer;

      //This determines if we interrupt the exchange by starting a gravity kernel on the GPU
      if(gravStream->isFinished())
      {
        LOGF(stderr,"GRAVFINISHED %d recvTree: %d  Time: %lg Since start: %lg\n",
            procId, recvTree, get_time()-t1, get_time()-t0);
        recvTree++;
        break;
      }
    }//end for each process



    z-=recvTree;

    //Now we have to merge the seperate tree-structures into one process

    //     double t1 = get_time();

    int PROCS = recvTree;

    procTrees += recvTree;

    if(mergeOwntree)
    {
      //Add the processors own tree to the LET tree
      int particleCount   = tree.n;
      int nodeCount       = tree.n_nodes;

      int realParticleCount = tree.n;
      int realNodeCount     = tree.n_nodes;

      particleCount += getTextureAllignmentOffset(particleCount, sizeof(real4));
      nodeCount     += getTextureAllignmentOffset(nodeCount    , sizeof(real4));

      int bufferSizeLocal = 1 + 2*particleCount + 5*nodeCount;

      treeBuffers[PROCS]  = new real4[bufferSizeLocal];

      //Note that we use the real*Counts otherwise we read out of the array boundaries!!
      int idx = 1;
      memcpy(&treeBuffers[PROCS][idx], &bodies[0],         sizeof(real4)*realParticleCount);
      idx += particleCount;
      memcpy(&treeBuffers[PROCS][idx], &velocities[0],     sizeof(real4)*realParticleCount);
      idx += particleCount;
      memcpy(&treeBuffers[PROCS][idx], &nodeSizeInfo[0],   sizeof(real4)*realNodeCount);
      idx += nodeCount;
      memcpy(&treeBuffers[PROCS][idx], &nodeCenterInfo[0], sizeof(real4)*realNodeCount);
      idx += nodeCount;
      memcpy(&treeBuffers[PROCS][idx], &multipole[0],      sizeof(real4)*realNodeCount*3);

      treeBuffers[PROCS][0].x = (float)particleCount;
      treeBuffers[PROCS][0].y = (float)nodeCount;
      treeBuffers[PROCS][0].z = (float)tree.level_list[level_start].x;
      treeBuffers[PROCS][0].w = (float)tree.level_list[level_start].y;
      PROCS                   = PROCS + 1; //Signal that we added one more tree-structure
      mergeOwntree            = false;     //Set it to false incase we do not merge all trees at once, we only inlcude our own once
    }

    //Arrays to store and compute the offsets
    int *particleSumOffsets  = new int[mpiGetNProcs()+1];
    int *nodeSumOffsets      = new int[mpiGetNProcs()+1];
    int *startNodeSumOffsets = new int[mpiGetNProcs()+1];
    uint2 *nodesBegEnd       = new uint2[mpiGetNProcs()+1];

    //Offsets start at 0 and then are increased by the number of nodes of each LET tree
    particleSumOffsets[0]           = 0;
    nodeSumOffsets[0]               = 0;
    startNodeSumOffsets[0]          = 0;
    nodesBegEnd[mpiGetNProcs()].x   = nodesBegEnd[mpiGetNProcs()].y = 0; //Make valgrind happy
    int totalTopNodes               = 0;


    //Calculate the offsets
    for(int i=0; i < PROCS ; i++)
    {
      int particles = (int)treeBuffers[i][0].x;
      int nodes     = (int)treeBuffers[i][0].y;

      nodesBegEnd[i].x = (int)treeBuffers[i][0].z;
      nodesBegEnd[i].y = (int)treeBuffers[i][0].w;

      totalTopNodes += nodesBegEnd[i].y-nodesBegEnd[i].x;

      particleSumOffsets[i+1]     = particleSumOffsets[i]  + particles;
      nodeSumOffsets[i+1]         = nodeSumOffsets[i]      + nodes - nodesBegEnd[i].y;    //Without the top-nodes
      startNodeSumOffsets[i+1]    = startNodeSumOffsets[i] + nodesBegEnd[i].y-nodesBegEnd[i].x;
    }

    //Compute total particles and total nodes, totalNodes is WITHOUT topNodes
    int totalParticles    = particleSumOffsets[PROCS];
    int totalNodes        = nodeSumOffsets[PROCS];

    //To bind parts of the memory to different textures, the memory start address
    //has to be aligned with XXX bytes, so nodeInformation*sizeof(real4) has to be
    //increased by an offset, so that the node data starts at a XXX byte boundary
    //this is already done on the sending process, but since we modify the structure
    //it has to be done again
    int nodeTextOffset = getTextureAllignmentOffset(totalNodes+totalTopNodes, sizeof(real4));

    //Compute the total size of the buffer
    int bufferSize     = 2*(totalParticles) + 5*(totalNodes+totalTopNodes + nodeTextOffset);

    thisPartLETExTime += get_time() - tStart;
    //Allocate memory on host and device to store the merged tree-structure
    if(bufferSize > remote.fullRemoteTree.get_size())
    {
      //Can only resize if we are sure the LET is not running
      if(letRunning)
      {
        gravStream->sync();     //Wait till the LET run is finished
      }
      remote.fullRemoteTree.cresize(bufferSize, false);  //Change the size but ONLY if we need more memory
    }
    tStart = get_time();

    real4 *combinedRemoteTree = &remote.fullRemoteTree[0];

    //Copy all the pieces of the different trees at the correct memory offsets
    for(int i=0; i < PROCS; i++)
    {
      //Get the properties of the LET
      int remoteP = (int) treeBuffers[i][0].x;    //Number of particles
      int remoteN = (int) treeBuffers[i][0].y;    //Number of nodes
      int remoteB = (int) treeBuffers[i][0].z;    //Begin id of top nodes
      int remoteE = (int) treeBuffers[i][0].w;    //End   id of top nodes
      int remoteNstart = remoteE-remoteB;

      //Particles
      memcpy(&combinedRemoteTree[particleSumOffsets[i]],   &treeBuffers[i][1], sizeof(real4)*remoteP);

      //Velocities
      memcpy(&combinedRemoteTree[(totalParticles) + particleSumOffsets[i]],
          &treeBuffers[i][1+remoteP], sizeof(real4)*remoteP);

      //The start nodes, nodeSizeInfo
      memcpy(&combinedRemoteTree[2*(totalParticles) + startNodeSumOffsets[i]],
          &treeBuffers[i][1+2*remoteP+remoteB], //From the start node onwards
          sizeof(real4)*remoteNstart);

      //Non start nodes, nodeSizeInfo
      memcpy(&combinedRemoteTree[2*(totalParticles) +  totalTopNodes + nodeSumOffsets[i]],
          &treeBuffers[i][1+2*remoteP+remoteE], //From the last start node onwards
          sizeof(real4)*(remoteN-remoteE));

      //The start nodes, nodeCenterInfo
      memcpy(&combinedRemoteTree[2*(totalParticles) + startNodeSumOffsets[i]
          + (totalNodes + totalTopNodes + nodeTextOffset)],
          &treeBuffers[i][1+2*remoteP+remoteB + remoteN], //From the start node onwards
          sizeof(real4)*remoteNstart);

      //Non start nodes, nodeCenterInfo
      memcpy(&combinedRemoteTree[2*(totalParticles) +  totalTopNodes
          + nodeSumOffsets[i] + (totalNodes + totalTopNodes + nodeTextOffset)],
          &treeBuffers[i][1+2*remoteP+remoteE + remoteN], //From the last start node onwards
          sizeof(real4)*(remoteN-remoteE));

      //The start nodes, multipole
      memcpy(&combinedRemoteTree[2*(totalParticles) + 3*startNodeSumOffsets[i] +
          2*(totalNodes+totalTopNodes + nodeTextOffset)],
          &treeBuffers[i][1+2*remoteP+2*remoteN + 3*remoteB], //From the start node onwards
          sizeof(real4)*remoteNstart*3);

      //Non start nodes, multipole
      memcpy(&combinedRemoteTree[2*(totalParticles) +  3*totalTopNodes +
          3*nodeSumOffsets[i] + 2*(totalNodes+totalTopNodes+nodeTextOffset)],
          &treeBuffers[i][1+2*remoteP+remoteE*3 + 2*remoteN], //From the last start node onwards
          sizeof(real4)*(remoteN-remoteE)*3);
      /*
         |real4| 2*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
         1 + 2*particleCount + nodeCount + nodeCount + 3*nodeCount

         Info about #particles, #nodes, start and end of tree-walk
         The particle positions
         velocities
         The nodeSizeData
         The nodeCenterData
         The multipole data, is 3x number of nodes (mono and quadrupole data)

         Now that the data is copied, modify the offsets of the tree so that everything works
         with the new correct locations and references. This takes place in two steps:
         First  the top nodes
         Second the normal nodes
         Has to be done in two steps since they are not continous in memory if NPROCS > 2
         */

      //Modify the top nodes
      int modStart = 2*(totalParticles) + startNodeSumOffsets[i];
      int modEnd   = modStart           + remoteNstart;

      for(int j=modStart; j < modEnd; j++)
      {
        real4 nodeCenter = combinedRemoteTree[j+totalTopNodes+totalNodes+nodeTextOffset];
        real4 nodeSize   = combinedRemoteTree[j];
        bool leaf        = nodeCenter.w <= 0;

        int childinfo = host_float_as_int(nodeSize.w);
        int child, nchild;

        if(!leaf)
        {
          //Node
          child    =    childinfo & 0x0FFFFFFF;                  //Index to the first child of the node
          nchild   = (((childinfo & 0xF0000000) >> 28)) ;        //The number of children this node has

          child = child - nodesBegEnd[i].y + totalTopNodes + nodeSumOffsets[i]; //Calculate the new start (non-leaf)
          child = child | (nchild << 28);                                       //Merging back in one int

          if(nchild == 0) child = 0;                             //To prevent incorrect negative values
        }else{ //Leaf
          child   =   childinfo & BODYMASK;                      //the first body in the leaf
          nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag

          child   =  child + particleSumOffsets[i];               //Increasing offset
          child   = child | ((nchild-1) << LEAFBIT);              //Merging back to one int
        }//end !leaf
        combinedRemoteTree[j].w =  host_int_as_float(child);      //store the modified offset
      }

      //Now the non-top nodes for this process
      modStart =  totalTopNodes + nodeSumOffsets[i] + 2*(totalParticles);
      modEnd   =  modStart      + remoteN-remoteE;
      for(int j=modStart; j < modEnd; j++)
      {
        real4 nodeCenter = combinedRemoteTree[j+totalTopNodes+totalNodes+nodeTextOffset];
        real4 nodeSize   = combinedRemoteTree[j];
        bool leaf        = nodeCenter.w <= 0;

        int childinfo = host_float_as_int(nodeSize.w);
        int child, nchild;

        if(!leaf) {  //Node
          child    =    childinfo & 0x0FFFFFFF;                   //Index to the first child of the node
          nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has

          //Calculate the new start (non-leaf)
          child = child - nodesBegEnd[i].y + totalTopNodes + nodeSumOffsets[i];  ;

          //Combine and store
          child = child | (nchild << 28);

          if(nchild == 0) child = 0;                              //To prevent incorrect negative values
        }else{ //Leaf
          child   =   childinfo & BODYMASK;                       //the first body in the leaf
          nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);      //number of bodies in the leaf masked with the flag

          child =  child + particleSumOffsets[i];                 //Modify the particle offsets
          child = child | ((nchild-1) << LEAFBIT);                //Merging the data back into one int
        }//end !leaf
        combinedRemoteTree[j].w =  host_int_as_float(child);      //Store the modified value
      }

      delete[] treeBuffers[i];    //Free the memory of this part of the LET
    }

    /*
       The final tree structure looks as follows:
       particlesT1, partcilesT2,...mparticlesTn |,
       topNodeSizeT1, topNodeSizeT2,..., topNodeSizeT2 | nodeSizeT1, nodeSizeT2, ...nodeSizeT3 |,
       topNodeCentT1, topNodeCentT2,..., topNodeCentT2 | nodeCentT1, nodeCentT2, ...nodeCentT3 |,
       topNodeMultT1, topNodeMultT2,..., topNodeMultT2 | nodeMultT1, nodeMultT2, ...nodeMultT3

       NOTE that the Multipole data consists of 3 float4 values per node

*/

    //Store the tree properties (number of particles, number of nodes, start and end topnode)
    remote.remoteTreeStruct.x = totalParticles;
    remote.remoteTreeStruct.y = totalNodes+totalTopNodes;
    remote.remoteTreeStruct.z = nodeTextOffset;
    totalTopNodes             = (0 << 16) | (totalTopNodes);  //If its a merged tree we start at 0
    remote.remoteTreeStruct.w = totalTopNodes;

    //     fprintf(stderr,"Modifying the LET took: %g \n", get_time()-t1);
    LOGF(stderr,"Number of local bodies: %d number LET bodies: %d number LET nodes: %d top nodes: %d Processed trees: %d (%d) \n",
        tree.n, totalParticles, totalNodes, remote.remoteTreeStruct.y-totalNodes, PROCS, procTrees);

    delete[] particleSumOffsets;
    delete[] nodeSumOffsets;
    delete[] startNodeSumOffsets;
    delete[] nodesBegEnd;


    thisPartLETExTime += get_time() - tStart;


    //Check if we need to summarize which particles are active,
    //only done during the last approximate_gravity_let call
    bool doActivePart = (procTrees == mpiGetNProcs() -1);

    approximate_gravity_let(this->localTree, this->remoteTree, bufferSize, doActivePart);

  } //end z
  delete[] treeBuffers;


  totalLETExTime += thisPartLETExTime;

  LOGF(stderr,"LETEX [%d] curStep: %g\t   Total: %g \n", procId, thisPartLETExTime, totalLETExTime);
}



inline double cust_fabs2(double a)
{
  return (a > 0) ? a : -a;
}


int globalCHECKCount;
//Improved Barnes Hut criterium
#ifdef INDSOFT
bool split_node_grav_impbh(float4 nodeCOM, double4 boxCenter, double4 boxSize,
    float group_eps, float node_eps)
#else
bool split_node_grav_impbh(float4 nodeCOM, double4 boxCenter, double4 boxSize)
#endif
{
  globalCHECKCount++;
#if 1
  //Compute the distance between the group and the cell
  float3 dr = make_float3(fabs((float)boxCenter.x - nodeCOM.x) - (float)boxSize.x,
      fabs((float)boxCenter.y - nodeCOM.y) - (float)boxSize.y,
      fabs((float)boxCenter.z - nodeCOM.z) - (float)boxSize.z);

  dr.x += fabs(dr.x); dr.x *= 0.5f;
  dr.y += fabs(dr.y); dr.y *= 0.5f;
  dr.z += fabs(dr.z); dr.z *= 0.5f;


  //Distance squared, no need to do sqrt since opening criteria has been squared
  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

#ifdef INDSOFT
  if(ds2      <= ((group_eps + node_eps ) * (group_eps + node_eps) ))           return true;
  //Limited precision can result in round of errors. Use this as extra safe guard
  if(fabs(ds2 -  ((group_eps + node_eps ) * (group_eps + node_eps) )) < 10e-04) return true;
#endif

  if (ds2     <= fabs(nodeCOM.w))           return true;
  if (fabs(ds2 - fabs(nodeCOM.w)) < 10e-04) return true; //Limited precision can result in round of errors. Use this as extra safe guard

  //   return true;
  return false;
#else

  //Compute the distance between the group and the cell
  float3 dr = make_float3(cust_fabs2((float)boxCenter.x - nodeCOM.x) - (float)boxSize.x,
      cust_fabs2((float)boxCenter.y - nodeCOM.y) - (float)boxSize.y,
      cust_fabs2((float)boxCenter.z - nodeCOM.z) - (float)boxSize.z);

  dr.x += cust_fabs2(dr.x); dr.x *= 0.5f;
  dr.y += cust_fabs2(dr.y); dr.y *= 0.5f;
  dr.z += cust_fabs2(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

#ifdef INDSOFT
  if(ds2      <= ((group_eps + node_eps ) * (group_eps + node_eps) ))           return true;
  //Limited precision can result in round of errors. Use this as extra safe guard
  if(cust_fabs2(ds2 -  ((group_eps + node_eps ) * (group_eps + node_eps) )) < 10e-04) return true;
#endif

  if (ds2     <= cust_fabs2(nodeCOM.w))           return true;
  if (cust_fabs2(ds2 - cust_fabs2(nodeCOM.w)) < 10e-04) return true; //Limited precision can result in round of errors. Use this as extra safe guard

  //   return true;
  return false;
#endif
}





//Minimal Distance version

//Minimum distance opening criteria
#ifdef INDSOFT
bool split_node(real4 nodeCenter, real4 nodeSize, double4 boxCenter, double4 boxSize,
    float group_eps, float node_eps)
#else
bool split_node(real4 nodeCenter, real4 nodeSize, double4 boxCenter, double4 boxSize)
#endif
{
  //Compute the distance between the group and the cell
  float3 dr = make_float3(fabs((float)boxCenter.x - nodeCenter.x) - (float)(boxSize.x + nodeSize.x),
      fabs((float)boxCenter.y - nodeCenter.y) - (float)(boxSize.y + nodeSize.y),
      fabs((float)boxCenter.z - nodeCenter.z) - (float)(boxSize.z + nodeSize.z));

  dr.x += fabs(dr.x); dr.x *= 0.5f;
  dr.y += fabs(dr.y); dr.y *= 0.5f;
  dr.z += fabs(dr.z); dr.z *= 0.5f;

  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

#ifdef INDSOFT
  if(ds2 <=      ((group_eps + node_eps ) * (group_eps + node_eps) ))           return true;
  if(fabs(ds2 -  ((group_eps + node_eps ) * (group_eps + node_eps) )) < 10e-04) return true;
#endif

  if (ds2     <= fabs(nodeCenter.w))           return true;
  if (fabs(ds2 - fabs(nodeCenter.w)) < 10e-04) return true; //Limited precision can result in round of errors. Use this as extra safe guard

  return false;

}



void octree::create_local_essential_tree(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
    vector<real4> &particles, vector<real4> &multipoleData,
    vector<real4> &nodeSizeData, vector<real4> &nodeCenterData)
{
  //Walk the tree as is done on device, level by level
  vector<int> curLevel;
  vector<int> nextLevel;


  double t1 = get_time();

  double massSum = 0;

  int nodeCount       = 0;

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    curLevel.push_back(i);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeSizeData.push_back(nodeSizeInfo[i]);
    nodeCenterData.push_back(nodeCenterInfo[i]);

    multipoleData.push_back(multipole[i*3 + 0]);
    multipoleData.push_back(multipole[i*3 + 1]);
    multipoleData.push_back(multipole[i*3 + 2]);
    nodeCount++;
  }

  //Start the tree-walk
  LOG("Start: %d end: %d \n", start, end);
  LOG("Sarting walk on: %d items! \n", (int)curLevel.size());


  int childNodeOffset         = end;
  int childParticleOffset     = 0;

  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      int node         = curLevel[i];
      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      bool split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      bool split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
      //Minimal distance version

#ifdef INDSOFT
      bool split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      bool split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif

#endif
      //       printf("Node %d is a leaf: %d  en childinfo: %d  \t\t-> %d en %d \t split: %d\n", node, leaf, childinfo, child, nchild, split);
      //          split = false;
      uint temp =0;
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          nextLevel.push_back(i);
        }

        temp = childNodeOffset | (nchild << 28);
        //Update reference to children
        childNodeOffset += nchild;
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particles.push_back(bodies[i]);
          massSum += bodies[i].w;
        }

        temp = childParticleOffset | ((nchild-1) << LEAFBIT);
        childParticleOffset += nchild;
      }


      //Add the node data to the appropriate arrays
      //and modify the node reference
      //start ofset for its children, should be nodeCount at start of this level +numberofnodes on this level
      //plus a counter that counts the number of childs of the nodes we have tested

      //New childoffset:
      union{int i; float f;} itof; //__int_as_float
      itof.i           = temp;
      float tempConv = itof.f;

      //Add node properties and update references
      real4 nodeSizeInfoTemp  = nodeSizeInfo[node];
      nodeSizeInfoTemp.w      = tempConv;             //Replace child reference
      nodeSizeData.push_back(nodeSizeInfoTemp);


      multipoleData.push_back(multipole[node*3 + 0]);
      multipoleData.push_back(multipole[node*3 + 1]);
      multipoleData.push_back(multipole[node*3 + 2]);

      if(!split)
      {
        massSum += multipole[node*3 + 0].w;
      }


    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();


  }//end while

  cout << "New tree structure: bodies: " << particles.size() << "\tnodes: " << nodeSizeData.size() << "\t took: " << get_time() -t1 << endl;
  cout << "Mass sum: " << massSum << endl;
  cout << "Mass sumtest: " << multipole[0*0 + 0].w << endl;

}

#if 1

typedef struct{
  int nodeID;
  vector<int> coarseIDs;
} combNodeCheck;


//void octree::create_local_essential_tree_fill(real4* bodies, real4* velocities, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
//                                         double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
//                                         int particleCount, int nodeCount, real4 *dataBuffer)
void octree::create_local_essential_tree_fill(real4* bodies, real4* velocities, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int particleCount, int nodeCount, real4 *dataBuffer)
{

#if 1
  create_local_essential_tree_fill_novector_startend4(bodies, velocities, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end,
      particleCount, nodeCount, dataBuffer);

  return;


#endif

  //Walk the tree as is done on device, level by level
  vector<combNodeCheck> curLevel;
  vector<combNodeCheck> nextLevel;

  curLevel.reserve(1024*128);
  nextLevel.reserve(1024*128);

  vector<int> coarseIDs;
  //     double t1 = get_time();

  double massSum = 0;

  int particleOffset     = 1;
  int velParticleOffset  = particleOffset      + particleCount;
  int nodeSizeOffset     = velParticleOffset   + particleCount;
  int nodeCenterOffset   = nodeSizeOffset      + nodeCount;
  int multiPoleOffset    = nodeCenterOffset    + nodeCount;

  //|real4| 2*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
  //Info about #particles, #nodes, start and end of tree-walk
  //The particle positions and velocities
  //The nodeSizeData
  //The nodeCenterData
  //The multipole data

  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs.push_back(globalCoarseGrpOffsets[remoteId] + i);
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck check;
    check.nodeID    = i;
    check.coarseIDs = coarseIDs;
    curLevel.push_back(check);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    dataBuffer[nodeSizeOffset++]   = nodeSizeInfo[i];
    dataBuffer[nodeCenterOffset++] = nodeCenterInfo[i];

    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 0];
    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 1];
    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 2];
  }

  //Start the tree-walk
  //Variables to rewrite the tree-structure indices
  int childNodeOffset         = end;
  int childParticleOffset     = 0;


  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      //Read node data
      combNodeCheck check = curLevel[i];
      int node           = check.nodeID;

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                   //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =    childinfo & BODYMASK;                     //the first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

      bool split = false;

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif

#if 0
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else

      vector<int> checkIDs;
      bool curSplit = false;

      for(int k=0; k < check.coarseIDs.size(); k++)
      {
        //Read box info
        int coarseGrpId = check.coarseIDs[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];

#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;
          checkIDs.push_back(coarseGrpId);
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }
      } //For globalCoarseGrpCount[remoteId]
#endif


      uint temp = 0;  //A node that is not split and is not a leaf will get childinfo 0
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          combNodeCheck check;
          check.nodeID    = i;
          check.coarseIDs = checkIDs;
          nextLevel.push_back(check);
        }

        temp = childNodeOffset | (nchild << 28);
        //Update reference to children
        childNodeOffset += nchild;
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          dataBuffer[particleOffset++] = bodies[i];
          dataBuffer[velParticleOffset++] = velocities[i];
          massSum += bodies[i].w;
        }

        temp = childParticleOffset | ((nchild-1) << LEAFBIT);
        childParticleOffset += nchild;
      }



      //Add the node data to the appropriate arrays and modify the node reference
      //start ofset for its children, should be nodeCount at start of this level +numberofnodes on this level
      //plus a counter that counts the number of childs of the nodes we have tested

      //New childoffset:
      union{int i; float f;} itof; //__int_as_float
      itof.i         = temp;
      float tempConv = itof.f;

      //Add node properties and update references
      real4 nodeSizeInfoTemp  = nodeSizeInfo[node];
      nodeSizeInfoTemp.w      = tempConv;             //Replace child reference

      dataBuffer[nodeSizeOffset++]   = nodeSizeInfoTemp;
      dataBuffer[nodeCenterOffset++] = nodeCenterInfo[node];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 0];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 1];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 2];

      if(!split)
      {
        massSum += multipole[node*3 + 0].w;
      }
    } //end for curLevel.size

    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();

  }//end while

  //   cout << "New offsets: "  << particleOffset << " \t" << nodeSizeOffset << " \t" << nodeCenterOffset << endl;
  //    cout << "Mass sum: " << massSum  << endl;
  //   cout << "Mass sumtest: " << multipole[0*0 + 0].w << endl;
}

#else

//void octree::create_local_essential_tree_fill(real4* bodies, real4* velocities, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
//                                         double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
//                                         int particleCount, int nodeCount, real4 *dataBuffer)
void octree::create_local_essential_tree_fill(real4* bodies, real4* velocities, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int particleCount, int nodeCount, real4 *dataBuffer)
{
  //Walk the tree as is done on device, level by level
  vector<int> curLevel;
  vector<int> nextLevel;


  //     double t1 = get_time();

  double massSum = 0;

  int particleOffset     = 1;
  int velParticleOffset  = particleOffset      + particleCount;
  int nodeSizeOffset     = velParticleOffset   + particleCount;
  int nodeCenterOffset   = nodeSizeOffset      + nodeCount;
  int multiPoleOffset    = nodeCenterOffset    + nodeCount;

  //|real4| 2*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
  //Info about #particles, #nodes, start and end of tree-walk
  //The particle positions and velocities
  //The nodeSizeData
  //The nodeCenterData
  //The multipole data

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    curLevel.push_back(i);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    dataBuffer[nodeSizeOffset++]   = nodeSizeInfo[i];
    dataBuffer[nodeCenterOffset++] = nodeCenterInfo[i];

    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 0];
    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 1];
    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 2];
  }

  //Start the tree-walk
  //Variables to rewrite the tree-structure indices
  int childNodeOffset         = end;
  int childParticleOffset     = 0;


  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      int node         = curLevel[i];
      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                   //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =    childinfo & BODYMASK;                     //the first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif
#if 0
#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      bool split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      bool split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
      //Minimal distance version
#ifdef INDSOFT
      bool split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      bool split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif

#endif

#endif //if0

      bool split = false;
#if 0
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      {
        //Compute this specific box...kinda expensive should just
        //make this list when receiving it and look it up.
        //For now just using for testing method
        //TODO NOTE BUG ERROR
        int coarseGrpId = globalCoarseGrpOffsets[remoteId] + i;

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];

#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(split) break;
      } //For globalCoarseGrpCount[remoteId]
#endif


      uint temp = 0;  //A node that is not split and is not a leaf will get childinfo 0
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          nextLevel.push_back(i);
        }

        temp = childNodeOffset | (nchild << 28);
        //Update reference to children
        childNodeOffset += nchild;
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          dataBuffer[particleOffset++] = bodies[i];
          dataBuffer[velParticleOffset++] = velocities[i];
          massSum += bodies[i].w;
        }

        temp = childParticleOffset | ((nchild-1) << LEAFBIT);
        childParticleOffset += nchild;
      }



      //Add the node data to the appropriate arrays and modify the node reference
      //start ofset for its children, should be nodeCount at start of this level +numberofnodes on this level
      //plus a counter that counts the number of childs of the nodes we have tested

      //New childoffset:
      union{int i; float f;} itof; //__int_as_float
      itof.i         = temp;
      float tempConv = itof.f;

      //Add node properties and update references
      real4 nodeSizeInfoTemp  = nodeSizeInfo[node];
      nodeSizeInfoTemp.w      = tempConv;             //Replace child reference

      dataBuffer[nodeSizeOffset++]   = nodeSizeInfoTemp;
      dataBuffer[nodeCenterOffset++] = nodeCenterInfo[node];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 0];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 1];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 2];

      if(!split)
      {
        massSum += multipole[node*3 + 0].w;
      }
    } //end for curLevel.size

    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();

  }//end while

  //   cout << "New offsets: "  << particleOffset << " \t" << nodeSizeOffset << " \t" << nodeCenterOffset << endl;
  //    cout << "Mass sum: " << massSum  << endl;
  //   cout << "Mass sumtest: " << multipole[0*0 + 0].w << endl;
}
#endif

#if 0
//Does not work,this one tries to order the boxes so the one needed is up front

//void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
//                                         double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
//                                         int &particles, int &nodes)
void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{

  //Walk the tree as is done on device, level by level
  vector<combNodeCheck> curLevel;
  vector<combNodeCheck> nextLevel;

  curLevel.reserve(1024*128);
  nextLevel.reserve(1024*128);

  vector<int> coarseIDs;

  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  LOGF(stderr,"Test loop value: %d", this->globalCoarseGrpCount[remoteId]);



  //Add the initial coarse boxes to this level
  for(int i=0; i < this->globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs.push_back(this->globalCoarseGrpOffsets[remoteId] + i);
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck check;
    check.nodeID    = i;
    check.coarseIDs = coarseIDs;
    curLevel.push_back(check);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      combNodeCheck check = curLevel[i];
      int node           = check.nodeID;

      //        LOGF(stderr, "LET count On level: %d\tNode: %d\tGoing to check: %d\n",
      //                          level,node, check.coarseIDs.size());

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

      bool split = false;

      int splitIdxToUse = 0;

      splitChecks++;

      vector<int> checkIDs;
      bool curSplit = false;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      float ds2;
      float ds2min = 10e10f;
      int   ds2min_idx = -1;

      for(int k=0; k < check.coarseIDs.size(); k++)
      {
        extraChecks++;
        //  particleCount++;
        //Test this specific box
        int coarseGrpId = check.coarseIDs[k];

        if(coarseGrpId < 0) LOGF(stderr,"FAIIIIIIIIIIIIIIIIIIL %d \n", coarseGrpId);

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh_SFCtest(nodeCOM, boxCenter, boxSize, ds2);
        //              curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;

          //            if(0)
          if(level < 2)
          {
            //Stop right away
            checkIDs.insert(checkIDs.begin(), &check.coarseIDs[k], &check.coarseIDs[0]+check.coarseIDs.size());
            break;

          }
          else
          {
            //Now order checkIDs
            if(ds2 < ds2min)
            {
              checkIDs.insert(checkIDs.begin(),coarseGrpId);
              //                LOGF(stderr,"TestSmaller: %f %f \t %d", ds2min, ds2, ds2min_idx);
              ds2min = ds2;
              //                ds2min_idx = coarseGrpId;

            }
            else
            {
              //                LOGF(stderr,"FAILED: %f %f \t %d", ds2min, ds2, ds2min_idx);
            }
          }//level > 30
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }//if curSplit
      } //For globalCoarseGrpCount[remoteId]

      //        if(ds2min_idx >= 0)
      //        checkIDs.push_back(ds2min_idx);
      //        if(ds2min_idx < 0)
      //        {
      //          LOGF(stderr, "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRROR %f %d split: %d\n", ds2min, ds2min_idx, split);
      //        }

      if(split == false)
      {
        //          uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
          combNodeCheck check;
          check.nodeID    = i;
          check.coarseIDs = checkIDs;
          nextLevel.push_back(check);
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();
    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}

#endif

#if 1
//This one goes over all coarse grps and removes the ones where the split
//fails. Does test them all

//void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
//                                         double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
//                                         int &particles, int &nodes)
void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  globalCHECKCount = 0;
#if 0
  create_local_essential_tree_count_recursive(
      bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);

  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);
  return;

#elif 0

  //Fastest so far, but recursive
  create_local_essential_tree_count_recursive_try2(
      bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);

  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);
  return;

#elif 0
  create_local_essential_tree_count_novector(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);
  return;

#elif 0
  create_local_essential_tree_count_vector_filter(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);
  return;


#elif 0

  //Was the Fastest non-recursive version untill create_local_essential_tree_count_novector_startend4
  create_local_essential_tree_count_novector_startend(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);

  return;

#elif 0

  //SLOW
  create_local_essential_tree_count_novector_startend2(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);

  return;

#elif 0

  //Second best
  create_local_essential_tree_count_novector_startend3(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);

  return;

#elif 1

  //Fastest so far, it sorts the boxes by putting most used ones in the back
  //Which reduces opening checks. Since there should be less unneeded checks
  //on the deeper levels of the tree
  create_local_essential_tree_count_novector_startend4(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);

  return;

#elif 0

  //Creates seperate box lists for differnt top nodes. Some sort of initial filter, does not help compared
  //to number startend4
  create_local_essential_tree_count_novector_startend5(bodies, multipole, nodeSizeInfo, nodeCenterInfo,
      remoteId, group_eps, start, end, particles, nodes);
  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);

  return;

#endif

  //Walk the tree as is done on device, level by level
  vector<combNodeCheck> curLevel;
  vector<combNodeCheck> nextLevel;

  curLevel.reserve(1024*64);
  nextLevel.reserve(1024*64);



  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  vector<int> coarseIDs;
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs.push_back(globalCoarseGrpOffsets[remoteId] + i);
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck check;
    check.nodeID    = i;
    check.coarseIDs = coarseIDs;
    curLevel.push_back(check);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      combNodeCheck check = curLevel[i];
      int node           = check.nodeID;

      //        LOGF(stderr, "LET count On level: %d\tNode: %d\tGoing to check: %d\n",
      //                          level,node, check.coarseIDs.size());

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;
      vector<int> checkIDs;
#if 0
      splitChecks++;
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      splitChecks++;


      bool curSplit = false;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      for(int k=0; k < check.coarseIDs.size(); k++)
      {
        extraChecks++;
        //  particleCount++;
        //Test this specific box
        int coarseGrpId = check.coarseIDs[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;

          if(leaf) break;

          checkIDs.push_back(coarseGrpId);
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }
      } //For globalCoarseGrpCount[remoteId]

      if(split == false)
      {
        //          uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
          combNodeCheck check;
          check.nodeID    = i;
          check.coarseIDs = checkIDs;
          nextLevel.push_back(check);
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();
    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr,"LET Number of total CHECKS: %d \n", globalCHECKCount);
  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}

typedef struct{
  int nodeID;
  int coarseIDOffset;
} combNodeCheck2;

void octree::create_local_essential_tree_count_novector(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level
  //    vector<combNodeCheck> curLevel;
  //    vector<combNodeCheck> nextLevel;

  combNodeCheck2 *curLevel = new combNodeCheck2[1024*64];
  combNodeCheck2 *nextLevel = new combNodeCheck2[1024*64];

  int curLevelCount = 0;
  int nextLevelCount = 0;


  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;


  double4 bigBoxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

  //    vector<int> coarseIDs;
  //    //Add the initial coarse boxes to this level
  //    for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  //    {
  //      coarseIDs.push_back(globalCoarseGrpOffsets[remoteId] + i);
  //    }

  int *coarseIDs = new int[globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck2 check;
    check.nodeID    = i;
    check.coarseIDOffset = 0;
    //      check.coarseIDs.insert(check.coarseIDs.begin(), coarseIDs, coarseIDs+globalCoarseGrpCount[remoteId]);
    curLevel[curLevelCount++] = check;
  }

  /*    //Filter out the initial boxes that will fail anyway
        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
        {
  // coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
  bool split;
  for(int j=start; j < end; j++)
  {
  real4 nodeCenter  = nodeCenterInfo[j];
  real4 nodeSize    = nodeSizeInfo[j];
  double4 boxCenter = coarseGroupBoxCenter[coarseIDs[i]];
  double4 boxSize   = coarseGroupBoxSize  [coarseIDs[i]];
  float4 nodeCOM    = multipole[j*3 + 0];
  nodeCOM.w         = nodeCenter.w;

  split |= split_node_grav_impbh(nodeCOM, boxCenter, boxSize);

  if(!split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
  {
  //          LOGF(stderr,"LET INITIAL Failed on box %d node: %d\n", i, j);
  }
  }
  if(split == false)
  {
  LOGF(stderr,"LET INITIAL Failed on %d \n", i);
  }
  }
  */
  curLevelCount = 0;
  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck2 check;
    check.nodeID    = i;
    check.coarseIDOffset = globalCoarseGrpCount[remoteId]+1; //Out of range default

    for(int j=0; j < globalCoarseGrpCount[remoteId]; j++)
    {
      real4 nodeCenter  = nodeCenterInfo[i];
      real4 nodeSize    = nodeSizeInfo[i];
      double4 boxCenter = coarseGroupBoxCenter[coarseIDs[j]];
      double4 boxSize   = coarseGroupBoxSize  [coarseIDs[j]];
      float4 nodeCOM    = multipole[j*3 + 0];
      nodeCOM.w         = nodeCenter.w;

      //Skip all previous not needed checks
      if(split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
      {
        check.coarseIDOffset = j;
        //          LOGF(stderr,"LET INITIAL Start node: %d at grp %d \n", i, j);
        break;
      }
    }
    curLevel[curLevelCount++] = check;
  }




  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck2 check = curLevel[i];
      int node           = check.nodeID;

      //        LOGF(stderr, "LET count On level: %d\tNode: %d\tGoing to check: %d\n",
      //                          level,node, check.coarseIDs.size());

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;
      //        vector<int> checkIDs;
#if 0
      splitChecks++;
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      splitChecks++;


      bool curSplit = false;

      int newOffset = 0;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      for(int k=check.coarseIDOffset; k < globalCoarseGrpCount[remoteId]; k++)
      {
        extraChecks++;
        //  particleCount++;
        //Test this specific box
        int coarseGrpId = coarseIDs[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;
          newOffset = k;
          break;
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }
        else
        {
          //Check the big box
          //            curSplit = split_node(nodeCenter, nodeSize, bigBoxCenter, bigBoxSize);
          //            if(curSplit == false) break;
        }
      } //For globalCoarseGrpCount[remoteId]

      if(split == false)
      {
        //          uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
          combNodeCheck2 check;
          check.nodeID    = i;
          check.coarseIDOffset = newOffset;
          nextLevel[nextLevelCount++] = check;
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    //      curLevel.clear();
    ////       cout << "Next level: " << nextLevel.size() << endl;
    //      curLevel.assign(nextLevel.begin(), nextLevel.end());
    //      nextLevel.clear();

    curLevelCount = nextLevelCount;
    combNodeCheck2 *temp = curLevel;
    curLevel = nextLevel;
    nextLevel = temp;
    nextLevelCount = 0;

    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}


typedef struct{
  int nodeID;
  int coarseIDOffset;
  int coarseIDEnd;
} combNodeCheck3;

void octree::create_local_essential_tree_count_novector_startend(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level
  //    vector<combNodeCheck> curLevel;
  //    vector<combNodeCheck> nextLevel;

  combNodeCheck3 *curLevel = new combNodeCheck3[1024*64];
  combNodeCheck3 *nextLevel = new combNodeCheck3[1024*64];

  int curLevelCount = 0;
  int nextLevelCount = 0;


  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;


  double4 bigBoxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

  //    vector<int> coarseIDs;
  //    //Add the initial coarse boxes to this level
  //    for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  //    {
  //      coarseIDs.push_back(globalCoarseGrpOffsets[remoteId] + i);
  //    }

  int *coarseIDs = new int[globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck3 check;
    check.nodeID    = i;
    check.coarseIDOffset = 0;
    check.coarseIDEnd = globalCoarseGrpCount[remoteId];
    //      check.coarseIDs.insert(check.coarseIDs.begin(), coarseIDs, coarseIDs+globalCoarseGrpCount[remoteId]);
    curLevel[curLevelCount++] = check;
  }

  //Filter out the initial boxes that will fail anyway
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    // coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
    bool split;
    for(int j=start; j < end; j++)
    {
      real4 nodeCenter  = nodeCenterInfo[j];
      real4 nodeSize    = nodeSizeInfo[j];
      double4 boxCenter = coarseGroupBoxCenter[coarseIDs[i]];
      double4 boxSize   = coarseGroupBoxSize  [coarseIDs[i]];
      float4 nodeCOM    = multipole[j*3 + 0];
      nodeCOM.w         = nodeCenter.w;

      split |= split_node_grav_impbh(nodeCOM, boxCenter, boxSize);

      if(!split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
      {
        //          LOGF(stderr,"LET INITIAL Failed on box %d node: %d\n", i, j);
      }
    }
    if(split == false)
    {
      LOGF(stderr,"LET INITIAL Failed on %d \n", i);
    }
  }

  //    curLevelCount = 0;
  //    //Add the initial nodes to the curLevel list
  //    for(int i=start; i < end; i++)
  //    {
  //      combNodeCheck3 check;
  //      check.nodeID    = i;
  //      check.coarseIDOffset = 0; //Out of range default
  //      check.coarseIDEnd    = globalCoarseGrpCount[remoteId]; //Out of range default
  //
  ////      for(int j=0; j < globalCoarseGrpCount[remoteId]; j++)
  ////      {
  ////        real4 nodeCenter  = nodeCenterInfo[i];
  ////        real4 nodeSize    = nodeSizeInfo[i];
  ////        double4 boxCenter = coarseGroupBoxCenter[coarseIDs[j]];
  ////        double4 boxSize   = coarseGroupBoxSize  [coarseIDs[j]];
  ////        float4 nodeCOM    = multipole[j*3 + 0];
  ////        nodeCOM.w         = nodeCenter.w;
  ////
  ////        //Skip all previous not needed checks
  ////        if(split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
  ////        {
  ////          check.coarseIDOffset = j;
  //////          LOGF(stderr,"LET INITIAL Start node: %d at grp %d \n", i, j);
  ////          break;
  ////        }
  ////      }
  //      curLevel[curLevelCount++] = check;
  //    }




  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck3 check = curLevel[i];
      int node           = check.nodeID;

      //        LOGF(stderr, "LET count On level: %d\tNode: %d\tGoing to check: %d\n",
      //                          level,node, check.coarseIDs.size());

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;
      //        vector<int> checkIDs;
#if 0
      splitChecks++;
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      splitChecks++;


      bool curSplit = false;

      int newOffset = -1;
      int newEnd  = 0;
      bool didBigCheck = false;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      for(int k=check.coarseIDOffset; k < check.coarseIDEnd; k++)
      {
        extraChecks++;
        //  particleCount++;
        //Test this specific box
        int coarseGrpId = coarseIDs[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;

          if(leaf)  break;  //Early out if this is a leaf, since we wont have to check any further

          if(newOffset < 0)
            newOffset = k;
          newEnd = k+1;


          //            break;
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }
        else
        {
          //Instead of checking all boxes we can just go over the tree-structure again
          //if we fail to check if there is any more grp that is required


          //Check the big box
          //            if(!didBigCheck)
          //            {
          //              curSplit = split_node(nodeCenter, nodeSize, bigBoxCenter, bigBoxSize);
          //              if(curSplit == false) break;
          //              didBigCheck = true;
          //            }
        }
      } //For globalCoarseGrpCount[remoteId]

      if(split == false)
      {
        //          uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
          combNodeCheck3 check;
          check.nodeID    = i;
          check.coarseIDOffset = newOffset;
          check.coarseIDEnd    = newEnd;
          nextLevel[nextLevelCount++] = check;
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    //      curLevel.clear();
    ////       cout << "Next level: " << nextLevel.size() << endl;
    //      curLevel.assign(nextLevel.begin(), nextLevel.end());
    //      nextLevel.clear();

    curLevelCount = nextLevelCount;
    combNodeCheck3 *temp = curLevel;
    curLevel = nextLevel;
    nextLevel = temp;
    nextLevelCount = 0;

    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}

void octree::create_local_essential_tree_count_novector_startend2(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level

  combNodeCheck3 *curLevel = new combNodeCheck3[1024*64];
  combNodeCheck3 *nextLevel = new combNodeCheck3[1024*64];

  int curLevelCount = 0;
  int nextLevelCount = 0;


  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;


  double4 bigBoxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};


  int *coarseIDs = new int[globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck3 check;
    check.nodeID    = i;
    check.coarseIDOffset = 0;
    check.coarseIDEnd = globalCoarseGrpCount[remoteId];
    curLevel[curLevelCount++] = check;
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck3 check = curLevel[i];
      int node           = check.nodeID;


      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;
      //        vector<int> checkIDs;
#if 0


#else
      splitChecks++;


      bool curSplit = false;

      int newOffset = -1;
      int newEnd  = 0;
      bool didBigCheck = false;

      int coarseGrpId = coarseIDs[check.coarseIDOffset];

      double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
      double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
      //Minimal distance version

#ifdef INDSOFT
      curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

      if(curSplit)
      {
        //Continue like this
        newOffset = check.coarseIDOffset;
        newEnd    = check.coarseIDEnd;
        split = true;
      }
      else
      {
        //Check others
        for(int k=check.coarseIDOffset+1; k < check.coarseIDEnd; k++)
        {
          //Test this specific box
          int coarseGrpId = coarseIDs[k];

          double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
          double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];

          //Improved barnes hut version
          float4 nodeCOM     = multipole[node*3 + 0];
          nodeCOM.w = nodeCenter.w;
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);

          //Early out if at least one box requires this info
          if(curSplit)
          {
            split = true;

            if(leaf)  break;  //Early out if this is a leaf, since we wont have to check any further

            newOffset = k;
            newEnd    = check.coarseIDEnd;
            break;
          }//if cursplit
        }//end for loop
      }//end else



#endif //if old method

      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
          combNodeCheck3 check;
          check.nodeID    = i;
          check.coarseIDOffset = newOffset;
          check.coarseIDEnd    = newEnd;
          nextLevel[nextLevelCount++] = check;
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    curLevelCount = nextLevelCount;
    combNodeCheck3 *temp = curLevel;
    curLevel = nextLevel;
    nextLevel = temp;
    nextLevelCount = 0;

    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

}


typedef struct{
  int nodeID;
  int coarseIDs[64];
  int coarseIDCount;
} combNodeCheck4;


void octree::create_local_essential_tree_count_novector_startend3(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level

  combNodeCheck4 *curLevel = new combNodeCheck4[1024*64];
  combNodeCheck4 *nextLevel = new combNodeCheck4[1024*64];

  int curLevelCount = 0;
  int nextLevelCount = 0;


  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;


  double4 bigBoxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};


  int *coarseIDs = new int[globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck4 check;
    check.nodeID    = i;
    for(int j=0; j < globalCoarseGrpCount[remoteId]; j++)
    {
      check.coarseIDs[j] = globalCoarseGrpOffsets[remoteId] + j;
    }
    check.coarseIDCount =  globalCoarseGrpCount[remoteId];
    curLevel[curLevelCount++] = check;
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck4 check = curLevel[i];
      int node           = check.nodeID;


      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;

#if 0

#else
      splitChecks++;


      bool curSplit = false;

      int newOffset = 0;
      bool didBigCheck = false;

      int tempList[128];

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      for(int k=0; k < check.coarseIDCount; k++)
      {
        extraChecks++;

        //Test this specific box
        int coarseGrpId = check.coarseIDs[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;

          if(leaf)  break;  //Early out if this is a leaf, since we wont have to check any further

          tempList[newOffset++] = check.coarseIDs[k];
        }
      } //For globalCoarseGrpCount[remoteId]


#endif //if old method

      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          combNodeCheck4 check;
          check.nodeID    = i;
          check.coarseIDCount = newOffset;

          memcpy(check.coarseIDs, tempList, sizeof(int)*newOffset);

          nextLevel[nextLevelCount++] = check;
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    curLevelCount = nextLevelCount;
    combNodeCheck4 *temp = curLevel;
    curLevel = nextLevel;
    nextLevel = temp;
    nextLevelCount = 0;

    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

}



struct cmp_key_value{
  bool operator () (const int2 &a, const int2 &b){
    return ( a.y < b.y);
  }
};

//This one sorts the groups by the most strict one as last
//and the least hit ones in the beginning, to quickly
//filter out things the deeper we go
void octree::create_local_essential_tree_count_novector_startend4(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level
  const int stackSize = 512;
  combNodeCheck3 *curLevel  = new combNodeCheck3[1024*stackSize];
  combNodeCheck3 *nextLevel = new combNodeCheck3[1024*stackSize];


  int curLevelCount  = 0;
  int nextLevelCount = 0;


  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  double4 bigBoxCenter = {  0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

  int2 *coarseIDsTest = new int2[globalCoarseGrpCount[remoteId]];
  int  *coarseIDs     = new int[globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;

    coarseIDsTest[i].x = globalCoarseGrpOffsets[remoteId] + i;
    coarseIDsTest[i].y = 0;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck3 check;
    check.nodeID    = i;
    check.coarseIDOffset = 0;
    check.coarseIDEnd = globalCoarseGrpCount[remoteId];
    curLevel[curLevelCount++] = check;
  }

  //Compute which boxes are most likely to be used and then sort
  //them. Saves another few percent
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    double4 boxCenter = coarseGroupBoxCenter[coarseIDs[i]];
    double4 boxSize   = coarseGroupBoxSize  [coarseIDs[i]];

    for(int j=start; j < end; j++)
    {
      real4 nodeCenter  = nodeCenterInfo[j];
      real4 nodeSize    = nodeSizeInfo[j];
      float4 nodeCOM    = multipole[j*3 + 0];
      nodeCOM.w         = nodeCenter.w;

      if(split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
      {
        coarseIDsTest[i].y++;
      }
    } //for j
  } // for i

  std::sort(coarseIDsTest, coarseIDsTest+globalCoarseGrpCount[remoteId], cmp_key_value());


  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    //       LOGF(stderr, "Box histo2 : %d \t %d \t %d \n", i, coarseIDsTest[i].x, coarseIDsTest[i].y);
    coarseIDs[i] = coarseIDsTest[i].x;
  }
  //Pre-processing done


  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  int maxSizeTemp = -1;

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    maxSizeTemp = max(maxSizeTemp, curLevelCount);

    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck3 check = curLevel[i];
      int node             = check.nodeID;

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;

      int newOffset = -1;
      int newEnd  = 0;

#if 0
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      bool curSplit = false;
      bool didBigCheck = false;

      //          if(split_node(nodeCenter, nodeSize, bigBoxCenter, bigBoxSize))
      {
        for(int k=check.coarseIDOffset; k < check.coarseIDEnd; k++)
        {
          //Test this specific box
          int coarseGrpId = coarseIDs[k];

          double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
          double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
          //Improved Barnes Hut version
          float4 nodeCOM     = multipole[node*3 + 0];
          nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
          //Minimal distance version

#ifdef INDSOFT
          curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
          curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

          //Check if this box needs to go along for the ride further
          //down the tree
          if(curSplit)
          {
            split = true;

            if(leaf)  break;  //Early out if this is a leaf, since we wont have to check any further

            if(newOffset < 0)
              newOffset = k;
            newEnd = k+1;
          } //if curSplit
        } //For globalCoarseGrpCount[remoteId]
      }//Big-check
#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          combNodeCheck3 check;
          check.nodeID                = i;
          check.coarseIDOffset        = newOffset;
          check.coarseIDEnd           = newEnd;
          nextLevel[nextLevelCount++] = check;
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevelCount         = nextLevelCount;
    combNodeCheck3 *temp  = curLevel;
    curLevel              = nextLevel;
    nextLevel             = temp;
    nextLevelCount        = 0;

    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  delete[] curLevel;
  delete[] nextLevel;
  delete[] coarseIDsTest;
  delete[] coarseIDs;


  //    LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d maxStack: %d\n", extraChecks, splitChecks, uselessChecks, maxSizeTemp);

}

void octree::create_local_essential_tree_fill_novector_startend4(real4* bodies, real4* velocities, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int particleCount, int nodeCount, real4 *dataBuffer)
{
  //Walk the tree as is done on device, level by level
  const int stackSize = 512;
  combNodeCheck3 *curLevel  = new combNodeCheck3[1024*stackSize];
  combNodeCheck3 *nextLevel = new combNodeCheck3[1024*stackSize];

  int curLevelCount  = 0;
  int nextLevelCount = 0;


  int level           = 0;

  double4 bigBoxCenter = {  0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

  int2 *coarseIDsTest = new int2[globalCoarseGrpCount[remoteId]];
  int *coarseIDs      = new int [globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;

    coarseIDsTest[i].x = globalCoarseGrpOffsets[remoteId] + i;
    coarseIDsTest[i].y = 0;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck3 check;
    check.nodeID    = i;
    check.coarseIDOffset = 0;
    check.coarseIDEnd = globalCoarseGrpCount[remoteId];
    curLevel[curLevelCount++] = check;
  }

  //Compute which boxes are most likely to be used and then sort
  //them. Saves another few percent
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    for(int j=start; j < end; j++)
    {
      real4 nodeCenter  = nodeCenterInfo[j];
      real4 nodeSize    = nodeSizeInfo[j];
      double4 boxCenter = coarseGroupBoxCenter[coarseIDs[i]];
      double4 boxSize   = coarseGroupBoxSize  [coarseIDs[i]];
      float4 nodeCOM    = multipole[j*3 + 0];
      nodeCOM.w         = nodeCenter.w;

      if(split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
      {
        coarseIDsTest[i].y++;
      }
    } //for j
  } // for i

  std::sort(coarseIDsTest, coarseIDsTest+globalCoarseGrpCount[remoteId], cmp_key_value());


  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    //       LOGF(stderr, "Box histo2 : %d \t %d \t %d \n", i, coarseIDsTest[i].x, coarseIDsTest[i].y);
    coarseIDs[i] = coarseIDsTest[i].x;
  }
  //Pre-processing done



  double massSum = 0;

  int particleOffset     = 1;
  int velParticleOffset  = particleOffset      + particleCount;
  int nodeSizeOffset     = velParticleOffset   + particleCount;
  int nodeCenterOffset   = nodeSizeOffset      + nodeCount;
  int multiPoleOffset    = nodeCenterOffset    + nodeCount;

  //|real4| 2*particleCount*real4| nodes*real4 | nodes*real4 | nodes*3*real4 |
  //Info about #particles, #nodes, start and end of tree-walk
  //The particle positions and velocities
  //The nodeSizeData
  //The nodeCenterData
  //The multipole data

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    dataBuffer[nodeSizeOffset++]   = nodeSizeInfo[i];
    dataBuffer[nodeCenterOffset++] = nodeCenterInfo[i];

    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 0];
    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 1];
    dataBuffer[multiPoleOffset++]  = multipole[i*3 + 2];
  }

  //Start the tree-walk
  //Variables to rewrite the tree-structure indices
  int childNodeOffset         = end;
  int childParticleOffset     = 0;

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck3 check = curLevel[i];
      int node             = check.nodeID;

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                   //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =    childinfo & BODYMASK;                     //the first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }


#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif

      bool split = false;

      int splitIdxToUse = 0;

      int newOffset = -1;
      int newEnd  = 0;


#if 0
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else

      bool curSplit = false;
      bool didBigCheck = false;

      //          if(split_node(nodeCenter, nodeSize, bigBoxCenter, bigBoxSize))
      {
        for(int k=check.coarseIDOffset; k < check.coarseIDEnd; k++)
        {
          //Test this specific box
          int coarseGrpId = coarseIDs[k];


          double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
          double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];

#ifdef IMPBH
          //Improved barnes hut version
          float4 nodeCOM     = multipole[node*3 + 0];
          nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
          //Minimal distance version

#ifdef INDSOFT
          curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
          curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

          //Check if this box needs to go along for the ride further
          //down the tree
          if(curSplit)
          {
            split = true;

            if(leaf)  break;  //Early out if this is a leaf, since we wont have to check any further

            if(newOffset < 0)
              newOffset = k;
            newEnd = k+1;
          } //if curSplit
        } //For globalCoarseGrpCount[remoteId]
      }//Big-check
#endif

      uint temp = 0;  //A node that is not split and is not a leaf will get childinfo 0
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          combNodeCheck3 check;
          check.nodeID                = i;
          check.coarseIDOffset        = newOffset;
          check.coarseIDEnd           = newEnd;
          nextLevel[nextLevelCount++] = check;
        }

        temp = childNodeOffset | (nchild << 28);
        //Update reference to children
        childNodeOffset += nchild;
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          dataBuffer[particleOffset++] = bodies[i];
          dataBuffer[velParticleOffset++] = velocities[i];
          massSum += bodies[i].w;
        }

        temp = childParticleOffset | ((nchild-1) << LEAFBIT);
        childParticleOffset += nchild;
      }



      //Add the node data to the appropriate arrays and modify the node reference
      //start ofset for its children, should be nodeCount at start of this level +numberofnodes on this level
      //plus a counter that counts the number of childs of the nodes we have tested

      //New childoffset:
      union{int i; float f;} itof; //__int_as_float
      itof.i         = temp;
      float tempConv = itof.f;

      //Add node properties and update references
      real4 nodeSizeInfoTemp  = nodeSizeInfo[node];
      nodeSizeInfoTemp.w      = tempConv;             //Replace child reference

      dataBuffer[nodeSizeOffset++]   = nodeSizeInfoTemp;
      dataBuffer[nodeCenterOffset++] = nodeCenterInfo[node];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 0];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 1];
      dataBuffer[multiPoleOffset++]  = multipole[node*3 + 2];

      if(!split)
      {
        massSum += multipole[node*3 + 0].w;
      }
    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevelCount         = nextLevelCount;
    combNodeCheck3 *temp  = curLevel;
    curLevel              = nextLevel;
    nextLevel             = temp;
    nextLevelCount        = 0;

    level++;
  }//end while

  delete[] curLevel;
  delete[] nextLevel;
  delete[] coarseIDsTest;
  delete[] coarseIDs;

  //   cout << "New offsets: "  << particleOffset << " \t" << nodeSizeOffset << " \t" << nodeCenterOffset << endl;
  //    cout << "Mass sum: " << massSum  << endl;
  //   cout << "Mass sumtest: " << multipole[0*0 + 0].w << endl;
}



typedef struct{
  int nodeID;
  int coarseIDList;
  int coarseIDOffset;
  int coarseIDEnd;
} combNodeCheck5;

//This one makes a seperate list for each top node
void octree::create_local_essential_tree_count_novector_startend5(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level
  const int stackSize = 128;
  combNodeCheck5 *curLevel  = new combNodeCheck5[1024*stackSize];
  combNodeCheck5 *nextLevel = new combNodeCheck5[1024*stackSize];

  int curLevelCount  = 0;
  int nextLevelCount = 0;


  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;


  double4 bigBoxCenter = {  0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 bigBoxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};


  int **coarseIDLists = new int*[end-start];

  int *coarseIDs = new int[globalCoarseGrpCount[remoteId]];
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs[i] = globalCoarseGrpOffsets[remoteId] + i;
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {

  }

  //Filters out the boxes for the topnodes. Idea is that there will be
  //less boxes to be checked further down the tree
  int coarseListIdx = 0;
  for(int j=start; j < end; j++)
  {
    coarseIDLists[coarseListIdx] = new int[globalCoarseGrpCount[remoteId]];
    int foundBoxes = 0;
    for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
    {
      real4 nodeCenter  = nodeCenterInfo[j];
      real4 nodeSize    = nodeSizeInfo[j];
      double4 boxCenter = coarseGroupBoxCenter[coarseIDs[i]];
      double4 boxSize   = coarseGroupBoxSize  [coarseIDs[i]];
      float4 nodeCOM    = multipole[j*3 + 0];
      nodeCOM.w         = nodeCenter.w;

      if(split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
      {
        coarseIDLists[coarseListIdx][foundBoxes++] = coarseIDs[i];
      }
    } //for i


    combNodeCheck5 check;
    check.nodeID          = j;
    check.coarseIDOffset  = 0;
    check.coarseIDList    = coarseListIdx;
    check.coarseIDEnd     = foundBoxes;
    curLevel[curLevelCount++] = check;

    coarseListIdx++;

  } // for j

  //    for(int j=0; j < end-start; j++)
  //    {
  //      combNodeCheck5 check = curLevel[j];
  //      LOGF(stderr ,"Top node info; %d %d %d\n", check.nodeID, check.coarseIDList, check.coarseIDEnd);
  //    }

  //Preprocessing done


  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  int maxSizeTemp = -1;

  //Start the tree-walk
  while(curLevelCount > 0)
  {
    maxSizeTemp = max(maxSizeTemp, curLevelCount);

    for(unsigned int i=0; i < curLevelCount; i++)
    {
      //Read node data
      combNodeCheck5 check = curLevel[i];
      int node           = check.nodeID;

      //        LOGF(stderr, "LET count On level: %d\tNode: %d\tGoing to check: %d\n",
      //                          level,node, check.coarseIDs.size());

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;

      int newOffset = -1;
      int newEnd  = 0;

#if 0 //Old big LET method
      splitChecks++;
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      splitChecks++;


      bool curSplit = false;


      bool didBigCheck = false;

      //          if(split_node(nodeCenter, nodeSize, bigBoxCenter, bigBoxSize))
      {
        for(int k=check.coarseIDOffset; k < check.coarseIDEnd; k++)
        {
          extraChecks++;
          //Test this specific box

          //            int coarseGrpId = coarseIDs[k];
          int coarseGrpId = coarseIDLists[check.coarseIDList][k];

          double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
          double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
          //Improved barnes hut version
          float4 nodeCOM     = multipole[node*3 + 0];
          nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
          curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
          //Minimal distance version

#ifdef INDSOFT
          curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
          curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

          //Check if this box needs to go along for the ride further
          //down the tree
          if(curSplit){
            split = true;

            if(leaf)  break;  //Early out if this is a leaf, since we wont have to check any further

            if(newOffset < 0)
              newOffset = k;
            newEnd = k+1;
          }
        } //For globalCoarseGrpCount[remoteId]
      }//Bigcheck


#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          combNodeCheck5 check2;
          check2.nodeID    = i;
          check2.coarseIDOffset = newOffset;
          check2.coarseIDEnd    = newEnd;
          check2.coarseIDList = check.coarseIDList;
          nextLevel[nextLevelCount++] = check2;
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevelCount = nextLevelCount;
    combNodeCheck5 *temp = curLevel;
    curLevel = nextLevel;
    nextLevel = temp;
    nextLevelCount = 0;

    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d maxStack: %d\n", extraChecks, splitChecks, uselessChecks, maxSizeTemp);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}

//This one does an initial filter but note that its not helping
//at all since we filter anyway on level further down....
void octree::create_local_essential_tree_count_vector_filter(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{

  //Walk the tree as is done on device, level by level
  vector<combNodeCheck> curLevel;
  vector<combNodeCheck> nextLevel;

  curLevel.reserve(1024*64);
  nextLevel.reserve(1024*64);



  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  vector<int> coarseIDs;
  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs.push_back(globalCoarseGrpOffsets[remoteId] + i);
  }

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    combNodeCheck check;
    check.nodeID    = i;

    for(int j=0; j < globalCoarseGrpCount[remoteId]; j++)
    {
      real4 nodeCenter  = nodeCenterInfo[i];
      real4 nodeSize    = nodeSizeInfo[i];
      double4 boxCenter = coarseGroupBoxCenter[coarseIDs[j]];
      double4 boxSize   = coarseGroupBoxSize  [coarseIDs[j]];
      float4 nodeCOM    = multipole[j*3 + 0];
      nodeCOM.w         = nodeCenter.w;

      //Skip all previous not needed checks
      if(split_node_grav_impbh(nodeCOM, boxCenter, boxSize))
      {
        check.coarseIDs.push_back(coarseIDs[j]);
      }
    }

    //      check.coarseIDs = coarseIDs;
    curLevel.push_back(check);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      combNodeCheck check = curLevel[i];
      int node           = check.nodeID;

      //        LOGF(stderr, "LET count On level: %d\tNode: %d\tGoing to check: %d\n",
      //                          level,node, check.coarseIDs.size());

      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;
      vector<int> checkIDs;
#if 0
      splitChecks++;
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      splitChecks++;


      bool curSplit = false;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      for(int k=0; k < check.coarseIDs.size(); k++)
      {
        extraChecks++;
        //  particleCount++;
        //Test this specific box
        int coarseGrpId = check.coarseIDs[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        curSplit = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(curSplit){
          split = true;
          checkIDs.push_back(coarseGrpId);
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }
      } //For globalCoarseGrpCount[remoteId]

      if(split == false)
      {
        //          uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
          combNodeCheck check;
          check.nodeID    = i;
          check.coarseIDs = checkIDs;
          nextLevel.push_back(check);
          //            nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size


    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();
    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}


void octree::create_local_essential_tree_count_recursive_part2(
    real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int nodeID,
    vector<int> &remoteGrps, uint remoteGrpStart,
    int &particles, int &nodes)
{
  //Read node data

  int node = nodeID;

  real4 nodeCenter = nodeCenterInfo[node];
  real4 nodeSize   = nodeSizeInfo[node];
  bool leaf        = nodeCenter.w <= 0;

  union{float f; int i;} u; //__float_as_int
  u.f           = nodeSize.w;
  int childinfo = u.i;

  int child, nchild;
  if(!leaf)
  {
    //Node
    child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
    nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
  }
  else
  {
    //Leaf
    child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
    nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
  }

  bool split = false;

  int splitIdxToUse = 0;

  bool curSplit = false;

  int coarseGrpStart;

  //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  for(int k=remoteGrpStart; k < remoteGrps.size(); k++)
  {
    //Test this specific box
    coarseGrpStart  = k;
    int coarseGrpId = remoteGrps[k];

    double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
    double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];

    //Improved barnes hut version
    float4 nodeCOM     = multipole[node*3 + 0];
    nodeCOM.w = nodeCenter.w;

    curSplit = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);

    //Early out if at least one box requires this info
    if(curSplit){
      split = true;
      break;
    }
  } //For globalCoarseGrpCount[remoteId]


  //if split & node add children to next lvl stack
  if(split && !leaf)
  {
    for(int i=child; i < child+nchild; i++)
    {
      nodes++;
      create_local_essential_tree_count_recursive_part2(
          bodies, multipole, nodeSizeInfo, nodeCenterInfo,
          i, remoteGrps, coarseGrpStart ,particles, nodes);
    }
  }

  //if split & leaf add particles to particle list
  if(split && leaf)
  {
    for(int i=child; i < child+nchild; i++)
    {
      particles++;
    }
  }
}


void octree::create_local_essential_tree_count_recursive(
    real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  vector<int> coarseIDs;

  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    coarseIDs.push_back(globalCoarseGrpOffsets[remoteId] + i);
  }

  nodeCount += start;
  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    nodeCount++;
    create_local_essential_tree_count_recursive_part2(
        bodies, multipole, nodeSizeInfo, nodeCenterInfo,
        i, coarseIDs, 0,particleCount, nodeCount);

  }


  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}


void octree::create_local_essential_tree_count_recursive_try2(
    real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  uint2 node_begend;
  node_begend.x   = this->localTree.level_list[2].x;
  node_begend.y   = this->localTree.level_list[2].y;

  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    //      if(procId == 1)   LOGF(stderr,"Going to check : %d \n", globalCoarseGrpOffsets[remoteId] + i);
    bool allDone = false;
    for(int k = node_begend.x; k < node_begend.y; k++)
    {
      create_local_essential_tree_count_recursive_part2_try2(
          bodies, multipole, nodeSizeInfo, nodeCenterInfo,
          k, globalCoarseGrpOffsets[remoteId] + i,particleCount, nodeCount, allDone);
    }

  }

  nodeCount += start;
  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    nodeCount++;
  }


  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}


//This one checks only one box
void octree::create_local_essential_tree_count_recursive_part2_try2(
    real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int nodeID, uint remoteGrpID, int &particles, int &nodes, bool &allDone)
{
  //Read node data

  int node = nodeID;

  real4 nodeCenter = nodeCenterInfo[node];
  real4 nodeSize   = nodeSizeInfo[node];
  bool leaf        = nodeCenter.w <= 0;

  union{float f; int i;} u; //__float_as_int
  u.f           = nodeSize.w;
  int childinfo = u.i;

  int child, nchild;
  if(!leaf)
  {
    //Node
    child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
    nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has

    //Early out if this node has been processed before
    if(childinfo == 0xFFFFFFFF) return;
  }
  else
  {
    //Leaf
    child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
    nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag

    //Early out if this leaf has been processed before
    if(childinfo == 0xFFFFFFFF) return;
  }

  bool split = false;

  //Test this specific box
  double4 boxCenter = coarseGroupBoxCenter[remoteGrpID];
  double4 boxSize   = coarseGroupBoxSize  [remoteGrpID];

  //Improved barnes hut version
  float4 nodeCOM     = multipole[node*3 + 0];
  nodeCOM.w = nodeCenter.w;

  split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);

  //if split & node add children to next lvl stack
  if(split && !leaf)
  {
    int nNodesDone = 0;
    for(int i=child; i < child+nchild; i++)
    {
      bool thisOneDone = false;
      nodes++;
      create_local_essential_tree_count_recursive_part2_try2(
          bodies, multipole, nodeSizeInfo, nodeCenterInfo,
          i, remoteGrpID ,particles, nodes, thisOneDone);
      if(thisOneDone) nNodesDone++;
    }
    if(nNodesDone == nchild)
    {
      //        LOGF(stderr, "Processed a full node! [%d - %d ] %d DONE? : %d\n",child, child+nchild, nNodesDone, nNodesDone == nchild);
      nodeSizeInfo[node].w = host_int_as_float(0xFFFFFFFF);
    }
    //      if(nNodesDone >= nchild/2)
    //      {
    ////        LOGF(stderr, "ALMOST a full node! [%d - %d ] %d DONE? : %d\n",
    ////            child, child+nchild, nNodesDone, nNodesDone == nchild);
    //        nodeSizeInfo[node].w = host_int_as_float(0xFFFFFFFF);
    //      }
  }

  //if split & leaf add particles to particle list
  if(split && leaf)
  {
    for(int i=child; i < child+nchild; i++)
    {
      particles++;
    }
    //Modify this leaf, so we do not process it anymore
    nodeSizeInfo[node].w = host_int_as_float(0xFFFFFFFF);
    allDone = true;
    //      LOGF(stderr, "Processed a leaf %d\n", node);
  }
}





#endif
#if 0
//This one remembers the index of where the split happend and uses this as start next time
//all before are ignored

//void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
//                                         double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
//                                         int &particles, int &nodes)
void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level
  vector<int2> curLevel;
  vector<int2> nextLevel;

  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  vector<int> boxIndicesToUse;

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    curLevel.push_back(make_int2(i,0));
  }

  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    boxIndicesToUse.push_back(globalCoarseGrpOffsets[remoteId] + i);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  //Start the tree-walk
  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      int node           = curLevel[i].x;
      int startCoarseBox = curLevel[i].y;
      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;
#if 0
      splitChecks++;
      double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
        0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
        0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
      double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
        fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
        fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

#ifdef IMPBH
      //Improved barnes hut version
      float4 nodeCOM     = multipole[node*3 + 0];
      nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
      split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif
#else
      //Minimal distance version
#ifdef INDSOFT
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
      split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

#else
      splitChecks++;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      for(int k=startCoarseBox; k < boxIndicesToUse.size(); k++)
      {
        //  particleCount++;
        //Test this specific box
        int coarseGrpId = boxIndicesToUse[k];

        double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(split){
          splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          extraChecks += splitIdxToUse-startCoarseBox;
          break;
        }
      } //For globalCoarseGrpCount[remoteId]

      if(split == false)
      {
        uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

#endif
      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size



    //      Hier gebleven dit werkt niet. Er is altijd wel 1 group die het heeft
    //      Oplossing, per node bijgaan houden waar we zitten in de lijst
    //      Dit kan door curLevel en nextLevel niet als int op te slaan maar als int2
    //      en dan in x het nodeID en in y de i waar we gebleven waren met de split


    //End reduce the boxes to use

    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();
    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}
#endif


#if 0
//This is the one that goes over the full box-grp

//void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
//                                         double4 boxCenter, double4 boxSize, float group_eps, int start, int end,
//                                         int &particles, int &nodes)
void octree::create_local_essential_tree_count(real4* bodies, real4* multipole, real4* nodeSizeInfo, real4* nodeCenterInfo,
    int remoteId, float group_eps, int start, int end,
    int &particles, int &nodes)
{
  //Walk the tree as is done on device, level by level
  vector<int2> curLevel;
  vector<int2> nextLevel;

  int particleCount   = 0;
  int nodeCount       = 0;

  int level           = 0;

  int extraChecks = 0;
  int uselessChecks = 0;
  int splitChecks = 0;

  vector<int> boxIndicesToUse;

  //Add the initial nodes to the curLevel list
  for(int i=start; i < end; i++)
  {
    curLevel.push_back(make_int2(i,0));
  }

  //Add the initial coarse boxes to this level
  for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
  {
    boxIndicesToUse.push_back(globalCoarseGrpOffsets[remoteId] + i);
  }

  //Add the nodes before the start and end to the node list
  for(int i=0; i < start; i++)
  {
    nodeCount++;
  }

  double4 boxCenter = {     0.5*(currentRLow[remoteId].x  + currentRHigh[remoteId].x),
    0.5*(currentRLow[remoteId].y  + currentRHigh[remoteId].y),
    0.5*(currentRLow[remoteId].z  + currentRHigh[remoteId].z), 0};
  double4 boxSize   = {fabs(0.5*(currentRHigh[remoteId].x - currentRLow[remoteId].x)),
    fabs(0.5*(currentRHigh[remoteId].y - currentRLow[remoteId].y)),
    fabs(0.5*(currentRHigh[remoteId].z - currentRLow[remoteId].z)), 0};

  LOGF(stderr,"TEST: %f %f %f || %f %f %f \n", boxCenter.x, boxCenter.y, boxCenter.z,
      boxSize.x, boxSize.y, boxSize.z);

  //Start the tree-walk
  while(curLevel.size() > 0)
  {
    for(unsigned int i=0; i < curLevel.size(); i++)
    {
      //Read node data
      int node           = curLevel[i].x;
      int startCoarseBox = curLevel[i].y;
      real4 nodeCenter = nodeCenterInfo[node];
      real4 nodeSize   = nodeSizeInfo[node];
      bool leaf        = nodeCenter.w <= 0;

      union{float f; int i;} u; //__float_as_int
      u.f           = nodeSize.w;
      int childinfo = u.i;

      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

#ifdef INDSOFT
      //Very inefficient this but for testing I have to live with it...
      float node_eps_val = multipole[node*3 + 1].w;
#endif


      bool split = false;

      int splitIdxToUse = 0;

      splitChecks++;

      //        for(int i=0; i < globalCoarseGrpCount[remoteId]; i++)
      //        for(int k=startCoarseBox; k < boxIndicesToUse.size(); k++)
      {
        //  particleCount++;
        //Test this specific box
        //          int coarseGrpId = boxIndicesToUse[k];
        //
        //          double4 boxCenter = coarseGroupBoxCenter[coarseGrpId];
        //          double4 boxSize   = coarseGroupBoxSize  [coarseGrpId];


#ifdef IMPBH
        //Improved barnes hut version
        float4 nodeCOM     = multipole[node*3 + 0];
        nodeCOM.w = nodeCenter.w;

#ifdef INDSOFT
        split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize, group_eps, node_eps_val);
#else
        split = split_node_grav_impbh(nodeCOM, boxCenter, boxSize);
#endif

#else
        //Minimal distance version

#ifdef INDSOFT
        split = split_node(nodeCenter, nodeSize, boxCenter, boxSize, group_eps, node_eps_val);  //Check if node should be split
#else
        split = split_node(nodeCenter, nodeSize, boxCenter, boxSize);
#endif
#endif //if IMPBH

        //Early out if at least one box requires this info
        if(split){
          //              splitIdxToUse = k;
          //              LOGF(stderr, "LET count On level: %d\tNode: %d\tStart: %d\tEnd: %d\tChecks: %d \n",
          //                  level,node, startCoarseBox, splitIdxToUse, splitIdxToUse-startCoarseBox+1);

          //              extraChecks += splitIdxToUse-startCoarseBox;
          //              break;
        }
      } //For globalCoarseGrpCount[remoteId]

      if(split == false)
      {
        uselessChecks +=  boxIndicesToUse.size()-startCoarseBox;
      }

      //if split & node add children to next lvl stack
      if(split && !leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          nextLevel.push_back(make_int2(i, splitIdxToUse));
        }
      }

      //if split & leaf add particles to particle list
      if(split && leaf)
      {
        for(int i=child; i < child+nchild; i++)
        {
          particleCount++;
        }
      }

      //Increase the nodeCount, since this node will be part of the tree-structure
      nodeCount++;
    } //end for curLevel.size



    //      Hier gebleven dit werkt niet. Er is altijd wel 1 group die het heeft
    //      Oplossing, per node bijgaan houden waar we zitten in de lijst
    //      Dit kan door curLevel en nextLevel niet als int op te slaan maar als int2
    //      en dan in x het nodeID en in y de i waar we gebleven waren met de split


    //End reduce the boxes to use

    //Put next level stack into current level and continue
    curLevel.clear();

    //       cout << "Next level: " << nextLevel.size() << endl;
    curLevel.assign(nextLevel.begin(), nextLevel.end());
    nextLevel.clear();
    level++;
  }//end while

  particles = particleCount;
  nodes     = nodeCount;

  LOGF(stderr, "LET Extra checks: %d SplitChecks: %d UselessChecks: %d\n", extraChecks, splitChecks, uselessChecks);

  /*    fprintf(stderr, "Count found: %d particles and %d nodes. Boxsize: (%f %f %f ) BoxCenter: (%f %f %f)\n",
        particles, nodes, boxSize.x ,boxSize.y, boxSize.z, boxCenter.x, boxCenter.y, boxCenter.z );  */
}
#endif


#if 0
//Exchange particles with other processes
int octree::gpu_exchange_particles_with_overflow_check(tree_structure &tree,
    bodyStruct *particlesToSend,
    my_dev::dev_mem<uint> &extractList,
    int nToSend)
{
  int myid      = procId;
  int nproc     = nProcs;
  int iloc      = 0;
  int nbody     = nToSend;


  bodyStruct  tmpp;

  int *firstloc   = new int[nProcs+1];
  int *nparticles = new int[nProcs+1];

  // Loop over particles and determine which particle needs to go where
  // reorder the bodies in such a way that bodies that have to be send
  // away are stored after each other in the array
  double t1 = get_time();

  //Array reserve some memory at forehand , 1%
  vector<bodyStruct> array2Send;
  array2Send.reserve((int)(nToSend*1.5));

  for(int ib=0;ib<nproc;ib++)
  {
    int ibox       = (ib+myid)%nproc;
    firstloc[ibox] = iloc;      //Index of the first particle send to proc: ibox

    for(int i=iloc; i<nbody;i++)
    {
      if(isinbox(particlesToSend[i].Ppos, domainRLow[ibox], domainRHigh[ibox]))
      {
        //Reorder the particle information
        tmpp                  = particlesToSend[iloc];
        particlesToSend[iloc] = particlesToSend[i];
        particlesToSend[i]    = tmpp;

        //Put the particle in the array of to send particles
        array2Send.push_back(particlesToSend[iloc]);

        iloc++;
      }// end if
    }//for i=iloc
    nparticles[ibox] = iloc-firstloc[ibox];//Number of particles that has to be send to proc: ibox
  } // for(int ib=0;ib<nproc;ib++)


  //   printf("Required search time: %lg ,proc: %d found in our own box: %d n: %d  to others: %ld \n",
  //          get_time()-t1, myid, nparticles[myid], tree.n, array2Send.size());


  if(iloc < nbody)
  {
    cerr << procId <<" exchange_particle error: particle in no box...iloc: " << iloc
      << " and nbody: " << nbody << "\n";
    exit(0);
  }


  /*totalsent = nbody - nparticles[myid];

    int tmp;
    MPI_Reduce(&totalsent,&tmp,1, MPI_INT, MPI_SUM,0,MPI_COMM_WORLD);
    if(procId == 0)
    {
    totalsent = tmp;
    cout << "Exchanged particles = " << totalsent << endl;
    }*/

  t1 = get_time();

  //Allocate two times the amount of memory of that which we send
  vector<bodyStruct> recv_buffer3(nbody*2);
  unsigned int recvCount = 0;

  //Exchange the data with the other processors
  int ibend = -1;
  int nsend;
  int isource = 0;
  for(int ib=nproc-1;ib>0;ib--)
  {
    int ibox = (ib+myid)%nproc; //index to send...

    if (ib == nproc-1)
    {
      isource= (myid+1)%nproc;
    }
    else
    {
      isource = (isource+1)%nproc;
      if (isource == myid)isource = (isource+1)%nproc;
    }

    if(MP_exchange_particle_with_overflow_check<bodyStruct>(ibox, &array2Send[0],
          recv_buffer3, firstloc[ibox] - nparticles[myid],
          nparticles[ibox], isource,
          nsend, recvCount))
    {
      ibend = ibox; //Here we get if exchange failed
      ib = 0;
    }//end if mp exchang
  }//end for all boxes


  if(ibend == -1){

  }else{
    //Something went wrong
    cerr << "ERROR in exchange_particles_with_overflow_check! \n"; exit(0);
  }


  LOG("Required inter-process communication time: %lg ,proc: %d\n",
      get_time()-t1, myid);

  //Compute the new number of particles:
  int newN = tree.n + recvCount - nToSend;

  execStream->sync();   //make certain that the particle movement on the device
  //is complete before we resize

  //Have to resize the bodies vector to keep the numbering correct
  //but do not reduce the size since we need to preserve the particles
  //in the oversized memory
  int memSize = newN*1.05; //5% extra
  tree.bodies_pos.cresize (memSize + 1, false);
  tree.bodies_acc0.cresize(memSize,     false);
  tree.bodies_acc1.cresize(memSize,     false);
  tree.bodies_vel.cresize (memSize,     false);
  tree.bodies_time.cresize(memSize,     false);
  tree.bodies_ids.cresize (memSize + 1, false);
  tree.bodies_Ppos.cresize(memSize + 1, false);
  tree.bodies_Pvel.cresize(memSize + 1, false);

  //This one has to be at least the same size as the number of particles inorder to
  //have enough space to store the other buffers
  //Can only be resized after we are done since we still have
  //parts of memory pointing to that buffer (extractList)
  //Note that we allocate some extra memory to make everything texture/memory alligned
  tree.generalBuffer1.cresize(3*(memSize)*4 + 4096, false);

  //Now we have to copy the data in batches incase the generalBuffer1 is not large enough
  //Amount we can store:
  int spaceInIntSize    = 3*(newN)*4;
  int newParticleSpace  = spaceInIntSize / (sizeof(bodyStruct) / sizeof(int));
  int stepSize = newParticleSpace;

  my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);

  int memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1,
      stepSize, 0);


  LOGF(stderr, "Exchange, received particles: (%d): %d \tnewN: %d\tItems that can be insert in one step: %d\n",
      procId, recvCount, newN, stepSize);

  int insertOffset = 0;
  for(unsigned int i=0; i < recvCount; i+= stepSize)
  {
    int items = min(stepSize, (int)(recvCount-i));

    if(items > 0)
    {
      //Copy the data from the MPI receive buffers into the GPU-send buffer
      memcpy(&bodyBuffer[0], &recv_buffer3[insertOffset], sizeof(bodyStruct)*items);

      bodyBuffer.h2d(items);

      //       int threads = max(nToSend, (int)recvCount);

      //Start the kernel that puts everything in place
      insertNewParticles.set_arg<int>(0,    &nToSend);
      insertNewParticles.set_arg<int>(1,    &items);
      insertNewParticles.set_arg<int>(2,    &tree.n);
      insertNewParticles.set_arg<int>(3,    &insertOffset);
      insertNewParticles.set_arg<cl_mem>(4, localTree.bodies_Ppos.p());
      insertNewParticles.set_arg<cl_mem>(5, localTree.bodies_Pvel.p());
      insertNewParticles.set_arg<cl_mem>(6, localTree.bodies_pos.p());
      insertNewParticles.set_arg<cl_mem>(7, localTree.bodies_vel.p());
      insertNewParticles.set_arg<cl_mem>(8, localTree.bodies_acc0.p());
      insertNewParticles.set_arg<cl_mem>(9, localTree.bodies_acc1.p());
      insertNewParticles.set_arg<cl_mem>(10, localTree.bodies_time.p());
      insertNewParticles.set_arg<cl_mem>(11, localTree.bodies_ids.p());
      insertNewParticles.set_arg<cl_mem>(12, bodyBuffer.p());
      insertNewParticles.setWork(items, 128);
      insertNewParticles.execute(execStream->s());
    }

    insertOffset += items;
  }

  //   printf("Required gpu malloc time step1: %lg \t Size: %d \tRank: %d \t Size: %d \n",
  //          get_time()-t1, newN, mpiGetRank(), tree.bodies_Ppos.get_size());
  //   t1 = get_time();


  tree.setN(newN);

  //Resize the arrays of the tree
  reallocateParticleMemory(tree);

  //   printf("Required gpu malloc tijd step 2: %lg \n", get_time()-t1);
  //   printf("Total GPU interaction time: %lg \n", get_time()-t2);

  int retValue = 0;

  delete[] firstloc;
  delete[] nparticles;

  return retValue;
}

#endif
//Exchange particles with other processes
int octree::gpu_exchange_particles_with_overflow_check(tree_structure &tree,
    bodyStruct *particlesToSend,
    my_dev::dev_mem<uint> &extractList,
    int nToSend)
{
  int myid      = procId;
  int nproc     = nProcs;
  int iloc      = 0;
  int nbody     = nToSend;


  bodyStruct  tmpp;

  int *firstloc   = new int[nProcs+1];
  int *nparticles = new int[nProcs+1];

  // Loop over particles and determine which particle needs to go where
  // reorder the bodies in such a way that bodies that have to be send
  // away are stored after each other in the array
  double t1 = get_time();

  //Array reserve some memory at forehand , 1%
  vector<bodyStruct> array2Send;
  array2Send.reserve((int)(nToSend*1.5));

  for(int ib=0;ib<nproc;ib++)
  {
    int ibox       = (ib+myid)%nproc;
    firstloc[ibox] = iloc;      //Index of the first particle send to proc: ibox

    for(int i=iloc; i<nbody;i++)
    {
      if(isinbox(particlesToSend[i].Ppos, domainRLow[ibox], domainRHigh[ibox]))
      {
        //Reorder the particle information
        tmpp                  = particlesToSend[iloc];
        particlesToSend[iloc] = particlesToSend[i];
        particlesToSend[i]    = tmpp;

        //Put the particle in the array of to send particles
        array2Send.push_back(particlesToSend[iloc]);

        iloc++;
      }// end if
    }//for i=iloc
    nparticles[ibox] = iloc-firstloc[ibox];//Number of particles that has to be send to proc: ibox
  } // for(int ib=0;ib<nproc;ib++)


  //   printf("Required search time: %lg ,proc: %d found in our own box: %d n: %d  to others: %ld \n",
  //          get_time()-t1, myid, nparticles[myid], tree.n, array2Send.size());


  if(iloc < nbody)
  {
    cerr << procId <<" exchange_particle error: particle in no box...iloc: " << iloc
      << " and nbody: " << nbody << "\n";
    exit(0);
  }


  /*totalsent = nbody - nparticles[myid];

    int tmp;
    MPI_Reduce(&totalsent,&tmp,1, MPI_INT, MPI_SUM,0,MPI_COMM_WORLD);
    if(procId == 0)
    {
    totalsent = tmp;
    cout << "Exchanged particles = " << totalsent << endl;
    }*/

  t1 = get_time();

  //Allocate two times the amount of memory of that which we send
  vector<bodyStruct> recv_buffer3(nbody*2);
  unsigned int recvCount = 0;

  //Exchange the data with the other processors
  int ibend = -1;
  int nsend;
  int isource = 0;
  for(int ib=nproc-1;ib>0;ib--)
  {
    int ibox = (ib+myid)%nproc; //index to send...

    if (ib == nproc-1)
    {
      isource= (myid+1)%nproc;
    }
    else
    {
      isource = (isource+1)%nproc;
      if (isource == myid)isource = (isource+1)%nproc;
    }

    if(MP_exchange_particle_with_overflow_check<bodyStruct>(ibox, &array2Send[0],
          recv_buffer3, firstloc[ibox] - nparticles[myid],
          nparticles[ibox], isource,
          nsend, recvCount))
    {
      ibend = ibox; //Here we get if exchange failed
      ib = 0;
    }//end if mp exchang
  }//end for all boxes


  if(ibend == -1){

  }else{
    //Something went wrong
    cerr << "ERROR in exchange_particles_with_overflow_check! \n"; exit(0);
  }


  LOG("Required inter-process communication time: %lg ,proc: %d\n",
      get_time()-t1, myid);

  //Compute the new number of particles:
  int newN = tree.n + recvCount - nToSend;

  execStream->sync();   //make certain that the particle movement on the device
  //is complete before we resize

  //Have to resize the bodies vector to keep the numbering correct
  //but do not reduce the size since we need to preserve the particles
  //in the oversized memory
  int memSize = newN*1.05; //5% extra
  tree.bodies_pos.cresize (memSize + 1, false);
  tree.bodies_acc0.cresize(memSize,     false);
  tree.bodies_acc1.cresize(memSize,     false);
  tree.bodies_vel.cresize (memSize,     false);
  tree.bodies_time.cresize(memSize,     false);
  tree.bodies_ids.cresize (memSize + 1, false);
  tree.bodies_Ppos.cresize(memSize + 1, false);
  tree.bodies_Pvel.cresize(memSize + 1, false);

  //This one has to be at least the same size as the number of particles inorder to
  //have enough space to store the other buffers
  //Can only be resized after we are done since we still have
  //parts of memory pointing to that buffer (extractList)
  //Note that we allocate some extra memory to make everything texture/memory alligned
  tree.generalBuffer1.cresize(3*(memSize)*4 + 4096, false);

  //Now we have to copy the data in batches incase the generalBuffer1 is not large enough
  //Amount we can store:
  int spaceInIntSize    = 3*(newN)*4;
  int newParticleSpace  = spaceInIntSize / (sizeof(bodyStruct) / sizeof(int));
  int stepSize = newParticleSpace;

  my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);

  int memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1,
      stepSize, 0);


  LOGF(stderr, "Exchange, received particles: (%d): %d \tnewN: %d\tItems that can be insert in one step: %d\n",
      procId, recvCount, newN, stepSize);

  int insertOffset = 0;
  for(unsigned int i=0; i < recvCount; i+= stepSize)
  {
    int items = min(stepSize, (int)(recvCount-i));

    if(items > 0)
    {
      //Copy the data from the MPI receive buffers into the GPU-send buffer
      memcpy(&bodyBuffer[0], &recv_buffer3[insertOffset], sizeof(bodyStruct)*items);

      bodyBuffer.h2d(items);

      //       int threads = max(nToSend, (int)recvCount);

      //Start the kernel that puts everything in place
      insertNewParticles.set_arg<int>(0,    &nToSend);
      insertNewParticles.set_arg<int>(1,    &items);
      insertNewParticles.set_arg<int>(2,    &tree.n);
      insertNewParticles.set_arg<int>(3,    &insertOffset);
      insertNewParticles.set_arg<cl_mem>(4, localTree.bodies_Ppos.p());
      insertNewParticles.set_arg<cl_mem>(5, localTree.bodies_Pvel.p());
      insertNewParticles.set_arg<cl_mem>(6, localTree.bodies_pos.p());
      insertNewParticles.set_arg<cl_mem>(7, localTree.bodies_vel.p());
      insertNewParticles.set_arg<cl_mem>(8, localTree.bodies_acc0.p());
      insertNewParticles.set_arg<cl_mem>(9, localTree.bodies_acc1.p());
      insertNewParticles.set_arg<cl_mem>(10, localTree.bodies_time.p());
      insertNewParticles.set_arg<cl_mem>(11, localTree.bodies_ids.p());
      insertNewParticles.set_arg<cl_mem>(12, bodyBuffer.p());
      insertNewParticles.setWork(items, 128);
      insertNewParticles.execute(execStream->s());
    }

    insertOffset += items;
  }

  //   printf("Required gpu malloc time step1: %lg \t Size: %d \tRank: %d \t Size: %d \n",
  //          get_time()-t1, newN, mpiGetRank(), tree.bodies_Ppos.get_size());
  //   t1 = get_time();


  tree.setN(newN);

  //Resize the arrays of the tree
  reallocateParticleMemory(tree);

  //   printf("Required gpu malloc tijd step 2: %lg \n", get_time()-t1);
  //   printf("Total GPU interaction time: %lg \n", get_time()-t2);

  int retValue = 0;

  delete[] firstloc;
  delete[] nparticles;

  return retValue;
}



//Improved Barnes Hut criterium
bool split_node_grav_impbh_SFCtest(float4 nodeCOM, double4 boxCenter, double4 boxSize, float &ds2)

{
  //Compute the distance between the group and the cell
  float3 dr = make_float3(fabs((float)boxCenter.x - nodeCOM.x) - (float)boxSize.x,
      fabs((float)boxCenter.y - nodeCOM.y) - (float)boxSize.y,
      fabs((float)boxCenter.z - nodeCOM.z) - (float)boxSize.z);

  dr.x += fabs(dr.x); dr.x *= 0.5f;
  dr.y += fabs(dr.y); dr.y *= 0.5f;
  dr.z += fabs(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  //  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;


#ifdef INDSOFT
  if(ds2      <= ((group_eps + node_eps ) * (group_eps + node_eps) ))           return true;
  //Limited precision can result in round of errors. Use this as extra safe guard
  if(fabs(ds2 -  ((group_eps + node_eps ) * (group_eps + node_eps) )) < 10e-04) return true;
#endif

  if (ds2     <= fabs(nodeCOM.w))           return true;
  if (fabs(ds2 - fabs(nodeCOM.w)) < 10e-04) return true; //Limited precision can result in round of errors. Use this as extra safe guard

  return false;
}




#if 0


//Sort using our custom merge sort, this requires that the subranges are sorted
//already!

#if 0
//Merge the received results with the already available data, this works
//for two processes only
uint4 *result = new uint4[totalNumberOfHashes];
merge_sort(result, &allHashes[0],nReceiveCnts[0] / sizeof(uint4),
    &allHashes[nReceiveDpls[1] / sizeof(uint4) ], nReceiveCnts[1] / sizeof(uint4));
memcpy(allHashes, result, sizeof(uint4)*totalNumberOfHashes);
#endif

#if 0

//Multiple merges to merge the different items
uint4 *result = new uint4[totalNumberOfHashes];
bool ping = true;
for(int i=0; i < nProcs-1; i++)
{
  if(ping)
  {
    merge_sort(result, &allHashes[0],nReceiveDpls[i+1] / sizeof(uint4),
        &allHashes[nReceiveDpls[i+1] / sizeof(uint4) ], //start
        nReceiveCnts[i+1] / sizeof(uint4)); //items
    ping = false;
    fprintf(stderr,"StartA at: %d  count: %d \n",nReceiveDpls[i+1] / sizeof(uint4), nReceiveCnts[i+1] / sizeof(uint4));
  }
  else
  {
    merge_sort(allHashes, &result[0],nReceiveDpls[i+1] / sizeof(uint4),
        &allHashes[nReceiveDpls[i+1] / sizeof(uint4) ], //start
        nReceiveCnts[i+1] / sizeof(uint4));             //items
    ping = true;
    fprintf(stderr,"StartB at: %d  count: %d \n",nReceiveDpls[i+1] / sizeof(uint4), nReceiveCnts[i+1] / sizeof(uint4));
  }
}

if(!ping)
{
  memcpy(allHashes, result, sizeof(uint4)*totalNumberOfHashes);
}

#endif


//Test ex
#if 0
//Multi-merge
uint4 *result = new uint4[totalNumberOfHashes]; //TEST
int *sizes  = new int[nProcs];
int *starts = new int[nProcs];

for(int z = 0 ; z < nProcs; z++)
{
  sizes[z] = nReceiveCnts[z] / sizeof(uint4);
  starts[z] = nReceiveDpls[z] / sizeof(uint4);
}

merge_sort2(result, allHashes,sizes, starts, nProcs);
//end test ex
memcpy(allHashes, result, sizeof(uint4)*totalNumberOfHashes);
#endif

#if 0
//Multi-merge using priority queue
uint4 *result = new uint4[totalNumberOfHashes]; //TEST
int *sizes  = new int[nProcs];
int *starts = new int[nProcs];

for(int z = 0 ; z < nProcs; z++)
{
  sizes[z] = nReceiveCnts[z] / sizeof(uint4);
  starts[z] = nReceiveDpls[z] / sizeof(uint4);
}

merge_sort3(result, allHashes,sizes, starts, nProcs);
//end test ex
memcpy(allHashes, result, sizeof(uint4)*totalNumberOfHashes);
#endif



LOGF(stderr, "Domain hash sort: %f on number of particles: %d\n",
    get_time()-t1, totalNumberOfHashes);

sumTime += get_time()-t1;


int sum = 0;
for(int i=0; i < totalNumberOfHashes-1; i++)
{
  int comp = cmp_uint4_host(allHashes[i], allHashes[i+1]);

  if(comp > 0)
  {
    LOGF(stderr, "Sorting FAILED to get the correct order :(  %d \n", comp);
    LOGF(stderr,"%d \t Key: %d %d %d \tsize->\t %d\n", i,
        allHashes[i].x, allHashes[i].y, allHashes[i].z, allHashes[i].w);

    LOGF(stderr,"%d \t Key: %d %d %d \tsize->\t %d\n", i+1,
        allHashes[i+1].x, allHashes[i+1].y, allHashes[i+1].z, allHashes[i+1].w);
  }


  // LOGF(stderr,"%d \t Key: %d %d %d \tsize->\t %d\n", i,
  //      allHashes[i].x, allHashes[i].y, allHashes[i].z, allHashes[i].w);
  sum += allHashes[i].w;
}

//   LOGF(stderr,"%d \t Key: %d %d %d \tsize->\t %d\n", totalNumberOfHashes-1,        allHashes[totalNumberOfHashes-1].x, allHashes[totalNumberOfHashes-1].y, allHashes[totalNumberOfHashes-1].z, allHashes[totalNumberOfHashes-1].w);


sum += allHashes[totalNumberOfHashes-1].w;
LOGF(stderr,"Total particlesA: %d \n", sum);





Old sorting codes

void merge_sort(uint4 *result, uint4 *left, int sizeLeft, uint4 *right, int sizeRight)
{
  uint iLeft = 0;
  uint iRight = 0;
  uint res = 0;

  while(iLeft < sizeLeft && iRight < sizeRight)
  {
    //      uint4 valLeft  = left[iLeft];
    //      uint4 valRight = right[iRight];

    int comp = cmp_uint4_host(left[iLeft], right[iRight]);

    if (comp <= 0)
    {
      result[res++] = left[iLeft];
      ++iLeft;
    }
    if (comp >= 0)
    {
      result[res++] = right[iRight];
      ++iRight;
    }
  }

  if(iLeft!=sizeLeft)
    memcpy(&result[res], &left[iLeft],   sizeof(uint4)*(sizeLeft-iLeft));
  else
    memcpy(&result[res], &right[iRight], sizeof(uint4)*(sizeRight-iRight));
}

void merge_sort2(uint4 *result, uint4 *data, int *sizes, int *starts, int nLists)
{
  uint iLeft = 0;
  uint iRight = 0;
  uint res = 0;

  fprintf(stderr, "Merge sorting, total lists: %d \n", nLists);

  uint4 *queue  = new uint4[nLists];
  int4 *readIdx = new int4[nLists];

  for(int i=0; i < nLists; i++)
  {
    queue[i]   = data[starts[i]];

    int4 read;
    read.x = i; //Source file
    read.y = starts[i]; //read Index in Data
    read.z = sizes[i]-1; //items left to process

    fprintf(stderr, "List: %d  Start: %d  Items: %d \n", i, starts[i], sizes[i]);

    readIdx[i] = read;
    //TODO should check on length !!! when adding first item
    //in case length is 0
  }

  int itemsInQueue = nLists;

  while(1)
  {
    int idxSmallest = 0;
    //Find the smallest item in the queue
    for(int j=1; j < itemsInQueue; j++)
    {
      //      fprintf(stderr, "Comparing; %d and %d  \t %d ",
      //          queue[idxSmallest].x, queue[j].x, cmp_uint4_host(queue[idxSmallest], queue[j]));

      if(cmp_uint4_host(queue[idxSmallest], queue[j]) >= 0)
        idxSmallest = j;
    }

    //Add items j to the list and refill queue
    result[res++] = queue[idxSmallest];

    if(readIdx[idxSmallest].z > 0)
    {
      readIdx[idxSmallest].z--; //decrease items left
      readIdx[idxSmallest].y++; //increase read location
      queue[idxSmallest] = data[readIdx[idxSmallest].y];
    }
    else
    {
      queue[idxSmallest] = queue[itemsInQueue-1];
      readIdx[idxSmallest] = readIdx[itemsInQueue-1];
      itemsInQueue -= 1; //decrease items in queue
    }

    if(itemsInQueue == 0) break;

  }//end while

  //  if(iLeft!=sizeLeft)
  //    memcpy(&result[res], &left[iLeft],   sizeof(uint4)*(sizeLeft-iLeft));
  //  else
  //    memcpy(&result[res], &right[iRight], sizeof(uint4)*(sizeRight-iRight));
}

typedef struct queueObject
{
  uint4 key;
  int4 val;
} queueObject;

struct cmp_ph_key_test{
  bool operator () (const queueObject &a, const queueObject &b){
    return ( cmp_uint4_host( b.key, a.key) < 1); //note reverse
  }
};
#include <queue>
void merge_sort3(uint4 *result, uint4 *data, int *sizes, int *starts, int nLists)
{
  std::priority_queue<queueObject, vector<queueObject>, cmp_ph_key_test> queue;

  uint iLeft = 0;
  uint iRight = 0;
  uint res = 0;

  fprintf(stderr, "Merge3 sorting, total lists: %d \n", nLists);

  //  uint4 *queue  = new uint4[nLists];
  //  int4 *readIdx = new int4[nLists];

  for(int i=0; i < nLists; i++)
  {
    queueObject obj;
    obj.key = data[starts[i]];

    int4 read;
    read.x = i; //Source file
    read.y = starts[i]; //read Index in Data
    read.z = sizes[i]-1; //items left to process

    obj.val = read;

    queue.push(obj);
    //TODO should check on length !!! when adding first item
    //in case length is 0
  }

  int itemsInQueue = nLists;

  while(1)
  {
    int idxSmallest = 0;
    //Find the smallest item in the queue

    queueObject obj = queue.top();

    //    fprintf(stderr, "Item 0: %d ", obj.key.x);
    //    queue.pop();
    //    obj = queue.top();
    //    fprintf(stderr, "Item 1: %d ", obj.key.x);
    //
    //    exit(0);


    //Add items j to the list and refill queue
    result[res++] = obj.key;

    queue.pop();

    if(obj.val.z > 0)
    {
      obj.val.z--; //decrease items left
      obj.val.y++; //increase read location
      obj.key = data[obj.val.y];
      queue.push(obj);
      //      fprintf(stderr, "Adding items from list: %d  from loc: %d  left: %d \n",
      //          readIdx[idxSmallest].x, readIdx[idxSmallest].y, readIdx[idxSmallest].z);
    }

    if(queue.empty()) break;

  }//end while

}
#endif


int octree::stackBasedTopLEvelsCheck(tree_structure &tree,
    real4 *treeBuffer,
    int proc,
    int nTopLevelTrees,
    uint2 *curLevelStack,
    uint2 *nextLevelStack,
    int &DistanceCheck)
{
  int DistanceCheckPP = 0;

  int ib       = (nProcs-1)-proc;
  int ibox = (ib+procId)%nProcs; //index to send...)

  //    LOGF(stderr,"Process %d Checking %d \t [%d %d %d ] \n", procId, ibox,i,ib,nProcs);

  int doFullGrp = fullGrpAndLETRequest[ibox];

  //Group info for this process
  int idx          =   globalGrpTreeOffsets[ibox];
  real4 *grpCenter =  &globalGrpTreeCntSize[idx];
  idx             += this->globalGrpTreeCount[ibox] / 2; //Divide by two to get halfway
  real4 *grpSize   =  &globalGrpTreeCntSize[idx];

  //Retrieve required for the tree-walk from the top node
  union{int i; float f;} itof; //float as int

  itof.f       = grpCenter[0].x;
  int startGrp = itof.i;
  itof.f       = grpCenter[0].y;
  int endGrp   = itof.i;

  if(!doFullGrp)
  {
    //This is a topNode only
    startGrp = 0;
    endGrp   = this->globalGrpTreeCount[ibox] / 2;
  }

  //Tree info
  const int nParticles = host_float_as_int(treeBuffer[0].x);
  const int nNodes     = host_float_as_int(treeBuffer[0].y);

  //    LOGF(stderr,"Working with %d and %d || %d %d\n", nParticles, nNodes, 1+nParticles+nNodes,nTopLevelTrees );

  real4* treeBoxSizes   = &treeBuffer[1+nParticles];
  real4* treeBoxCenters = &treeBuffer[1+nParticles+nNodes];
  real4* treeBoxMoments = &treeBuffer[1+nParticles+2*nNodes];

  int maxLevel = 0;

  //Walk these groups along our tree
  //Add the topNode to the stack
  int nexLevelCount = 0;
  int curLevelCount = 1;
  curLevelStack[0]  = make_uint2(0, startGrp); //Add top node

  while(curLevelCount > 0)
  {
    //        LOGF(stderr,"Processing level: %d  with %d %d  and %d Source: %d\n", maxLevel, curLevelCount, startGrp,endGrp, ibox);
    for(int idx = 0; idx < curLevelCount; idx++)
    {
      int nodeID = curLevelStack[idx].x;
      int grpID  = curLevelStack[idx].y;
      //Check this node against the groups
      real4 nodeCOM  = treeBoxMoments[nodeID*3];
      real4 nodeSize = treeBoxSizes  [nodeID];
      real4 nodeCntr = treeBoxCenters[nodeID];

      nodeCOM.w = nodeCntr.w;
      for(int grp=grpID; grp < endGrp; grp++)
      {
        real4 grpcntr = grpCenter[grp];
        real4 grpsize = grpSize[grp];

        bool split = false;
        {
          DistanceCheck++;
          DistanceCheckPP++;
          //Compute the distance between the group and the cell
          float3 dr = make_float3(fabs((float)grpcntr.x - nodeCOM.x) - (float)grpsize.x,
              fabs((float)grpcntr.y - nodeCOM.y) - (float)grpsize.y,
              fabs((float)grpcntr.z - nodeCOM.z) - (float)grpsize.z);

          dr.x += fabs(dr.x); dr.x *= 0.5f;
          dr.y += fabs(dr.y); dr.y *= 0.5f;
          dr.z += fabs(dr.z); dr.z *= 0.5f;

          //Distance squared, no need to do sqrt since opening criteria has been squared
          float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

          if (ds2     <= fabs(nodeCOM.w))           split = true;
          if (fabs(ds2 - fabs(nodeCOM.w)) < 10e-04) split = true; //Limited precision can result in round of errors. Use this as extra safe guard

          //          LOGF(stderr,"Node: %d grp: %d  split: %d || %f %f\n", curLevelStack[idx], grp, split, ds2, nodeCOM.w);
        }


        if(split)
        {
          if(host_float_as_int(nodeSize.w) == 0xFFFFFFFF)
          {
            //We want to split, but then we go to deep. So we need a full tree-walk
            return -1;
          }

          int child, nchild;
          int childinfo = host_float_as_int(nodeSize.w);
          bool leaf = nodeCntr.w <= 0;

          if(!leaf)
          {
            //Node
            child    =    childinfo & 0x0FFFFFFF;           //Index to the first child of the node
            nchild   = (((childinfo & 0xF0000000) >> 28)) ; //The number of children this node has
            //Add the child-nodes to the next stack
            for(int y=child; y < child+nchild; y++)
              nextLevelStack[nexLevelCount++] = make_uint2(y,grpID);
          }//!leaf
          //Skip rest of the groups
          grp = endGrp;
        }//if split
        if(maxLevel < 0) break;
      }//for groups

      if(maxLevel < 0) break;
    }//for curLevelCount
    if(maxLevel < 0) break;

    if(nexLevelCount == 0)
      return maxLevel;

    //Check if we continue
    if(nexLevelCount > 0)
    {
      curLevelCount   = nexLevelCount; nexLevelCount = 0;
      uint2 *temp     = nextLevelStack;
      nextLevelStack  = curLevelStack;
      curLevelStack   = temp;
      maxLevel++;
      //          LOGF(stderr, "Max level found: %d \n", maxLevel)
    }
  }//while curLevelCount > 0

  //  LOGF(stderr, "Finally Max level found: %d Process : %d \n", maxLevel, ibox)
  return maxLevel;
}

#endif












#include "octree.h"

#define USE_MPI

//#define USE_AVX


#ifdef USE_MPI
#include <xmmintrin.h>
#include "radix.h"
#include <parallel/algorithm>
#include <map>
#include "dd2d.h"


extern "C" uint2 thrust_partitionDomains( my_dev::dev_mem<uint2> &validList,
                                          my_dev::dev_mem<uint2> &validList2,
                                          my_dev::dev_mem<uint> &idList,
                                          my_dev::dev_mem<uint2> &outputKeys,
                                          my_dev::dev_mem<uint> &outputValues,
                                          const int N,
                                          my_dev::dev_mem<uint> &generalBuffer, const int currentOffset);


#define USE_GROUP_TREE  //If this is defined we convert boundaries into a group

#define NMAXPROC 32768


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

/*
 *
 * OpenMP magic / chaos here, to prevent realloc of
 * buffers which seems to be notoriously slow on
 * HA-Pacs
 */
struct GETLETBUFFERS
{
  std::vector<int2> LETBuffer_node;
  std::vector<int > LETBuffer_ptcl;

  std::vector<uint4>  currLevelVecUI4;
  std::vector<uint4>  nextLevelVecUI4;

  std::vector<int>  currLevelVecI;
  std::vector<int>  nextLevelVecI;


  std::vector<int>    currGroupLevelVec;
  std::vector<int>    nextGroupLevelVec;

  //These are for getLET(Quick) only
  std::vector<v4sf> groupCentreSIMD;
  std::vector<v4sf> groupSizeSIMD;

  std::vector<v4sf> groupCentreSIMDSwap;
  std::vector<v4sf> groupSizeSIMDSwap;

  std::vector<int>  groupSIMDkeys;


#ifdef USE_AVX /* AVX */
    #ifndef __AVX__
        #error "AVX is not defined"
    #endif
    std::vector< std::pair<v4sf,v4sf> > groupSplitFlag;
#else
  std::vector<v4sf> groupSplitFlag;
#endif


  char padding[512 -
               ( sizeof(LETBuffer_node) +
                 sizeof(LETBuffer_ptcl) +
                 sizeof(currLevelVecUI4) +
                 sizeof(nextLevelVecUI4) +
                 sizeof(currLevelVecI) +
                 sizeof(nextLevelVecI) +
                 sizeof(currGroupLevelVec) +
                 sizeof(nextGroupLevelVec) +
                 sizeof(groupSplitFlag) +
                 sizeof(groupCentreSIMD)
               )];
};
/* End of Magic */



#include "hostTreeBuild.h"
#include "mpi.h"
#include <omp.h>


MPIComm *myComm;

//#include "MPIComm.h"
//template <> MPI_Datatype MPIComm_datatype<float>() {return MPI_FLOAT; }

//static MPI_Datatype MPI_V4SF = 0;

//  template <>
//MPI_Datatype MPIComm_datatype<v4sf>()
//{
//  if (MPI_V4SF) return MPI_V4SF;
//  else {
//    int ss = sizeof(v4sf) / sizeof(float);
//    assert(0 == sizeof(v4sf) % sizeof(float));
//    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_V4SF);
//    MPI_Type_commit(&MPI_V4SF);
//    return MPI_V4SF;
//  }
//}
//void MPIComm_free_type()
//{
//  if (MPI_V4SF) MPI_Type_free(&MPI_V4SF);
//}

#endif //USE_MPI


double get_time2() {
  struct timeval Tvalue;
  struct timezone dummy;
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
}



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


//SSE stuff for local tree-walk
#ifdef USE_MPI



#if 1

static inline float2 extract_opening_criteria(float val)
{
    __half2 temp = *((__half2*)&val);
    float2 res = {_cvtsh_ss(temp.x), _cvtsh_ss(temp.y)};
    return res;
}


#else

// Future Power8 version

#endif





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



//We want to know for each of the 4 groups if it needs an interaction
//with this node. Therefore we have to test every possible combination
//until either all combinations are exhausted or each group requires an interaction
template<bool SPHCheck>
inline _v4sf split_node_grav_sph_impbh_box4a( // takes 4 tree nodes and returns 4-bit integer
        _v4sf  ret,
  const _v4sf  checkValue,
  const _v4sf  nodeCNTRorCOM,
  const _v4sf  nodeSIZE,
  const _v4sf  boxCenter[4],
  const _v4sf  boxSize  [4],
  const float3 periodicBoundaries,
  int2 xP, int2 yP, int2 zP)
{
    const _v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};

    _v4sf ncx2  = __builtin_ia32_shufps(nodeCNTRorCOM, nodeCNTRorCOM, 0x00);
    _v4sf ncy2  = __builtin_ia32_shufps(nodeCNTRorCOM, nodeCNTRorCOM, 0x55);
    _v4sf ncz2  = __builtin_ia32_shufps(nodeCNTRorCOM, nodeCNTRorCOM, 0xaa);

    _v4sf bcx   =  (boxCenter[0]);
    _v4sf bcy   =  (boxCenter[1]);
    _v4sf bcz   =  (boxCenter[2]);
    _v4sf bcw   =  (boxCenter[3]);      //Not used at the moment
    _v4sf_transpose(bcx, bcy, bcz, bcw);

    _v4sf bsx   =  (boxSize[0]);
    _v4sf bsy   =  (boxSize[1]);
    _v4sf bsz   =  (boxSize[2]);
    _v4sf bsw   =  (boxSize[3]);
    _v4sf_transpose(bsx, bsy, bsz, bsw);

    _v4sf size;
    if(SPHCheck)
    {
        _v4sf nsx    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x00);
        _v4sf nsy    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x55);
        _v4sf nsz    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0xaa);

        bsx += nsx;
        bsy += nsy;
        bsz += nsz;
    }

    for(int ix=xP.x; ix <= xP.y; ix++)     //Periodic around X
    {
      for(int iy=yP.x; iy <= yP.y; iy++)   //Periodic around Y
      {
          for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
          {
              _v4sf ncx = __builtin_ia32_addps(ncx2,  _mm_set_ps1(periodicBoundaries.x*ix));
              _v4sf ncy = __builtin_ia32_addps(ncy2,  _mm_set_ps1(periodicBoundaries.y*iy));
              _v4sf ncz = __builtin_ia32_addps(ncz2,  _mm_set_ps1(periodicBoundaries.z*iz));


              _v4sf dx = __abs(bcx - ncx) - (bsx);
              _v4sf dy = __abs(bcy - ncy) - (bsy);
              _v4sf dz = __abs(bcz - ncz) - (bsz);


              dx = __builtin_ia32_maxps(dx, zero);
              dy = __builtin_ia32_maxps(dy, zero);
              dz = __builtin_ia32_maxps(dz, zero);

              const _v4sf ds2 = dx*dx + dy*dy + dz*dz;

              //Test all 4 distances against 4 smoothing lengths and or with previous result
              ret =  _mm_or_ps (ret,__builtin_ia32_cmpleps(ds2, checkValue));

              //Test if all 4 bits are set (then all groups active ) this is 4 lower bits set to 1: 1+2+4+8=15
              //Early out if this is the case since we won't learn anything new :)
              if(__builtin_ia32_movmskps(ret) == 15) return ret;
          }//iz
      }//iy
    }//ix

    return ret;
}


template<bool TRANSPOSE, bool SPHCheck>
inline int split_node_grav_sph_impbh_box4simd1_periodic( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  ncx2,
    const _v4sf  ncy2,
    const _v4sf  ncz2,
    const _v4sf  nsx,
    const _v4sf  nsy,
    const _v4sf  nsz,
    const _v4sf  checkValue,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4],
    const float3 periodicBoundaries,
    int2 xP, int2 yP, int2 zP)
{
  const _v4sf zero = {0.0, 0.0, 0.0, 0.0};

  _v4sf bcx =  (boxCenter[0]);
  _v4sf bcy =  (boxCenter[1]);
  _v4sf bcz =  (boxCenter[2]);
  _v4sf bcw =  (boxCenter[3]);

  _v4sf bsx =  (boxSize[0]);
  _v4sf bsy =  (boxSize[1]);
  _v4sf bsz =  (boxSize[2]);
  _v4sf bsw =  (boxSize[3]);

  int ret = 0;

  if (TRANSPOSE)
  {
    _v4sf_transpose(bcx, bcy, bcz, bcw);
    _v4sf_transpose(bsx, bsy, bsz, bsw);
  }

  if(SPHCheck)
  {
      bsx +=nsx;
      bsy +=nsy;
      bsz +=nsz;
  }


  for(int ix=xP.x; ix <= xP.y; ix++)     //Periodic around X
  {
    for(int iy=yP.x; iy <= yP.y; iy++)   //Periodic around Y
    {
      for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
      {
          _v4sf ncx = __builtin_ia32_addps(ncx2,  _mm_set_ps1(periodicBoundaries.x*ix));
          _v4sf ncy = __builtin_ia32_addps(ncy2,  _mm_set_ps1(periodicBoundaries.y*iy));
          _v4sf ncz = __builtin_ia32_addps(ncz2,  _mm_set_ps1(periodicBoundaries.z*iz));

          _v4sf dx = __abs(bcx - ncx) - (bsx);
          _v4sf dy = __abs(bcy - ncy) - (bsy);
          _v4sf dz = __abs(bcz - ncz) - (bsz);

          dx = __builtin_ia32_maxps(dx, zero);
          dy = __builtin_ia32_maxps(dy, zero);
          dz = __builtin_ia32_maxps(dz, zero);

          const _v4sf ds2 = dx*dx + dy*dy + dz*dz;

          ret = __builtin_ia32_movmskps(__builtin_ia32_cmpleps(ds2, checkValue));
          if(ret) return ret;
      }//iz
    }//iy
  }//ix

  return ret;
}




#ifdef USE_AVX
template<bool SPHCheck>
inline _v8sf split_node_grav_sph_impbh_box8a( // takes 4 tree nodes and returns 4-bit integer
                _v8sf  ret,
          const _v8sf  checkValue,
          const _v4sf  nodeCNTRorCOM,
          const _v4sf  nodeSIZE,
          const _v4sf  boxCenter[8],
          const _v4sf  boxSize  [8],
          const float3 periodicBoundaries,
          int2 xP, int2 yP, int2 zP)
{
  _v8sf com = pack_2xmm(nodeCNTRorCOM, nodeCNTRorCOM);
  _v8sf ncx2 = __builtin_ia32_shufps256(com, com, 0x00);
  _v8sf ncy2 = __builtin_ia32_shufps256(com, com, 0x55);
  _v8sf ncz2 = __builtin_ia32_shufps256(com, com, 0xaa);

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

  if(SPHCheck)
  {
      _v8sf size   = pack_2xmm(nodeSIZE, nodeSIZE);
      _v8sf nsx    = __builtin_ia32_shufps256(size, size, 0x00);
      _v8sf nsy    = __builtin_ia32_shufps256(size, size, 0x55);
      _v8sf nsz    = __builtin_ia32_shufps256(size, size, 0xaa);

      bsx += nsx;
      bsy += nsy;
      bsz += nsz;
  }

  for(int ix=xP.x; ix <= xP.y; ix++)       //Periodic around X
  {
    for(int iy=yP.x; iy <= yP.y; iy++)     //Periodic around Y
    {
        for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
        {
            _v8sf ncx = __builtin_ia32_addps256(ncx2,  _mm256_set1_ps(periodicBoundaries.x*ix));
            _v8sf ncy = __builtin_ia32_addps256(ncy2,  _mm256_set1_ps(periodicBoundaries.y*iy));
            _v8sf ncz = __builtin_ia32_addps256(ncz2,  _mm256_set1_ps(periodicBoundaries.z*iz));

            _v8sf dx = __abs8(bcx - ncx) - bsx;
            _v8sf dy = __abs8(bcy - ncy) - bsy;
            _v8sf dz = __abs8(bcz - ncz) - bsz;

            const _v8sf zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,0.0f,0.0f,0.0f};
            dx = __builtin_ia32_maxps256(dx, zero);
            dy = __builtin_ia32_maxps256(dy, zero);
            dz = __builtin_ia32_maxps256(dz, zero);

            const _v8sf ds2 = dx*dx + dy*dy + dz*dz;
            ret = _mm256_or_ps(ret,__builtin_ia32_cmpps256(ds2, checkValue, 18)); //18 indicates is Less or Equal OP

            //Test if all 8 mask bits are set (then all groups active ) this is 8 lower bits set to 1: 0xFF
            //Early out if this is the case since we won't learn anything new :)
            if(_mm256_movemask_ps(ret) ==  0xFF)    return ret;
        }//iz
    }//iy
  }//ix
  return ret;
}

template<bool TRANSPOSE, bool SPHCheck>
inline int split_node_grav_sph_impbh_box8simd1_periodic( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  ncx2,
    const _v4sf  ncy2,
    const _v4sf  ncz2,
    const _v4sf  nsx,
    const _v4sf  nsy,
    const _v4sf  nsz,
    const _v4sf  checkValue1,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4],
    const float3 periodicBoundaries,
    int2 xP, int2 yP, int2 zP)
{
  const _v8sf zero = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const _v8sf checkValue = pack_2xmm(checkValue1, checkValue1);
  int ret = 0;

  _v4sf bcx1 =  (boxCenter[0]);
  _v4sf bcy1 =  (boxCenter[1]);
  _v4sf bcz1 =  (boxCenter[2]);
  _v4sf bcw1 =  (boxCenter[3]);

  _v4sf bcx2 =  (boxCenter[4]);
  _v4sf bcy2 =  (boxCenter[5]);
  _v4sf bcz2 =  (boxCenter[6]);
  _v4sf bcw2 =  (boxCenter[7]);

  _v8sf bcx = pack_2xmm(bcx1, bcx2);
  _v8sf bcy = pack_2xmm(bcy1, bcy2);
  _v8sf bcz = pack_2xmm(bcz1, bcz2);
  _v8sf bcw = pack_2xmm(bcw1, bcw2);

  _v4sf bsx1 =  (boxSize[0]);
  _v4sf bsy1 =  (boxSize[1]);
  _v4sf bsz1 =  (boxSize[2]);
  _v4sf bsw1 =  (boxSize[3]);

  _v4sf bsx2 =  (boxSize[4]);
  _v4sf bsy2 =  (boxSize[5]);
  _v4sf bsz2 =  (boxSize[6]);
  _v4sf bsw2 =  (boxSize[7]);

  _v8sf bsx = pack_2xmm(bsx1, bsx2);
  _v8sf bsy = pack_2xmm(bsy1, bsy2);
  _v8sf bsz = pack_2xmm(bsz1, bsz2);
  _v8sf bsw = pack_2xmm(bsw1, bsw2);

  if (TRANSPOSE) assert(0); //Arrays should be set as v8sf to get this to work


  if(SPHCheck)
  {
      bsx += pack_2xmm(nsx, nsx);
      bsy += pack_2xmm(nsy, nsy);
      bsz += pack_2xmm(nsz, nsz);
  }


  for(int ix=xP.x; ix <= xP.y; ix++)     //Periodic around X
  {
    for(int iy=yP.x; iy <= yP.y; iy++)   //Periodic around Y
    {
      for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
      {
          _v4sf ncx3 = __builtin_ia32_addps(ncx2,  _mm_set_ps1(periodicBoundaries.x*ix));
          _v4sf ncy3 = __builtin_ia32_addps(ncy2,  _mm_set_ps1(periodicBoundaries.y*iy));
          _v4sf ncz3 = __builtin_ia32_addps(ncz2,  _mm_set_ps1(periodicBoundaries.z*iz));

          _v8sf ncx = pack_2xmm(ncx3, ncx3);
          _v8sf ncy = pack_2xmm(ncy3, ncy3);
          _v8sf ncz = pack_2xmm(ncz3, ncz3);

          _v8sf dx = __abs8(bcx - ncx) - (bsx);
          _v8sf dy = __abs8(bcy - ncy) - (bsy);
          _v8sf dz = __abs8(bcz - ncz) - (bsz);


          dx = __builtin_ia32_maxps256(dx, zero);
          dy = __builtin_ia32_maxps256(dy, zero);
          dz = __builtin_ia32_maxps256(dz, zero);

          const _v8sf ds2 = dx*dx + dy*dy + dz*dz;
          ret = _mm256_movemask_ps( __builtin_ia32_cmpps256(ds2, checkValue, 18)); //18 indicates is Less or Equal OP
          if(ret) return ret;
      }//iz
    }//iy
  }//ix

  return ret;
}
#endif



template<typename T, int STRIDE>
void shuffle2vecAllocated(
    std::vector<T>   &data1,
    std::vector<T>   &data2,
    std::vector<T>   &rdata1,
    std::vector<T>   &rdata2,
    std::vector<int> &keys)
{
  const int n = data1.size();

  assert(n%STRIDE == 0);
  keys.resize(n/STRIDE);
  for (int i = 0, idx=0; i < n; i += STRIDE, idx++)
    keys[idx] = i;
  std::random_shuffle(keys.begin(), keys.end());

  rdata1.resize(n); //Safety only
  rdata2.resize(n); //Safety only
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

#endif //USE_MPI

void octree::mpiSetup()
{
#ifdef USE_MPI
  int  namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_size(mpiCommWorld, &this->nProcs);
  MPI_Comm_rank(mpiCommWorld, &this->procId);

  myComm = new MPIComm(procId, nProcs,mpiCommWorld);

  MPI_Get_processor_name(processor_name,&namelen);
#else
  char processor_name[] = "Default";
#endif


  LOGF(   stderr, "Proc id: %d @ %s , total processes: %d (mpiInit) \n", procId, processor_name, nProcs);
  fprintf(stderr, "Proc id: %d @ %s , total processes: %d (mpiInit) \n", procId, processor_name, nProcs);


  currentRLow          = new double4[nProcs];
  currentRHigh         = new double4[nProcs];
  curSysState          = new sampleRadInfo[nProcs];
  globalGrpTreeCount   = new uint[nProcs];
  globalGrpTreeOffsets = new uint[nProcs];
  globalGrpTreeStatistics = new int4[nProcs];
}



//Utility functions
void octree::mpiSync(){
#ifdef USE_MPI
  MPI_Barrier(mpiCommWorld);
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
  MPI_Allreduce(&value,&tmp,1, MPI_DOUBLE, MPI_SUM,mpiCommWorld);
  value = tmp;
#endif
}

double octree::AllMin(double value)
{
#ifdef USE_MPI
  double tmp = -1;
  MPI_Allreduce(&value,&tmp,1, MPI_DOUBLE, MPI_MIN,mpiCommWorld);
  return tmp;
#endif
}


//Sum the number of particles on all processes
void octree::mpiSumParticleCount(int numberOfParticles)
{
  nTotalFreq_ull = numberOfParticles;
#ifdef USE_MPI
  unsigned long long tmp  = 0;
  unsigned long long tmp2 = numberOfParticles;
  MPI_Allreduce(&tmp2,&tmp,1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,mpiCommWorld);
  nTotalFreq_ull = tmp;
#endif

  if(procId == 0) LOG("Total number of particles: %llu\n", nTotalFreq_ull);
}



double octree::SumOnRootRank(double value)
{
 double temp = value;
#ifdef USE_MPI
  MPI_Reduce(&value,&temp,1, MPI_DOUBLE, MPI_SUM,0, mpiCommWorld);
#endif
  return temp;
}

int octree::SumOnRootRank(int value)
{
  int temp = value;
#ifdef USE_MPI
  MPI_Reduce(&value,&temp,1, MPI_INT, MPI_SUM,0, mpiCommWorld);
#endif
  return temp;
}
//end utility



//Main functions


//Functions related to domain decomposition


void octree::exchangeSamplesAndUpdateBoundarySFC(uint4 *sampleKeys2,    int  nSamples2,
    uint4 *globalSamples2, int  *nReceiveCnts2, int *nReceiveDpls2,
    int    totalCount2,   uint4 *parallelBoundaries, float lastExecTime,
    bool initialSetup)
{
#ifdef USE_MPI


#if 1
 //Start of 2D


  /* evghenii: 2d sampling comes here,
   * make sure that locakTree.bodies_key.d2h in src/build.cpp.
   * if you don't see my comment there, don't use this version. it will be
   * blow up :)
   */

  {
    const double t0 = get_time();

    const int nkeys_loc = localTree.n;
    assert(nkeys_loc > 0);
    const int nloc_mean = nTotalFreq_ull/nProcs;

    /* LB step */

    double f_lb = 1.0;
#if 1  /* LB: use load balancing */
    {
      static double prevDurStep = -1;
      static int prevSampFreq = -1;
      prevDurStep = (prevDurStep <= 0) ? lastExecTime : prevDurStep;

      double timeLocal = (lastExecTime + prevDurStep) / 2;
      double timeSum = 0.0;

      //JB We should not forget to set prevDurStep
      prevDurStep = timeLocal;

      MPI_Allreduce( &timeLocal, &timeSum, 1,MPI_DOUBLE, MPI_SUM, mpiCommWorld);

      double fmin = 0.0;
      double fmax = HUGE;

/* evghenii: updated LB and MEMB part, works on the following synthetic test
      double lastExecTime = (double)nkeys_loc/nloc_mean;
      const double imb_min = 0.5;
      const double imb_max = 2.0;
      const double imb = imb_min + (imb_max - imb_min)/(nProcs-1) * procId;

      lastExecTime *= imb;

      lastExecTime becomes the same on all procs after about 20 iterations: passed
      with MEMB enabled, single proc doesn' incrase # particles by more than mem_imballance: passed
*/
#if 1  /* MEMB: constrain LB to maintain ballanced memory use */
      {
        const double mem_imballance = 0.3;

        double fac = 1.0;

        fmin = fac/(1.0+mem_imballance);
        fmax = HUGE;
#if 0   /* use this to limit # of exported particles */
        fmax = fac*(1.0+mem_imballance);
#endif
      }

#endif  /* MEMB: end memory balance */

      f_lb  = timeLocal / timeSum * nProcs;
      f_lb *= (double)nloc_mean/(double)nkeys_loc;
      f_lb  = std::max(std::min(fmax, f_lb), fmin);
    }
#endif

    /*** particle sampling ***/

    const int npx = myComm->n_proc_i;  /* number of procs doing domain decomposition */

    int nsamples_glb;
    if(initialSetup)
    {
      nsamples_glb = nTotalFreq_ull / 1000;
      nsamples_glb = std::max(nsamples_glb, nloc_mean / 3);
      if(procId == 0) fprintf(stderr,"TEST Nsamples_gbl: %d \n", nsamples_glb);

      //nsamples_glb = nloc_mean / 3; //Higher rate in first steps to get proper distribution
    }
    else
      nsamples_glb = nloc_mean / 30;

    std::vector<DD2D::Key> key_sample1d, key_sample2d;
    key_sample1d.reserve(nsamples_glb);
    key_sample2d.reserve(nsamples_glb);

    const double nsamples1d_glb = (f_lb * nsamples_glb);
    const double nsamples2d_glb = (f_lb * nsamples_glb) * npx;

    const double nTot = nTotalFreq_ull;
    const double stride1d = std::max(nTot/nsamples1d_glb, 1.0);
    const double stride2d = std::max(nTot/nsamples2d_glb, 1.0);
    for (double i = 0; i < (double)nkeys_loc; i += stride1d)
    {
      const uint4 key = localTree.bodies_key[(int)i];
      key_sample1d.push_back(DD2D::Key(
            (static_cast<unsigned long long>(key.y) ) |
            (static_cast<unsigned long long>(key.x) << 32)
            ));
    }
    for (double i = 0; i < (double)nkeys_loc; i += stride2d)
    {
      const uint4 key = localTree.bodies_key[(int)i];
      key_sample2d.push_back(DD2D::Key(
            (static_cast<unsigned long long>(key.y) ) |
            (static_cast<unsigned long long>(key.x) << 32)
            ));
    }

    //JB, TODO check if this is the correct location to put this
    //and or use parallel sort
    std::sort(key_sample1d.begin(), key_sample1d.end(), DD2D::Key());
    std::sort(key_sample2d.begin(), key_sample2d.end(), DD2D::Key());

    const DD2D dd(procId, npx, nProcs, key_sample1d, key_sample2d, mpiCommWorld);

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


#if 0 /* evghenii: disable 1D to enable 2D domain decomposition below */
  {
    //Send actual data
    MPI_Gatherv(&sampleKeys[0],    nSamples*sizeof(uint4), MPI_BYTE,
        &globalSamples[0], nReceiveCnts, nReceiveDpls, MPI_BYTE,
        0, mpiCommWorld);


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
    MPI_Bcast(&parallelBoundaries[0], sizeof(uint4)*(nProcs+1), MPI_BYTE, 0, mpiCommWorld);
  }

  //End of 1D
#endif
}

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
      sizeof(sampleRadInfo), MPI_BYTE, mpiCommWorld);
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




//Function that uses the GPU to get a set of particles that have to be
//send to other processes
void octree::gpuRedistributeParticles_SFC(uint4 *boundaries)
{
#ifdef USE_MPI
  double tStart = get_time();

  uint4 lowerBoundary = boundaries[this->procId];
  uint4 upperBoundary = boundaries[this->procId+1];

  static std::vector<uint>  nParticlesPerDomain(nProcs);
  static std::vector<uint2> domainId           (nProcs);

  my_dev::dev_mem<uint2>  validList2;
  my_dev::dev_mem<uint2>  validList3;
  my_dev::dev_mem<uint4>  boundariesGPU;
  my_dev::dev_mem<uint>   idList;
  my_dev::dev_mem<uint>   atomicBuff; //Used for counting during particle movement

  int tempOffset1 = validList2.   cmalloc_copy(localTree.generalBuffer1, localTree.n, 0);
      tempOffset1 = validList3.   cmalloc_copy(localTree.generalBuffer1, localTree.n, tempOffset1);
  int tempOffset  = idList.       cmalloc_copy(localTree.generalBuffer1, localTree.n, tempOffset1);
                    boundariesGPU.cmalloc_copy(localTree.generalBuffer1, nProcs+2,    tempOffset);
  tempOffset1     = atomicBuff.   cmalloc_copy(localTree.generalBuffer1, 1,           tempOffset1);


  for(int idx=0; idx <= nProcs; idx++)
  {
    boundariesGPU[idx] = boundaries[idx];
  }
  boundariesGPU.h2d();


  domainCheckSFCAndAssign.set_args(0, &localTree.n, &nProcs, &lowerBoundary, &upperBoundary,
                                      boundariesGPU.p(), localTree.bodies_key.p(), validList2.p(),
                                      idList.p(), &procId);
  domainCheckSFCAndAssign.setWork(localTree.n, 128);
  domainCheckSFCAndAssign.execute2(execStream->s());
  execStream->sync();

  //After this we don't need boundariesGPU anymore so can overwrite that memory space
  my_dev::dev_mem<uint>   outputValues;
  my_dev::dev_mem<uint2>  outputKeys;
  tempOffset = outputValues.cmalloc_copy(localTree.generalBuffer1, nProcs, tempOffset );
  tempOffset = outputKeys  .cmalloc_copy(localTree.generalBuffer1, nProcs, tempOffset );

  double tCheck = get_time();
  uint2 res = thrust_partitionDomains(validList2, validList3,
                                      idList,
                                      outputKeys, outputValues,
                                      localTree.n,
                                      localTree.generalBuffer1, tempOffset);
  double tSort = get_time();
  LOGF(stderr,"Sorting preparing took: %lg nExport: %d  nDomains: %d Since start: %lg\n", get_time()-tCheck, res.x, res.y, get_time()-tStart);

  const int nExportParticles = res.x;
  const int nToSendToDomains = res.y;

  nParticlesPerDomain.clear();   nParticlesPerDomain.resize(nToSendToDomains);
  domainId.           clear();   domainId.resize           (nToSendToDomains);

  outputKeys  .d2h(nToSendToDomains, &domainId[0]);
  outputValues.d2h(nToSendToDomains, &nParticlesPerDomain[0]);

  bodyStruct *extraBodyBuffer = NULL;
  bool doInOneGo              = true;
  double tExtract             = 0;
  double ta2aSize             = 0;

  int *nparticles  = &exchangePartBuffer[0*(nProcs+1)]; //nParticles to send per domain
  int *nreceive    = &exchangePartBuffer[2*(nProcs+1)]; //nParticles to receive per domain
  int *nsendDispls = &exchangePartBuffer[1*(nProcs+1)]; //Prefix sum for storage

  //TODO
  // This can be changed by a copy per domain. That way we do not have to wait till everything is
  // copied and can start sending whenever one domain is done. Note we can also use GPUdirect for
  // sending when we use it that way

  //Overlap the particle extraction with the all2all size communication

  const int curOMPMax = omp_get_max_threads();
  omp_set_nested(1);
  omp_set_num_threads(2);


#pragma omp parallel
  {
    const int tid =  omp_get_thread_num();
    //Thread 0, has GPU context and is responsible for particle extraction/GPU steering
    //Thread 1, will do the MPI all2all stuff
    if(tid == 0)
    {
        //Check if the memory size, of the generalBuffer is large enough to store the exported particles
        //if not allocate a buffer on the host that will store the data instead
        int validCount = nExportParticles;
        int tempSize   = localTree.generalBuffer1.get_size() - tempOffset1;
        int stepSize   = (tempSize / (sizeof(bodyStruct) / sizeof(int)))-512; //Available space in # of bodyStructs

        if(stepSize > nExportParticles)
        {
          doInOneGo = true; //We can do it in one go
        }
        else
        {
          doInOneGo       = false; //We need an extra CPU buffer
          extraBodyBuffer = new bodyStruct[validCount];
          assert(extraBodyBuffer != NULL);
        }


        my_dev::dev_mem<bodyStruct>  bodyBuffer;
        int memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1, stepSize, tempOffset1);

        double tx  = get_time();
        int extractOffset = 0;
        for(unsigned int i=0; i < validCount; i+= stepSize)
        {
          int items = min(stepSize, (int)(validCount-i));

          if(items > 0)
          {
            extractOutOfDomainParticlesAdvancedSFC2.set_args(
                    0, &extractOffset, &items, validList2.p(),
                    localTree.bodies_Ppos.p(), localTree.bodies_Pvel.p(), localTree.bodies_pos.p(),
                    localTree.bodies_vel.p(),  localTree.bodies_acc0.p(), localTree.bodies_acc1.p(),
                    localTree.bodies_time.p(), localTree.bodies_ids.p(),  localTree.bodies_key.p(),
                    localTree.bodies_h.p(),
                    localTree.bodies_dens.p(),
                    localTree.bodies_grad.p(),
                    localTree.bodies_hydro.p(),
                    bodyBuffer.p());
            extractOutOfDomainParticlesAdvancedSFC2.setWork(items, 128);
            extractOutOfDomainParticlesAdvancedSFC2.execute2(execStream->s());

            if(!doInOneGo)
            {
      #if 0
              bodyBuffer.d2h(items, &extraBodyBuffer[extractOffset]); //Copy to our custom buffer, non-pinned
      #else
              bodyBuffer.d2h(items);
              omp_set_num_threads(4); //Experiment with this number to see what is fastest
      #pragma omp parallel for
              for(int cpIdx=0; cpIdx < items; cpIdx++)
                extraBodyBuffer[extractOffset+cpIdx] = bodyBuffer[cpIdx];
      #endif
              extractOffset += items;
            }
            else
            {
      //        double tx  = get_time();
              bodyBuffer.d2h(items);
      //        double ty = get_time();
      //        LOGF(stderr,"ToDev B: Took: %lg Size: %ld  MB/s: %lg \n", ty-tx, (items*sizeof(bodyStruct)) / (1024*1024), (1/(ty-tx))*(items*sizeof(bodyStruct)) / (1024*1024));

            }
          }//if items > 0
        }//end for


        tExtract = get_time();

        LOGF(stderr,"Exported particles from device. In one go: %d  Took: %lg Size: %ld  MB/s: %lg \n",
            doInOneGo, tExtract-tx, (validCount*sizeof(bodyStruct)) / (1024*1024), (1/(tExtract-tx))*(validCount*sizeof(bodyStruct)) / (1024*1024));


        if(doInOneGo)
        {
          extraBodyBuffer = &bodyBuffer[0]; //Assign correct pointer
        }

        //Now we have to move particles from the back of the array to the invalid spots
        //this can be done in parallel with exchange operation to hide some time
        atomicBuff.zeroMem();

        double t3 = get_time();
        //Internal particle movement
        internalMoveSFC2.set_args(0, &validCount, &localTree.n, &lowerBoundary, &upperBoundary,
                validList3.p(), atomicBuff.p(), localTree.bodies_Ppos.p(),
                localTree.bodies_Pvel.p(), localTree.bodies_pos.p(),  localTree.bodies_vel.p(),
                localTree.bodies_acc0.p(), localTree.bodies_acc1.p(), localTree.bodies_time.p(),
                localTree.bodies_ids.p(),  localTree.bodies_key.p(),  localTree.bodies_h.p(),
                localTree.bodies_dens.p(), localTree.bodies_grad.p(), localTree.bodies_hydro.p());
        internalMoveSFC2.setWork(validCount, 128);
        internalMoveSFC2.execute2(execStream->s());
        //execStream->sync(); LOGF(stderr,"Internal move: %lg  Since start: %lg \n", get_time()-t3,get_time()-tStart);

    } //if tid == 0
    else if(tid == 1)
    {
      //The MPI thread, performs a2a during memory copies
      memset(nparticles,  0, sizeof(int)*(nProcs+1));
      memset(nreceive,    0, sizeof(int)*(nProcs));
      memset(nsendDispls, 0, sizeof(int)*(nProcs));

      int sendOffset = 0;
      for(int i=0; i < domainId.size(); i++)
      {
        const int domain = domainId[i].x & 0x0FFFFFF;
        assert(domain != procId); //Should not send to ourselves

        nparticles [domain] = nParticlesPerDomain[i];
        nsendDispls[domain] = sendOffset;
        sendOffset         += nParticlesPerDomain[i];
      }

      double tStarta2a = get_time();
      MPI_Alltoall(nparticles, 1, MPI_INT, nreceive, 1, MPI_INT, mpiCommWorld);
      ta2aSize = get_time()-tStarta2a;
    }//if tid == 1
  } //omp section

  omp_set_num_threads(curOMPMax); //Restore the number of OMP threads

  //LOGF(stderr,"Particle extraction took: %lg \n", get_time()-tStart);

  int currentN = localTree.n;

  this->gpu_exchange_particles_with_overflow_check_SFC2(localTree, &extraBodyBuffer[0],
                                                        nparticles, nsendDispls, nreceive,
                                                        nExportParticles);
  double tEnd = get_time();

  char buff5[1024];
  sprintf(buff5,"EXCHANGE-%d: tCheckDomain: %lg ta2aSize: %lg tSort: %lg tExtract: %lg tDomainEx: %lg nExport: %d nImport: %d \n",
      procId, tCheck-tStart, ta2aSize, tSort-tCheck, tExtract-tSort, tEnd-tExtract,nExportParticles, localTree.n - (currentN-nExportParticles));
  devContext->writeLogEvent(buff5);

  if(!doInOneGo) delete[] extraBodyBuffer;


#if 0
  First step, partition?
  IDs:     [0,1,2,4,5,6,7,3, 8  ,9  ]
  Domains: [0,1,3,1,1,0,3,0xF,0xF,0xF]

  Second step, sort by exported domain
   IDs:     [0,6,1,4,5,3,7,3, 8  ,9  ]
   Domains: [0,0,1,1,1,3,3,0xF,0xF,0xF]

 Third step, reduce the domains
   Domains/Key  [0,0,1,1,1,3,3]
   Values       [1,1,1,1,1,1,1]
   reducebykey  [0,1,3] domain IDs
                [2,3,2] # particles per domain
#endif


#endif
} //End gpuRedistributeParticles

//Exchange particles with other processes
int octree::gpu_exchange_particles_with_overflow_check_SFC2(tree_structure &tree,
                                                            bodyStruct *particlesToSend,
                                                            int *nparticles, int *nsendDispls,
                                                            int *nreceive, int nToSend)
{
#ifdef USE_MPI

  double tStart = get_time();

  unsigned int recvCount  = nreceive[0];
  for (int i = 1; i < nProcs; i++)
  {
    recvCount     += nreceive[i];
  }

  static std::vector<bodyStruct> recv_buffer3;
  recv_buffer3.resize(recvCount);

  int recvOffset = 0;

  static MPI_Status stat[NMAXPROC];
  static MPI_Request req[NMAXPROC*2];
  assert(nProcs < NMAXPROC);

  //TODO this loop could overflow if scount > INT_MAX (same for rcount)
  int nreq = 0;
  for (int dist = 1; dist < nProcs; dist++)
  {
    const int src    = (nProcs + procId - dist) % nProcs;
    const int dst    = (nProcs + procId + dist) % nProcs;
    const int scount = nparticles[dst] * (sizeof(bodyStruct) / sizeof(double));
    const int rcount = nreceive  [src] * (sizeof(bodyStruct) / sizeof(double));

    assert(scount >= 0);
    assert(rcount >= 0);

    if (scount > 0)
    {
      MPI_Isend(&particlesToSend[nsendDispls[dst]], scount, MPI_DOUBLE, dst, 1, mpiCommWorld, &req[nreq++]);
    }
    if(rcount > 0)
    {
      MPI_Irecv(&recv_buffer3[recvOffset], rcount, MPI_DOUBLE, src, 1, mpiCommWorld, &req[nreq++]);
      recvOffset += nreceive[src];
    }
  }

  double t94 = get_time();
  MPI_Waitall(nreq, req, stat);
  double tSendEnd = get_time();

  //If we arrive here all particles have been exchanged, move them to the GPU
  LOGF(stderr,"Required inter-process communication time: %lg ,proc: %d\n", get_time()-tStart, procId);

  //Compute the new number of particles:
  int newN = tree.n + recvCount - nToSend;

  LOGF(stderr, "Exchange, received %d \tSend: %d newN: %d\n", recvCount, nToSend, newN);

  //make certain that the particle movement on the device is complete before we resize
  execStream->sync();

  double tSyncGPU = get_time();

  //Allocate MULTI_GPU_MEM_INCREASE% extra if we have to allocate, to reduce the total number of memory allocations
  int memSize = newN;
  if(tree.bodies_acc0.get_size() < newN)
    memSize = newN * MULTI_GPU_MEM_INCREASE;

  //LOGF(stderr,"Going to allocate memory for %d particles \n", newN);

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
  tree.bodies_h.   cresize(memSize + 1, false);
  tree.bodies_dens.cresize(memSize + 1, false);
  tree.bodies_grad.cresize(memSize + 1, false);
  tree.bodies_hydro.cresize(memSize + 1, false);

  memSize = tree.bodies_acc0.get_size();
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

  my_dev::dev_mem<bodyStruct>  bodyBuffer;

  int memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1, stepSize, 0);

  double tAllocComplete = get_time();

  int insertOffset = 0;
  for(unsigned int i=0; i < recvCount; i+= stepSize)
  {
    int items = min(stepSize, (int)(recvCount-i));

    if(items > 0)
    {
      //Copy the data from the MPI receive buffers into the GPU-send buffer
#pragma omp parallel for
        for(int cpIdx=0; cpIdx < items; cpIdx++)
          bodyBuffer[cpIdx] = recv_buffer3[insertOffset+cpIdx]; //TODO can't we just copy directly from recv_buffer3?

      bodyBuffer.h2d(items);

//      for(int z=0; z<items; z++)
//      {
//    	  LOGF(stderr,"RECV: %d\t%f %f %f %f\t%f %f %f %f\n",
//      		z,
//      		bodyBuffer[z].Ppos.x,bodyBuffer[z].Ppos.y,bodyBuffer[z].Ppos.z,bodyBuffer[z].Ppos.w,
//      		bodyBuffer[z].Pvel.x,bodyBuffer[z].Pvel.y,bodyBuffer[z].Pvel.z,bodyBuffer[z].Pvel.w);
//      }


      //Start the kernel that puts everything in place
      insertNewParticlesSFC.set_args(0,
              &nToSend, &items, &tree.n, &insertOffset, localTree.bodies_Ppos.p(),
              localTree.bodies_Pvel.p(), localTree.bodies_pos.p(), localTree.bodies_vel.p(),
              localTree.bodies_acc0.p(), localTree.bodies_acc1.p(), localTree.bodies_time.p(),
              localTree.bodies_ids.p(), localTree.bodies_key.p(), localTree.bodies_h.p(),
              localTree.bodies_dens.p(), localTree.bodies_grad.p(), localTree.bodies_hydro.p(), bodyBuffer.p());
      insertNewParticlesSFC.setWork(items, 128);
      insertNewParticlesSFC.execute2(execStream->s());
    }// if items > 0
    insertOffset += items;
  } //for recvCount

  //Resize the arrays of the tree
  tree.setN(newN);
  reallocateParticleMemory(tree);

  double tEnd = get_time();

  char buff5[1024];
  sprintf(buff5,"EXCHANGEB-%d: tExSend: %lg tExGPUSync: %lg tExGPUAlloc: %lg tExGPUSend: %lg tISendIRecv: %lg tWaitall: %lg\n",
                procId, tSendEnd-tStart, tSyncGPU-tSendEnd,
                tAllocComplete-tSyncGPU, tEnd-tAllocComplete,
                t94-tStart, tSendEnd-t94);
  devContext->writeLogEvent(buff5);

#endif

//  localTree.bodies_Ppos.d2h();
//  localTree.bodies_pos.d2h();

//  for(int i=0; i < tree.n; i++)
//  {
//	  LOGF(stderr,"CURRENT: %d %f %f  \t %f %f\n",
//			  i,
//			  localTree.bodies_pos[i].x, localTree.bodies_pos[i].y,
//			  localTree.bodies_Ppos[i].x, localTree.bodies_Ppos[i].y);
//  }


  return 0;
}



/********************************************************
 *                                                      *
 * Functions related to the LET Creation and Exchange   *
 *                                                      *
 ********************************************************/

/*
 *
 *  Broadcast the boundaries of the tree-structure (will be used during the LET creation)
 *  First gather the size, then create/allocate memory
 *  followed by a broadcast to send and receive the structures.
 *  This is a sync-operation and therefore can be quite costly
 *  Maybe we want to do this in a separate thread
 *
 */
int4 octree::getSearchPropertiesBoundaryTrees()
{
    int searchDepthUsed = 99;


    const int maxNLarge = 512;  //Maximum number of processes to which we send the full group tree.


    static int4 boundaryTreeSettings;  //x = depth                [1..5],
                                       //y = startLevelReduction  [0..6],
                                       //z = doTuning             [0: no, 1 tune]

    //TODO This does the tuning of when to send the full group tree and when to send only
    //the small group. This is based on the previous iteration statistics on what is send

    //TODO this version stops tuning after iteration 16. Adjust method to make
    //this a per-process tuning

    std::vector<int2> globalGroupSizeArray    (nProcs);  //x is fullGroup, y = smallGroup
    std::vector<int2> globalGroupSizeArrayRecv(nProcs);  //x is fullGroup, y = smallGroup
    globalGroupSizeArray[procId] = make_int2(0,0);       //Nothing to ourselves


    //Count the number of processes that used the boundary tree and compute the average level
    int smallTreeStart = localTree.level_list[localTree.startLevelMin].x;
    int smallTreeEnd   = localTree.level_list[localTree.startLevelMin].y;

    if(iter < 8)
    {
      //For iteration 0 to 8 always send the full tree
      for(int i=0; i < nProcs; i++)
      {
        fullGrpAndLETRequestStatistics[i].x = 0;
        fullGrpAndLETRequestStatistics[i].y = 0;
      }

      //Set the default values
      boundaryTreeSettings.x = 5;
      boundaryTreeSettings.y = 0;
      boundaryTreeSettings.z = 1;
    }
    else
    {
      //Check if we have to restore some settings
      if(boundaryTreeSettings.z > 0)
      {
        int countFailed = 0;
        for(int i=0; i < nProcs; i++)
        {
          if(fullGrpAndLETRequestStatistics[i].x <= 0)
            countFailed++;
        }

        //Test if we have to restore
        if(countFailed > maxNLarge)
        {
          //Restore the marked processes
          for(int i=0; i < nProcs; i++)
            if(fullGrpAndLETRequestStatistics[i].x < 0) fullGrpAndLETRequestStatistics[i].x = 1;

          //Restore workable settings, by modifying the start and max depth of the boundaryTree
          if(boundaryTreeSettings.y > 0)
          {
            boundaryTreeSettings.y--;
          }
          else
          {
            boundaryTreeSettings.x = std::min(5, boundaryTreeSettings.x+1);
          }

          fprintf(stderr,"Proc: %d COUNTFAIL  %d  iter: %d back to: %d %d \n",
              procId, countFailed, iter, boundaryTreeSettings.x, boundaryTreeSettings.y);

          //Mark as not changeable anymore, we found the 'optimal' settings
          boundaryTreeSettings.z = 0;
        }//if countFailed > maxNLarge
        else
        {
          //Mark processes permanently and just keep tuning
          for(int i=0; i < nProcs; i++)
            if(fullGrpAndLETRequestStatistics[i].x < 0) fullGrpAndLETRequestStatistics[i].x = 0;
        }
      } //if(boundaryTreeSettings.z > 0)
      else
      {
        //Force the ones set to 0 for counters further on
        for(int i=0; i < nProcs; i++)
          if(fullGrpAndLETRequestStatistics[i].x < 0) fullGrpAndLETRequestStatistics[i].x = 0;
      }



      //Start on the root node
      if(iter >= 12 && boundaryTreeSettings.z == 1)
      {
        //First decrease the depth, how far we go from the start.
        searchDepthUsed =  boundaryTreeSettings.x-1;
        boundaryTreeSettings.x--;

        boundaryTreeSettings.x = std::max(1, boundaryTreeSettings.x);
        searchDepthUsed        = std::max(searchDepthUsed,1); //Minimum level 1

        if(searchDepthUsed == 1)
        {
          //Start decreasing the start level as we hit a minimum depth level
          boundaryTreeSettings.y++;
        }

        //Set the start and end node of start level
        if(boundaryTreeSettings.y > localTree.startLevelMin) boundaryTreeSettings.y = localTree.startLevelMin;

        smallTreeStart  = localTree.level_list[localTree.startLevelMin-boundaryTreeSettings.y].x;
        smallTreeEnd    = localTree.level_list[localTree.startLevelMin-boundaryTreeSettings.y].y;
      }

      if(boundaryTreeSettings.z == 0)
      {
        searchDepthUsed = boundaryTreeSettings.x;
        searchDepthUsed = std::max(searchDepthUsed,1); //Minimum level 1
        //Set the start and end node of start level
        if(boundaryTreeSettings.y > localTree.startLevelMin) boundaryTreeSettings.y = localTree.startLevelMin;

        smallTreeStart  = localTree.level_list[localTree.startLevelMin-boundaryTreeSettings.y].x;
        smallTreeEnd    = localTree.level_list[localTree.startLevelMin-boundaryTreeSettings.y].y;
      }


      //if(iter > 20)
      if(0)
      {
        smallTreeStart  = 0;
        smallTreeEnd    = 1;
        searchDepthUsed = 1;
      }
    }

    return make_int4(smallTreeStart, smallTreeEnd, 0, 0); //TODO remove, testing only, force a small Grp tree
    return make_int4(smallTreeStart, smallTreeEnd, searchDepthUsed, 0);
}


void octree::updateCurrentInfoGrpTree()
{
    //This function will send updated smoothing values in order to upgrade
    //the existing boundary trees.

assert(0); //This has to be double checked that the new indices are correct now that we also send dens and hydro props of partiles


    //Launch GPU kernel to extract and copy updates values to host
    //NOTE/TODO this should be moved to somewhere else such that it can be hidden behind local computations
    bool smoothingOnly = true; //We want the full tree at this point
    gpuBoundaryTreeExtract.set_args(0, &smoothingOnly,
                                   localTree.boxSizeInfo.p(), localTree.boxCenterInfo.p(), localTree.boxSmoothing.p(), localTree.multipole.p(),
                                   localTree.bodies_Ppos.p(), localTree.bodies_Pvel.p(),   localTree.bodies_dens.p(),  localTree.bodies_hydro.p(),
                                   localTree.smallBoundaryTreeIndices.p(), localTree.smallBoundaryTree.p());
    gpuBoundaryTreeExtract.setWork(32, 32, 1);
    gpuBoundaryTreeExtract.execute2(execStream->s());
    gpuBoundaryTreeExtract.set_args(0, &smoothingOnly,
            localTree.boxSizeInfo.p(), localTree.boxCenterInfo.p(), localTree.boxSmoothing.p(), localTree.multipole.p(),
            localTree.bodies_Ppos.p(), localTree.bodies_Pvel.p(),   localTree.bodies_dens.p(),  localTree.bodies_hydro.p(),
            localTree.fullBoundaryTreeIndices.p(), localTree.fullBoundaryTree.p());
    gpuBoundaryTreeExtract.execute2(execStream->s());
    execStream->sync();

    //Compute the startOffset and number of items for the memory copies
    int2 nInfoSmall = make_int2(host_float_as_int(localTree.smallBoundaryTree[0].x), host_float_as_int(localTree.smallBoundaryTree[0].y));
    int2 nInfoFull  = make_int2(host_float_as_int(localTree.fullBoundaryTree[0].x),  host_float_as_int(localTree.fullBoundaryTree[0].y));

    const int nBodyProps = 2;
    int offsetSmall = 1+nBodyProps*nInfoSmall.x+2*nInfoSmall.y;
    int offsetFull  = 1+nBodyProps*nInfoFull.x +2*nInfoFull.y;

    //Only start the copy after the execStream has been completed, otherwise the buffers aint filled yet
    localTree.smallBoundaryTree.d2h(nInfoSmall.y, offsetSmall, false, LETDataToHostStream->s());
    localTree.fullBoundaryTree. d2h(nInfoFull.y,  offsetFull , false, LETDataToHostStream->s());
    localTree.smallBoundaryTree.waitForCopyEvent();
    localTree.fullBoundaryTree.waitForCopyEvent();

    //  uint4 grpBoundStats;
    //x: Number of nodes to receive during the allGatherV
    //y: The displacement (in bytes) to where the smoothing values will be stored (both for allgather and isend/irecv)
    //z: The number of items we send to this process during iSend/iRecv
    //w: The number of items we receive from this process during iSend/iRecv, offset is computed during the execution.
    //[procId].w = onlyUseAllGather (so 1 if we only use the allGather option and 0 otherwise, to get proper arrays)


    bool onlyUseAllGather = globalGrpTreeStatistics[procId].w;
    std::vector<int> recvSizes          (nProcs,0);
    std::vector<int> displacement       (nProcs,0);
    for(int i=0; i < nProcs; i++)
    {
        recvSizes[i]    = globalGrpTreeStatistics[i].x;
        displacement[i] = globalGrpTreeStatistics[i].y;
    }

    //MPI_Allgatherv for the small tree and Isend/IRecv for the fulltree
    //When only all gather we send fullTree otherwise smallTree
    if(onlyUseAllGather)
    {
        MPI_Allgatherv(&localTree.fullBoundaryTree[offsetFull], sizeof(real4)*nInfoFull.y,      MPI_BYTE,
                        globalGrpTreeCntSize,          &recvSizes[0],   &displacement[0], MPI_BYTE, mpiCommWorld);
    }
    else
    {
        MPI_Allgatherv(&localTree.smallBoundaryTree[offsetSmall], sizeof(real4)*nInfoSmall.y,      MPI_BYTE,
                       globalGrpTreeCntSize,            &recvSizes[0],   &displacement[0], MPI_BYTE, mpiCommWorld);
    }


    //Send / receive loop like particle exchange, for the full boundary trees
    static MPI_Status stat[NMAXPROC];
    static MPI_Request req[NMAXPROC*2];
    assert(nProcs < NMAXPROC);

    int nreq = 0;
    for (int dist = 1; dist < nProcs; dist++)
    {
      const int src    = (nProcs + procId - dist) % nProcs;
      const int dst    = (nProcs + procId + dist) % nProcs;
      const int scount = globalGrpTreeStatistics[dst].z * sizeof(float4);
      const int rcount = globalGrpTreeStatistics[src].w * sizeof(float4);
      const int offset = globalGrpTreeStatistics[src].y / sizeof(float4);

      if (scount > 0)
      {
        fprintf(stderr,"Sending %d bytes to %d \n", scount, dst);
        MPI_Isend(&localTree.fullBoundaryTree[offsetFull], scount, MPI_BYTE, dst, 1, mpiCommWorld, &req[nreq++]);
      }
      if(rcount > 0)
      {
        fprintf(stderr,"Receiving %d bytes from %d \n", rcount, src);
        MPI_Irecv(&globalGrpTreeCntSize[offset], rcount, MPI_BYTE, src, 1, mpiCommWorld, &req[nreq++]);
      }
    }
    MPI_Waitall(nreq, req, stat);

    //Boundary trees have been updated, we can use them now :)
}


void octree::sendCurrentInfoGrpTree()
{
#ifdef USE_MPI

  double tStartGrp = get_time(); //TODO delete

  std::vector<int2> globalGroupSizeArray    (nProcs);  //x is fullGroup, y = smallGroup
  std::vector<int2> globalGroupSizeArrayRecv(nProcs);  //x is fullGroup, y = smallGroup
  globalGroupSizeArray[procId] = make_int2(0,0);       //Nothing to ourselves

  for(int i=0; i < nProcs; i++)
  {
      globalGrpTreeStatistics[i] = make_int4(0,0,0,0); //Reset the array
  }


  int nGroupsSmallSet = boundaryTreeDimensions.x;
  int nGroupsFullSet  = boundaryTreeDimensions.y;


  bool onlyUseAllGather = false;

  //If we have to send full-groups to a large number of processes, then we use the allgaterv
  //and do not use ISend/Irecv at all. So send the full groups through the allgather path.
  int tempCount = 0;
  for(int i=0; i < nProcs; i++)
  {
    if(fullGrpAndLETRequestStatistics[i].x == 0) tempCount++;

    //if(tempCount == nProcs) //if(tempCount > ((int) (0.75*nProcs)))
    if(tempCount > 2048)
    {
      onlyUseAllGather = true;
      break;
    }
  }


  //Assign the sizes of the boundary trees
  for(int i=0; i < nProcs; i++)
  {
    //TODO check that we don't overwrite i with a negative nGroupsSmallSet
    if(i == procId) globalGroupSizeArray[i].y = nGroupsSmallSet;

    //If a smallGroup was enough or when small and big are equal/same data then only use
    //MPI_Allgatherv this reduces the number of iSend/Irecv calls
    if(fullGrpAndLETRequestStatistics[i].x > 0 || (nGroupsSmallSet == nGroupsFullSet))
    {
      globalGroupSizeArray[i].y = nGroupsSmallSet; //Use the small set
      globalGroupSizeArray[i].x = 0;
    }
    else
    {
      globalGroupSizeArray[i].y = -nGroupsSmallSet; //use the big set
      globalGroupSizeArray[i].x = nGroupsFullSet;
    }

    //Ignore the above if we use allGatherV and do not use Isend/Irecv
    if(onlyUseAllGather)
    {
      globalGroupSizeArray[i].y = nGroupsFullSet;
      globalGroupSizeArray[i].x = 0;
    }
  }


  //Statistics
  int nGroupsSmall = 0, nGroupsLarge = 0;
  for(int i=0; i < nProcs; i++)
  {
    if(globalGroupSizeArray[i].x == 0)  nGroupsSmall++;
    else                                nGroupsLarge++;
  }

  LOGF(stderr,"GRP Iter: %d small: %d %d  d: %d  full: %d %d || sizeS: %d sizeN: %d  nSmall: %d nlarge: %d  \n",
      iter, -1, -1, -1,
      localTree.level_list[localTree.startLevelMin].x,
      localTree.level_list[localTree.startLevelMin].y,
      nGroupsSmallSet, nGroupsFullSet,nGroupsSmall, nGroupsLarge);

  double t0 = get_time();

  //Communicate the sizes
  MPI_Alltoall(&globalGroupSizeArray[0],     2, MPI_INT,
               &globalGroupSizeArrayRecv[0], 2, MPI_INT, mpiCommWorld);

  std::vector<int> groupRecvSizesSmall(nProcs, 0);
  std::vector<int> displacement       (nProcs,0);

  /* compute displacements for allgatherv for fullGrps */
  int runningOffset  = 0;
  int allGatherVSize = 0;
  for (int i = 0; i < nProcs; i++)
  {
    groupRecvSizesSmall[i] = sizeof(real4)*abs(globalGroupSizeArrayRecv[i].y); //Size of small tree

    allGatherVSize        += groupRecvSizesSmall[i];

    this->globalGrpTreeCount[i]   = std::max(globalGroupSizeArrayRecv[i].x, globalGroupSizeArrayRecv[i].y);
    this->globalGrpTreeOffsets[i] = runningOffset;
    displacement[i]               = runningOffset*sizeof(real4);
    runningOffset                += this->globalGrpTreeCount[i];
  }

  if (globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
  globalGrpTreeCntSize = new real4[runningOffset]; /* total Number Of Groups = runningOffset */

  //Two methods
  //1) use MPI_Alltoallv
  //2) use combination of allgatherv (for small) + isend/irecv for large


  localTree.smallBoundaryTree.waitForCopyEvent();
  localTree.fullBoundaryTree.waitForCopyEvent();

  LOGF(stderr, "BoundaryTreeProps iter: %d small-tree nBody: %d nNodes: %d Start-End: %d-%d full-tree nBody: %d nNodes: %d Start-End: %d-%d \n",
          iter,
          host_float_as_int(localTree.smallBoundaryTree[0].x), host_float_as_int(localTree.smallBoundaryTree[0].y),
          host_float_as_int(localTree.smallBoundaryTree[0].z), host_float_as_int(localTree.smallBoundaryTree[0].w),
          host_float_as_int(localTree.fullBoundaryTree[0].x),  host_float_as_int(localTree.fullBoundaryTree[0].y),
          host_float_as_int(localTree.fullBoundaryTree[0].z),  host_float_as_int(localTree.fullBoundaryTree[0].w));

  double t1 = get_time();



  //MPI_Allgatherv for the small tree and Isend/IRecv for the fulltree
  //When only all gather we send fullTree otherwise smallTree
  if(onlyUseAllGather)
  {
      MPI_Allgatherv(&localTree.fullBoundaryTree[0], sizeof(real4)*boundaryTreeDimensions.y,      MPI_BYTE,
                      globalGrpTreeCntSize,          &groupRecvSizesSmall[0],   &displacement[0], MPI_BYTE, mpiCommWorld);
  }
  else
  {
      MPI_Allgatherv(&localTree.smallBoundaryTree[0], sizeof(real4)*boundaryTreeDimensions.x,      MPI_BYTE,
                     globalGrpTreeCntSize,            &groupRecvSizesSmall[0],   &displacement[0], MPI_BYTE, mpiCommWorld);
  }



  for(int i=0; i < nProcs; i++)
  {
      float4 header = globalGrpTreeCntSize[displacement[i]/sizeof(real4)];
      //Store the number of nodes in bytes received during the allGatherV step
      globalGrpTreeStatistics[i].x = host_float_as_int(header.y)*sizeof(float4);
  }


  double t2 = get_time();

  //Send / receive loop like particle exchange, for the full boundary trees
  static MPI_Status stat[NMAXPROC];
  static MPI_Request req[NMAXPROC*2];
  assert(nProcs < NMAXPROC);

  int nreq = 0;
  for (int dist = 1; dist < nProcs; dist++)
  {
    const int src    = (nProcs + procId - dist) % nProcs;
    const int dst    = (nProcs + procId + dist) % nProcs;
    const int scount = (globalGroupSizeArray    [dst].y <= 0) ? boundaryTreeDimensions.y        * sizeof(real4) : 0;
    const int rcount = (globalGroupSizeArrayRecv[src].y <= 0) ? globalGroupSizeArrayRecv[src].x * sizeof(real4) : 0;
    const int offset = this->globalGrpTreeOffsets[src];

    if (scount > 0)
    {
      MPI_Isend(&localTree.fullBoundaryTree[0], scount, MPI_BYTE, dst, 1, mpiCommWorld, &req[nreq++]);
      LOGF(stderr,"Sending to: %d size: %d \n", dst, (int)(scount / sizeof(real4)));

      //Store the number of nodes we send to this process
      globalGrpTreeStatistics[dst].z = host_float_as_int(localTree.fullBoundaryTree[0].y);
    }
    if(rcount > 0)
    {
      MPI_Irecv(&globalGrpTreeCntSize[offset], rcount, MPI_BYTE, src, 1, mpiCommWorld, &req[nreq++]);
      LOGF(stderr,"Receiving from: %d size: %d Offset: %d \n", src, globalGroupSizeArrayRecv[src].x, offset);
    }
  }
  fprintf(stderr,"Full send, nreq: %d\n", nreq);
  MPI_Waitall(nreq, req, stat);

  //Rerun the loop to build the list of receive properties
  for (int dist = 0; dist < nProcs; dist++)
  {
    const int src    = (nProcs + procId - dist) % nProcs;
    const int rcount = (globalGroupSizeArrayRecv[src].y <= 0) ? globalGroupSizeArrayRecv[src].x * sizeof(real4) : 0;

    if(rcount > 0 && src != procId)
    {
      //Store the number of nodes we received from this process
      globalGrpTreeStatistics[src].w = host_float_as_int(globalGrpTreeCntSize[this->globalGrpTreeOffsets[src]].y);
      //We cannot store the offset, not enough items, but we can compute it during receive via the globalGrpTreeCntSize
    }

    //Compute the part where the smoothing starts. We need this to make sure the allGatherv does not overwrite other values
    const int nBodyProps = 4;
    float4 header = globalGrpTreeCntSize[this->globalGrpTreeOffsets[src]];
    int displacementSmooth         = (1+nBodyProps*host_float_as_int(header.x)+2*host_float_as_int(header.y))*sizeof(float4);
    globalGrpTreeStatistics[src].y = displacement[src] + displacementSmooth;
  }

  globalGrpTreeStatistics[procId].w = onlyUseAllGather;

  //  uint4 grpBoundStats;
  //x: Number of nodes to receive during the allGatherV
  //y: The displacement (in bytes) to where the smoothing values will be stored in the allGatherV
  //z: The number of items we send to this process during iSend/iRecv
  //w: The number of items we receive from this process during iSend/iRecv, offset is computed during the execution.
  //[procId].w = onlyUseAllGather (so 1 if we only use the allGather option and 0 otherwise, to get proper arrays)


  double tEndGrp = get_time();
  char buff5[1024];
  sprintf(buff5,"BLETTIME-%d: Iter: %d tGrpSend: %lg nGrpSizeSmall: %d nGrpSizeLarge: %d nSmall: %d nLarge: %d tAllgather: %lg tAllGatherv: %lg tSendRecv: %lg AllGatherVSize: %f\n",
                 procId, iter, tEndGrp-tStartGrp, nGroupsSmallSet, nGroupsFullSet, nGroupsSmall, nGroupsLarge, t1-t0, t2-t1, tEndGrp-t2, allGatherVSize / (1024*1024.));
  devContext->writeLogEvent(buff5);
#endif
}



//////////////////////////////////////////////////////
// ***** Local essential tree functions ************//
//////////////////////////////////////////////////////
#ifdef USE_MPI



int getLEToptQuickTreevsTree(
    GETLETBUFFERS &bufferStruct,
    const real4 *nodeCentre,        //Remote
    const real4 *nodeSize,          //Remote
    const real4 *multipole,         //Remote
    const int cellBeg,
    const int cellEnd,
    const real4 *groupSizeInfo,     //Local
    const real4 *groupCentreInfo,   //Local
    const int groupBeg,
    const int groupEnd,
    const int nNodes,
    const int procId,
    const int ibox,
    double &timeFunction, int &depth,
    const float3 periodicDomainSize,
    const int    periodicMethod,
    const int    selectionMethod,
    const int    LETMethod)
{
  double tStart = get_time2();

  depth = 0;

  const _v4sf*          nodeSizeV = (const _v4sf*)nodeSize;
  const _v4sf*        nodeCentreV = (const _v4sf*)nodeCentre;
  const _v4sf*         multipoleV = (const _v4sf*)multipole;
  const _v4sf*   grpNodeSizeInfoV = (const _v4sf*)groupSizeInfo;
  const _v4sf* grpNodeCenterInfoV = (const _v4sf*)groupCentreInfo;


#ifdef USE_AVX  /* AVX */
    #ifndef __AVX__
        #error "AVX is not defined"
    #endif
      const int SIMDW  = 8;
      #define AVXIMBH
    #else
      const int SIMDW  = 4; //#define SSEIMBH
#endif

  bool usePeriodic = false;
  int2 xP = {0,0}, yP = {0,0}, zP = {0,0};
  if(periodicMethod & 1) { xP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & 2) { yP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & 4) { zP = {-1,1}; usePeriodic = true;}

  bufferStruct.LETBuffer_node.clear();
  bufferStruct.LETBuffer_ptcl.clear();
  bufferStruct.currLevelVecUI4.clear();
  bufferStruct.nextLevelVecUI4.clear();
  bufferStruct.currGroupLevelVec.clear();
  bufferStruct.nextGroupLevelVec.clear();
  bufferStruct.groupSplitFlag.clear();

  Swap<std::vector<uint4> > levelList(bufferStruct.currLevelVecUI4, bufferStruct.nextLevelVecUI4);
  Swap<std::vector<int> > levelGroups(bufferStruct.currGroupLevelVec, bufferStruct.nextGroupLevelVec);

  /* copy group info into current level buffer */
  for (int group = groupBeg; group < groupEnd; group++)
    levelGroups.first().push_back(group);

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back((uint4){(uint)cell, 0, (uint)levelGroups.first().size(),0});

  double tPrep = get_time2();

  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint4       nodePacked = levelList.first()[i];
      const uint  nodeIdx          = nodePacked.x;

      const float2 openings  = extract_opening_criteria(nodeCentre[nodeIdx].w);
      const float nodeInfo_x = openings.x;
      const float nodeSmth   = fabs(openings.y);

      const uint  nodeInfo_y       = host_float_as_int(nodeSize[nodeIdx].w);
      const _v4sf nodeCOM          = __builtin_ia32_vec_set_v4sf(multipoleV[nodeIdx*3], nodeInfo_x, 3);

      //SPH uses physical center and size of the box to test overlap and distance
      const _v4sf nodeSIZE = nodeSizeV  [nodeIdx];
      const _v4sf nodeCNTR = __builtin_ia32_vec_set_v4sf(nodeCentreV[nodeIdx], nodeSmth, 3); //Move smoothing to the .w of cntr

#ifdef USE_AVX
      _v8sf checkValueGrav = {nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x,nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x};
            checkValueGrav = __abs8(checkValueGrav);
      _v8sf checkValueSPH  = {nodeSmth, nodeSmth, nodeSmth, nodeSmth, nodeSmth, nodeSmth, nodeSmth, nodeSmth};
#else
      _v4sf checkValueGrav = {nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x};
            checkValueGrav = __abs(checkValueGrav);
       //nodeSmth
      _v4sf checkValueSPH  = {nodeSmth, nodeSmth, nodeSmth, nodeSmth};
#endif

      const bool lleaf   = nodeInfo_x <= 0.0f;
      const int groupBeg = nodePacked.y;
      const int groupEnd = nodePacked.z;

      bufferStruct.groupSplitFlag.clear();
      for (int ib = groupBeg; ib < groupEnd; ib += SIMDW)
      {
        _v4sf centre[SIMDW], size[SIMDW];
        for (int laneIdx = 0; laneIdx < SIMDW; laneIdx++)
        {
          const int group = levelGroups.first()[std::min(ib+laneIdx, groupEnd-1)];
          centre[laneIdx] = grpNodeCenterInfoV[group];
          size  [laneIdx] =   grpNodeSizeInfoV[group];

          //For hydro we need mutual forces so add the group and node smoothing
          if(LETMethod == LET_METHOD_HYDR)
          {
              const float2 openings  = extract_opening_criteria(groupCentreInfo[group].w);
              const float grpSmth    = fabs(openings.y);

              float max_distance = std::max(nodeSmth,grpSmth);
              switch(laneIdx)
              {
                case 0: checkValueSPH = __builtin_ia32_vec_set_v4sf(checkValueSPH, max_distance, 0); break;
                case 1: checkValueSPH = __builtin_ia32_vec_set_v4sf(checkValueSPH, max_distance, 1); break;
                case 2: checkValueSPH = __builtin_ia32_vec_set_v4sf(checkValueSPH, max_distance, 2); break;
                case 3: checkValueSPH = __builtin_ia32_vec_set_v4sf(checkValueSPH, max_distance, 3); break;
#ifdef USE_AVX
                case 4: checkValueSPH = __builtin_ia32_vec_set_v8sf(checkValueSPH, max_distance, 4); break;
                case 5: checkValueSPH = __builtin_ia32_vec_set_v8sf(checkValueSPH, max_distance, 5); break;
                case 6: checkValueSPH = __builtin_ia32_vec_set_v8sf(checkValueSPH, max_distance, 6); break;
                case 7: checkValueSPH = __builtin_ia32_vec_set_v8sf(checkValueSPH, max_distance, 7); break;
#endif
              }
          }
        }




#ifdef AVXIMBH
        _v8sf resGrav = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        if(selectionMethod & SELECT_GRAV) resGrav = split_node_grav_sph_impbh_box8a<false>(resGrav,checkValueGrav, nodeCOM,  nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
        if(selectionMethod & SELECT_SPH)  resGrav = split_node_grav_sph_impbh_box8a<true >(resGrav,checkValueSPH,  nodeCNTR, nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);

        const _v4sf ret1 = __builtin_ia32_vextractf128_ps256(resGrav, 0);
        const _v4sf ret2 = __builtin_ia32_vextractf128_ps256(resGrav, 1);
        bufferStruct.groupSplitFlag.push_back(std::make_pair(ret1,ret2));
#else

       _v4sf resGrav = {0.0f, 0.0f, 0.0f, 0.0f};
       if(selectionMethod & SELECT_GRAV) resGrav = split_node_grav_sph_impbh_box4a<false>(resGrav, checkValueGrav, nodeCOM,  nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
       if(selectionMethod & SELECT_SPH)  resGrav = split_node_grav_sph_impbh_box4a<true >(resGrav, checkValueSPH,  nodeCNTR, nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
       bufferStruct.groupSplitFlag.push_back(resGrav);
#endif
      }

      const int groupNextBeg = levelGroups.second().size();
      int split = false;
      for (int idx = groupBeg; idx < groupEnd; idx++)
      {
        const bool gsplit = ((uint*)&bufferStruct.groupSplitFlag[0])[idx - groupBeg];
        if (gsplit)
        {
          split = true;
          const int group = levelGroups.first()[idx];
          if (!lleaf)
          {
            const bool gleaf = groupCentreInfo[group].w <= 0.0f; //This one does not go down leaves, since it are particles
            if (!gleaf)
            {
              const int childinfoGrp  = ((uint4*)groupSizeInfo)[group].w;
              const int gchild  =   childinfoGrp & 0x0FFFFFFF;
              const int gnchild = ((childinfoGrp & 0xF0000000) >> 28) ;

              //for (int i = gchild; i <= gchild+gnchild; i++) //old tree
              for (int i = gchild; i < gchild+gnchild; i++) //GPU-tree TODO JB: I think this is the correct one, verify in treebuild code
              {
                levelGroups.second().push_back(i);
              }
            }
            else
              levelGroups.second().push_back(group);
          }
          else
            break;
        }
      }

      real4 size  = nodeSize[nodeIdx];
      int sizew   = 0xFFFFFFFF;

      if (split)
      {
        //Return -1 if we need to split something for which we
        //don't have any data-available
        if(nodeInfo_y == 0xFFFFFFFF)
          return -1;

        if (!lleaf)
        {
          const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
          const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has
          for (int i = lchild; i < lchild + lnchild; i++)
            levelList.second().push_back((uint4){(uint)i,(uint)groupNextBeg,(uint)levelGroups.second().size()});
        }
        else
        {
            //It's a leaf, and as it can be opened we got the data. Do nothing.
        }
      }//if split
    }//for
    depth++;
    levelList.swap();
    levelList.second().clear();

    levelGroups.swap();
    levelGroups.second().clear();
  }

  return 0;
}
int3 getLET1(
          GETLETBUFFERS     &bufferStruct,
          real4            **LETBuffer_ptr,
    const real4             *nodeCentre,
    const real4             *nodeSize,
    const real4             *multipole,
    const int                cellBeg,
    const int                cellEnd,
    const real4             *bodies_pos,
    const real4             *bodies_vel,
    const real2             *bodies_dens,
    const real4             *bodies_hydro,
    const int                nParticles,
    const real4             *groupSizeInfo,
    const real4             *groupCentreInfo,
    const int                nGroups,
    const int                nNodes,
          unsigned long long &nflops,
    const float3             periodicDomainSize,
    const int                periodicMethod,
    const int                selectionMethod,
    const int                LETMethod)
{
  bufferStruct.LETBuffer_node.clear();
  bufferStruct.LETBuffer_ptcl.clear();
  bufferStruct.currLevelVecI.clear();
  bufferStruct.nextLevelVecI.clear();

  nflops = 0;

  int nExportPtcl = 0;
  int nExportCell = 0;
  int nExportCellOffset = cellEnd;

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    bufferStruct.LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});

  const int SIMDW   = 4;

  #ifdef USE_AVX
      //We use a mix of 4 and 8 widths
      const int SIMDW2  = 8;
  #else
      const int SIMDW2  = 4;
  #endif


  const _v4sf*      bodies_posV     = (const _v4sf*)bodies_pos;
  const _v4sf*      bodies_velV     = (const _v4sf*)bodies_vel;
  const _v4sf*      bodies_HydroV   = (const _v4sf*)bodies_hydro;
  const _v4sf*      nodeSizeV       = (const _v4sf*)nodeSize;
  const _v4sf*      nodeCentreV     = (const _v4sf*)nodeCentre;
  const _v4sf*      multipoleV      = (const _v4sf*)multipole;
  const _v4sf*      groupSizeV      = (const _v4sf*)groupSizeInfo;
  const _v4sf*      groupCenterV    = (const _v4sf*)groupCentreInfo;


  Swap<std::vector<int> > levelList(bufferStruct.currLevelVecI, bufferStruct.nextLevelVecI);

  //const int nGroupsV = ((nGroups-1)/SIMDW + 1)*SIMDW;
  const int nGroupsV = ((nGroups-1)/SIMDW2 + 1)*SIMDW2;

  //We need a bunch of buffers to act as swap space
  const int allocSize = (int)(nGroupsV*1.10);
  bufferStruct.groupCentreSIMD.reserve(allocSize);
  bufferStruct.groupSizeSIMD.reserve(allocSize);

  bufferStruct.groupCentreSIMD.resize(nGroupsV);
  bufferStruct.groupSizeSIMD.resize(nGroupsV);

  bufferStruct.groupCentreSIMDSwap.reserve(allocSize);
  bufferStruct.groupSizeSIMDSwap.reserve(allocSize);

  bufferStruct.groupCentreSIMDSwap.resize(nGroupsV);
  bufferStruct.groupSizeSIMDSwap.resize(nGroupsV);

  bufferStruct.groupSIMDkeys.resize((int)(1.10*(nGroupsV/SIMDW)));

  bool usePeriodic = false;
  int2 xP = {0,0}, yP = {0,0}, zP = {0,0};
  if(periodicMethod & 1) { xP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & 2) { yP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & 4) { zP = {-1,1}; usePeriodic = true;}


#if 1
  const bool TRANSPOSE_SPLIT = false;
#else
  const bool TRANSPOSE_SPLIT = true;
#endif
  for (int ib = 0; ib < nGroupsV; ib += SIMDW)
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

    bufferStruct.groupCentreSIMD[ib+0] = bcx;
    bufferStruct.groupCentreSIMD[ib+1] = bcy;
    bufferStruct.groupCentreSIMD[ib+2] = bcz;
    bufferStruct.groupCentreSIMD[ib+3] = bcw;

    bufferStruct.groupSizeSIMD[ib+0] = bsx;
    bufferStruct.groupSizeSIMD[ib+1] = bsy;
    bufferStruct.groupSizeSIMD[ib+2] = bsz;
    bufferStruct.groupSizeSIMD[ib+3] = bsw;
  }

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);

  int depth = 0;
  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
#if 1
    if (nGroups > 128)   /* randomizes algo, can give substantial speed-up */
        shuffle2vecAllocated<v4sf,SIMDW>(bufferStruct.groupCentreSIMD,
                                         bufferStruct.groupSizeSIMD,
                                         bufferStruct.groupCentreSIMDSwap,
                                         bufferStruct.groupSizeSIMDSwap,
                                         bufferStruct.groupSIMDkeys);
#endif
    for (int i = 0; i < csize; i++)
    {
      const uint    nodeIdx  = levelList.first()[i];

      const float2 openings  = extract_opening_criteria(nodeCentre[nodeIdx].w);
      const float nodeInfo_x = openings.x;
      const float nodeSmth   = fabs(openings.y);

      const uint  nodeInfo_y = host_float_as_int(nodeSize[nodeIdx].w);

      _v4sf nodeCOM = multipoleV[nodeIdx*3];
      nodeCOM       = __builtin_ia32_vec_set_v4sf (nodeCOM, nodeInfo_x, 3);

      _v4sf checkValue = {0.0,0.0,0.0,0.0};


      //Gravity uses COM
      const _v4sf vncx  = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x00);
      const _v4sf vncy  = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0x55);
      const _v4sf vncz  = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xaa);
      const _v4sf vncw  = __builtin_ia32_shufps(nodeCOM, nodeCOM, 0xff);
      checkValue        = __abs(vncw);


      //SPH uses physical box center and box size to test overlap and distance
      _v4sf nodeCNTR = nodeCentreV[nodeIdx];
      _v4sf nodeSIZE = nodeSizeV  [nodeIdx];
      const _v4sf vnsx    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x00);
      const _v4sf vnsy    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x55);
      const _v4sf vnsz    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0xaa);
      const _v4sf vncntrx = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x00);
      const _v4sf vncntry = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x55);
      const _v4sf vncntrz = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xaa);

      int split     = false;

      nflops += nGroups*20;  /* effective flops, can be less */
      for (int ib = 0; ib < nGroupsV && !split; ib += SIMDW2)
      {
//          //Gravity test
          if(selectionMethod & SELECT_GRAV)
          {
#ifdef USE_AVX
             split |= split_node_grav_sph_impbh_box8simd1_periodic<TRANSPOSE_SPLIT, false>(
#else
             split |= split_node_grav_sph_impbh_box4simd1_periodic<TRANSPOSE_SPLIT, false>(
#endif
                                                                     vncx,vncy,vncz,
                                                                     vnsx,vnsy,vnsz,
                                                                     checkValue,
                                                                    (_v4sf*)&bufferStruct.groupCentreSIMD[ib],
                                                                    (_v4sf*)&bufferStruct.groupSizeSIMD[ib],
                                                                    periodicDomainSize,
                                                                    xP,yP,zP);
          } //if grav
          if(split) break;

          //SPH test if distance between boxes is smaller than the smoothing range
          if(selectionMethod & SELECT_SPH)
          {
              float temp_smth[4] = {0.0,0.0,0.0,0.0};

              for(int k=0; k < SIMDW2; k++)
              {
                  const float2 openings  = extract_opening_criteria(bufferStruct.groupCentreSIMD[ib+3].data[k]);
                  temp_smth[k]   = fabs(openings.y);

                  //For hydro we need mutual forces, so grap the larger of the node and group smoothing values
                  if(LETMethod == LET_METHOD_HYDR)
                  {
                      temp_smth[k] = std::max(nodeSmth, temp_smth[k]);
                  }
              }

              checkValue       = __builtin_ia32_vec_set_v4sf (checkValue, temp_smth[0], 0);
              checkValue       = __builtin_ia32_vec_set_v4sf (checkValue, temp_smth[1], 1);
              checkValue       = __builtin_ia32_vec_set_v4sf (checkValue, temp_smth[2], 2);
              checkValue       = __builtin_ia32_vec_set_v4sf (checkValue, temp_smth[3], 3);

              #ifdef USE_AVX
                             split |= split_node_grav_sph_impbh_box8simd1_periodic<TRANSPOSE_SPLIT, true>(
              #else
                             split |= split_node_grav_sph_impbh_box4simd1_periodic<TRANSPOSE_SPLIT, true>(
              #endif
                                                                   vncntrx,vncntry,vncntrz,
                                                                   vnsx,vnsy,vnsz,
                                                                   checkValue,
                                                                   (_v4sf*)&bufferStruct.groupCentreSIMD[ib],
                                                                   (_v4sf*)&bufferStruct.groupSizeSIMD[ib],
                                                                   periodicDomainSize,
                                                                   xP,yP,zP);
          } //if (selectionMethod & SELECT_SPH)
      } //for nGroupsV

      /**************/

      real4 size  = nodeSize[nodeIdx];
      int sizew   = 0xFFFFFFFF;

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
            bufferStruct.LETBuffer_ptcl.push_back(i);
          nExportPtcl += np;
        }
      }

      bufferStruct.LETBuffer_node.push_back((int2){(int)nodeIdx, sizew});
      nExportCell++;
    }
    depth++;
    levelList.swap();
    levelList.second().clear();
  }

  assert((int)bufferStruct.LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)bufferStruct.LETBuffer_node.size() == nExportCell);

  int bodyPropsCount = 1;                                //Position
  if(LETMethod == LET_METHOD_DRVT)  bodyPropsCount +=1;  //Position and Velocity
  if(LETMethod == LET_METHOD_HYDR)  bodyPropsCount +=3;  //Position, Velocity, Density and Hydro

  /* now copy data into LETBuffer */
  {
    //LETBuffer.resize(nExportPtcl + 5*nExportCell);
#pragma omp critical //Malloc seems to be not so thread safe..
    *LETBuffer_ptr     = (real4*)malloc(sizeof(real4)*(1+ bodyPropsCount*nExportPtcl + 5*nExportCell));
    real4 *LETBuffer   = *LETBuffer_ptr;
    _v4sf *vLETBuffer  = (_v4sf*)(&LETBuffer[1]); //Start at 1, since 0 contains the header

    real2 *densBuffer   = (real2*)(&LETBuffer[1+2*nExportPtcl]);


    for (int i = 0; i < nExportPtcl; i++)
    {
      const int idx = bufferStruct.LETBuffer_ptcl[i];
      vLETBuffer[i] = bodies_posV[idx];
    }
    if(LETMethod == LET_METHOD_DRVT ||LETMethod == LET_METHOD_HYDR)
    {
        for (int i = 0; i < nExportPtcl; i++)
        {
          const int idx = bufferStruct.LETBuffer_ptcl[i];
          vLETBuffer[nExportPtcl+i] = bodies_velV[idx];
        }
    }
    if(LETMethod == LET_METHOD_HYDR)
    {

        //Changing this if 1 to if 0 also requires changes in mergeAndLaunch
#if 0
        int storeIdx = 0;
        for(int i=0; i < nExportPtcl; i+=2)
        {
            int idx1             = bufferStruct.LETBuffer_ptcl[i];
            int idx2             = bufferStruct.LETBuffer_ptcl[std::min(i+1, nExportPtcl-1)];

            _v4sf dens = {0.0f,0.0f,0.0f,0.0f};
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx1].x, 0); //Move density to the .x component
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx1].y, 1); //Move smth to the .y component
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx2].x, 2); //Move density to the .z component
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx2].y, 3); //Move smth to the .w component
            vLETBuffer[2*nExportPtcl+storeIdx] = dens;
            storeIdx++;
        }
#endif

        for (int i = 0; i < nExportPtcl; i++)
        {
          const int idx             = bufferStruct.LETBuffer_ptcl[i];

          //Do not use the 4 lines below if above we set #if 1
          _v4sf dens = {0.0f,0.0f,0.0f,0.0f};
          dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx].x, 0); //Move density to the .x component
          dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx].y, 1); //Move density to the .y component
          vLETBuffer[2*nExportPtcl+i] = dens;


          vLETBuffer[3*nExportPtcl+i] = bodies_HydroV[idx];
        }
    }

    int nStoreIdx     = nExportPtcl*bodyPropsCount;
    int multiStoreIdx = nStoreIdx + 2*nExportCell;
    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = bufferStruct.LETBuffer_node[i];
      const int idx         = packed_idx.x;
      const float sizew     = host_int_as_float(packed_idx.y);
      const _v4sf size      = __builtin_ia32_vec_set_v4sf(nodeSizeV[idx], sizew, 3);


      vLETBuffer[nStoreIdx+nExportCell] = nodeCentreV[idx];     /* centre */
      vLETBuffer[nStoreIdx            ] = size;                 /*  size  */

      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+0];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+1];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+2];  /* multipole.x */
      nStoreIdx++;
    }
  }

  return (int3){nExportCell, nExportPtcl, depth};
}


//April 3, 2014. JB: Disabled the copy/creation of tree. Since we don't do alltoallV sends
//it now only counts/tests
int getLEToptQuickFullTree(
    GETLETBUFFERS   &bufferStruct,
    const int        NCELLMAX,
    const int        NDEPTHMAX,
    const real4     *nodeCentre,    //Local
    const real4     *nodeSize,      //Local
    const real4     *multipole,     //Local
    const int        cellBeg,
    const int        cellEnd,
    const real4     *bodies,
    const int        nParticles,
    const real4     *groupSizeInfo,    //Remote
    const real4     *groupCentreInfo,  //Remote
    const int        groupBeg,
    const int        groupEnd,
    const int        nNodes,
    const int        procId,
    const int        ibox,
    unsigned long long &nflops,
    double &time,
    const float3 periodicDomainSize,
    const int    periodicMethod,
    const int    selectionMethod,
    const int    LETMethod)
{
  double tStart = get_time2();

  int depth = 0;
  nflops    = 0;

  int nExportCell = 0;
  int nExportPtcl = 0;
  int nExportCellOffset = cellEnd;

//  const _v4sf*            bodiesV = (const _v4sf*)bodies;
  const _v4sf*          nodeSizeV = (const _v4sf*)nodeSize;
  const _v4sf*        nodeCentreV = (const _v4sf*)nodeCentre;
  const _v4sf*         multipoleV = (const _v4sf*)multipole;
  const _v4sf*   grpNodeSizeInfoV = (const _v4sf*)groupSizeInfo;
  const _v4sf* grpNodeCenterInfoV = (const _v4sf*)groupCentreInfo;


#ifdef USE_AVX
    #ifndef __AVX__
    #error "AVX is not defined"
    #endif
      const int SIMDW  = 8;
    #else
      const int SIMDW  = 4;
#endif

  bool usePeriodic = false;
  int2 xP = {0,0}, yP = {0,0}, zP = {0,0};
  if(periodicMethod & 1) { xP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & 2) { yP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & 4) { zP = {-1,1}; usePeriodic = true;}


  bufferStruct.LETBuffer_node.clear();
  bufferStruct.LETBuffer_ptcl.clear();
  bufferStruct.currLevelVecUI4.clear();
  bufferStruct.nextLevelVecUI4.clear();
  bufferStruct.currGroupLevelVec.clear();
  bufferStruct.nextGroupLevelVec.clear();
  bufferStruct.groupSplitFlag.clear();

  Swap<std::vector<uint4> > levelList(bufferStruct.currLevelVecUI4, bufferStruct.nextLevelVecUI4);
  Swap<std::vector<int> > levelGroups(bufferStruct.currGroupLevelVec, bufferStruct.nextGroupLevelVec);

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    bufferStruct.LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});

  /* copy group info into current level buffer */
  for (int group = groupBeg; group < groupEnd; group++)
    levelGroups.first().push_back(group);

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back((uint4){(uint)cell, 0, (uint)levelGroups.first().size(),0});

  double tPrep = get_time2();

  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      /* play with criteria to fit what's best */
      if (depth > NDEPTHMAX && nExportCell > NCELLMAX){
        return -1;
      }
      if (nExportCell > NCELLMAX){
        return -1;
      }

      const uint4 nodePacked       = levelList.first()[i];
      const uint  nodeIdx          = nodePacked.x;

      const float2 openings  = extract_opening_criteria(nodeCentre[nodeIdx].w);
      const float nodeInfo_x = openings.x;
      const float nodeSmth   = fabs(openings.y);

      const uint  nodeInfo_y       = host_float_as_int(nodeSize[nodeIdx].w);

      const _v4sf nodeCOM          = __builtin_ia32_vec_set_v4sf(multipoleV[nodeIdx*3], nodeInfo_x, 3);
      const bool lleaf             = nodeInfo_x <= 0.0f;

      const _v4sf nodeSIZE         = nodeSizeV  [nodeIdx];
      const _v4sf nodeCNTR         = nodeCentreV[nodeIdx]; //Move smoothing to the .w of cntr

      _v4sf checkValueSPH0  = {0,0,0,0};
      #ifdef USE_AVX
          _v8sf checkValueSPH  = {0,0,0,0,0,0,0,0};
          _v8sf checkValueGrav = {nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x,nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x};
          checkValueGrav       = __abs8(checkValueGrav);

          _v8sf checkValueSPH01  = {0,0,0,0,0,0,0,0};
          _v4sf checkValueSPH1   = {0,0,0,0};
      #else
          _v4sf checkValueGrav = {nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x};
          checkValueGrav       = __abs(checkValueGrav);
      #endif


      const int groupBeg = nodePacked.y;
      const int groupEnd = nodePacked.z;
      nflops += 20*((groupEnd - groupBeg-1)/SIMDW+1)*SIMDW;

      bufferStruct.groupSplitFlag.clear();
      for (int ib = groupBeg; ib < groupEnd; ib += SIMDW)
      {
        _v4sf centre[SIMDW], size[SIMDW];
        for (int laneIdx = 0; laneIdx < SIMDW; laneIdx++)
        {
          const int group     = levelGroups.first()[std::min(ib+laneIdx, groupEnd-1)];
          size  [laneIdx]     = grpNodeSizeInfoV[group];
          centre[laneIdx]     = grpNodeCenterInfoV[group];
          if(selectionMethod & SELECT_SPH)
          {
              const float2 openings  = extract_opening_criteria(groupCentreInfo[laneIdx].w);
              float grpSmth          = fabs(openings.y);

              if(LETMethod == LET_METHOD_HYDR)
              {
                  //Hydro part requires mutual forces hence include the node smoothing in our test-criteria
                  grpSmth = std::max(grpSmth, nodeSmth);
              }


              if(laneIdx == 0) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, grpSmth, 0); //Set smoothing distance
              if(laneIdx == 1) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, grpSmth, 1); //Set smoothing distance
              if(laneIdx == 2) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, grpSmth, 2); //Set smoothing distance
              if(laneIdx == 3) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, grpSmth, 3); //Set smoothing distance
#ifdef USE_AVX
              if(laneIdx == 4) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, grpSmth, 0); //Set smoothing distance
              if(laneIdx == 5) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, grpSmth, 1); //Set smoothing distance
              if(laneIdx == 6) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, grpSmth, 2); //Set smoothing distance
              if(laneIdx == 7) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, grpSmth, 3); //Set smoothing distance
#endif
          }

        }

#ifdef USE_AVX
        checkValueSPH01 = pack_2xmm(checkValueSPH0,checkValueSPH1);
        _v8sf resGrav   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        if(selectionMethod & SELECT_GRAV) resGrav = split_node_grav_sph_impbh_box8a<false>(resGrav,checkValueGrav,   nodeCOM,  nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
        if(selectionMethod & SELECT_SPH)  resGrav = split_node_grav_sph_impbh_box8a<true >(resGrav,checkValueSPH01,  nodeCNTR, nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);

        const _v4sf ret1 = __builtin_ia32_vextractf128_ps256(resGrav, 0);
        const _v4sf ret2 = __builtin_ia32_vextractf128_ps256(resGrav, 1);
        bufferStruct.groupSplitFlag.push_back(std::make_pair(ret1,ret2));
#else
        _v4sf resGrav = {0.0f, 0.0f, 0.0f, 0.0f};
       if(selectionMethod & SELECT_GRAV) resGrav = split_node_grav_sph_impbh_box4a<false>(resGrav, checkValueGrav, nodeCOM,  nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
       if(selectionMethod & SELECT_SPH)  resGrav = split_node_grav_sph_impbh_box4a<true >(resGrav, checkValueSPH0, nodeCNTR, nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
        bufferStruct.groupSplitFlag.push_back(resGrav);
#endif
      }

      const int groupNextBeg = levelGroups.second().size();
      int split = false;
      for (int idx = groupBeg; idx < groupEnd; idx++)
      {
        const bool gsplit = ((uint*)&bufferStruct.groupSplitFlag[0])[idx - groupBeg];

        if (gsplit)
        {
          split = true;
          const int group = levelGroups.first()[idx];
          if (!lleaf)
          {
            bool gleaf = groupCentreInfo[group].w <= 0.0f; //This one does not go down leaves
            //const bool gleaf = groupCentreInfo[group].w == 0.0f; //Old tree This one goes up to including actual groups
            //const bool gleaf = groupCentreInfo[group].w == -1; //GPU-tree This one goes up to including actual groups

            //Do an extra check on size.w to test if this is an end-point. If it is an end-point
            //we can do no further splits.
            if(!gleaf)
            {
              gleaf = (host_float_as_int(groupSizeInfo[group].w) == 0xFFFFFFFF);
            }

            if (!gleaf)
            {
              const int childinfoGrp  = ((uint4*)groupSizeInfo)[group].w;
              const int gchild  =   childinfoGrp & 0x0FFFFFFF;
              const int gnchild = ((childinfoGrp & 0xF0000000) >> 28) ;


              //for (int i = gchild; i <= gchild+gnchild; i++) //old tree
              for (int i = gchild; i < gchild+gnchild; i++) //GPU-tree TODO JB: I think this is the correct one, verify in treebuild code
              {
                levelGroups.second().push_back(i);
              }
            }
            else
              levelGroups.second().push_back(group);
          } //if(!lleaf)
          else
          {
            break;
          }
        }
      }//for groupBeg to groupEnd test splits

      real4 size  = nodeSize[nodeIdx];
      int sizew   = 0xFFFFFFFF;

      if (split)
      {
        if (!lleaf)
        {
          const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
          const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has
          sizew = (nExportCellOffset | (lnchild << LEAFBIT));
          nExportCellOffset += lnchild;
          for (int i = lchild; i < lchild + lnchild; i++)
            levelList.second().push_back((uint4){(uint)i,(uint)groupNextBeg,(uint)levelGroups.second().size()});
        }
        else
        {
          const int pfirst =    nodeInfo_y & BODYMASK;
          const int np     = (((nodeInfo_y & INVBMASK) >> LEAFBIT)+1);
          sizew = (nExportPtcl | ((np-1) << LEAFBIT));
          nExportPtcl += np;
        }
      }//if split
      nExportCell++;
    }
    depth++;
    levelList.swap();
    levelList.second().clear();

    levelGroups.swap();
    levelGroups.second().clear();
  }

  double tCalc = get_time2();
  // LOGF(stderr,"getLETOptQuick P: %d N: %d  Calc took: %lg Prepare: %lg Copy: %lg Total: %lg \n",nExportPtcl, nExportCell, tCalc-tStart, tPrep - tStart, tEnd-tCalc, tEnd-tStart);
  //  fprintf(stderr,"[Proc: %d ] getLETOptQuick P: %d N: %d  Calc took: %lg Prepare: %lg (calc: %lg ) Copy: %lg Total: %lg \n",
  //    procId, nExportPtcl, nExportCell, tCalc-tStart, tPrep - tStart, tCalc-tPrep,  tEnd-tCalc, tEnd-tStart);

    return  1 + nExportPtcl + 5*nExportCell;

#if 0 //Disabled tree-copy
  assert((int)bufferStruct.LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)bufferStruct.LETBuffer_node.size() == nExportCell);

  /* now copy data into LETBuffer */
  {
    _v4sf *vLETBuffer;
    {
      const size_t oldSize     = LETBuffer.size();
      const size_t oldCapacity = LETBuffer.capacity();
      LETBuffer.resize(oldSize + 1 + nExportPtcl + 5*nExportCell);
      const size_t newCapacity = LETBuffer.capacity();
      /* make sure memory is not reallocated */
      assert(oldCapacity == newCapacity);

      /* fill tree info */
      real4 &data4 = *(real4*)(&LETBuffer[oldSize]);
      data4.x      = host_int_as_float(nExportPtcl);
      data4.y      = host_int_as_float(nExportCell);
      data4.z      = host_int_as_float(cellBeg);
      data4.w      = host_int_as_float(cellEnd);

      //LOGF(stderr, "LET res for: %d  P: %d  N: %d old: %ld  Size in byte: %d\n",procId, nExportPtcl, nExportCell, oldSize, (int)(( 1 + nExportPtcl + 5*nExportCell)*sizeof(real4)));
      vLETBuffer = (_v4sf*)(&LETBuffer[oldSize+1]);
    }

    int nStoreIdx     = nExportPtcl;
    int multiStoreIdx = nStoreIdx + 2*nExportCell;
    for (int i = 0; i < nExportPtcl; i++)
    {
      const int idx = bufferStruct.LETBuffer_ptcl[i];
      vLETBuffer[i] = bodiesV[idx];
    }
    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = bufferStruct.LETBuffer_node[i];
      const int idx = packed_idx.x;
      const float sizew = host_int_as_float(packed_idx.y);
      const _v4sf size = __builtin_ia32_vec_set_v4sf(nodeSizeV[idx], sizew, 3);

      vLETBuffer[nStoreIdx+nExportCell] = nodeCentreV[idx];     /* centre */
      vLETBuffer[nStoreIdx            ] = size;                 /*  size  */

      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+0];  /* multipole com */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+1];  /* multipole q0 */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+2];  /* multipole q1 */
      nStoreIdx++;
    } //for
  } //now copy data into LETBuffer

  time = get_time2() - tStart;
  double tEnd = get_time2();
#endif
}


/*
 * Create the LET structure for the remote process using the full tree-layout of the remote boundaries
 * where as getLET1 uses only the boundaries and not the tree-structure that comes with it
 *
 * SPH version does additional boundary checks because of smoothing range
 */


int3 getLEToptFullTree(
    GETLETBUFFERS   &bufferStruct,
    real4          **LETBuffer_ptr,
    const real4     *nodeCentre,
    const real4     *nodeSize,
    const real4     *multipole,
    const int        cellBeg,
    const int        cellEnd,
    const real4     *bodies_pos,
    const real4     *bodies_vel,
    const real2     *bodies_dens,
    const real4     *bodies_hydro,
    const int        nParticles,
    const real4     *groupSizeInfo,
    const real4     *groupCentreInfo,
    const int        groupBeg,
    const int        groupEnd,
    const int        nNodes,
    unsigned long long &nflops,
    const float3     periodicDomainSize,
    const int        periodicMethod,
    const int        selectionMethod,
    const int        LETMethod)
{
  bufferStruct.LETBuffer_node.clear();
  bufferStruct.LETBuffer_ptcl.clear();
  bufferStruct.currLevelVecUI4.clear();
  bufferStruct.nextLevelVecUI4.clear();
  bufferStruct.currGroupLevelVec.clear();
  bufferStruct.nextGroupLevelVec.clear();
  bufferStruct.groupSplitFlag.clear();

  nflops = 0;

  int nExportCell       = 0;
  int nExportPtcl       = 0;
  int nExportCellOffset = cellEnd;

  const _v4sf*         bodiesPosV = (const _v4sf*)bodies_pos;
  const _v4sf*         bodiesVelV = (const _v4sf*)bodies_vel;
  const _v4sf*         bodiesHydroV = (const _v4sf*)bodies_hydro;
  const _v4sf*          nodeSizeV = (const _v4sf*)nodeSize;
  const _v4sf*        nodeCentreV = (const _v4sf*)nodeCentre;
  const _v4sf*         multipoleV = (const _v4sf*)multipole;
  const _v4sf*   grpNodeSizeInfoV = (const _v4sf*)groupSizeInfo;
  const _v4sf* grpNodeCenterInfoV = (const _v4sf*)groupCentreInfo;

#ifdef USE_AVX  /* AVX */
    #ifndef __AVX__
        #error "AVX is not defined"
    #endif
    const int SIMDW  = 8;
    _v8sf checkValueSPH01;
#else
    const int SIMDW  = 4; //#define SSEIMBH
#endif


  bool usePeriodic = false;
  int2 xP = {0,0}, yP = {0,0}, zP = {0,0};
  if(periodicMethod & PERIODIC_X) { xP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & PERIODIC_Y) { yP = {-1,1}; usePeriodic = true;}
  if(periodicMethod & PERIODIC_Z) { zP = {-1,1}; usePeriodic = true;}


  Swap<std::vector<uint4> > levelList  (bufferStruct.currLevelVecUI4,   bufferStruct.nextLevelVecUI4);
  Swap<std::vector<int> >   levelGroups(bufferStruct.currGroupLevelVec, bufferStruct.nextGroupLevelVec);

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    bufferStruct.LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});


  /* copy group info into current level buffer */
  for (int group = groupBeg; group < groupEnd; group++)
    levelGroups.first().push_back(group);

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back((uint4){(uint)cell, 0, (uint)levelGroups.first().size(),0});

  int depth = 0;
  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint4 nodePacked = levelList.first()[i];

      const uint  nodeIdx    = nodePacked.x;

      const float2 openings  = extract_opening_criteria(nodeCentre[nodeIdx].w);
      const float nodeInfo_x = openings.x;
      const float nodeSmth   = fabs(openings.y);

      const uint  nodeInfo_y = host_float_as_int(nodeSize[nodeIdx].w);

      const _v4sf nodeCOM  = __builtin_ia32_vec_set_v4sf(multipoleV[nodeIdx*3], nodeInfo_x, 3); //Set opening angle
      const bool lleaf     = nodeInfo_x <= 0.0f;


      //SPH uses physical center and size of the box to test overlap and distance
      const _v4sf nodeSIZE   = nodeSizeV  [nodeIdx];
      const _v4sf nodeCNTR   = nodeCentreV[nodeIdx];

      _v4sf checkValueGrav = {nodeInfo_x, nodeInfo_x, nodeInfo_x, nodeInfo_x};
            checkValueGrav = __abs(checkValueGrav);
      _v4sf checkValueSPH0  = {0,0,0,0};
      _v4sf checkValueSPH1;

      const int groupBeg = nodePacked.y;
      const int groupEnd = nodePacked.z;
      nflops += 20*((groupEnd - groupBeg-1)/SIMDW+1)*SIMDW;

      bufferStruct.groupSplitFlag.clear();
      for (int ib = groupBeg; ib < groupEnd; ib += SIMDW)
      {
        _v4sf centre[SIMDW], size[SIMDW];
        for (int laneIdx = 0; laneIdx < SIMDW; laneIdx++)
        {
          const int group  = levelGroups.first()[std::min(ib+laneIdx, groupEnd-1)];
          centre[laneIdx]  = grpNodeCenterInfoV[group];
          size  [laneIdx]  =   grpNodeSizeInfoV[group];
          if(selectionMethod & SELECT_SPH)
          {
              const float2 openings  = extract_opening_criteria(groupCentreInfo[laneIdx].w);
              float smth   = fabs(openings.y); //This is for density computation, there we do not use the distance of the node

              //Smth is the maximum allowed distance (already squared) before a cell is opened
              if(LETMethod == LET_METHOD_HYDR)
              {
                  smth = std::max(nodeSmth, smth); //For hydro we need mutual forces so take the longest distance
              }

              if(laneIdx == 0) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, smth, 0); //Set smoothing distance
              if(laneIdx == 1) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, smth, 1); //Set smoothing distance
              if(laneIdx == 2) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, smth, 2); //Set smoothing distance
              if(laneIdx == 3) checkValueSPH0 = __builtin_ia32_vec_set_v4sf(checkValueSPH0, smth, 3); //Set smoothing distance
#ifdef USE_AVX
              if(laneIdx == 4) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, smth, 0); //Set smoothing distance
              if(laneIdx == 5) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, smth, 1); //Set smoothing distance
              if(laneIdx == 6) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, smth, 2); //Set smoothing distance
              if(laneIdx == 7) checkValueSPH1 = __builtin_ia32_vec_set_v4sf(checkValueSPH1, smth, 3); //Set smoothing distance
              checkValueSPH01 = pack_2xmm(checkValueSPH0,checkValueSPH1);
#endif
          }
        }


#ifdef USE_AVX
        _v8sf resGrav = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        _v8sf check2 = pack_2xmm(checkValueGrav,checkValueGrav);

        if(selectionMethod & SELECT_GRAV) resGrav = split_node_grav_sph_impbh_box8a<false>(resGrav,check2, nodeCOM,  nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
        if(selectionMethod & SELECT_SPH)  resGrav = split_node_grav_sph_impbh_box8a<true >(resGrav,checkValueSPH01,  nodeCNTR, nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);

        const _v4sf ret1 = __builtin_ia32_vextractf128_ps256(resGrav, 0);
        const _v4sf ret2 = __builtin_ia32_vextractf128_ps256(resGrav, 1);
        bufferStruct.groupSplitFlag.push_back(std::make_pair(ret1,ret2));
#else
          _v4sf resGrav = {0.0f, 0.0f, 0.0f, 0.0f};
          if(selectionMethod & SELECT_GRAV) resGrav = split_node_grav_sph_impbh_box4a<false>(resGrav,checkValueGrav, nodeCOM,  nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
          if(selectionMethod & SELECT_SPH)  resGrav = split_node_grav_sph_impbh_box4a<true >(resGrav,checkValueSPH0,  nodeCNTR, nodeSIZE, centre, size,periodicDomainSize, xP, yP,zP);
          bufferStruct.groupSplitFlag.push_back(resGrav);
#endif
      }

      const int groupNextBeg = levelGroups.second().size();
      int split = false;
      for (int idx = groupBeg; idx < groupEnd; idx++)
      {
        const bool gsplit = ((uint*)&bufferStruct.groupSplitFlag[0])[idx - groupBeg];
        if (gsplit)
        {
          split = true;
          const int group = levelGroups.first()[idx];
          if (!lleaf)
          {
            //Test on if its a leaf and on if it's and end-point
            bool gleaf = groupCentreInfo[group].w <= 0.0f;
            if(!gleaf)
            {
              gleaf = (host_float_as_int(groupSizeInfo[group].w) == 0xFFFFFFFF);
            }

            if (!gleaf)
            {
              const int childinfoGrp  = ((uint4*)groupSizeInfo)[group].w;
              const int gchild  =   childinfoGrp & 0x0FFFFFFF;
              const int gnchild = ((childinfoGrp & 0xF0000000) >> 28) ;

              //for (int i = gchild; i <= gchild+gnchild; i++)
              for (int i = gchild; i < gchild+gnchild; i++)
              {
                levelGroups.second().push_back(i);
              }
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
            levelList.second().push_back((uint4){(uint)i,(uint)groupNextBeg,(uint)levelGroups.second().size()});
        }
        else
        {
          const int pfirst =    nodeInfo_y & BODYMASK;
          const int np     = (((nodeInfo_y & INVBMASK) >> LEAFBIT)+1);
          sizew = (nExportPtcl | ((np-1) << LEAFBIT));
          for (int i = pfirst; i < pfirst+np; i++)
            bufferStruct.LETBuffer_ptcl.push_back(i);
          nExportPtcl += np;
        }
      }

      bufferStruct.LETBuffer_node.push_back((int2){(int)nodeIdx, sizew});
      nExportCell++;
    }

    depth++;

    levelList.swap();
    levelList.second().clear();

    levelGroups.swap();
    levelGroups.second().clear();
  }

  assert((int)bufferStruct.LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)bufferStruct.LETBuffer_node.size() == nExportCell);

  /* now copy data into LETBuffer */
  {

      int bodyPropsCount = 1;                                //Position
      if(LETMethod == LET_METHOD_DRVT)  bodyPropsCount +=1;  //Position and Velocity
      if(LETMethod == LET_METHOD_HYDR)  bodyPropsCount +=3;  //Position, Velocity, Density and Hydro
    //LETBuffer.resize(nExportPtcl + 5*nExportCell);
#pragma omp critical //Malloc seems to be not so thread safe..
    *LETBuffer_ptr      = (real4*)malloc(sizeof(real4)*(1+ bodyPropsCount*nExportPtcl + 5*nExportCell));
    real4 *LETBuffer    = *LETBuffer_ptr;

    real2 *densBuffer   = (real2*)(&LETBuffer[1+2*nExportPtcl]);


    _v4sf *vLETBuffer   = (_v4sf*)(&LETBuffer[1]);


    int nStoreIdx     = bodyPropsCount*nExportPtcl;
    int multiStoreIdx = nStoreIdx + 2*nExportCell;
    for (int i = 0; i < nExportPtcl; i++)
    {
      const int idx = bufferStruct.LETBuffer_ptcl[i];
      vLETBuffer[i] = bodiesPosV[idx];
    }
    if(LETMethod == LET_METHOD_DRVT ||LETMethod == LET_METHOD_HYDR)
    {
        for (int i = 0; i < nExportPtcl; i++)
        {
          const int idx = bufferStruct.LETBuffer_ptcl[i];
          vLETBuffer[nExportPtcl+i] = bodiesVelV[idx];
        }
    }
    if(LETMethod == LET_METHOD_HYDR)
    {

        //Changing this if 1 to if 0 also requires changes in mergeAndLaunch
#if 0
        int storeIdx = 0;
        for(int i=0; i < nExportPtcl; i+=2)
        {
            int idx1             = bufferStruct.LETBuffer_ptcl[i];
            int idx2             = bufferStruct.LETBuffer_ptcl[std::min(i+1, nExportPtcl-1)];

            _v4sf dens = {0.0f,0.0f,0.0f,0.0f};
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx1].x, 0); //Move density to the .x component
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx1].y, 1); //Move smth to the .y component
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx2].x, 2); //Move density to the .z component
            dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx2].y, 3); //Move smth to the .w component
            vLETBuffer[2*nExportPtcl+storeIdx] = dens;
            storeIdx++;
        }
#endif

        for (int i = 0; i < nExportPtcl; i++)
        {
          const int idx             = bufferStruct.LETBuffer_ptcl[i];

          //Uncomment this if the above #if 1 is set to #if 0
          _v4sf dens = {0.0f,0.0f,0.0f,0.0f};
          dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx].x, 0); //Move density to the .x component
          dens = __builtin_ia32_vec_set_v4sf(dens, bodies_dens[idx].y, 1); //Move density to the .y component
          vLETBuffer[2*nExportPtcl+i] = dens;


          vLETBuffer[3*nExportPtcl+i] = bodiesHydroV[idx];
        }
    }

    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = bufferStruct.LETBuffer_node[i];
      const int idx         = packed_idx.x;
      const float sizew     = host_int_as_float(packed_idx.y);
      const _v4sf size      = __builtin_ia32_vec_set_v4sf(nodeSizeV[idx], sizew, 3);


      vLETBuffer[nStoreIdx+nExportCell] = nodeCentreV[idx];     /* centre */
      vLETBuffer[nStoreIdx            ] = size;                 /*  size  */

      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+0];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+1];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+2];  /* multipole.x */
      nStoreIdx++;
    }
  }

  return (int3){nExportCell, nExportPtcl, depth};
}


#endif //USE MPI



void octree::checkGPUAndStartLETComputation(tree_structure &tree,
                                            tree_structure &remote,
                                            int            &topNodeOnTheFlyCount,
                                            int            &nReceived,
                                            int            &procTrees,
                                            double         &tStart,
                                            double         &totalLETExTime,
                                            bool            mergeOwntree,
                                            int            *treeBuffersSource,
                                            real4         **treeBuffers,
                                            const  int      LETMethod)
{
  //This determines if we interrupt the LET computation by starting a gravity kernel on the GPU
  //if(gravStream->isFinished())
    if(gravStream->isFinished() || (nReceived == nProcs -1))
  {
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
          topNodeCount, recvTree, mergeOwntree, procTrees, tStart, LETMethod);
      LOGF(stderr, "Merging and launchingA iter: %d took: %lg \n", iter, get_time()-t000);

      //Correct the topNodeOnTheFlyCounter
  #pragma omp critical(updateReceivedProcessed)
      {
        //Compute how many are left, and add these back to the globalCounter
        topNodeOnTheFlyCount += (oriTopCount-topNodeCount);
      }

      totalLETExTime += thisPartLETExTime;
    }// if (nReceived - procTrees) > 0)
  }// if isFinished
}



struct letObject
{
  real4       *buffer;
  int          size;
  int          destination;
#ifdef USE_MPI
  MPI_Request  req;
#endif
};


void octree::essential_tree_exchangeV2(tree_structure &tree,
                                       tree_structure &remote,
                                       vector<real4>  &topLevelTrees,
                                       vector<uint2>  &topLevelTreesSizeOffset,
                                       int             nTopLevelTrees,
                                       const    int    LETMethod)
{
#ifdef USE_MPI

  devContext->pushNVTX("essential_tree_exchangeV2");


  //This should be moved to where we actually need it
  localTree.boxSizeInfo.waitForCopyEvent();
  localTree.boxCenterInfo.waitForCopyEvent();
  localTree.boxSmoothing.waitForCopyEvent();
  localTree.multipole.waitForCopyEvent();


  const float3 periodicDomainSize =  {6.8593750000000000, 4.0594940802395563E-002f, 3.8273277230987154E-002f};   //Hardcoded for phantom tube
//  const float3 periodicDomainSize = {1.0f, 0.125f, 0.125f};   //Hardcoded for our testing IC
//  const float3 periodicDomainSize = {100.0f, 100.0f, 100.0f};   //Hardcoded for our testing IC
  const int    periodicMethod     = PERIODIC_X | PERIODIC_Y | PERIODIC_Z;
  const int    selectionMethod    = SELECT_SPH;//GRAV;
  const int    nBodyProps         = 4; //Position, velocity, density and hydro props per particle

  double t0         = get_time();

  double tStatsStartUpStart = get_time(); //TODO DELETE

  bool mergeOwntree = false;              //Default do not include our own tree-structure, thats mainly used for testing
  int procTrees     = 0;                  //Number of trees that we've received and processed

  real4  *bodies_pos          = &tree.bodies_Ppos[0];
  real4  *bodies_vel          = &tree.bodies_Pvel[0];
  real2  *bodies_dens         = &tree.bodies_dens[0];
  real4  *bodies_hydro        = &tree.bodies_hydro[0];
  real4  *multipole           = &tree.multipole[0];
  real4  *nodeSizeInfo        = &tree.boxSizeInfo[0];
  real4  *nodeCenterInfo      = &tree.boxCenterInfo[0];

  real4 **treeBuffers;

  //creates a new array of pointers to int objects, with space for the local tree
  treeBuffers            = new real4*[mpiGetNProcs()];
  int *treeBuffersSource = new int[nProcs];

  //Timers for the LET Exchange
  static double totalLETExTime    = 0;
  thisPartLETExTime               = 0;
  double tStart                   = get_time();


  int topNodeOnTheFlyCount = 0;

  this->fullGrpAndLETRequestStatistics[procId] = make_int2(0, 0); //Reset our box
  //For the first 8 iterations mark boundary as used
  for(int i=0; i < nProcs; i++)
  {
    if(iter < 8)
      this->fullGrpAndLETRequestStatistics[i] = make_int2(1, 1);
  }

  uint2 node_begend;
  node_begend.x   = tree.level_list[tree.startLevelMin].x;
  node_begend.y   = tree.level_list[tree.startLevelMin].y;

  int resultOfQuickCheck[nProcs];

  int4 quickCheckSendSizes [nProcs];
  int  quickCheckSendOffset[nProcs];

  int4 quickCheckRecvSizes [nProcs];
  int quickCheckRecvOffset[nProcs];


  int nCompletedQuickCheck      = 0;

  resultOfQuickCheck[procId]    = 99; //Mark ourself
  quickCheckSendSizes[procId].x =  0;
  quickCheckSendSizes[procId].y =  0;
  quickCheckSendSizes[procId].z =  0;
  quickCheckSendOffset[procId]  =  0;

  //For statistics
  int nQuickCheckSends          = 0;
  int nQuickCheckRealSends      = 0;
  int nQuickCheckReceives       = 0;
  int nQuickBoundaryOk          = 0;


  omp_set_num_threads(8); //8 Piz-Daint, 16 Titan

  letObject *computedLETs = new letObject[nProcs-1];

  int omp_ticket      = 0;
  int omp_ticket2     = 0;
  int omp_ticket3     = 0;
  int nComputedLETs   = 0;
  int nReceived       = 0;
  int nSendOut        = 0;
  int nToSend	      = 0;

  //Use getLETQuick instead of recursiveTopLevelCheck
  //#define doGETLETQUICK


  const int NCELLMAX  = 1024;
  const int NDEPTHMAX = 30;
  const int NPROCMAX  = 32768;
  assert(nProcs <= NPROCMAX);


  const static int MAX_THREAD = 64;
  assert(MAX_THREAD >= omp_get_num_threads());
  static __attribute__(( aligned(64) )) GETLETBUFFERS getLETBuffers[MAX_THREAD];


//  static std::vector<v4sf> quickCheckData[NPROCMAX];


  std::vector<int> communicationStatus(nProcs);
  for(int i=0; i < nProcs; i++) communicationStatus[i] = 0;


  double tStatsStartUpEnd = get_time();


  //TODO DELETE
  double tX1, tXA, tXB, tXC, tXD, tXE, tYA, tYB, tYC;
  double ZA1, tXC2, tXD2;
  double tA1 = 0, tA2 = 0, tA3 = 0, tA4, tXD3;
  int nQuickRecv = 0;

  double tStatsStartLoop = get_time(); //TODO DELETE


  double tStatsEndQuickCheck, tStatsEndWaitOnQuickCheck;
  double tStatsEndAlltoAll, tStatsEndGetLET;
  double tStartsEndGetLETSend;
  double tStatsStartAlltoAll, tStartsStartGetLETSend;


  int receivedLETCount = 0;
  int expectedLETCount = 0;
  int nBoundaryOk      = 0;

  std::vector<int>   requiresFullLET;              //Build from quick-check results
  std::vector<int>   requiresFullLETExtra;         //Build from received boundary status info.
                                                   //contains IDs that are not in requiresFullLET, but are in
                                                   //list of IDs for which boundary is not good enough
  std::vector<int>   idsThatNeedExtraLET;          //Processes on this list need getLET data
  std::vector<int>   idsThatNeedMoreThanBoundary;  //Processes on this list need getLET data

  requiresFullLET.reserve(nProcs);
  idsThatNeedMoreThanBoundary.reserve(nProcs);
  int requiresFullLETCount = 0;

  bool completedA2A = false; //Barrier for the getLET threads

  //Use multiple OpenMP threads in parallel to build and exchange LETs
#pragma omp parallel
  {
    int tid      = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    if(tid != 1) //Thread 0, does LET creation and GPU control, Thread == 1 does MPI communication, all others do LET creation
    {
      int DistanceCheck = 0;
      double tGrpTest = get_time();

      const int allocSize = (int)(tree.n_nodes*1.10);

      //Resize the buffers
      getLETBuffers[tid].currLevelVecI.reserve(allocSize);
      getLETBuffers[tid].nextLevelVecI.reserve(allocSize);
      getLETBuffers[tid].currLevelVecUI4.reserve(allocSize);
      getLETBuffers[tid].nextLevelVecUI4.reserve(allocSize);
      getLETBuffers[tid].currGroupLevelVec.reserve(allocSize);
      getLETBuffers[tid].nextGroupLevelVec.reserve(allocSize);
      getLETBuffers[tid].groupSplitFlag.reserve(allocSize);


      if(tid == 0) devContext->pushNVTX("StartCheckingTheBoundaries");

      while(true) //Continue until everything is computed
      {
        int currentTicket = 0;

        //Check if we can start some GPU work
        if(tid == 0) //Check if GPU is free
        {
          if(omp_ticket > (nProcs - 1))
          {
            checkGPUAndStartLETComputation(tree, remote, topNodeOnTheFlyCount,
                                           nReceived, procTrees,  tStart, totalLETExTime,
                                           mergeOwntree,  treeBuffersSource, treeBuffers, LETMethod);
          }
        }//tid == 0

        #pragma omp critical
          currentTicket = omp_ticket++; //Get a unique ticket to determine which process to build the LET for

        if(currentTicket >= (nProcs-1)) //Break out if we processed all nodes
          break;

        bool doQuickLETCheck = (currentTicket < (nProcs - 1));
        int ib               = (nProcs-1)-(currentTicket%nProcs);
        int ibox             = (ib+procId)%nProcs; //index to send...
        //Above could be replaced by a priority list, based on previous
        //loops (eg nearest neighbours first)


        //Group info for this process
        int idx          =   globalGrpTreeOffsets[ibox];
        real4 *grpCenter =  &globalGrpTreeCntSize[idx];


        if(doQuickLETCheck) //Perform the quick-check tests
        {
            unsigned long long nflops;


            int nbody = host_float_as_int(grpCenter[0].x);
            int nnode = std::abs(host_float_as_int(grpCenter[0].y));

            real4 *grpSize2   = &grpCenter[1+nBodyProps*nbody];
            real4 *grpCenter2 = &grpCenter[1+nBodyProps*nbody+nnode];
//            real4 *grpSmth    = &grpCenter[1+nBodyProps*nbody+nnode+nnode];


            //The return value sizeTree is used to determine for which processes we have to make
            //proper LETs without having to wait on communication.
            //If the tree becomes too big or too deep then we will do direct LET for sure, otherwise
            //it depends on what the other side tells us.

            //Test if we have to send data to the remote process?
            //Returns -1 if there is not enough data and we have to generate a LET, otherwise all is ok?
            double bla3;
            int sizeTree =  getLEToptQuickFullTree(
                                            getLETBuffers[tid],
                                            256, //NCELLMAX,
                                            NDEPTHMAX,
                                            &nodeCenterInfo[0],
                                            &nodeSizeInfo[0],
                                            &multipole[0],
                                            0,                //Cell begin
                                            1,                //Cell end
                                            &bodies_pos[0],
                                            tree.n,
                                            grpSize2,         //size
                                            grpCenter2,       //center
                                            0,                //group begin
                                            1,                //group end
                                            tree.n_nodes,
                                            procId, ibox,
                                            nflops, bla3,
                                            periodicDomainSize, periodicMethod,
                                            selectionMethod,
                                            LETMethod);

            //Test if the boundary tree sent by the remote tree is sufficient for us
            double tBoundaryCheck;
            int depthSearch = 0;
            const int resultTree = getLEToptQuickTreevsTree(
                                              getLETBuffers[tid],
                                              &grpCenter[1+nBodyProps*nbody+nnode],    //center
                                              &grpCenter[1+nBodyProps*nbody],          //size
                                              &grpCenter[1+nBodyProps*nbody+nnode*3],  //multipole
                                              0, 1,                         //Start at the root of remote boundary tree
                                              &nodeSizeInfo[0],             //Local tree-sizes
                                              &nodeCenterInfo[0],           //Local tree-centers
                                              0, 1,                         //start at the root of local tree
                                              nnode,
                                              procId,
                                              ibox,
                                              tBoundaryCheck, depthSearch,
                                              periodicDomainSize, periodicMethod,
                                              selectionMethod,
                                              LETMethod);
//            int resultTree = 1; //TODO remove once we updated the boundary tree to contain velocity information

            if(resultTree == 0)
            {
              //We can use this tree to compute gravity, no further info needed of the remote domain
              #pragma omp critical
              {
                //Add the boundary as a LET tree
                treeBuffers[nReceived] = &grpCenter[0];

                //Make the #nodes negative to indicate that this tree is sourced from a boundary
//                treeBuffers[nReceived][0].y = std::abs(host_int_as_float(-nnode));

                //Increase the top-node count
                int topStart = host_float_as_int(treeBuffers[nReceived][0].z);
                int topEnd   = host_float_as_int(treeBuffers[nReceived][0].w);

                topNodeOnTheFlyCount        += (topEnd-topStart);
                treeBuffersSource[nReceived] = 2; //2 indicate quick boundary check source
                nReceived++;
                nBoundaryOk++;

                communicationStatus[ibox] = 2;    //2 Indicate we used the boundary
              }//omp critical

              quickCheckSendSizes[ibox].y = 1;    //1 To indicate we used this processes boundary
              quickCheckSendSizes[ibox].z = depthSearch;
            }//resultTree == 0
            else
            {
              quickCheckSendSizes[ibox].y = 0; //0 to indicate we do not use this processes boundary
            }

            if (sizeTree != -1)
            {
              quickCheckSendSizes[ibox].x = sizeTree;
              resultOfQuickCheck [ibox]   = 1;
            }
            else
            { //Quickcheck failed, requires point to point LET
              quickCheckSendSizes[ibox].x =  0;
              resultOfQuickCheck [ibox]   = -1;

              #pragma omp critical
              {
                requiresFullLET.push_back(ibox);
                requiresFullLETCount++;
              }
            } //if (sizeTree != -1)

            #pragma omp critical
              nCompletedQuickCheck++;

        } //if(doQuickLETCheck)
      } //end while, this part does the quickListCreation

      //Only continue if all quickChecks are done, otherwise some threads might still be
      //executing the quick-check! Wait till nCompletedQuickCheck equals number of checks to be done
      while(1)
      {
        if(nCompletedQuickCheck == nProcs-1)  break;
        usleep(10);
      }
      if(tid == 2) tStatsEndQuickCheck = get_time();

      if(tid == 0)
      {
          devContext->popNVTX();
          devContext->pushNVTX("StartLETExtraction");
      }

      while(1)
      {
        if(tid == 0)
        {
          checkGPUAndStartLETComputation(tree, remote, topNodeOnTheFlyCount,
                                         nReceived, procTrees,  tStart, totalLETExTime,
                                         mergeOwntree,  treeBuffersSource, treeBuffers,
                                         LETMethod);
        }//tid == 0

        bool breakOutOfFullLoop = false;

        int ibox          = 0;
        int currentTicket = 0;

        #pragma omp critical
                currentTicket = omp_ticket2++; //Get a unique ticket to determine which process to build the LET for

        if(currentTicket >= requiresFullLET.size())
        {
          //We processed the nodes we identified ourself using quickLET, next we
          //continue with the LETs that we need to do after the A2A.

          while(1)
          { //Wait till the A2a communication is complete
            if(completedA2A == true)  break;
            usleep(10);
          }


          #pragma omp critical
                  currentTicket = omp_ticket3++; //Get a unique ticket to determine which process to build the LET for

          if(currentTicket >= idsThatNeedMoreThanBoundary.size())
            breakOutOfFullLoop = true;
          else
            ibox = idsThatNeedMoreThanBoundary[currentTicket]; //From the A2A result list
        }
        else
        {
          ibox = requiresFullLET[currentTicket];             //From the quickTest result list
        }

        //Jump out of the LET creation while
        if(breakOutOfFullLoop == true) break;

        //Group info for this process
        int idx          =   globalGrpTreeOffsets[ibox];
        real4 *grpCenter =  &globalGrpTreeCntSize[idx];

        //Start and endGrp, only used when not using a tree-structure for the groups
        int startGrp = 0;
        int endGrp   = this->globalGrpTreeCount[ibox] / 2;  //This is an upper bound for the reserve calls and has nothing to do with # of groups.

        int countNodes = 0, countParticles = 0;

        double tz = get_time();
        real4   *LETDataBuffer;
        unsigned long long int nflops = 0;

        double tStartEx = get_time();

        devContext->pushNVTX("CreatingLetForProcP1");

        //Extract the boundaries from the tree-structure. This reduces the boundaries from a tree to a list of boxes
        #ifdef USE_GROUP_TREE
          std::vector<float4> boundaryCentres;
          std::vector<float4> boundarySizes;
          std::vector<float4> boundarySmoothing;

          boundarySizes.reserve(endGrp);
          boundaryCentres.reserve(endGrp);
          boundarySmoothing.reserve(endGrp);
          boundarySizes.clear();
          boundaryCentres.clear();

          int nbody = host_float_as_int(grpCenter[0].x);
          int nnode = host_float_as_int(grpCenter[0].y);

          real4 * grpSize   = &grpCenter[1+nBodyProps*nbody];
          real4 * grpSmooth = &grpCenter[1+nBodyProps*nbody+nnode*2];
          grpCenter         = &grpCenter[1+nBodyProps*nbody+nnode];

          for(int startSearch=0; startSearch < nnode; startSearch++)
          {
            //Two tests, if its a  leaf, and/or if its a node and marked as end-point
            if((host_float_as_int(grpSize[startSearch].w) == 0xFFFFFFFF) || grpCenter[startSearch].w <= 0) //Tree extract
            {
              //For SPH it might be useful to filter out boxes that fall outside our domain, before we start constructing
              //the LET tree. For gravity this is more tricky so only test this for SPH.

              bool useGroup = true;

#if 0
              Does not work when using periodic boundaries
              if(selectionMethod == SELECT_SPH)
              {
                  //Compute minimum distance and test if this is smaller than the softening radius
                  float4 groupCenter = grpCenter[startSearch];
                  float4 groupSize   = grpSize  [startSearch];
                  float4 nodeCenter  = nodeCenterInfo[0];
                  float4 nodeSize    = nodeSizeInfo[0];

                  //Compute the distance between the group and the root node
                  float3 dr = {fabs(groupCenter.x - nodeCenter.x) - (groupSize.x + nodeSize.x),
                               fabs(groupCenter.y - nodeCenter.y) - (groupSize.y + nodeSize.y),
                               fabs(groupCenter.z - nodeCenter.z) - (groupSize.z + nodeSize.z)};

                  dr.x += fabs(dr.x); dr.x *= 0.5f;
                  dr.y += fabs(dr.y); dr.y *= 0.5f;
                  dr.z += fabs(dr.z); dr.z *= 0.5f;
                  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                  useGroup = (ds2 <= grpSmooth[startSearch].x);
              }
#endif
              if(useGroup)
              {
                  boundarySizes.    push_back(grpSize  [startSearch]);
                  boundaryCentres.  push_back(grpCenter[startSearch]);
                  boundarySmoothing.push_back(grpSmooth[startSearch]);
              }
            }
          }//end for

          endGrp    = boundarySizes.size();
          grpCenter = &boundaryCentres  [0];
          grpSize   = &boundarySizes    [0];
          grpSmooth = &boundarySmoothing[0];
        #endif


       //Optionally extend the list of boundaries, use: extendGroupsWithPeriodicGroups()

        double tEndEx = get_time();
        devContext->popNVTX();
        devContext->pushNVTX("CreatingLetForProcP2");

        int2 usedStartEndNode = {(int)node_begend.x, (int)node_begend.y};

#if 0
        tz = get_time();
        assert(startGrp == 0);
        int3  nExport = getLET1(getLETBuffers[tid],
                                &LETDataBuffer,
                                &nodeCenterInfo[0],
                                &nodeSizeInfo[0],
                                &multipole[0],
                                usedStartEndNode.x, usedStartEndNode.y,
                                &bodies_pos[0],
                                &bodies_vel[0],
                                &bodies_dens[0],
                                &bodies_hydro[0],
                                tree.n,
                                grpSize, grpCenter,
                                endGrp,
                                tree.n_nodes, nflops,
                                periodicDomainSize, periodicMethod,
                                selectionMethod,
                                LETMethod);
#endif


#if 1
        real4 *grpCenter2    =  &globalGrpTreeCntSize[globalGrpTreeOffsets[ibox]];
        int start            = host_float_as_int(grpCenter2[0].z);
        int end              = host_float_as_int(grpCenter2[0].w);
            nbody            = host_float_as_int(grpCenter2[0].x);
            nnode            = host_float_as_int(grpCenter2[0].y);

        double tJB = get_time();

        tz = get_time();
        assert(startGrp == 0);


        grpSize    = &grpCenter2[1+nBodyProps*nbody];
        grpSmooth  = &grpCenter2[1+nBodyProps*nbody+nnode*2];
        grpCenter2 = &grpCenter2[1+nBodyProps*nbody+nnode];

        int3 nExport = getLEToptFullTree(getLETBuffers[tid],
                                         &LETDataBuffer,
                                         &nodeCenterInfo[0],
                                         &nodeSizeInfo[0],
                                         &multipole[0],
                                         usedStartEndNode.x, usedStartEndNode.y,
                                         &bodies_pos[0],
                                         &bodies_vel[0],
                                         &bodies_dens[0],
                                         &bodies_hydro[0],
                                         tree.n,
                                         grpSize, grpCenter2,
                                         start, end, tree.n_nodes, nflops,
                                         periodicDomainSize, periodicMethod,
                                         selectionMethod,
                                         LETMethod);
#endif


        int bodyPropsCount = 1;
        if(LETMethod == LET_METHOD_DRVT) bodyPropsCount +=1;  //Velocity included
        if(LETMethod == LET_METHOD_HYDR) bodyPropsCount +=3;  //Velocity, Density and Hydro included

        countParticles  = nExport.y;
        countNodes      = nExport.x;
        int bufferSize  = 1 + bodyPropsCount*countParticles + 5*countNodes;
        //Use count of exported particles and nodes, but let particles count more heavy.
        //Used during particle exchange / domain update to speedup particle-box assignment
//        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);
//        if (ENABLE_RUNTIME_LOG)
        if(1)
        {
          fprintf(stderr,"Proc: %d LET getLetOp count&fill [%d,%d]: Depth: %d Dest: %d Total : %lg (#P: %d \t#N: %d) nBody= %d nNodes= %d  nGroups= %d \tsince start: %lg size: %d\n",
                          procId, procId, tid, nExport.z, ibox, get_time()-tz,countParticles,
                          countNodes, tree.n, tree.n_nodes, endGrp, get_time()-t0, bufferSize);
        }

        //Set the tree properties, before we exchange the data
        LETDataBuffer[0].x = host_int_as_float(countParticles);         //Number of particles in the LET
        LETDataBuffer[0].y = host_int_as_float(countNodes);             //Number of nodes     in the LET
        LETDataBuffer[0].z = host_int_as_float(usedStartEndNode.x);     //First node on the level that indicates the start of the tree walk
        LETDataBuffer[0].w = host_int_as_float(usedStartEndNode.y);     //last  node on the level that indicates the start of the tree walk


        //In a critical section to prevent multiple threads writing to the same location
        #pragma omp critical
        {
          computedLETs[nComputedLETs].buffer      = LETDataBuffer;
          computedLETs[nComputedLETs].destination = ibox;
          computedLETs[nComputedLETs].size        = sizeof(real4)*bufferSize;
          nComputedLETs++;
        }



        devContext->popNVTX();

        if(tid == 0)
        {
          checkGPUAndStartLETComputation(tree, remote, topNodeOnTheFlyCount,
                                         nReceived, procTrees,  tStart, totalLETExTime,
                                         mergeOwntree,  treeBuffersSource, treeBuffers,
                                         LETMethod);
        }//tid == 0
      }//end while that surrounds LET computations


      //ALL LET-trees are built and sent to remote domains/MPI thread

      if(tid != 0) tStatsEndGetLET = get_time(); //TODO delete

      if(tid == 0) devContext->popNVTX();


      //All data that has to be send out is computed
      if(tid == 0)
      {
        devContext->pushNVTX("WaitOnAndProcessIncommingData");
        //Thread 0 starts the GPU work so it stays alive until that is complete
        while(procTrees != nProcs-1) //Exit when everything is processed
        {
          bool startGrav = false;

          //Indicates that we have received all there is to receive
          if(nReceived == nProcs-1)       startGrav = true;
          //Only start if there actually is new data
          if((nReceived - procTrees) > 0) startGrav = true;

          if(startGrav) //Only start if there is new data
          {
            checkGPUAndStartLETComputation(tree, remote, topNodeOnTheFlyCount,
                                           nReceived, procTrees,  tStart, totalLETExTime,
                                           mergeOwntree,  treeBuffersSource, treeBuffers,
                                           LETMethod);
          }
          else //if startGrav
          {
            usleep(10);
          }//if startGrav
        }//while 1

        devContext->popNVTX(); //WaitOnAndProcessIncommingData
      }//if tid==0

    }//if tid != 1
    else if(tid == 1)
    {
      //MPI communication thread

      //Do nothing until we are finished with the quickLet/boundary-test computation
      while(1)
      {
        if(nCompletedQuickCheck == nProcs-1)
          break;
        usleep(10);
      }

      tStatsEndWaitOnQuickCheck = get_time();
      mpiSync(); //TODO DELETE
      tStatsStartAlltoAll = get_time();



      //Send the sizes
      LOGF(stderr, "Going to do the alltoall size communication! Iter: %d Since begin: %lg \n", iter, get_time()-tStart);
      double t100 = get_time();
      MPI_Alltoall(quickCheckSendSizes, 4, MPI_INT, quickCheckRecvSizes, 4, MPI_INT, mpiCommWorld);
      LOGF(stderr, "Completed_alltoall size communication! Iter: %d Took: %lg ( %lg )\n", iter, get_time()-t100, get_time()-t0);

      //If quickCheckRecvSizes[].y == 1 then the remote process used the boundary.
      //do not send our quickCheck result!
      int recvCountItems = 0;
      for (int i = 0; i < nProcs; i++)
      {
        //Did the remote process use the boundaries, if so do not send LET data
        if(quickCheckRecvSizes[i].y == 1)
        { //Clear the size/data
          quickCheckSendSizes[i].x = 0;
          quickCheckSendOffset[i] = 0;
          nQuickBoundaryOk++;

          //Mark as we can use small boundary
          //this->fullGrpAndLETRequestStatistics[i] = make_uint2(1, 1);
        }
        else
        {
          //Did not use boundary, mark that for next run, so it sends full boundary
          //if(iter  < 16) //TODO this stops updating this list after iteration 16, make dynamic
	        //  this->fullGrpAndLETRequestStatistics[i] = make_uint2(0, 0);
          this->fullGrpAndLETRequestStatistics[i] = make_int2(-1, -1);

          if(i != procId) idsThatNeedExtraLET.push_back(i);
        }

     /*   LOGF(stderr,"A2A data: %d %d  | %d %d | %d\n",
            quickCheckRecvSizes[i].x, quickCheckRecvSizes[i].y,
            quickCheckSendSizes[i].x, quickCheckSendSizes[i].y,
            expectedLETCount);*/

        if(quickCheckRecvSizes[i].x == 0 || quickCheckSendSizes[i].y == 0)
          expectedLETCount++; //Increase the number of incoming trees


        //Did we use the boundary of that tree, if so it should not send us anything
        if(quickCheckSendSizes[i].y == 1)
        {
          quickCheckRecvSizes[i].x = 0;
        }


        quickCheckRecvOffset[i]   = recvCountItems*sizeof(real4);
        recvCountItems           += quickCheckRecvSizes[i].x;
        quickCheckRecvSizes[i].x  = quickCheckRecvSizes[i].x*sizeof(real4);
      } //for i




      expectedLETCount -= 1; //Don't count ourself

      nQuickCheckSends = nProcs-idsThatNeedExtraLET.size()-1;

      for(unsigned int i=0; i < idsThatNeedExtraLET.size(); i++)
      {
        int boxID = idsThatNeedExtraLET[i];

        //Check if this process is already on our list of processes that require extra data
         if(resultOfQuickCheck[boxID] != -1) idsThatNeedMoreThanBoundary.push_back(boxID);
      }

      completedA2A = true;
      LOGF(stderr,"Proc: %d Has to proces additional lets: %ld Already did: %d Used bound: %d\n",
          procId,idsThatNeedMoreThanBoundary.size(), requiresFullLETCount, nQuickBoundaryOk);

      nToSend = idsThatNeedMoreThanBoundary.size() + requiresFullLETCount;


      tStatsEndAlltoAll = get_time();

      LOGF(stderr, "Received trees using alltoall: %d qRecvSum %d  top-nodes: %d Send with alltoall: %d qSndSum: %d \tnBoundary: %d\n",
                    nQuickCheckReceives, nReceived, topNodeOnTheFlyCount,
                    nQuickCheckRealSends, nQuickCheckRealSends+nQuickBoundaryOk,nBoundaryOk);

      tStartsStartGetLETSend = get_time();
      while(1)
      {
        bool sleepAtTheEnd = true;  //Will be set to false if we did anything in here. If true we wait a bit

        //Send out individual LETs that are computed and ready to be send
        int tempComputed = nComputedLETs;

        if(tempComputed > nSendOut)
        {
          sleepAtTheEnd = false;
          for(int i=nSendOut; i < tempComputed; i++)
          {
            MPI_Isend(&(computedLETs[i].buffer)[0],computedLETs[i].size,
                MPI_BYTE, computedLETs[i].destination, 999,
                mpiCommWorld, &(computedLETs[i].req));
          }
          nSendOut = tempComputed;
        }

        //Receiving
        MPI_Status probeStatus;
        MPI_Status recvStatus;
        int flag  = 0;

        do
        {
          MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, mpiCommWorld, &flag, &probeStatus);

          if(flag)
          {
            sleepAtTheEnd = false;  //We do something here
            int count;
            MPI_Get_count(&probeStatus, MPI_BYTE, &count);

            double tY = get_time();
            real4 *recvDataBuffer = new real4[count / sizeof(real4)];
            double tZ = get_time();
            MPI_Recv(&recvDataBuffer[0], count, MPI_BYTE, probeStatus.MPI_SOURCE, probeStatus.MPI_TAG, mpiCommWorld,&recvStatus);

            LOGF(stderr, "Receive complete from: %d  || recvTree: %d since start: %lg ( %lg ) alloc: %lg Recv: %lg Size: %d\n",
                          recvStatus.MPI_SOURCE, 0, get_time()-tStart,get_time()-t0,tZ-tY, get_time()-tZ, count);

            receivedLETCount++;

//            this->fullGrpAndLETRequestStatistics[probeStatus.MPI_SOURCE] = make_uint2(0, 0);

            if( communicationStatus[probeStatus.MPI_SOURCE] == 2)
            {
              //We already used the boundary for this remote process, so don't use the custom tree
              delete[] recvDataBuffer;

              fprintf(stderr,"Proc: %d , Iter: %d we received UNNEEDED LET data from proc: %d \n", procId,iter,probeStatus.MPI_SOURCE );
            }
            else
            {
              treeBuffers[nReceived]       = recvDataBuffer;
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
            }

            flag = 0;
          }//if flag
        }while(flag); //TODO, if we reset flag after do, we keep receiving untill we emptied waiting list



//        LOGF(stderr,"TEST %d == %d ||  %d+%d == %d || %d == %d  || %d == %d \n",
//            nReceived, nProcs-1,
//            nSendOut,nQuickCheckSends,nProcs-1,
//            receivedLETCount,expectedLETCount, nSendOut, nToSend);

        //Exit if we have send and received all there is
        if(nReceived == nProcs-1)                    //if we received data for all processes
          if((nSendOut == nToSend))                  //If we sent out all the LETs we need to send
            if(receivedLETCount == expectedLETCount) //If we received all LETS that we expect, which
              break;                                 //can be more than nReceived if we get double data

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

        if(sleepAtTheEnd)   usleep(10); //Only sleep when we did not send or receive anything
      } //while (1) surrounding the thread-id==1 code

      //Wait till all outgoing sends have been completed
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
      tStartsEndGetLETSend = get_time();
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

  char buff5[1024];
  sprintf(buff5,"LETTIME-%d: tInitLETEx: %lg tQuickCheck: %lg tQuickCheckWait: %lg tGetLET: %lg \
tAlltoAll: %lg tGetLETSend: %lg tTotal: %lg mbSize-a2a: %f nA2AQsend: %d nA2AQrecv: %d nBoundRemote: %d nBoundLocal: %d\n",
     procId,
     tStatsStartUpEnd-tStatsStartUpStart, tStatsEndQuickCheck-tStatsStartUpEnd,
     tStatsEndWaitOnQuickCheck-tStatsStartUpEnd, tStatsEndGetLET-tStatsEndQuickCheck,
     tStatsEndAlltoAll-tStatsStartAlltoAll, tStartsEndGetLETSend-tStartsStartGetLETSend,
     get_time()-tStatsStartUpStart,
     ZA1, nQuickCheckRealSends, nQuickCheckReceives, nQuickBoundaryOk, nBoundaryOk);
     //ZA1, nQuickCheckSends, nQuickRecv, nBoundaryOk);
   devContext->writeLogEvent(buff5); //TODO DELETE

//  if(recvAllToAllBuffer) delete[] recvAllToAllBuffer;
  delete[] treeBuffersSource;
  delete[] computedLETs;
  delete[] treeBuffers;
  LOGF(stderr,"LET Creation and Exchanging time [%d] curStep: %g\t   Total: %g  Full-step: %lg  since last start: %lg\n", procId, thisPartLETExTime, totalLETExTime, get_time()-t0, get_time()-tStart);

  devContext->popNVTX();

#endif
}//essential tree-exchange


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


void octree::mergeAndLaunchLETStructures(
    tree_structure &tree, tree_structure &remote,
    real4 **treeBuffers, int *treeBuffersSource,
    int &topNodeOnTheFlyCount,
    int &recvTree, bool &mergeOwntree, int &procTrees, double &tStart,
    const int METHOD)
{
  //Now we have to merge the separate tree-structures into one big-tree

   devContext->pushNVTX("mergeAndLaunchLETStructures");

  int PROCS  = recvTree-procTrees;

  double t0 = get_time();

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

  int bodyPropsCount = 1;                            //Position
  if(METHOD == LET_METHOD_DRVT) bodyPropsCount +=1;  //Position and Velocity
  if(METHOD == LET_METHOD_HYDR) bodyPropsCount +=3;  //Position, Velocity, Dens and Hydro TODO


  int nParticlesCounted   = 0;
  int nNodesCounted       = 0;
  int nProcsProcessed     = 0;
  bool continueProcessing = true;

  //Calculate the offsets
  for(int i=0; i < PROCS ; i++)
  {
    int particles = host_float_as_int(treeBuffers[procTrees+i][0].x);
    int nodes     = host_float_as_int(treeBuffers[procTrees+i][0].y);

    int bodyPropsCountSource = bodyPropsCount;
    if(treeBuffersSource[procTrees+i] == 2) bodyPropsCountSource = 4; //Boundary tree is a full tree with all particle properties

    nParticlesCounted += particles;
    nNodesCounted     += nodes;

    //Check if we go over the limit, if so, we have two options:
    // - Ignore this last one, if we have processed nodes before (nProcsProcessed > 0)
    // - Process this one anyway and hope we have enough memory, do this if nProcsProcessed == 0
    //   otherwise we would make no progress

    int localLimit   =  tree.n                           + 5*tree.n_nodes; //TODO should this include the bodyPropsCount factor?
    int currentCount =  bodyPropsCountSource*nParticlesCounted + 5*nNodesCounted;

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
    int nTop      = nodesBegEnd[i].y-nodesBegEnd[i].x;
    int startTree = 1+bodyPropsCountSource*particles;

    memcpy(&topBoxSizes[totalTopNodes],
           &treeBuffers[procTrees+i][startTree+nodesBegEnd[i].x],             sizeof(real4)*nTop);
    memcpy(&topBoxCenters[totalTopNodes],
           &treeBuffers[procTrees+i][startTree+nodes+nodesBegEnd[i].x],       sizeof(real4)*nTop);
    memcpy(&topMultiPoles[3*totalTopNodes],
           &treeBuffers[procTrees+i][startTree+2*nodes+3*nodesBegEnd[i].x], 3*sizeof(real4)*nTop);
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
  build_NewTopLevels(topNodeOnTheFlyCount,  &keys[0],          nodes,
                     nodeKeys,              node_levels,       topTree_n_levels,
                     topTree_n_nodes,       topTree_startNode, topTree_endNode);

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

  //Tree properties computed, now do some (spaghetti) magic to put everything in one array

#else
  int topTree_n_nodes = 0;
#endif //DO_NOT_USE_TOP_TREE

  //Modify the offsets of the children to fix the index references to their children
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
        child   =    childinfo & BODYMASK;                      //the first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag

        child   = child + particleSumOffsets[srcProc];         //Increasing offset
        child   = child | ((nchild-1) << LEAFBIT);             //Merging back to one integer
      }//end !leaf
    }//if endpoint

    topBoxSizes[i].w =  host_int_as_float(child);      //store the modified offset
  }//For topNodeOnTheFly


  //Compute total particles and total nodes, totalNodes is WITHOUT topNodes
  int totalParticles    = particleSumOffsets[PROCS];
  int totalNodes        = nodeSumOffsets[PROCS];

  //To bind parts of the memory to different textures, the memory start address
  //has to be aligned with a certain amount of bytes, so nodeInformation*sizeof(real4) has to be
  //increased by an offset, so that the node data starts at aligned byte boundary
  //this is already done on the sending process, but since we modify the structure
  //it has to be done again
  int nodeTextOffset = getTextureAllignmentOffset(totalNodes+totalTopNodes+topTree_n_nodes, sizeof(real4));
  int partTextOffset = getTextureAllignmentOffset(totalParticles                          , sizeof(real4));

  totalParticles    += partTextOffset;

  //Compute the total size of the buffer
  int bufferSize     = bodyPropsCount*(totalParticles) + 5*(totalNodes+totalTopNodes+topTree_n_nodes + nodeTextOffset);


  double t1 = get_time();


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

  double t2 = get_time();

  //First copy the properties of the top_tree nodes and the original top-nodes

#ifndef DO_NOT_USE_TOP_TREE
  //The top-tree node properties
  //Sizes
  memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles)],
      topTreeSizes, sizeof(real4)*topTree_n_nodes);
  //Centers
  memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) + (totalNodes + totalTopNodes + topTree_n_nodes + nodeTextOffset)],
      topTreeCenters, sizeof(real4)*topTree_n_nodes);
  //Multipoles
  memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) +
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
  memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) + topTree_n_nodes],
      &topBoxSizes[0], sizeof(real4)*topNodeOnTheFlyCount);
  //Node center information
  memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) + (totalNodes + totalTopNodes + topTree_n_nodes + nodeTextOffset) + topTree_n_nodes],
      &topBoxCenters[0], sizeof(real4)*topNodeOnTheFlyCount);
  //Multipole information
  memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) +
      2*(totalNodes+totalTopNodes+topTree_n_nodes+nodeTextOffset)+3*topTree_n_nodes],
      &topMultiPoles[0], sizeof(real4)*topNodeOnTheFlyCount*3);

  //Copy all the 'normal' pieces of the different trees at the correct memory offsets
  for(int i=0; i < PROCS; i++)
  {
    //Get the properties of the LET
    int remoteP      = host_float_as_int(treeBuffers[i+procTrees][0].x);    //Number of particles
    int remoteN      = host_float_as_int(treeBuffers[i+procTrees][0].y);    //Number of nodes
    int remoteB      = host_float_as_int(treeBuffers[i+procTrees][0].z);    //Begin id of top nodes
    int remoteE      = host_float_as_int(treeBuffers[i+procTrees][0].w);    //End   id of top nodes
    int remoteNstart = remoteE-remoteB;

    int bodyPropsCountSource = bodyPropsCount;
    if(treeBuffersSource[i+procTrees] == 2) bodyPropsCountSource = 4; //Boundary tree is a full tree with all particle properties

    //Particles
    memcpy(&combinedRemoteTree[particleSumOffsets[i]],   &treeBuffers[i+procTrees][1], sizeof(real4)*remoteP);

    if(METHOD == LET_METHOD_DRVT || METHOD == LET_METHOD_HYDR) //Insert the velocity
    {
        memcpy(&combinedRemoteTree[1*totalParticles+particleSumOffsets[i]],         //Write after the positions
               &treeBuffers[i+procTrees][1+1*remoteP], sizeof(real4)*remoteP);      // Read after the positions

        if(METHOD == LET_METHOD_HYDR) //Insert density and hydro
        {
#if 0
            float2 *dsrc = (float2*) &treeBuffers[i+procTrees][1+2*remoteP];
            float2 *ddst = (float2*) &remote.fullRemoteTree[2*totalParticles];
            ddst         =           &ddst[particleSumOffsets[i]];

            memcpy(ddst, dsrc, sizeof(real2)*remoteP);
#else
            //Convert the density from their float4 storage to float2 locations. They have to be stored float2,float2,float2
            //otherwise the indices inside the compute kernels will be incorrect.
            float2 *dens = (float2*) &remote.fullRemoteTree[2*totalParticles];
            for(int j=0; j < remoteP; j++)
            {
                float4 d = treeBuffers[i+procTrees][j+1+2*remoteP];
                dens[particleSumOffsets[i]+j]  = make_float2(d.x, d.y);
            }
#endif
            memcpy(&combinedRemoteTree[3*totalParticles+particleSumOffsets[i]],         //Write hydro after the density
                   &treeBuffers[i+procTrees][1+3*remoteP], sizeof(real4)*remoteP);      //Read after the density

        }
    }

    //Non start nodes, nodeSizeInfo
    memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) +  totalTopNodes + topTree_n_nodes + nodeSumOffsets[i]],
        &treeBuffers[i+procTrees][1+bodyPropsCountSource*remoteP+remoteE], //From the last start node onwards
        sizeof(real4)*(remoteN-remoteE));

    //Non start nodes, nodeCenterInfo
    memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) + totalTopNodes + topTree_n_nodes + nodeSumOffsets[i] +
        (totalNodes + totalTopNodes + topTree_n_nodes + nodeTextOffset)],
        &treeBuffers[i+procTrees][1+bodyPropsCountSource*remoteP+remoteE + remoteN], //From the last start node onwards
        sizeof(real4)*(remoteN-remoteE));

    //Non start nodes, multipole
    memcpy(&combinedRemoteTree[bodyPropsCount*(totalParticles) +  3*(totalTopNodes+topTree_n_nodes) +
        3*nodeSumOffsets[i] + 2*(totalNodes+totalTopNodes+topTree_n_nodes+nodeTextOffset)],
        &treeBuffers[i+procTrees][1+bodyPropsCountSource*remoteP+remoteE*3 + 2*remoteN], //From the last start node onwards
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
    int modStart =  totalTopNodes + topTree_n_nodes + nodeSumOffsets[i] + bodyPropsCount*(totalParticles);
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

  double t3 = get_time();

  //Check if we need to summarize which particles are active,
  //only done during the last approximate_gravity_let call
  bool doActivePart = (procTrees == mpiGetNProcs() -1);

  switch(METHOD)
  {
      case LET_METHOD_GRAV:
          approximate_gravity_let(this->localTree, this->remoteTree, bufferSize, doActivePart);
          break;
      case LET_METHOD_DENS:
          approximate_density_let(this->localTree, this->remoteTree, bufferSize, doActivePart);
          break;
      case LET_METHOD_DRVT:
          approximate_density_let(this->localTree, this->remoteTree, bufferSize, doActivePart);
//          approximate_derivative_let(this->localTree, this->remoteTree, bufferSize, doActivePart);
          break;
      case LET_METHOD_HYDR:
          approximate_hydro_let(this->localTree, this->remoteTree, bufferSize, doActivePart);
          break;
      default:
          assert(0);
  }

  double t4 = get_time();
  //Statistics about the tree-merging
  char buff5[512];
  sprintf(buff5, "LETXTIME-%d Iter: %d Processed: %d topTree: %lg Alloc: %lg  Copy/Update: %lg TotalC: %lg Wait: %lg TotalRun: %lg \n",
                  procId, iter, procTrees, t1-t0, t2-t1,t3-t2,t3-t0, t4-t3, t4-t0);
  devContext->writeLogEvent(buff5); //TODO DELETE

  devContext->popNVTX();
}



#if 0
void octree::ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities,  ullong *bodiesIDs, int toSend)
{
#ifdef USE_MPI
  //First send the number of particles, then the actual sample data
  MPI_Send(&toSend, 1, MPI_INT, destination, destination*2 , mpiCommWorld);

  //Send the positions, velocities and ids
  MPI_Send( bodyPositions,  toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+1, mpiCommWorld);
  MPI_Send( bodyVelocities, toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+2, mpiCommWorld);
  MPI_Send( bodiesIDs,      toSend*sizeof(ullong), MPI_BYTE, destination, destination*2+3, mpiCommWorld);
#endif
}


void octree::ICRecv(int recvFrom, vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<ullong> &bodiesIDs)
{
#ifdef USE_MPI
  MPI_Status status;
  int nreceive;

  //First receive the number of particles, then the actual sample data
  MPI_Recv(&nreceive, 1, MPI_INT, recvFrom, procId*2, mpiCommWorld,&status);

  bodyPositions.resize(nreceive);
  bodyVelocities.resize(nreceive);
  bodiesIDs.resize(nreceive);

  //Receive the positions, velocities and ids
  MPI_Recv( (real*  )&bodyPositions[0],  nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+1, mpiCommWorld,&status);
  MPI_Recv( (real*  )&bodyVelocities[0], nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+2, mpiCommWorld,&status);
  MPI_Recv( (ullong*)&bodiesIDs[0],      nreceive*sizeof(ullong), MPI_BYTE, recvFrom, procId*2+3, mpiCommWorld,&status);
#endif
}
#endif


#if 0
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
      sizeof(sampleRadInfo), MPI_BYTE, mpiCommWorld);
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
#endif


#if 0
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
#endif

#if 0
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
#endif


#if 0
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
    MPI_Allreduce( &timeLocal, &timeSum, 1,MPI_DOUBLE, MPI_SUM, mpiCommWorld);

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

      MPI_Allreduce(&nrate, &nrate2_sum, 1, MPI_DOUBLE, MPI_SUM, mpiCommWorld);

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
  fprintf(stderr, "NSAMP [%d]: sample: %d nrate: %f final sampleRate: %f localTree.n: %d\tprevious: %d timeLocal: %f prevTimeLocal: %f  Took: %lg\n",
      procId, nSamples, nrate, sampleRate, localTree.n, prevSampFreq,
      timeLocal, prevDurStep,get_time()-t00);
  assert(sampleRate > 1);

  prevDurStep  = timeLocal;
  prevSampFreq = sampleRate;

#endif
}


/*
 * Code for testing performance of getLet1 and getlETOptFullTree_sph
 */

#if 0
  //Expand the groups when using periodic boundaries, this might or might not be the most
  //efficient method as we increase the total number of checks significantly

  //For periodic boundaries, mirror the group and body positions along the periodic axis
  //float3 domainSize = {1.0f, 0.125f, 0.125f};   //Hardcoded for our testing IC
  float3 domainSize = {100.0f, 100.0f, 100.0f};   //Hardcoded for our testing IC

  for(int ix=-1; ix <= 1; ix++)     //Periodic around X
  {
      for(int iy=-1; iy <= 1; iy++)   //Periodic around Y
      {
          for(int iz=-1; iz <= 1; iz++) //Periodic around Z
          {
              for(int grp=0; grp < endGrp; grp++)
              {

                    float4 boundaryPos = boundaryCentres[grp];

                    boundaryPos.x   += (domainSize.x*ix);
                    boundaryPos.y   += (domainSize.y*iy);
                    boundaryPos.z   += (domainSize.z*iz);

                    if(ix == 1 && iy == 1 && iz == 1)
                    {
                        fprintf(stderr,"Current values: %f %f %f \n", boundaryPos.x,boundaryPos.y,boundaryPos.z);
                    }

                    boundaryCentres.  push_back(boundaryPos);
                    boundarySizes.    push_back(boundarySizes  [grp]);
                    boundarySmoothing.push_back(boundarySmoothing[grp]);
              }//grp
          }//iz
      }//iy
  }//ix
  //Update the group count
  endGrp    = boundarySizes.size();
  grpCenter = &boundaryCentres  [0];
  grpSize   = &boundarySizes    [0];
  grpSmooth = &boundarySmoothing[0];
#endif


#if 1
void extendGroupsWithPeriodicGroups(){

//Filter the boundaries while creating a list of periodic boundary groups.
std::vector<float4> boundaryCentresFiltered;
std::vector<float4> boundarySizesFiltered;
std::vector<float4> boundarySmoothingFiltered;

boundarySizesFiltered.reserve(endGrp*4);
boundaryCentresFiltered.reserve(endGrp*4);
boundarySmoothingFiltered.reserve(endGrp*4);
boundarySizesFiltered.clear();
boundaryCentresFiltered.clear();
boundarySmoothingFiltered.clear();


for(int ix=-1; ix <= 1; ix++)             //Periodic around X
{
   for(int iy=-1; iy <= 1; iy++)         //Periodic around Y
   {
       for(int iz=-1; iz <= 1; iz++)     //Periodic around Z
       {
           for(int grp=0; grp < endGrp; grp++)
           {
                 float4 boundaryPos = boundaryCentres[grp];
                 float4 groupSize   = boundarySizes  [grp];
                 float  grpSmooth   = boundarySmoothing[grp].x;

                 boundaryPos.x   += (periodicDomainSize.x*ix);
                 boundaryPos.y   += (periodicDomainSize.y*iy);
                 boundaryPos.z   += (periodicDomainSize.z*iz);

                 //Test this group against the top node of our tree
                 float4 nCenter = nodeCenterInfo[0];
                 float4 nSize   = nodeSizeInfo[0];
                 float4 nCOM    = multipole[0];

                 //Compute the distance between the group and the cell
                 float3 dr = {fabs(boundaryPos.x - nCenter.x) - (groupSize.x + nSize.x),
                              fabs(boundaryPos.y - nCenter.y) - (groupSize.y + nSize.y),
                              fabs(boundaryPos.z - nCenter.z) - (groupSize.z + nSize.z)};

                 dr.x += fabs(dr.x); dr.x *= 0.5f;
                 dr.y += fabs(dr.y); dr.y *= 0.5f;
                 dr.z += fabs(dr.z); dr.z *= 0.5f;

                 //Distance squared, no need to do sqrt since opening criteria has been squared
                 float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                 if(ds2 <= grpSmooth)
                 {
                     boundaryCentresFiltered.  push_back(boundaryPos);
                     boundarySizesFiltered.    push_back(boundarySizes  [grp]);
                     boundarySmoothingFiltered.push_back(boundarySmoothing[grp]);
                 }
                 else
                 {
                     //Test if we need it for gravity

                     //Compute the distance between the group and the cell
                       float3 dr = make_float3(
                           fabsf(boundaryPos.x - nCOM.x) - (groupSize.x),
                           fabsf(boundaryPos.y - nCOM.y) - (groupSize.y),
                           fabsf(boundaryPos.z - nCOM.z) - (groupSize.z)
                           );

                       dr.x += fabsf(dr.x); dr.x *= 0.5f;
                       dr.y += fabsf(dr.y); dr.y *= 0.5f;
                       dr.z += fabsf(dr.z); dr.z *= 0.5f;

                       //Distance squared, no need to do sqrt since opening criteria has been squared
                       const float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

                       if(ds2 <= fabsf(nCOM.w))
                       {
                           boundaryCentresFiltered.  push_back(boundaryPos);
                           boundarySizesFiltered.    push_back(boundarySizes  [grp]);
                           boundarySmoothingFiltered.push_back(boundarySmoothing[grp]);
                       }
                 }
           }//grp
       }//iz
   }//iy
}//ix

//Update the group count
endGrp    = boundarySizesFiltered.size();
grpCenter = &boundaryCentresFiltered  [0];
grpSize   = &boundarySizesFiltered    [0];
grpSmooth = &boundarySmoothingFiltered[0];
}
#endif


#if 1
#define SLEEP 5


        if(procId == 0)
        {
            fprintf(stderr,"Smoothing: %f %f %f %f \n", grpSmooth[0].x,grpSmooth[1].x,grpSmooth[2].x,grpSmooth[3].x);
            double tJB = get_time();
            for(int i=0; i < 10; i++){
            tz = get_time();
             assert(startGrp == 0);
             int3  nExport = getLET1(
                                     getLETBuffers[tid],
                                     &LETDataBuffer,
                                     &nodeCenterInfo[0],
                                     &nodeSizeInfo[0],
                                     &multipole[0],
                                     usedStartEndNode.x, usedStartEndNode.y,
                                     &bodies[0],
                                     tree.n,
                                     grpSize, grpCenter, grpSmooth,
                                     endGrp,
                                     tree.n_nodes, nflops,
                                     periodicDomainSize, periodicMethod);


             //int3 getLEToptFullTree;



             countParticles  = nExport.y;
             countNodes      = nExport.x;
             bufferSize  = 1 + 1*countParticles + 5*countNodes;
             //Use count of exported particles and nodes, but let particles count more heavy.
             //Used during particle exchange / domain update to speedup particle-box assignment
     //        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);
             if (ENABLE_RUNTIME_LOG)
             {
               fprintf(stderr,"Proc: %d LET getLetOp count&fill [%d,%d]: Depth: %d Dest: %d Total : %lg (#P: %d \t#N: %d) nNodes= %d  nGroups= %d \tsince start: %lg \n",
                               procId, procId, tid, nExport.z, ibox, get_time()-tz,countParticles,
                               countNodes, tree.n_nodes, endGrp, get_time()-t0);
             }
            }
            fprintf(stderr,"Proc: %d Total: %lg AVG: %lg \n", procId, get_time()-tJB, (get_time()-tJB) / 10);
        }
        else{sleep(SLEEP);}

        if(procId == 1){
            double tJB = get_time();
            for(int i=0; i < 10; i++){
        tz = get_time();
             assert(startGrp == 0);
             int3  nExport = getLET1(
                                     getLETBuffers[tid],
                                     &LETDataBuffer,
                                     &nodeCenterInfo[0],
                                     &nodeSizeInfo[0],
                                     &multipole[0],
                                     usedStartEndNode.x, usedStartEndNode.y,
                                     &bodies[0],
                                     tree.n,
                                     grpSize, grpCenter, grpSmooth,
                                     endGrp,
                                     tree.n_nodes, nflops,
                                     periodicDomainSize, periodicMethod);


             countParticles  = nExport.y;
             countNodes      = nExport.x;
             bufferSize  = 1 + 1*countParticles + 5*countNodes;
             //Use count of exported particles and nodes, but let particles count more heavy.
             //Used during particle exchange / domain update to speedup particle-box assignment
     //        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);
             if (ENABLE_RUNTIME_LOG)
             {
               fprintf(stderr,"Proc: %d LET getLetOp count&fill [%d,%d]: Depth: %d Dest: %d Total : %lg (#P: %d \t#N: %d) nNodes= %d  nGroups= %d \tsince start: %lg \n",
                               procId, procId, tid, nExport.z, ibox, get_time()-tz,countParticles,
                               countNodes, tree.n_nodes, endGrp, get_time()-t0);
             }
        }
            fprintf(stderr,"Proc: %d Total: %lg AVG: %lg \n", procId, get_time()-tJB, (get_time()-tJB) / 10);
        }
        else{sleep(SLEEP);}

#endif

        sleep(1);
#if 1
        {
        int idx          =   globalGrpTreeOffsets[ibox];
        real4 *grpCenter =  &globalGrpTreeCntSize[idx];
        int start = host_float_as_int(grpCenter[0].z);
        int end = host_float_as_int(grpCenter[0].w);

        int nbody = host_float_as_int(grpCenter[0].x);
        int nnode = host_float_as_int(grpCenter[0].y);

#define SLEEP 5
        int bufferSize;


        if(procId == 0)
        {
            double tJB = get_time();
            for(int i=0; i < 10; i++){
            tz = get_time();
             assert(startGrp == 0);


             real4 * grpSize    = &grpCenter[1+nbody];
             real4 * grpSmooth  = &grpCenter[1+nbody+nnode*2];
             real4 * grpCenter2 = &grpCenter[1+nbody+nnode];

             int3  nExport = getLEToptFullTree_sph(
                                     getLETBuffers[tid],
                                     &LETDataBuffer,
                                     &nodeCenterInfo[0],
                                     &nodeSizeInfo[0],
                                     &multipole[0],
                                     usedStartEndNode.x, usedStartEndNode.y,
                                     &bodies[0],
                                     tree.n,
                                     grpSize, grpCenter2,grpSmooth,
                                     start,
                                     end,
                                     tree.n_nodes, nflops);


             //int3 ;



             countParticles  = nExport.y;
             countNodes      = nExport.x;
             bufferSize  = 1 + 1*countParticles + 5*countNodes;
             //Use count of exported particles and nodes, but let particles count more heavy.
             //Used during particle exchange / domain update to speedup particle-box assignment
     //        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);
             if (ENABLE_RUNTIME_LOG)
             {
               fprintf(stderr,"Proc: %d LET getLetOp count&fill [%d,%d]: Depth: %d Dest: %d Total : %lg (#P: %d \t#N: %d) nNodes= %d  nGroups= %d \tsince start: %lg \n",
                               procId, procId, tid, nExport.z, ibox, get_time()-tz,countParticles,
                               countNodes, tree.n_nodes, endGrp, get_time()-t0);
             }
            }
            fprintf(stderr,"Proc: %d Total: %lg AVG: %lg \n", procId, get_time()-tJB, (get_time()-tJB) / 10);
        }
        else{sleep(SLEEP);}

        if(procId == 1){
            double tJB = get_time();
            for(int i=0; i < 10; i++){
        tz = get_time();
             assert(startGrp == 0);
             real4 * grpSize    = &grpCenter[1+nbody];
             real4 * grpSmooth  = &grpCenter[1+nbody+nnode*2];
             real4 * grpCenter2 = &grpCenter[1+nbody+nnode];

             int3  nExport = getLEToptFullTree_sph(
                                     getLETBuffers[tid],
                                     &LETDataBuffer,
                                     &nodeCenterInfo[0],
                                     &nodeSizeInfo[0],
                                     &multipole[0],
                                     usedStartEndNode.x, usedStartEndNode.y,
                                     &bodies[0],
                                     tree.n,
                                     grpSize, grpCenter2,grpSmooth,
                                     start,
                                     end,
                                     tree.n_nodes, nflops);


             countParticles  = nExport.y;
             countNodes      = nExport.x;
             bufferSize  = 1 + 1*countParticles + 5*countNodes;
             //Use count of exported particles and nodes, but let particles count more heavy.
             //Used during particle exchange / domain update to speedup particle-box assignment
     //        this->fullGrpAndLETRequestStatistics[ibox] = make_uint2(countParticles*10 + countNodes, ibox);
             if (ENABLE_RUNTIME_LOG)
             {
               fprintf(stderr,"Proc: %d LET getLetOp count&fill [%d,%d]: Depth: %d Dest: %d Total : %lg (#P: %d \t#N: %d) nNodes= %d  nGroups= %d \tsince start: %lg \n",
                               procId, procId, tid, nExport.z, ibox, get_time()-tz,countParticles,
                               countNodes, tree.n_nodes, endGrp, get_time()-t0);
             }
        }
            fprintf(stderr,"Proc: %d Total: %lg AVG: %lg \n", procId, get_time()-tJB, (get_time()-tJB) / 10);
        }
        else{sleep(SLEEP);}

}
#endif





#if 0

template<typename T>
int getLETquick(
    GETLETBUFFERS &bufferStruct,
    std::vector<T> &LETBuffer,
    const int NCELLMAX,
    const int NDEPTHMAX,
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
    const int nNodes, const int procId,
    double &time)
{
  bufferStruct.LETBuffer_node.clear();
  bufferStruct.LETBuffer_ptcl.clear();
  bufferStruct.currLevelVecI.clear();
  bufferStruct.nextLevelVecI.clear();

  unsigned long long nflops = 0;
  double t0 = get_time2();

  int nExportPtcl = 0;
  int nExportCell = 0;
  int nExportCellOffset = cellEnd;

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    bufferStruct.LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});


  const _v4sf*            bodiesV = (const _v4sf*)bodies;
  const _v4sf*          nodeSizeV = (const _v4sf*)nodeSize;
  const _v4sf*        nodeCentreV = (const _v4sf*)nodeCentre;
  const _v4sf*         multipoleV = (const _v4sf*)multipole;
  const _v4sf*   groupSizeV = (const _v4sf*)groupSizeInfo;
  const _v4sf* groupCenterV = (const _v4sf*)groupCentreInfo;


  Swap<std::vector<int> > levelList(bufferStruct.currLevelVecI, bufferStruct.nextLevelVecI);

  const int SIMDW  = 4;

  //We need a bunch of buffers to act as swap space
  const int nGroups4  = ((nGroups-1)/SIMDW + 1)*SIMDW;
  const int allocSize = (int)(nGroups4*1.10);
  bufferStruct.groupCentreSIMD.reserve(allocSize);
  bufferStruct.groupSizeSIMD.  reserve(allocSize);
  bufferStruct.groupCentreSIMD.resize(nGroups4);
  bufferStruct.groupSizeSIMD.  resize(nGroups4);

  bufferStruct.groupCentreSIMDSwap.reserve(allocSize);
  bufferStruct.groupSizeSIMDSwap.  reserve(allocSize);
  bufferStruct.groupCentreSIMDSwap.resize(nGroups4);
  bufferStruct.groupSizeSIMDSwap.  resize(nGroups4);

  bufferStruct.groupSIMDkeys.resize((int)(1.10*(nGroups4/SIMDW)));

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

    bufferStruct.groupCentreSIMD[ib+0] = bcx;
    bufferStruct.groupCentreSIMD[ib+1] = bcy;
    bufferStruct.groupCentreSIMD[ib+2] = bcz;
    bufferStruct.groupCentreSIMD[ib+3] = bcw;

    bufferStruct.groupSizeSIMD[ib+0] = bsx;
    bufferStruct.groupSizeSIMD[ib+1] = bsy;
    bufferStruct.groupSizeSIMD[ib+2] = bsz;
    bufferStruct.groupSizeSIMD[ib+3] = bsw;
  }

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);

  int relativeDepth = 0;

  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
#if 1
    if (nGroups > 64)   /* randomizes algo, can give substantial speed-up */
      shuffle2vecAllocated<v4sf,SIMDW>(bufferStruct.groupCentreSIMD,
                                       bufferStruct.groupSizeSIMD,
                                       bufferStruct.groupCentreSIMDSwap,
                                       bufferStruct.groupSizeSIMDSwap,
                                       bufferStruct.groupSIMDkeys);
//      shuffle2vec<v4sf,SIMDW>(bufferStruct.groupCentreSIMD, bufferStruct.groupSizeSIMD);
#endif
    for (int i = 0; i < csize; i++)
    {
      /* play with criteria to fit what's best */
      if (relativeDepth > NDEPTHMAX && nExportCell > NCELLMAX)
      {
        time = get_time2()-t0;
        return -1;
      }
      if (nExportCell > NCELLMAX)
      {
        time = get_time2()-t0;
        return -1;
      }


      const uint    nodeIdx  = levelList.first()[i];
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
      {
        split |= split_node_grav_impbh_box4simd1<TRANSPOSE_SPLIT>(vncx,vncy,vncz,vsize, (_v4sf*)&bufferStruct.groupCentreSIMD[ib], (_v4sf*)&bufferStruct.groupSizeSIMD[ib]);
      }

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
            bufferStruct.LETBuffer_ptcl.push_back(i);
          nExportPtcl += np;
        }
      }

      bufferStruct.LETBuffer_node.push_back((int2){nodeIdx, sizew});
      nExportCell++;
    }

    levelList.swap();
    levelList.second().clear();
    relativeDepth++;
  }

  double t1=get_time2();
  assert((int)bufferStruct.LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)bufferStruct.LETBuffer_node.size() == nExportCell);

  /* now copy data into LETBuffer */
  {
    //LETBuffer.resize(nExportPtcl + 5*nExportCell);
    _v4sf *vLETBuffer;

//#pragma omp critical //Malloc seems to be not so thread safe..
    {
      const size_t oldSize     = LETBuffer.size();
      const size_t oldCapacity = LETBuffer.capacity();
      LETBuffer.resize(oldSize + 1 + nExportPtcl + 5*nExportCell);
      const size_t newCapacity = LETBuffer.capacity();
      /* make sure memory is not reallocated */
      assert(oldCapacity == newCapacity);

      /* fill int tree info */
      real4 &data4 = *(real4*)(&LETBuffer[oldSize]);
      data4.x = host_int_as_float(nExportPtcl);
      data4.y = host_int_as_float(nExportCell);
      data4.z = host_int_as_float(cellBeg);
      data4.w = host_int_as_float(cellEnd);

      /* write info */

//      LOGF(stderr, "LET res for: %d  P: %d  N: %d old: %ld  Size in byte: %d\n",procId, nExportPtcl, nExportCell, oldSize, (int)(( 1 + nExportPtcl + 5*nExportCell)*sizeof(real4)));
      vLETBuffer = (_v4sf*)(&LETBuffer[oldSize+1]);
    }



    int nStoreIdx     = nExportPtcl;
    int multiStoreIdx = nStoreIdx + 2*nExportCell;
    for (int i = 0; i < nExportPtcl; i++)
    {
      const int idx = bufferStruct.LETBuffer_ptcl[i];
      vLETBuffer[i] = bodiesV[idx];
    }
    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = bufferStruct.LETBuffer_node[i];
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

//double t2 = get_time2();
time = get_time2()-t0;
//LOGF(stderr,"LETQ proc: %d Took: %lg  Calc: %lg  Copy: %lg P: %d N: %d \n",procId, t2-t0, t1-t0, t2-t1, nExportPtcl, nExportCell);

  return  1 + nExportPtcl + 5*nExportCell;
}

#endif


#if 0
template<bool TRANSPOSE>
inline int split_node_grav_impbh_box4simd1( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  ncx,
    const _v4sf  ncy,
    const _v4sf  ncz,
    const _v4sf  boxSmooth,
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

//  float3 domainSize = {100.0f, 100.0f, 100.0f};   //Hardcoded for our testing IC
//    for(int ix=-1; ix <= 1; ix++)     //Periodic around X
//    {
//      for(int iy=-1; iy <= 1; iy++)   //Periodic around Y
//      {
//        for(int iz=-1; iz <= 1; iz++) //Periodic around Z
//        {
//            _v4sf bcx = __builtin_ia32_addss(bcx2,  _mm_set_ps1(domainSize.x*ix));
//            _v4sf bcy = __builtin_ia32_addss(bcy2,  _mm_set_ps1(domainSize.y*iy));
//            _v4sf bcz = __builtin_ia32_addss(bcz2,  _mm_set_ps1(domainSize.z*iz));



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
      __builtin_ia32_cmpleps(ds2, boxSmooth));
#endif
//  if(ret) return ret;
//        }
//      }
//    }
  return ret;
}



template<bool TRANSPOSE>
inline int split_node_sph_impbh_box4simd1( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  ncx,
    const _v4sf  ncy,
    const _v4sf  ncz,
    const _v4sf  nsx,
    const _v4sf  nsy,
    const _v4sf  nsz,
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

  _v4sf dx = __abs(bcx - ncx) - (bsx+nsx);
  _v4sf dy = __abs(bcy - ncy) - (bsy+nsy);
  _v4sf dz = __abs(bcz - ncz) - (bsz+nsz);

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
#endif

#if 0

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
//Here each group has a different smoothing length, encoded in boxCenter.w
//We want to know for each of the 4 groups if it needs an interaction
//with this node. Therefore we have to test every possible combination
//until either all combinations are exhausted or each group requires an interaction
template<bool periodic>
inline _v4sf split_node_sph_impbh_box4a( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  nodeCNTR,
    const _v4sf  nodeSIZE,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4],
    const float3 periodicBoundaries,
    int2 xP, int2 yP, int2 zP)
{
  _v4sf ncx2  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x00);
  _v4sf ncy2  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x55);
  _v4sf ncz2  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xaa);
  _v4sf ncw   = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xff);
  _v4sf size = __abs(ncw);  //smoothing
  _v4sf ret  {0.0f, 0.0f, 0.0f, 0.0f};


  _v4sf nsx    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x00);
  _v4sf nsy    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x55);
  _v4sf nsz    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0xaa);

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

  bsx +=nsx;
  bsy +=nsy;
  bsz +=nsz;


  for(int ix=xP.x; ix <= xP.y; ix++)     //Periodic around X
  {
    for(int iy=yP.x; iy <= yP.y; iy++)   //Periodic around Y
    {
      for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
      {
           _v4sf ncx = __builtin_ia32_addps(ncx2,  _mm_set_ps1(periodicBoundaries.x*ix));
           _v4sf ncy = __builtin_ia32_addps(ncy2,  _mm_set_ps1(periodicBoundaries.y*iy));
           _v4sf ncz = __builtin_ia32_addps(ncz2,  _mm_set_ps1(periodicBoundaries.z*iz));


          _v4sf dx = __abs(bcx - ncx) - (bsx);
          _v4sf dy = __abs(bcy - ncy) - (bsy);
          _v4sf dz = __abs(bcz - ncz) - (bsz);

          const _v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
          dx = __builtin_ia32_maxps(dx, zero);
          dy = __builtin_ia32_maxps(dy, zero);
          dz = __builtin_ia32_maxps(dz, zero);

          const _v4sf ds2 = dx*dx + dy*dy + dz*dz;

          //Test all 4 distances against 4 smoothing lengths and or with previous result
          ret =  _mm_or_ps (ret,__builtin_ia32_cmpleps(ds2, size));

          //Test if all 4 bits are set (then all groups active ) this is 4 lower bits set to 1: 1+2+4+8=15
          //Early out if this is the case since we won't learn anything new :)
          if(__builtin_ia32_movmskps(ret) == 15) return ret;
       }//iz
     }//iy
   }//ix


  return ret;
}


inline _v4sf split_node_sph_impbh_box4a( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  nodeCNTR,
    const _v4sf  nodeSIZE,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4])
{
  _v4sf ncx  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x00);
  _v4sf ncy  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x55);
  _v4sf ncz  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xaa);
  _v4sf ncw  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xff);
  _v4sf size = __abs(ncw);  //smoothing

  _v4sf nsx    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x00);
  _v4sf nsy    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x55);
  _v4sf nsz    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0xaa);

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

  _v4sf dx = __abs(bcx - ncx) - (bsx+nsx);
  _v4sf dy = __abs(bcy - ncy) - (bsy+nsy);
  _v4sf dz = __abs(bcz - ncz) - (bsz+nsz);

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



//Here each group has a different smoothing length, encoded in boxCenter.w
//We want to know for each of the 4 groups if it needs an interaction
//with this node. Therefore we have to test every possible combination
//until either all combinations are exhausted or each group requires an interaction
template<bool periodic>
inline _v4sf split_node_sph_grp_impbh_box4a( // takes 4 tree nodes and returns 4-bit integer
          _v4sf  ret,
    const _v4sf  nodeCNTR,
    const _v4sf  nodeSIZE,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4],
    const float3 periodicBoundaries,
    int2 xP_, int2 yP_, int2 zP_)
{
    _v4sf ncx2  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x00);
    _v4sf ncy2  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x55);
    _v4sf ncz2  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xaa);
    _v4sf ncw   = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xff);

    _v4sf nsx    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x00);
    _v4sf nsy    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x55);
    _v4sf nsz    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0xaa);

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

    _v4sf size = __abs(bcw);

    bsx += nsx;
    bsy += nsy;
    bsz += nsz;

    const _v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};


    int2 xP = {0,0}, yP = {0,0}, zP = {0,0};
    if(periodic)
    {
        xP = xP_; yP = yP_; zP = zP_;
    }

    for(int ix=xP.x; ix <= xP.y; ix++)     //Periodic around X
    {
        for(int iy=yP.x; iy <= yP.y; iy++)   //Periodic around Y
        {
            for(int iz=zP.x; iz <= zP.y; iz++) //Periodic around Z
            {
                _v4sf ncx = __builtin_ia32_addps(ncx2,  _mm_set_ps1(periodicBoundaries.x*ix));
                _v4sf ncy = __builtin_ia32_addps(ncy2,  _mm_set_ps1(periodicBoundaries.y*iy));
                _v4sf ncz = __builtin_ia32_addps(ncz2,  _mm_set_ps1(periodicBoundaries.z*iz));


                _v4sf dx = __abs(bcx - ncx) - (bsx+nsx);
                _v4sf dy = __abs(bcy - ncy) - (bsy+nsy);
                _v4sf dz = __abs(bcz - ncz) - (bsz+nsz);


                dx = __builtin_ia32_maxps(dx, zero);
                dy = __builtin_ia32_maxps(dy, zero);
                dz = __builtin_ia32_maxps(dz, zero);

                const _v4sf ds2 = dx*dx + dy*dy + dz*dz;

                //Test all 4 distances against 4 smoothing lengths and or with previous result
                ret =  _mm_or_ps (ret,__builtin_ia32_cmpleps(ds2, size));

                //Test if all 4 bits are set (then all groups active ) this is 4 lower bits set to 1: 1+2+4+8=15
                //Early out if this is the case since we won't learn anything new :)
                if(__builtin_ia32_movmskps(ret) == 15) return ret;
            }//iz
        }//iy
    }//ix


  return ret;
}
#endif


#if 0

/*
 * Create the LET structure for the remote process using the full tree-layout of the remote boundaries
 * where as getLET1 uses only the boundaries and not the tree-structure that comes with it
 *
 */

int3 getLEToptFullTree(
    GETLETBUFFERS &bufferStruct,
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
  bufferStruct.LETBuffer_node.clear();
  bufferStruct.LETBuffer_ptcl.clear();
  bufferStruct.currLevelVecUI4.clear();
  bufferStruct.nextLevelVecUI4.clear();
  bufferStruct.currGroupLevelVec.clear();
  bufferStruct.nextGroupLevelVec.clear();
  bufferStruct.groupSplitFlag.clear();

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


#if 0 /* AVX */
#ifndef __AVX__
#error "AVX is not defined"
#endif
  const int SIMDW  = 8;
#define AVXIMBH
#else
  const int SIMDW  = 4; //#define SSEIMBH
#endif


  Swap<std::vector<uint4> > levelList(bufferStruct.currLevelVecUI4, bufferStruct.nextLevelVecUI4);
  Swap<std::vector<int> > levelGroups(bufferStruct.currGroupLevelVec, bufferStruct.nextGroupLevelVec);

  nExportCell += cellBeg;
  for (int node = 0; node < cellBeg; node++)
    bufferStruct.LETBuffer_node.push_back((int2){node, host_float_as_int(nodeSize[node].w)});



  /* copy group info into current level buffer */
  for (int group = groupBeg; group < groupEnd; group++)
    levelGroups.first().push_back(group);

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back((uint4){(uint)cell, 0, (uint)levelGroups.first().size(),0});

  int depth = 0;
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

      bufferStruct.groupSplitFlag.clear();
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
        bufferStruct.groupSplitFlag.push_back(split_node_grav_impbh_box8a(nodeCOM, centre, size));
#else
        bufferStruct.groupSplitFlag.push_back(split_node_grav_impbh_box4a(nodeCOM, centre, size));
#endif
      }

      const int groupNextBeg = levelGroups.second().size();
      int split = false;
      for (int idx = groupBeg; idx < groupEnd; idx++)
      {
        const bool gsplit = ((uint*)&bufferStruct.groupSplitFlag[0])[idx - groupBeg];
        if (gsplit)
        {
          split = true;
          const int group = levelGroups.first()[idx];
          if (!lleaf)
          {
            //Test on if its a leaf and on if it's and end-point
            bool gleaf = groupCentreInfo[group].w <= 0.0f;
            if(!gleaf)
            {
              gleaf = (host_float_as_int(groupSizeInfo[group].w) == 0xFFFFFFFF);
            }

            if (!gleaf)
            {
              const int childinfoGrp  = ((uint4*)groupSizeInfo)[group].w;
              const int gchild  =   childinfoGrp & 0x0FFFFFFF;
              const int gnchild = ((childinfoGrp & 0xF0000000) >> 28) ;

              //for (int i = gchild; i <= gchild+gnchild; i++)
              for (int i = gchild; i < gchild+gnchild; i++)
              {
                levelGroups.second().push_back(i);
              }
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
            levelList.second().push_back((uint4){(uint)i,(uint)groupNextBeg,(uint)levelGroups.second().size()});
        }
        else
        {
          const int pfirst =    nodeInfo_y & BODYMASK;
          const int np     = (((nodeInfo_y & INVBMASK) >> LEAFBIT)+1);
          sizew = (nExportPtcl | ((np-1) << LEAFBIT));
          for (int i = pfirst; i < pfirst+np; i++)
            bufferStruct.LETBuffer_ptcl.push_back(i);
          nExportPtcl += np;
        }
      }

      bufferStruct.LETBuffer_node.push_back((int2){(int)nodeIdx, sizew});
      nExportCell++;
    }

    depth++;

    levelList.swap();
    levelList.second().clear();

    levelGroups.swap();
    levelGroups.second().clear();
  }

  assert((int)bufferStruct.LETBuffer_ptcl.size() == nExportPtcl);
  assert((int)bufferStruct.LETBuffer_node.size() == nExportCell);

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
      const int idx = bufferStruct.LETBuffer_ptcl[i];
      vLETBuffer[i] = bodiesV[idx];
    }
    for (int i = 0; i < nExportCell; i++)
    {
      const int2 packed_idx = bufferStruct.LETBuffer_node[i];
      const int idx = packed_idx.x;
      const float sizew = host_int_as_float(packed_idx.y);
      const _v4sf size = __builtin_ia32_vec_set_v4sf(nodeSizeV[idx], sizew, 3);

      vLETBuffer[nStoreIdx+nExportCell] = nodeCentreV[idx];     /* centre */
      vLETBuffer[nStoreIdx            ] = size;                 /*  size  */

      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+0];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+1];  /* multipole.x */
      vLETBuffer[multiStoreIdx++      ] = multipoleV[3*idx+2];  /* multipole.x */
      nStoreIdx++;
    }
  }

  return (int3){nExportCell, nExportPtcl, depth};
}

#endif

#if 0
//In the below version the smoothing is in the boxes so take the smoothing
//value from the .w component of the boxCenter
inline _v4sf split_node_sph_impbh_box4a_grp( // takes 4 tree nodes and returns 4-bit integer
    const _v4sf  nodeCNTR,
    const _v4sf  nodeSIZE,
    const _v4sf  boxCenter[4],
    const _v4sf  boxSize  [4])
{
  _v4sf ncx  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x00);
  _v4sf ncy  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0x55);
  _v4sf ncz  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xaa);
  _v4sf ncw  = __builtin_ia32_shufps(nodeCNTR, nodeCNTR, 0xff);


  _v4sf nsx    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x00);
  _v4sf nsy    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0x55);
  _v4sf nsz    = __builtin_ia32_shufps(nodeSIZE, nodeSIZE, 0xaa);

  _v4sf bcx =  (boxCenter[0]);
  _v4sf bcy =  (boxCenter[1]);
  _v4sf bcz =  (boxCenter[2]);
  _v4sf bcw =  (boxCenter[3]);
  _v4sf_transpose(bcx, bcy, bcz, bcw);
  _v4sf size = __abs(bcw);  //smoothing

  _v4sf bsx =  (boxSize[0]);
  _v4sf bsy =  (boxSize[1]);
  _v4sf bsz =  (boxSize[2]);
  _v4sf bsw =  (boxSize[3]);
  _v4sf_transpose(bsx, bsy, bsz, bsw);

  _v4sf dx = __abs(bcx - ncx) - (bsx+nsx);
  _v4sf dy = __abs(bcy - ncy) - (bsy+nsy);
  _v4sf dz = __abs(bcz - ncz) - (bsz+nsz);

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
#endif


#if 0
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


//Exports a full-structure including the multipole moments
//Note this one should only Count and does not contain smoothing info
//TODO remove this function?
int extractGroupsTreeFullCount2(
    std::vector<real4> &groupCentre,
    std::vector<real4> &groupSize,

    std::vector<real4> &groupMulti,
    std::vector<real4> &groupBody,
    const real4 *nodeCentre,
    const real4 *nodeSize,
    const real4 *nodeMulti,
    const real4 *nodeBody,
    const int cellBeg,
    const int cellEnd,
    const int nNodes,
    const int maxDepth)
{
  groupCentre.clear();
  groupCentre.reserve(nNodes);

  groupSize.clear();
  groupSize.reserve(nNodes);

  groupBody.clear();
  groupBody.reserve(nNodes); //We only select leaves with child==1, so cant ever have more than this

  const int levelCountMax = nNodes;
  std::vector<int> currLevelVec, nextLevelVec;
  currLevelVec.reserve(levelCountMax);
  nextLevelVec.reserve(levelCountMax);
  Swap<std::vector<int> > levelList(currLevelVec, nextLevelVec);

  //These are top level nodes. And everything before
  //should be added. Nothing has to be changed
  //since we keep the structure
  for(int cell = 0; cell < cellBeg; cell++)
  {
    groupCentre.push_back(nodeCentre[cell]);
    groupSize  .push_back(nodeSize[cell]);
  }

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);


  int childOffset    = cellEnd;
  int childBodyCount = 0;

  int depth = 0;
  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint   nodeIdx = levelList.first()[i];
      const float4 centre  = nodeCentre[nodeIdx];
      const float4 size    = nodeSize[nodeIdx];
      const float nodeInfo_x = centre.w;
      const uint  nodeInfo_y = host_float_as_int(size.w);

      const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
      const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has

      const bool lleaf = nodeInfo_x <= 0.0f;
      if (!lleaf)
      {
        //We mark this as an end-point
        if (lnchild == 8)
        {
          float4 size1 = size;
          size1.w = host_int_as_float(0xFFFFFFFF);
          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
        }
        else
        {
          //We pursue this branch, mark the offsets and add the parent
          //to our list and the children to next level process
          float4 size1   = size;
          uint newOffset   = childOffset | ((uint)(lnchild) << LEAFBIT);
          childOffset     += lnchild;
          size1.w         = host_int_as_float(newOffset);

          if(depth <  maxDepth){
            size1.w = host_int_as_float(0xFFFFFFFF); //mark as end point
          }

          groupCentre.push_back(centre);
          groupSize  .push_back(size1);

          if(depth <  maxDepth){
            for (int i = lchild; i < lchild + lnchild; i++)
              levelList.second().push_back(i);
          }
        }
      }
      else
      {
        //We always open leafs with nchild == 1 so check and possibly add child
        if(lnchild == 0)
        { //1 child
          float4 size1;
          uint newOffset  = childBodyCount | ((uint)(lnchild) << LEAFBIT);
          childBodyCount += 1;
          size1.w         = host_int_as_float(newOffset);

          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
          groupBody  .push_back(nodeBody[lchild]);
        }
        else
        { //More than 1 child, mark as END point
          float4 size1 = size;
          size1.w = host_int_as_float(0xFFFFFFFF);
          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
        }
      }
    }

    levelList.swap();
    levelList.second().clear();
    depth++;
  }

  //Required space Multi*3 + size+cntr+smooth
  return (1 + groupBody.size() + 6*groupSize.size());
}


//Exports a full-structure including the multipole moments
//TODO remove double lines by making the if/else structures cleaner
void extractGroupsTreeFull(
    std::vector<real4> &groupCentre,
    std::vector<real4> &groupSize,
    std::vector<real4> &groupSmoothing,
    std::vector<real4> &groupMulti,
    std::vector<real4> &groupBody,
    const real4 *nodeCentre,
    const real4 *nodeSize,
    const real  *nodeSmoothing,
    const real4 *nodeMulti,
    const real4 *nodeBody,
    const int cellBeg,
    const int cellEnd,
    const int nNodes,
    const int maxDepth)
{
  groupCentre.clear();
  groupSize.clear();
  groupMulti.clear();
  groupSmoothing.clear();
  groupBody.clear();


  groupCentre.   reserve(nNodes);
  groupSize.     reserve(nNodes);
  groupSmoothing.reserve(nNodes);
  groupMulti.    reserve(3*nNodes);
  groupBody.     reserve(nNodes); //We only select leaves with child==1, so cant ever have more than this

  const int levelCountMax = nNodes;
  std::vector<int> currLevelVec, nextLevelVec;
  currLevelVec.reserve(levelCountMax);
  nextLevelVec.reserve(levelCountMax);
  Swap<std::vector<int> > levelList(currLevelVec, nextLevelVec);

  //These are top level nodes. And everything before
  //should be added. Nothing has to be changed
  //since we keep the structure
  for(int cell = 0; cell < cellBeg; cell++)
  {
    groupCentre.push_back(nodeCentre[cell]);
    groupSize  .push_back(nodeSize[cell]);
    groupSmoothing  .push_back(make_float4(nodeSmoothing[cell], 0,0,0));
    groupMulti .push_back(nodeMulti[cell*3+0]);
    groupMulti .push_back(nodeMulti[cell*3+1]);
    groupMulti .push_back(nodeMulti[cell*3+2]);
  }

  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);


  int childOffset    = cellEnd;
  int childBodyCount = 0;

  int depth = 0;
  while (!levelList.first().empty())
  {
//    LOGF(stderr, " depth= %d Store offset: %d cursize: %d\n", depth++, childOffset, groupSize.size());
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint   nodeIdx = levelList.first()[i];
      const float4 centre  = nodeCentre[nodeIdx];
      const float4 size    = nodeSize[nodeIdx];

      const float nodeInfo_x = centre.w;
      const uint  nodeInfo_y = host_float_as_int(size.w);

//      LOGF(stderr,"BeforeWorking on %d \tLeaf: %d \t %f [%f %f %f]\n",nodeIdx, nodeInfo_x <= 0.0f, nodeInfo_x, centre.x, centre.y, centre.z);
      const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
      const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has

      const bool lleaf = nodeInfo_x <= 0.0f;
      if (!lleaf)
      {
#if 1
        //We mark this as an end-point
        if (lnchild == 8)
        {
          float4 size1 = size;
          size1.w = host_int_as_float(0xFFFFFFFF);
          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
          groupSmoothing  .push_back(make_float4(nodeSmoothing[nodeIdx], 0,0,0));
          groupMulti .push_back(nodeMulti[nodeIdx*3+0]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+1]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+2]);
        }
        else
#endif
        {
//          LOGF(stderr,"ORIChild info: Node: %d stored at: %d  info:  %d %d \n",nodeIdx, groupSize.size(), lchild, lnchild);
          //We pursue this branch, mark the offsets and add the parent
          //to our list and the children to next level process
          float4 size1   = size;
          uint newOffset   = childOffset | ((uint)(lnchild) << LEAFBIT);
          childOffset     += lnchild;
          size1.w         = host_int_as_float(newOffset);

          if(depth >=  maxDepth){
            size1.w = host_int_as_float(0xFFFFFFFF); //mark as end point
          }

          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
          groupSmoothing  .push_back(make_float4(nodeSmoothing[nodeIdx], 0,0,0));
          groupMulti .push_back(nodeMulti[nodeIdx*3+0]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+1]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+2]);

          if(depth <  maxDepth){
            for (int i = lchild; i < lchild + lnchild; i++)
              levelList.second().push_back(i);
          }
        }
      }
      else
      {
        //We always open leafs with nchild == 1 so check and possibly add child
        if(lnchild == 0)
        { //1 child
          float4 size1 = size;
          uint newOffset  = childBodyCount | ((uint)(lnchild) << LEAFBIT);
          childBodyCount += 1;
          size1.w         = host_int_as_float(newOffset);

          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
          groupSmoothing  .push_back(make_float4(nodeSmoothing[nodeIdx], 0,0,0));
          groupMulti .push_back(nodeMulti[nodeIdx*3+0]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+1]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+2]);
          groupBody  .push_back(nodeBody[lchild]);

//          LOGF(stderr,"Adding a leaf with only 1 child!! Grp cntr: %f %f %f body: %f %f %f\n",
//              centre.x, centre.y, centre.z, nodeBody[lchild].x, nodeBody[lchild].y, nodeBody[lchild].z);
        }
        else
        { //More than 1 child, mark as END point
          float4 size1 = size;
          size1.w = host_int_as_float(0xFFFFFFFF);
          groupCentre.push_back(centre);
          groupSize  .push_back(size1);
          groupSmoothing  .push_back(make_float4(nodeSmoothing[nodeIdx], 0,0,0));

          groupMulti .push_back(nodeMulti[nodeIdx*3+0]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+1]);
          groupMulti .push_back(nodeMulti[nodeIdx*3+2]);
        }
      }
    }

//    LOGF(stderr, "  done depth= %d Store offset: %d cursize: %d\n", depth, childOffset, groupSize.size());
    levelList.swap();
    levelList.second().clear();
    depth++;
  }


#if 0 //Verification during testing, compare old and new method

//
//  char buff[20*128];
//  sprintf(buff,"Proc: ");
//  for(int i=0; i < grpIds.size(); i++)
//  {
//    sprintf(buff,"%s %d, ", buff, grpIds[i]);
//  }
//  LOGF(stderr,"%s \n", buff);


  //Verify our results
  int checkCount = 0;
  for(int j=0; j < grpIdsNormal.size(); j++)
  {
    for(int i=0; i < grpIds.size(); i++)
    {
        if(grpIds[i] == grpIdsNormal[j])
        {
          checkCount++;
          break;
        }
    }
  }

  if(checkCount == grpIdsNormal.size()){
    LOGF(stderr,"PASSED grpTest %d \n", checkCount);
  }else{
    LOGF(stderr, "FAILED grpTest %d \n", checkCount);
  }


  std::vector<real4> groupCentre2;
  std::vector<real4> groupSize2;
  std::vector<int> grpIdsNormal2;

  extractGroupsPrint(
     groupCentre2,
     groupSize2,
     grpIdsNormal2,
     &groupCentre[0],
     &groupSize[0],
     cellBeg,
     cellEnd,
     nNodes);

#endif

}



//Only counts the items in a full-structure
//We basically count the nodes that form the external
//structure of the tree
int extractGroupsTreeFullCount(
    const real4 *nodeCentre,
    const real4 *nodeSize,
    const int cellBeg,
    const int cellEnd,
    const int nNodes,
    const int maxDepth,
          int &depth)
{
  const int levelCountMax = nNodes;
  std::vector<int> currLevelVec, nextLevelVec;
  currLevelVec.reserve(levelCountMax);
  nextLevelVec.reserve(levelCountMax);
  Swap<std::vector<int> > levelList(currLevelVec, nextLevelVec);

  int exportBodyCount = 0;
  int exportNodeCount = cellBeg;


  //Add the start level to the queue
  for (int cell = cellBeg; cell < cellEnd; cell++)
    levelList.first().push_back(cell);

  //Walk through the tree levels
  while (!levelList.first().empty())
  {
    const int csize = levelList.first().size();
    for (int i = 0; i < csize; i++)
    {
      const uint   nodeIdx = levelList.first()[i];
      const float4 centre  = nodeCentre[nodeIdx];
      const float4 size    = nodeSize[nodeIdx];
      const float nodeInfo_x = centre.w;
      const uint  nodeInfo_y = host_float_as_int(size.w);

      const int lchild  =    nodeInfo_y & 0x0FFFFFFF;            //Index to the first child of the node
      const int lnchild = (((nodeInfo_y & 0xF0000000) >> 28)) ;  //The number of children this node has

      const bool lleaf = nodeInfo_x <= 0.0f;
      exportNodeCount++;
      if (!lleaf)
      {
        //We treat this as an end-point if it has 8 children, otherwise continue down the tree
        if (lnchild != 8)
        {
          if(depth <  maxDepth){
            //We pursue this branch, mark the offsets and add the parent
            //to our list and the children to next level process
            for (int i = lchild; i < lchild + lnchild; i++)
              levelList.second().push_back(i);
          }
        }
      }
      else
      {
        //We always open leafs with nchild == 1 so check and possibly add child
        if(lnchild == 0)
        {
          exportBodyCount++;
        }
      }
    }
    depth++;
    levelList.swap();
    levelList.second().clear();
  }

  //Required space Multi*3 + size+cntr+smooth = 6
  return (1 + exportBodyCount + 6*exportNodeCount);
}

#endif


#if 0
inline int split_node_grav_impbh(
    const _v4sf nodeCOM1,
    const _v4sf boxCenter1,
    const _v4sf boxSize1)
{
  const int fullMask = static_cast<int>(0xFFFFFFFF);
  const int zeroMask = static_cast<int>(0x0);
  const _v4si mask = {fullMask, fullMask, fullMask, zeroMask};
  const _v4sf size = __abs(__builtin_ia32_shufps(nodeCOM1, nodeCOM1, 0xFF));

  //mask to prevent NaN signaling / Overflow ? Required to get good pre-SB performance
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
#endif

#if 0
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

template<typename T, int STRIDE>
void shuffle2vecAllocated(
    std::vector<T>   &data1,
    std::vector<T>   &data2,
    std::vector<T>   &rdata1,
    std::vector<T>   &rdata2,
    std::vector<int> &keys)
{
  const int n = data1.size();

  assert(n%STRIDE == 0);
  keys.resize(n/STRIDE);
  for (int i = 0, idx=0; i < n; i += STRIDE, idx++)
    keys[idx] = i;
  std::random_shuffle(keys.begin(), keys.end());

  rdata1.resize(n); //Safety only
  rdata2.resize(n); //Safety only
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

template<typename T, int STRIDE>
void shuffle3vecAllocated(
    std::vector<T>   &data1,
    std::vector<T>   &data2,
    std::vector<T>   &data3,
    std::vector<T>   &rdata1,
    std::vector<T>   &rdata2,
    std::vector<T>   &rdata3,
    std::vector<int> &keys)
{
  const int n = data1.size();

  assert(n%STRIDE == 0);
  keys.resize(n/STRIDE);
  for (int i = 0, idx=0; i < n; i += STRIDE, idx++)
    keys[idx] = i;
  std::random_shuffle(keys.begin(), keys.end());

  rdata1.resize(n); //Safety only
  rdata2.resize(n); //Safety only
  rdata3.resize(n); //Safety only
  for (int i = 0, idx=0; i < n; i += STRIDE, idx++)
  {
    const int key = keys[idx];
    for (int j = 0; j < STRIDE; j++)
    {
      rdata1[i+j] = data1[key+j];
      rdata2[i+j] = data2[key+j];
      rdata3[i+j] = data3[key+j];
    }
  }

  data1.swap(rdata1);
  data2.swap(rdata2);
  data3.swap(rdata3);
}

#endif

#if 0
  //Build the small-tree
  {

     extractGroupsTreeFull(
                           groupCentre, groupSize, groupSmoothing,
                           groupMulti, groupBody,
                           &localTree.boxCenterInfo[0],
                           &localTree.boxSizeInfo[0],
                           &localTree.boxSmoothing[0],
                           &localTree.multipole[0],
                           &localTree.bodies_Ppos[0],
                           smallTreeStart,
                           smallTreeEnd,
                           localTree.n_nodes, searchDepthUsed);

     int nGroups = groupCentre.size();
     assert(nGroups*3 == groupMulti.size());

     //Merge all data into a single array, store offsets
     const int nbody = groupBody.size();
     const int nnode = groupSize.size();

     LOGF(stderr, "ExtractGroupsTreeFull (small) n: %d [%d] Multi: %d body: %d Tot: %d \tTook: %lg\n",
            nGroups, (int)groupSize.size(), (int)groupMulti.size(), (int)groupBody.size(),
            1 + nbody + 6*nnode, get_time() - t1);

     SmallBoundaryTree.reserve(1 + nbody + 6*nnode); //header+bodies+size+cntr+smth+3*multi
     SmallBoundaryTree.clear();

     //Set the tree properties, before we exchange the data
     float4 description;
     description.x = host_int_as_float(nbody);
     description.y = host_int_as_float(nnode);
     description.z = host_int_as_float(smallTreeStart);
     description.w = host_int_as_float(smallTreeEnd);

     SmallBoundaryTree.push_back(description);
     SmallBoundaryTree.insert(SmallBoundaryTree.end(), groupBody.begin()  , groupBody.end());         //Particles
     SmallBoundaryTree.insert(SmallBoundaryTree.end(), groupSize.begin()  , groupSize.end());         //Sizes
     SmallBoundaryTree.insert(SmallBoundaryTree.end(), groupCentre.begin(), groupCentre.end());       //Centres
     SmallBoundaryTree.insert(SmallBoundaryTree.end(), groupSmoothing.begin(), groupSmoothing.end()); //Smoothing values
     SmallBoundaryTree.insert(SmallBoundaryTree.end(), groupMulti.begin() , groupMulti.end());        //Multipoles


     if(procId == 2){
         FILE *f = fopen("grpRef.txt", "w");
  for(int i=0; i < nnode; i++)
  {
      fprintf(f,"Node: %d Size\t%f %f %f %d\t| Cntr\t%f %f %f %f Smth\t%f Mult\t%f %f %f\n",
              i,
              groupSize[i].x,
              groupSize[i].y,
              groupSize[i].z,
              host_float_as_int(groupSize[i].w),
              groupCentre[i].x,
              groupCentre[i].y,
              groupCentre[i].z,
              groupCentre[i].w,

              groupSmoothing[i].x,

              groupMulti[i*3+0].x,
              groupMulti[i*3+1].x,
              groupMulti[i*3+2].x);
  }
  for(int i=0;  i < nbody; i++)
  {
      fprintf(f,"Body: %d Cntr\t%f %f %f %f\n",
              i,
              groupBody[i].x,
              groupBody[i].y,
              groupBody[i].z,
              groupBody[i].w);
  }
  fclose(f);
     }
     assert(SmallBoundaryTree.size() == (1 + nbody + 6*nnode));
//     fprintf(stderr,"Proc: %d Smalltree: %d %d %d %d \n",
//         procId, nbody,nnode, localTree.level_list[localTree.startLevelMin].x,
//         localTree.level_list[localTree.startLevelMin].y);
  }
#endif


#if 0
  //Build the full-tree
  {
     extractGroupsTreeFull(
       groupCentre, groupSize, groupSmoothing,
       groupMulti, groupBody,
       &localTree.boxCenterInfo[0],
       &localTree.boxSizeInfo[0],
       &localTree.boxSmoothing[0],
       &localTree.multipole[0],
       &localTree.bodies_Ppos[0],
       localTree.level_list[localTree.startLevelMin].x,
       localTree.level_list[localTree.startLevelMin].y,
       localTree.n_nodes, 99);

     int nGroups = groupCentre.size();
     assert(nGroups*3 == groupMulti.size());

     //Merge all data into a single array, store offsets
     const int nbody           = groupBody.size();
     const int nnode           = groupSize.size();

     LOGF(stderr, "ExtractGroupsTreeFull n: %d [%d] Multi: %d body: %d Tot: %d \tTook: %lg\n",
            nGroups, (int)groupSize.size(), (int)groupMulti.size(),(int)groupBody.size(),
            1 + nbody + 6*nnode, get_time() - t1);

     fullBoundaryTree.reserve(1 + nbody + 6*nnode); //header+bodies+size+cntr+3*multi
     fullBoundaryTree.clear();

     //Set the tree properties, before we exchange the data
     float4 description;
     description.x = host_int_as_float(nbody);
     description.y = host_int_as_float(nnode);
     description.z = host_int_as_float(localTree.level_list[localTree.startLevelMin].x);
     description.w = host_int_as_float(localTree.level_list[localTree.startLevelMin].y);

     fullBoundaryTree.push_back(description);
     fullBoundaryTree.insert(fullBoundaryTree.end(), groupBody.begin()  , groupBody.end());   //Particles
     fullBoundaryTree.insert(fullBoundaryTree.end(), groupSize.begin()  , groupSize.end());   //Sizes
     fullBoundaryTree.insert(fullBoundaryTree.end(), groupCentre.begin(), groupCentre.end()); //Centres
     fullBoundaryTree.insert(fullBoundaryTree.end(), groupSmoothing.begin(), groupSmoothing.end()); //Smoothing values
     fullBoundaryTree.insert(fullBoundaryTree.end(), groupMulti.begin() , groupMulti.end());  //Multipoles

     assert(fullBoundaryTree.size() == (1 + nbody + 6*nnode));

//     fprintf(stderr,"Proc: %d Bigtree: %d %d %d %d \n",
//         procId, nbody,nnode, localTree.level_list[localTree.startLevelMin].x,
//         localTree.level_list[localTree.startLevelMin].y);
  }
#endif

#endif

#ifndef _NODE_SPECS_H_
#define _NODE_SPECS_H_

typedef unsigned int uint;

typedef float real;
typedef float4 real4;

//Dont uncomment this yet
#define DO_BLOCK_TIMESTEP


//This will depend on the SPH method, I stick it here to be able to prevent
//having the loose constants around

/* Quinitc kernel */
//#define KERNEL_QUINTIC
//#define SPH_KERNEL_SIZE 3.0f
//#define PARAM_SMTH 1.0

/* M4 Cubic kernel */
#define KERNEL_M_4
#define SPH_KERNEL_SIZE 2.0f
#define  PARAM_SMTH 1.2


/* Phantom Wendland C6 kernel */
//#define KERNEL_W_C6
//#define SPH_KERNEL_SIZE 2.0f
//#define  PARAM_SMTH 1.6

/* Natsuki Wendland C6 kernel , works*/
//#define SPH_KERNEL_SIZE 3.5f
//#define  PARAM_SMTH 1.2


//TODO Remove this once we removed reference to dev_approximate_grvity_sph
#define SPH_KERNEL_SIZE2 (SPH_KERNEL_SIZE*SPH_KERNEL_SIZE)

//#define ADIABATIC_INDEX 1.4f
#define ADIABATIC_INDEX 5.0f/3.0f

//Enabling the following increases the number of particle properties
//exchanged during mpi particle exchange. Only required if you run
//block time steps. Not needed in the default shared time-step mode.
//#define DO_BLOCK_TIMESTEP_EXCHANGE_MPI

//Uncomment the next line to use thrust radix sort instead of built in one
// #define USE_THRUST

#if USE_DUST
  #if USE_MPI
    #error "Fatal, USE DUST does not work when using MPI. Its for demo only"
  #endif
#endif

class domainInformation
{
public:
    float4 domainSize;
    float3 minrange;
    float3 maxrange;

    domainInformation(){}

    domainInformation(const float3 _min, const float3 _max, const float periodicity) :
        minrange(_min), maxrange(_max)
    {
        domainSize.x = maxrange.x-minrange.x;
        domainSize.y = maxrange.y-minrange.y;
        domainSize.z = maxrange.z-minrange.z;
        domainSize.w = periodicity;
    }
};


struct bodyProps
{
    real4  *body_pos;
    real4  *body_vel;
    float2 *body_dens;
    float4 *body_grad;
    float4 *body_hydro;
    unsigned long long *ID;
};




typedef struct bodyStruct
{
  real4  pos;
  real4  vel;
  real4  acc0;
  real4  Ppos;
  real4  Pvel;
  float2 time;
  unsigned long long id;

  //SPH values
  float2 rhoH;
  float4 hydro;
  float4 drvt;




#ifdef DO_BLOCK_TIMESTEP_EXCHANGE_MPI

  uint4 key;
  real4 acc1;
#endif
} bodyStruct;




#define IMPBH   //Improved barnes hut opening method
//#define INDSOFT //Individual softening using cubic spline kernel

//Tree-walk and stack configuration
//#define LMEM_STACK_SIZE            3072         //Number of storage places PER thread, MUST be power 2 !!!!
//#define LMEM_STACK_SIZE             2048        //Number of storage places PER thread, MUST be power 2 !!!!

#define LMEM_EXTRA_SIZE            512
#define LMEM_STACK_SIZE            512         //Number of storage places PER thread, MUST be power 2 !!!!
#define CELL_LIST_MEM_PER_WARP     (LMEM_STACK_SIZE*32)
#if ((CELL_LIST_MEM_PER_WARP-1) & CELL_LIST_MEM_PER_WARP) != 0
#error "CELL_LIST_MEM_PER_WARP must be power of 2"
#endif

//#define LMEM_STACK_SIZE            1024         //Number of storage places PER thread, MUST be power 2 !!!!
//#define LMEM_STACK_SIZE            512         //Number of storage places PER thread, MUST be power 2 !!!!
// #define TREE_WALK_BLOCKS_PER_SM    32           //Number of GPU thread-blocks used for tree-walk
                                                //this is per SM, 8 is ok for Fermi architecture, 16 is save side

//Put this in this file since it is a setting
inline int getTreeWalkBlocksPerSM(int devMajor, int devMinor)
{
  switch(devMajor)
  {
    case 1:
      fprintf(stderr, "Sorry devices with compute capability < 2.0 are not supported \n");
      exit(0);
    case 2:     //Fermi
      return 16;     
    case 3:     //Kepler
      return 32;
    default:    //Future proof...
      return 32;
  }  
}

//Factor of extra memory we allocate during multi-GPU runs. By allocating a bit extra
//we reduce the number of memory allocations when particle numbers fluctuate. 1.1 == 10% extra
#define MULTI_GPU_MEM_INCREASE 1.1

//If USE_HASH_TABLE_DOMAIN_DECOMP is set to 1 we build a hash-table, otherwise we use
//sampling particles to get an idea of the domain space used to compute the domain
//decomposition
#define USE_HASH_TABLE_DOMAIN_DECOMP 0

//Number of processors to which we exchange the full domain. Also means if nProcs <= this
//number we will always do a full exchange
#define NUMBER_OF_FULL_EXCHANGE 16


#define TEXTURE_BOUNDARY  512   //Fermi architecture boundary for textures

#define MAXLEVELS 30

//Minimum number of nodes that is required  before we make leafs
#define START_LEVEL_MIN_NODES 16

#define BITLEVELS 27
#define ILEVELMASK 0x07FFFFFF
#define  LEVELMASK 0xF8000000

#define NLEAF 16
#define NCRIT 8    //No larger than 32 for SPH
//BEST PERF #define NCRIT 8
#define NTHREAD 128

#if NLEAF == 1

#define NLEAF2 1
#define LEAFBIT 31
#define BODYMASK 0x7FFFFFFF
#define INVBMASK 0x80000000


#elif NLEAF == 2

#define NLEAF2 2
#define LEAFBIT 30
#define BODYMASK 0x3FFFFFFF
#define INVBMASK 0xC0000000

#elif NLEAF == 4

#define NLEAF2 2
#define LEAFBIT 30
#define BODYMASK 0x3FFFFFFF
#define INVBMASK 0xC0000000


#elif NLEAF == 8

#define NLEAF2 3
#define LEAFBIT 29
#define BODYMASK 0x1FFFFFFF
#define INVBMASK 0xE0000000

#elif NLEAF == 16

#define NLEAF2 4
#define LEAFBIT 28
#define BODYMASK 0x0FFFFFFF
#define INVBMASK 0xF0000000

#elif NLEAF == 32

#define NLEAF2 5
#define LEAFBIT 27
#define BODYMASK 0x07FFFFFF
#define INVBMASK 0xF8000000

#elif NLEAF == 64

#define NLEAF2 6
#define LEAFBIT 26
#define BODYMASK 0x03FFFFFF
#define INVBMASK 0xFC000000

#elif NLEAF == 128

#define NLEAF2 7
#define LEAFBIT 25
#define BODYMASK 0x01FFFFFF
#define INVBMASK 0xFE000000

#else
#error "Please choose correct NLEAF available in node_specs.h"
#endif


#if NCRIT == 1

#define NCRIT2 2
#define CRITBIT 30
#define CRITMASK 0x3FFFFFFF
#define INVCMASK 0xC0000000


#elif NCRIT == 2

#define NCRIT2 2
#define CRITBIT 30
#define CRITMASK 0x3FFFFFFF
#define INVCMASK 0xC0000000


#elif NCRIT == 4

#define NCRIT2 2
#define CRITBIT 30
#define CRITMASK 0x3FFFFFFF
#define INVCMASK 0xC0000000


#elif NCRIT == 8

#define NCRIT2 3
#define CRITBIT 29
#define CRITMASK 0x1FFFFFFF
#define INVCMASK 0xE0000000

#elif NCRIT == 16

#define NCRIT2 4
#define CRITBIT 28
#define CRITMASK 0x0FFFFFFF
#define INVCMASK 0xF0000000

#elif NCRIT == 32

#define NCRIT2 5
#define CRITBIT 27
#define CRITMASK 0x07FFFFFF
#define INVCMASK 0xF8000000

#elif NCRIT == 64

#define NCRIT2 6
#define CRITBIT 26
#define CRITMASK 0x03FFFFFF
#define INVCMASK 0xFC000000

#elif NCRIT == 128

#define NCRIT2 7
#define CRITBIT 25
#define CRITMASK 0x01FFFFFF
#define INVCMASK 0xFE000000

#else
#error "Please choose correct NCRIT available in node_specs.h"
#endif

#if NTHREAD == 8

#define NTHREAD2 3

#elif NTHREAD == 16

#define NTHREAD2 4

#elif NTHREAD == 32

#define NTHREAD2 5

#elif NTHREAD == 64

#define NTHREAD2 6

#elif NTHREAD == 96

#define NTHREAD2 7

#elif NTHREAD == 128

#define NTHREAD2 7

#elif NTHREAD == 256

#define NTHREAD2 8

#else
#error "Please choose correct NTHREAD available in node_specs.h"
#endif





#if NCRIT < NLEAF
//#error "Fatal, NCRIT < NLEAF. Please check that NCRIT >= NLEAF"
#endif

#endif /* _NODE_SPECS_H_ */

#ifndef _NODE_SPECS_H_
#define _NODE_SPECS_H_

typedef unsigned int uint;

typedef float real;
typedef float4 real4;

//Dont uncomment this yet
#define DO_BLOCK_TIMESTEP

//Uncomment the next line to use thrust radix sort instead of built in one
// #define USE_THRUST

//If you uncomment the next line dust/massless particles will be treated
//like normal particles and used in the tree-construction
#if 0  /* DEFINED in CMakeLists.txt */
#define USE_DUST
#endif


#define IMPBH   //Improved barnes hut opening method
//#define INDSOFT //Individual softening using cubic spline kernel

//Tree-walk and stack configuration
//#define LMEM_STACK_SIZE            3072         //Number of storage places PER thread, MUST be power 2 !!!!
#define LMEM_STACK_SIZE            2048         //Number of storage places PER thread, MUST be power 2 !!!!
#define LMEM_EXTRA_SIZE            2048
//#define LMEM_STACK_SIZE            1024         //Number of storage places PER thread, MUST be power 2 !!!!
//#define LMEM_STACK_SIZE            512         //Number of storage places PER thread, MUST be power 2 !!!!
#define TREE_WALK_BLOCKS_PER_SM    32           //Number of GPU thread-blocks used for tree-walk
                                                //this is per SM, 8 is ok for Fermi architecture, 16 is save side

#define TEXTURE_BOUNDARY  512   //Fermi architecture boundary for textures

#define MAXLEVELS 30


#define BITLEVELS 27
#define ILEVELMASK 0x07FFFFFF
#define  LEVELMASK 0xF8000000

#define NLEAF 16
#define NCRIT 64
#define NTHREAD 128

#define NLEAFTEST 8

#if NLEAF == 8

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


#if NCRIT == 8

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
#error "Fatal, NCRIT < NLEAF. Please check that NCRIT >= NLEAF"
#endif

#endif /* _NODE_SPECS_H_ */

//  BONSAI.H
//
//  Every CUDA file includes this. It contains the wrapper for __global__
//  functions to allow easy redefinition of names and types.
//
//  The wrapper defaults evaluates to a standard definition.
//
#ifndef BONSAI_H
#define BONSAI_H

// Macro to map between separate compilation and non-separate compilation names
#ifndef KERNEL_DECLARE     // Avoids redefinition errors from multiple includes
#ifndef KERNEL_SEPARATE    // Separate declaration option
#define KERNEL_NAME(funcname)    funcname
#define KERNEL_DECLARE(funcname) extern "C" __global__ void KERNEL_NAME(funcname)
#else
#define KERNEL_NAME(funcname)    gpu_ ## funcname
#define KERNEL_DECLARE(funcname) __global__ void KERNEL_NAME(funcname)
#endif  // KERNEL_SEPARATE
#endif  // KERNEL_DECLARE


// These are the call-out routines to do kernel launches separately from when
// embedded inside classes. It allows for alternate launch paths.
#include "my_cuda_rt.h"
class octree;

void build_tree_node_levels(octree &tree, 
                            my_dev::dev_mem<uint>  &validList,
                            my_dev::dev_mem<uint>  &compactList,
                            my_dev::dev_mem<uint>  &levelOffset,
                            my_dev::dev_mem<uint>  &maxLevel,
                            cudaStream_t           stream);

#endif

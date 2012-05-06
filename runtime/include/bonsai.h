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
#define KERNEL_DECLARE(funcname) extern "C" __global__ void funcname
#else
#define KERNEL_SEPARATE(funcname) __global__ void gpu_ ## funcname
#endif  // KERNEL_SEPARATE
#endif  // KERNEL_DECLARE


#endif

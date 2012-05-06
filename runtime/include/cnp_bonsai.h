//  CNP_BONSAI.H
//
//  Every CUDA file includes this. It contains the wrapper for __global__
//  functions along with any other CNP-related definitions.
//
//  In a normal build, the wrapper evaluates to a standard definition.
//  In a CNP build, it wraps a static global with a cnp_ prefix.
#ifndef CNP_BONSAI_H
#define CNP_BONSAI_H

// Macro to map between separate compilation and non-separate compilation names
#ifndef CNP_DECLARE     // Avoids redefinition errors from multiple includes
#ifndef CNP_SEPARATE    // CNP files define this, then re-include the .cu file to duplicate kernels
#define CNP_DECLARE(funcname) extern "C" __global__ void funcname
#else
#define CNP_SEPARATE(funcname) __global__ void cnp_ ## funcname
#endif  // CNP_SEPARATE
#endif  // CNP_DECLARE


#endif

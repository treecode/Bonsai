/*
 * Copyright 2010 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

//
//  DERIVED_ATOMIC_FUNCTIONS.H
//
//  Certain 64-bit atomic functions are not available, so this defines
//  intrinsics which implement them as efficiently as possible via CAS.
//
//  NOTE: *** These do NOT work for shared-memory atomics at this time! ***
//
//  Sorry that it's a mess - supporting all architectures is a huge
//  pile of code spaghetti.
//
//  Functions added in this package are:
//      +------------------------------------------+
//      | Function  | int64 | uint64 | fp32 | fp64 |
//      +-----------+-------+--------+------+------+
//      | atomicOr  |       |   X    |      |      |
//      | atomicAnd |       |   X    |      |      |
//      | atomicXor |       |   X    |      |      |
//      | atomicMin |   X   |   X    |  X   |  X   |
//      | atomicMax |   X   |   X    |  X   |  X   |
//      | atomicAdd*|       |        |  X   |  X   |
//      +-----------+-------+--------+------+------+
//  *note for atomicAdd:  int64/uint64 already available on sm_13
//                        fp32 already available on sm_20
//                        int64/uint64 atomic min/max already on sm_35
//                        uint64 atomic and/or/xor already on sm_35
//
//  NOTE: Architectural limits still apply. i.e.:
//      sm_10 - supports no atomics
//      sm_11 - supports only 32-bit atomics, and no doubles
//      sm_12 - supports 64-bit integer atomics, but not doubles
//      sm_13 - supports everything
//      sm_20 - supports everything
//
//  TODO: Make these work with shared memory atomics by separating
//        out warp contention
//

#ifndef DERIVED_ATOMIC_FUNCTIONS_H
#define DERIVED_ATOMIC_FUNCTIONS_H

// Dummy functions for unsupported architecture
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 100)
__device__ unsigned long long atomicOr(unsigned long long *address, unsigned long long val) { return 0; }
__device__ unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) { return 0; }
__device__ unsigned long long atomicXor(unsigned long long *address, unsigned long long val) { return 0; }
__device__ long long atomicMin(long long *address, long long val) { return 0; }
__device__ unsigned long long atomicMin(unsigned long long *address, unsigned long long val) { return 0; }
__device__ long long atomicMax(long long *address, long long val) { return 0; }
__device__ unsigned long long atomicMax(unsigned long long *address, unsigned long long val) { return 0; }
__device__ float atomicMin(float *address, float val) { return 0; }
__device__ float atomicMax(float *address, float val) { return 0; }
__device__ double atomicMin(double *address, double val) { return 0; }
__device__ double atomicMax(double *address, double val) { return 0; }
__device__ double atomicAdd(double *address, double val) { return 0; }
#else

/**** Prototypes ****/
// longlong versions of int32 functions
#if (__CUDA_ARCH__ >= 120) && (__CUDA_ARCH__ < 350)
__device__ __forceinline__ unsigned long long atomicOr(unsigned long long *address, unsigned long long val);
__device__ __forceinline__ unsigned long long atomicAnd(unsigned long long *address, unsigned long long val);
__device__ __forceinline__ unsigned long long atomicXor(unsigned long long *address, unsigned long long val);

__device__ __forceinline__ long long atomicMin(long long *address, long long val);
__device__ __forceinline__ unsigned long long atomicMin(unsigned long long *address, unsigned long long val);
__device__ __forceinline__ long long atomicMax(long long *address, long long val);
__device__ __forceinline__ unsigned long long atomicMax(unsigned long long *address, unsigned long long val);
#endif

// Floating-point versions of int32 functions
__device__ __forceinline__ float atomicMin(float *address, float val);
__device__ __forceinline__ float atomicMax(float *address, float val);
#if __CUDA_ARCH__ >= 130
__device__ __forceinline__ double atomicMin(double *address, double val);
__device__ __forceinline__ double atomicMax(double *address, double val);

// Double-precision version of float functions
__device__ __forceinline__ double atomicAdd(double *address, double val);
#endif

// arch < sm_20 needs single precision atomicAdd as well
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ <= 130
__device__ __forceinline__ float atomicAdd(float *address, float val);
#endif


/**** Implementation ****/
#if (__CUDA_ARCH__ >= 120) && (__CUDA_ARCH__ < 350)        // Gives 64-bit atomic CAS
// uint64 atomicOr
__device__ __forceinline__ unsigned long long atomicOr(unsigned long long *address, unsigned long long val)
{
    unsigned long long old, ret = *address;
    do {
        old = ret;
    } while((ret = atomicCAS(address, old, old | val)) != old);
    return ret;
}

// uint64 atomicAnd
__device__ __forceinline__ unsigned long long atomicAnd(unsigned long long *address, unsigned long long val)
{
    unsigned long long old, ret = *address;
    do {
        old = ret;
    } while((ret = atomicCAS(address, old, old & val)) != old);
    return ret;
}

// uint64 atomicXor
__device__ __forceinline__ unsigned long long atomicXor(unsigned long long *address, unsigned long long val)
{
    unsigned long long old, ret = *address;
    do {
        old = ret;
    } while((ret = atomicCAS(address, old, old ^ val)) != old);
    return ret;
}


// int64 atomicMin
__device__ __forceinline__ long long atomicMin(long long *address, long long val)
{
    long long ret = *address;
    while(val < ret)
    {
        long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
            break;
    }
    return ret;
}

// uint64 atomicMin
__device__ __forceinline__ unsigned long long atomicMin(unsigned long long *address, unsigned long long val)
{
    unsigned long long ret = *address;
    while(val < ret)
    {
        unsigned long long old = ret;
        if((ret = atomicCAS(address, old, val)) == old)
            break;
    }
    return ret;
}

// int64 atomicMax
__device__ __forceinline__ long long atomicMax(long long *address, long long val)
{
    long long ret = *address;
    while(val > ret)
    {
        long long old = ret;
        if((ret = (long long)atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
            break;
    }
    return ret;
}

// uint64 atomicMax
__device__ __forceinline__ unsigned long long atomicMax(unsigned long long *address, unsigned long long val)
{
    unsigned long long ret = *address;
    while(val > ret)
    {
        unsigned long long old = ret;
        if((ret = atomicCAS(address, old, val)) == old)
            break;
    }
    return ret;
}
#endif // (__CUDA_ARCH__ >= 120) && (__CUDA_ARCH__ < 350)

// For all float & double atomics:
//      Must do the compare with integers, not floating point,
//      since NaN is never equal to any other NaN

// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

#if __CUDA_ARCH__ >= 130
// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// Double-precision floating point atomic add
__device__ __forceinline__ double atomicAdd(double *address, double val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    unsigned long long *ptr = (unsigned long long *)address;
    unsigned long long old, newdbl, ret = *ptr;
    do {
        old = ret;
        newdbl = __double_as_longlong(__longlong_as_double(old)+val);
    } while((ret = atomicCAS(ptr, old, newdbl)) != old);
    
    return __longlong_as_double(ret);
}
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 130)
// Single-precision floating point atomic add
__device__ __forceinline__ float atomicAdd(float *address, float val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    unsigned int *ptr = (unsigned int *)address;
    unsigned int old, newint, ret = *ptr;
    do {
        old = ret;
        newint = __float_as_int(__int_as_float(old)+val);
    } while((ret = atomicCAS(ptr, old, newint)) != old);
    
    return __int_as_float(ret);
}
#endif

#endif  // DERIVED_ATOMIC_FUNCTIONS_H

#endif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 100)


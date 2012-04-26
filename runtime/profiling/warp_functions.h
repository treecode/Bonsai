/*
 * Copyright 2010 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

//
//  WARP_FUNCTIONS.H
//
//  A bunch of convenience functions for manipulating warp-level programming
//  in CUDA. It's not officially supported and so use of *any* of these could
//  result in unspecified behaviour. However, it's useful to be able to do
//  some of these things and here they are.
//
//  With the exception of __flo() and __clz(), all functions here work on any
//  architecture (sm_10 -> sm_20, at time of writing).
//
//  In brief, the functions contained here are:
//
//  Indexing functions. Not that these are *physical* indices, which means
//      they relate to hardware state. Thus if 10 warps are running on an SM,
//      it may be that none of them have an ID of "0".
//  IMPORTANT: These are all *mutable* numbers. That means that they may
//      change at any time. On future architectures they will do so. This
//      makes them not very useful at all for robust, future-proof code.
//
//    __laneid() - Returns the calling thread's index within its warp (0-31)
//    __warpid() - Returns the calling thread's warp ID within the SM (Fermi: 0-47, Tesla: 0-31)
//    __smid()   - Returns the calling thread's SM ID within the GPU (Fermi: 0-16, Tesla: 0-30)
//    __gridid() - Returns a unique grid identifier, as maintained by the hardware
//
//  Fermi-specific evaluation functions for which no CUDA intrinsic exists
//    __flo()    - Retuns the index of the first "1" in a 32-bit word,
//                 starting from bit 31 counting down
//    __clz()    - Returns the number of zeros before the first "1" in a 32-bit
//                 word, starting from bit 0
//    __activeid() - Returns the 0-based index of this thread in the warp,
//                   counting only the active threads (i.e if threads 2,3,18,19 & 22
//                   are active, the index of thread 19 is "3")
//
//  Fermi-specific functions adapted for Tesla architecture
//    __ballot()  - Requires vote intrinsics (sm >= 1.2) but is very handy
//
//  Utility functions for warp-synchronous programming. These are extremely
//  dangerous because the compiler is permitted to move code around, making
//  code which looks correct behave incorrectly. Use these at your own risk.
//    __masterlane()   - Elects one active lane in the calling thread's warp,
//                       and returns its lane ID
//    __masterthread() - Returns TRUE for exactly one thread in a warp
//                       (i.e. if my lane ID == master lane ID)
//
//  NOTE: The __masterlane() and __masterthread() functions use 32 words
//      of shared memory for arch <= sm_11, because of the lack of warp vote
//      instruction.
//
//  NOTE: __masterlane() returns different values for different architectures.
//        In particular, sm_12 & sm_13 will return 0 if all lanes are active, while
//        sm_20 will return 31. sm_10 and sm_11 will return a random value in this
//        case.
//

#ifndef WARP_FUNCTIONS_H
#define WARP_FUNCTIONS_H

// Inline PTX to retrieve lane, warp, sm and grid id
__device__ __forceinline__ unsigned int __laneid() { unsigned int laneid; asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid)); return laneid; }
__device__ __forceinline__ unsigned int __warpid() { unsigned int warpid; asm volatile ("mov.u32 %0, %warpid;" : "=r"(warpid)); return warpid; }
__device__ __forceinline__ unsigned int   __smid() { unsigned int smid;   asm volatile ("mov.u32 %0, %smid;"   : "=r"(smid));   return smid;   }
__device__ __forceinline__ unsigned int __gridid() { unsigned int gridid; asm volatile ("mov.u32 %0, %gridid;" : "=r"(gridid)); return gridid; }

#if __CUDA_ARCH__ >= 200            // Fermi only
// Inline PTX to return index of first "1" (MSB-first) in a unsigned int
__device__ __forceinline__ unsigned int __flo(unsigned int word) { unsigned int ret; asm volatile ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word)); return ret; }
__device__ __forceinline__ unsigned int __flo(unsigned long long dword) { unsigned int ret; asm volatile ("bfind.u64 %0, %1;" : "=r"(ret) : "l"(dword)); return ret; }

// Inline PTX to return number of leading zeroes (from MSB) in a unsigned int
__device__ __forceinline__ unsigned int __clz(unsigned int word) { unsigned int ret; asm volatile ("clz.b32 %0, %1;" : "=r"(ret) : "r"(word)); return ret; }
__device__ __forceinline__ unsigned int __clz(unsigned long long dword) { unsigned int ret; asm volatile ("clz.b64 %0, %1;" : "=r"(ret) : "l"(dword)); return ret; }
#endif

// Often you want to select a single active thread from a warp and
// use it as the proxy for the entire warp (i.e. when taking a lock).
// This is possible without shared memory on any architecture >= 1.2
// but requires shared memory for 1.1 and 1.0. Fermi has specific
// instructions to help with this.

#if __CUDA_ARCH__ >= 100        // Device-side code
// We start with a common "am I the master thread" macro which returns
// TRUE for exactly one thread in a warp. All the work is done in __masterlane().
#define __masterthread()    (__masterlane() == __laneid())

#if __CUDA_ARCH__ >= 200            // Fermi
// For Fermi we can just make an inline PTX macro
#define __masterlane()      __flo(__ballot(1))
#elif __CUDA_ARCH__ >= 120          // GT2xx
// For GT2xx we have "vote.any" so we can do a reduction
__device__ __forceinline__ int __masterlane()
{
    unsigned int myid = __laneid();
    
	// Acceleration for lane=0, because I think it'll be common...
	if(__any(myid == 0))
		return 0;

	// ...otherwise do the 5-step reduction to get active lane ID
	unsigned int ret = 16;
    ret += 8 + (-16 * __any(myid < ret));   // Avoid conditionals
    ret += 4 + ( -8 * __any(myid < ret));
    ret += 2 + ( -4 * __any(myid < ret));
    ret += 1 + ( -2 * __any(myid < ret));
    ret -= __any(myid < ret);
	return ret;
}

// Ballot can be (sequentially) enacted through repeated voting
__device__ __forceinline__ unsigned int __ballot(int condition)
{
    unsigned int ballot = 0;
    unsigned int mylane = __laneid();
    for(int lane=0; lane<32; lane++)
    {
        if(__any(condition && (mylane == lane)))
            ballot |= (1 << lane);
    }
    return ballot;
}

#else                               // G8x & G9x (no vote intrinsic)
// For G80 & G9x we have no vote so must use shared memory.
// Shared memory is okay because we're warp-synchronous so no syncthreads is needed.
// This takes 128 bytes of shared memory per CTA
// 32 warps on G80/G9x. Volatile ensures compiler does not fuse "lanevote = " with "return lanevote".
__shared__ volatile unsigned int __cuda_warp_lanevote[32];
__device__ __forceinline__ int __masterlane()
{
    unsigned int mywarp = __warpid();
    unsigned int mylane = __laneid();
    __cuda_warp_lanevote[mywarp] = mylane;

    // No need to syncthreads because this is warp-synchronous programming
    // (and we are assuming divergent code so it wouldn't work anyway!)
    return __cuda_warp_lanevote[mywarp];
}

// Ballot on G80/G9x means using shared memory. Otherwise works the same way.
__device__ __forceinline__ unsigned int __ballot(int condition)
{
    unsigned int mywarp = __warpid();
    unsigned int mylane = __laneid();
    __cuda_warp_lanevote[mywarp] = 0;

    for(int lane=0; lane<32; lane++)
    {
        if(condition && (lane == mylane))
             __cuda_warp_lanevote[mywarp] |= (1 << lane);
    }
    return __cuda_warp_lanevote[mywarp];
}
#endif

#if __CUDA_ARCH__ >= 200
// The "active ID" is the index of this thread among those which are
// currently active in the warp.
#define __activeid() __popc(__ballot(1) << (warpSize-__laneid()))
#endif

#else       // Host-side code
// Host side has only one thread, so 
#define __masterlane()      0
#define __masterthread()    1
#define __activeid()        0
#endif

#endif

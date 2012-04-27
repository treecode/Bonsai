//
//  cuxTimer_private.h
//
//  Private definitions for internal use within cuxTimer - not visible to
//  the outside world.
//

#ifndef CUXTIMER_PRIVATE_H
#define CUXTIMER_PRIVATE_H

// Development only: Define this for internal debug messages
//#define _CUXTIMER_DEBUG

// Architecture-specific stuff (fermi has clock64_t, tesla has only clock_t, etc)
#if __CUDA_ARCH__ > 0
#if __CUDA_ARCH__ >= 200        // GF1xx
#define AtomicType  unsigned long long
#define ClockType   unsigned long long
#define timerNowTime() clock64()

#else                           // GT2xx
#define ClockType clock_t
#define timerNowTime() clock()

// SM1.1 has no 64-bit atomics, so all our accumulators must be 32 bits for sm11
#if __CUDA_ARCH__ >= 120
#define AtomicType  unsigned long long
#else
#define AtomicType  unsigned int
#endif

#endif  // __CUDA_ARCH__ >= 200
#endif  // __CUDA_ARCH__ >= 0

// For defining the data structures, we have a problem: different
// architectures have different sizes. This means the host cannot
// know what we compiled for and needs a "one-size-fits-all" solution.
// We'll therefore use the maximum #sms on all architectures, etc,
// to size our data structures. It won't matter in global memory anyway
// but in local we'd have to care.
#define __CUXTIMER_MAXSMS     30          // 30 SMs on Tesla
#define __CUXTIMER_MAXWARPS   48          // 48 warps on Fermi
#define __CUXTIMER_STACKLEN   32          // Max call depth of 32!

// This is the per-warp tracking structure. It holds the stack of calls
// which the thread has made.
typedef struct _timerStack_st {
    unsigned long long curr_time;               // Tick at which we began tracking this ID (note longlong, even on Tesla, to ease host-side code)
    unsigned long long accum[__CUXTIMER_STACKLEN];// Total accumulated time at this level for "max" value  (so as not to count time spent in child functions)
    unsigned int previd[__CUXTIMER_STACKLEN];     // Stack for storing previous IDs
    unsigned int occupancy[__CUXTIMER_STACKLEN];  // Stack for storing occupancy info at a given level (to handle divergent exit)
    unsigned int stackoffs;                     // Offset into stack
    unsigned int curr_id;                       // Current function ID that's being tracked
} _timerStack;

// This is the tracking data. We aggregate this across the
// entire machine. We track:
//  1. Maximum time spent by any one warp on a given function
//  2. Number of times a given function was entered
//  3. Total time, aggregated over all warps, during which a function was active
//  4. A string associated with this timing ID (_timerStrings, below)
typedef struct _timerData_st {
    unsigned long long idtime[__CUXTIMER_IDS];        // Total (summed over all threads) time spent in this ID
    unsigned long long maximum[__CUXTIMER_IDS];       // Maximum time spent by any one thread
    unsigned int count[__CUXTIMER_IDS];               // Number of times this ID was called (per-warp, NOT per-thread!)
} _timerData;


// This holds string names of functions, for convenience of reading
typedef struct _timerStrings_st {
    unsigned int nameMask[(__CUXTIMER_IDS >> 5)+2];                           // Mask of used ID function names
    char idNames[__CUXTIMER_IDS][256];                                        // Copy of function names for display
    unsigned int descrMask[(__CUXTIMER_IDS >> 5)+2];                          // Mask of used ID descriptions
    char idDescrs[__CUXTIMER_IDS][256];                                       // Copy of descriptions for display
} _timerStrings;

// Macro to determine if an id bit in the mask array is set. We index
// the array by id/32, and then pick a bit based on the shift of id%32.
#define _timer_isNameSet(mask, id)   ((mask[(id) >> 5] & (1 << ((id) & 31))) != 0)

// Initialisation flow, to allow only one thread to do the init. We use defines
// not an enum so as to avoid namespace pollution
#define __CUXTIMERINIT_NOTINITIALISED 0     // Constants start set to 0, so "not initialised" should be 0
#define __CUXTIMERINIT_INITIALISING   (__CUXTIMERINIT_NOTINITIALISED+1)
#define __CUXTIMERINIT_INITIALISED    (__CUXTIMERINIT_NOTINITIALISED+2)
#define __CUXTIMERINIT_FAILED         (__CUXTIMERINIT_NOTINITIALISED+3)


// These are separate configuration values, which must be marked volatile
typedef struct _timerConfig_st {
    volatile unsigned int init;
} _timerConfig;


// The overall storage is the function-tracking info, and the stack
// management info. We could do this per-SM, but in this iteration
// of cuxTimer I want to track all counters globally. I *always* need
// to track the stack per-SM, however (it already has per-warp info).
// All storage should start out zeroed-out.
// NOTE: This struct **must** start with _timerData.
typedef struct _timerGlobals_st {
    _timerData funcdata/*[__CUXTIMER_MAXSMS]*/;                          // Per-ID data
    _timerStack stack[__CUXTIMER_MAXSMS][__CUXTIMER_MAXWARPS];           // Per-warp stack
    _timerStrings strings;                                               // User-defined timing data
    unsigned int autocount;                                              // Auto-display counter
} _timerGlobals;

#undef __CUXTIMER_MAXSMS
#undef __CUXTIMER_MAXWARPS

#endif      // CUXTIMER_PRIVATE_H

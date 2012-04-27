//
//  cuxTimer.h
//
//  Enables per-function (or per-{}-section) timing on Fermi.
//
//  Basic timing functions, wrapped in a C++ class to have the convenience
//  of constructors and destructors handle scoping. This means you only have
//  to insert a single line to time a function, instead of a pair of start/end
//  calls.
//
//  Remember to include cuxTimer.cu somewhere in your program for the
//  implementation of these.
//

#ifndef CUXTIMER_H
#define CUXTIMER_H

#include <stdio.h>

// If the timer has been globally disabled, we define some basic equivalents
// to avoid warnings.
#ifdef CUXTIMER_DISABLE
// Allows "cuxTimer _p(xxx, yyy, zzz)": compiler will eliminate as dead code
class cuxTimer {      
public:
    __device__ cuxTimer(unsigned int id=0, const char *name=NULL, const char *descr=NULL) {}
};
#define cudaxTimerReset()
#define cudaxTimerDisplay(...)
#define CUXTIMER(...)

// Here are the real definitions which run the whole show
#else

// Define this to enable per-warp stats, instead of per-thread stats (given all threads
// in a warp wait for the slowest one anyway). It actually gives slightly more accurate
// results and runs faster, but is less intutitive for humans to read.
// Note we always define it for SM1.1, because 32-bit atomics will too easily overflow
// for per-thread counter tracking
#if __CUDA_ARCH__ == 110
#define __CUXTIMER_PER_WARP_STATS
#endif

// This is the maximum number of IDs which can be supported
#define __CUXTIMER_IDS        1024        // Must be a power of 2!

// Cannot noinline pointers on SM <= 1.3, so this define eliminates warnings
#if __CUDA_ARCH__ >= 200
#define TIMER_NOINLINE   __noinline__
#else
#define TIMER_NOINLINE
#endif

// Class definition so we can use auto-scoping to call the destructor
class cuxTimer {
public:
    // Note no-inline functions to avoid register over-use problems. 
    __device__ TIMER_NOINLINE  cuxTimer(const char *autoname, unsigned int autoid, const char *autodesc, const char *username, unsigned int userid=(unsigned)-1, const char *userdesc=NULL);
    __device__ TIMER_NOINLINE  cuxTimer(const char *autoname, unsigned int autoid, const char *autodesc, unsigned int id, const char *username=NULL, const char *userdesc=NULL);
    __device__ TIMER_NOINLINE  cuxTimer(const char *name, unsigned int id, const char *desc=NULL);
    __device__ TIMER_NOINLINE ~cuxTimer();
    static __device__ __noinline__ int Init();
    static __device__ __noinline__ void Reset();
    static __device__ __noinline__ void Display(int csv=false, int show_headings=true);

private:
    int invalid;               // Set to TRUE if this is not a validly-constructed timing entry
    static __device__ __forceinline__ void Start(unsigned int id, const char *name, const char *descr, unsigned long long nowtime);    // Call with a unique ID per function
    static __device__ __forceinline__ void Stop();
};

// Auto-display class runs along similar lines, with auto-destruct doing the work
// Because of how macro expansion works, we need an "unused" argument in the constructor.
class cuxTimerAutoDisplay {
public:
    __device__ TIMER_NOINLINE cuxTimerAutoDisplay(int unused, int csv=false, int show_headings=true, int reset=false);
    __device__ TIMER_NOINLINE ~cuxTimerAutoDisplay();
private:
    // Store incoming variables so we can use them later
    int auto_csv;
    int auto_show_headings;
    int auto_reset;
};


// Host-callable functions
extern __host__ void cudaxTimerReset();
extern __host__ void cudaxTimerDisplay(FILE *fp=stdout, int csv=false, int show_headings=true);


// Convenience macros

// Create a timing variable with a named, unique ID. We include function name
// and line number for auto-naming. The varargs are the user-specified extras
// which fall into default arguments in the cuxTimer constructor which handles
// these things.
//
// We support the following CUXTIMER macro args (all args are optional):
//  CUXTIMER()                 // Fully-automatic name/id/desc generation
//  CUXTIMER(name)             // Automatic ID generation with given name. No desc.
//  CUXTIMER(name, id)         // Uses given name/id, no desc
//  CUXTIMER(name, id, desc)   // Uses all things as passed in
//
// Arguments are: (id, file:line description, function:line description, user args)
// If the user does not give a string, we use the "function:line" as the string.
// If the user does give an ID, we use it instead of the passed-in ID.
// Auto-generated IDs start at __CUXTIMER_IDS and count down, so as to minimise
// likelihood of intersecting with user-selected IDs.

// Note the weird macros which appear to do nothing - this is because the preprocessor
// expands down to a depth of 2, so "__COUNTER__" only converts to an integer when it
// hits the second macro down. Likewise for TOSTRING/STRINGIFY. Nasty hack, but it works.
#define __CUXTIMER_VARNAME(id) _cuxTime_##id
#define __CUXTIMER_STRINGIFY(arg) #arg
#define __CUXTIMER_TOSTRING(arg) __CUXTIMER_STRINGIFY(arg)

// Linux cannot use __FUNCTION__, so we replace the default name with the filename.
// It will at least guarantee uniqueness, even if it's not as pretty. Note that for
// CUDA 4.0 and later, __FUNCTION__ does work under linux, but I can't figure out how
// to detect that.
#if defined(_WIN32)
#define __CUXTIMER_DEFAULT_NAME (__FUNCTION__ ":" __CUXTIMER_TOSTRING(__LINE__))
#else
#define __CUXTIMER_DEFAULT_NAME (__FILE__ ":" __CUXTIMER_TOSTRING(__LINE__))
#endif

// Finally, we have the building blocks needed to create the timing point.
#define __CUXTIMER_CREATE(id, ...) \
    cuxTimer __CUXTIMER_VARNAME(id)(__CUXTIMER_DEFAULT_NAME, \
                                    __CUXTIMER_IDS-2-id, \
                                    "[" __FILE__ ":" __CUXTIMER_TOSTRING(__LINE__) "]", \
                                    ##__VA_ARGS__)

// __COUNTER__ requires gcc 4.3, or windows: the CUXTIMER macro won't work without
#if  defined(_WIN32) || (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ > 2)
#define CUXTIMER(...) __CUXTIMER_CREATE(__COUNTER__, ##__VA_ARGS__)

// Without __COUNTER__, we limit what the CUXTIMER macro can do:
//  - If you do not specify "name" and "id", the current line number is
//    used, which runs the risk of an ID conflict from another file.
//    Also, this breaks auto-ID if a file has more than __CUXTIMER_IDS lines.
//    It's a poor hack, therefore, but I see no other way around it.
#else
#define CUXTIMER(...) __CUXTIMER_CREATE(__LINE__, ##__VA_ARGS__)
#endif      // WIN32 || >=GCC4.3

// CUXTIMER_AUTODISPLAY is simple - it's only called at the top of __global__
// so we can play fast and loose with the name.
#define CUXTIMER_AUTODISPLAY(...) cuxTimerAutoDisplay _cuxTime_AutoDisplay(0, ##__VA_ARGS__)

#endif      // CUXTIMER_DISABLE
#endif      // CUXTIMER_H

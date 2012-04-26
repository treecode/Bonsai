//
//  cuxTimer.cu
//
//  Basic timing functions, wrapped in a C++ class to have the convenience
//  of constructors and destructors handle scoping. This means you only have
//  to insert a single line to time a function, instead of a pair of 
//  start/end calls.
//
//  Timing is done by "ID", which is a unique identifier number (32-bit)
//  passed as argument to the cuxTimer() constructor. ID=0 is reserved.
//  You may also pass a string after the ID, for more legible output.
//
//  Note you must call cudaxTimerInit() (from cuxTimer_host.cu) before
//  the first kernel launch to use any of these timing functions.
//

#include "cuxTimer.h"   // Always include this - it's protected internally of course

#ifndef CUXTIMER_CU
#define CUXTIMER_CU

#ifndef CUXTIMER_DISABLE
#include "cuxTimer_private.h"
#include "warp_functions.h"
#include "derived_atomic_functions.h"

// Macros for printf output
#ifdef _CUXTIMER_DEBUG
    #if __CUDA_ARCH__ > 0 && __CUDA_ARCH__ < 200
        #include "../cuPrintf.cu"
        #define _CUXTIMER_PRINTF cuPrintf
    #else
        #define _CUXTIMER_PRINTF printf
        #define cudaPrintfInit(...)
        #define cudaPrintfDisplay(...)
    #endif
    #define TIMERPRINTF(fmt, ...) _CUXTIMER_PRINTF("[%d, %d] " fmt, blockIdx.x, threadIdx.x, ##__VA_ARGS__)
#else
#define TIMERPRINTF(...)
#endif

// Data structures used to record timings
// Our main data is of fixed, known size so we can just globally declare it
static __device__ _timerGlobals _cuxtimer_globals;

// We also have a small amount of constant data for fast lookup. It's mirrored
// in device memory for cases where we're not initialised from the host.
static __constant__ _timerConfig _cuxtimer_constants;
static __device__ volatile _timerConfig _cuxtimer_config;


// Only build device code during device compilation (but need above for host code)
#if __CUDA_ARCH__ > 0

// Macros to make access of these more readable.
// Each "funcdata" needs to be suffixed by [id_value]
#define timerMax    _cuxtimer_globals.funcdata/*[smid][warpid]*/.maximum
#define timerCount  _cuxtimer_globals.funcdata/*[smid][warpid]*/.count
#define timerTimes  _cuxtimer_globals.funcdata/*[smid][warpid]*/.idtime

// Macros for use of the stack. The first three need an array indexed
// by stack offset.
#define timerPrevID    _cuxtimer_globals.stack[smid][warpid].previd//[stackOffset]
#define timerOccupancy _cuxtimer_globals.stack[smid][warpid].occupancy//[stackOffset]
#define timerAccum     _cuxtimer_globals.stack[smid][warpid].accum//[stackOffset]
#define stackOffset  (_cuxtimer_globals.stack[smid][warpid].stackoffs)
#define currId       (_cuxtimer_globals.stack[smid][warpid].curr_id)
#define currTime     (_cuxtimer_globals.stack[smid][warpid].curr_time)

// Macros to use the pure global data in the timing structure
#define globalNameMask  _cuxtimer_globals.strings.nameMask
#define globalIDName    _cuxtimer_globals.strings.idNames
#define globalDescrMask _cuxtimer_globals.strings.descrMask
#define globalIDDescr   _cuxtimer_globals.strings.idDescrs

// Init-state access macros
#define timerInitState       (_cuxtimer_config.init)

// Auto-display variable
#define timerAutoDisplayCount   (_cuxtimer_globals.autocount)


//
//  timerStrCpy
//
//  Basic string copy
//
static __device__ __forceinline__ void timerStrCpy(char *dst, const char *src, int n)
{
    for(int j=0; j<n; j++)
    {
        if((*dst++ = *src++) == 0)
            break;
    }
    *dst = 0;
}


//
//  timerPush
//
//  Pushes info onto the per-thread stack (for tracking returns).
//  We track the ID at the new stack offset, and how many threads
//  are entering it (in case they leave it at different times).
//  If we overflow the stack, nothing gets pushed.
//
static __device__ __forceinline__ void timerPush(int smid, int warpid, unsigned int id, unsigned int count)
{
    int offset = ++stackOffset;
    //TIMERPRINTF("Pushed id %d at stack offset %d (%d threads)\n", id, offset, count);
    if(offset < __CUXTIMER_STACKLEN)
    {
        timerPrevID[offset] = id;
        timerOccupancy[offset] = count;
        timerAccum[offset] = 0;             // No time accumulated at this depth so far
    }
}


//
//  timerPop
//
//  Pops the most recent ID off the stack. If the stack is in overflow,
//  the same ID is returned. If the stack is empty then we're done
//  timing.
//
//  If we're done timing (i.e. ID becomes 0) then we use this chance
//  to copy into global memory all timer names which have been used.
//  This allows the host-side display to print the name string correctly.
//
static __device__ __forceinline__ unsigned int timerPop(int smid, int warpid)
{
    int offset = stackOffset--;
    //TIMERPRINTF("Popping [%d,%d] from stack offset %d\n", timerPrevID[offset], timerOccupancy[offset], offset);

    // If stack is in overflow, return the last-known ID
    if(offset >= __CUXTIMER_STACKLEN)
        return timerPrevID[__CUXTIMER_STACKLEN-1];

    // If we're in legal stack range, return the top
    if(offset > 0)
        return timerPrevID[offset];

    // At this point, we've drained the stack. That means our
    // very first timing datapoint is out of range. This should
    // never happen, but reset the stack pointer just in case.
    stackOffset = 0;
    return 0;
}


//
// Constructor. This actually checkpoints the old ID and begins timing the new.
//  We use "start()" to make things happen.
//
//  We have three versions: one where there's a simple "name, id, desc",
//  one with "id, name, desc", and one permitting complex macro operation.
//  This allows for a fastest-possible-path and minimal register use
//  (compared to having a single function and determining how it was called
//  at runtime).
//
//  Note, coming in here, we're *guaranteed* that the user has used the CUXTIMER()
//  macro, with at least one argument. In this case, we allow auto-assignment of
//  the ID number but *not* auto-assignment of the description.
//
//  All empty invocations of CUXTIMER() will go through the fast-path below.
//
__device__ TIMER_NOINLINE cuxTimer::cuxTimer(const char *autoname, unsigned int autoid, const char *autodesc, const char *username, unsigned int userid, const char *userdesc) : invalid(0)
{
    // First thing: record the time!
    ClockType now = timerNowTime();

    //TIMERPRINTF("Auto-construct: %s,%d,%s -> %s,%d,%s\n", autoname?autoname:"(null)", autoid, autodesc?autodesc:"(null)", username?username:"(null)", userid, userdesc?userdesc:"(null)");
    // Figure out if we should use the auto-args or the user-supplied args
    unsigned int id = ((userid == (unsigned)-1) ? autoid : userid) + 1;
    if((id > 0) && (id < __CUXTIMER_IDS))
        Start(id, username, userdesc, now);  // "username" is guaranteed valid here. Always uses "userdesc"!
    else
        invalid = 1;
}

__device__ TIMER_NOINLINE cuxTimer::cuxTimer(const char *autoname, unsigned int autoid, const char *autodesc, unsigned int id, const char *username, const char *userdesc) : invalid(0)
{
    // First thing: record the time!
    ClockType now = timerNowTime();

    //TIMERPRINTF("Auto-construct: %s,%d,%s -> %s,%d,%s\n", autoname?autoname:"(null)", autoid, autodesc?autodesc:"(null)", username?username:"(null)", userid, userdesc?userdesc:"(null)");
    // Figure out if we should use the auto-args or the user-supplied args
    id++;
    if((id > 0) && (id < __CUXTIMER_IDS))
        Start(id, (username == NULL) ? autoname : username, userdesc, now);  // Select autoname if we weren't given a username.
    else
        invalid = 1;
}

// Fast-path version
__device__ TIMER_NOINLINE cuxTimer::cuxTimer(const char *name, unsigned int id, const char *desc) : invalid(0)
{
    // First thing: record the time!
    ClockType now = timerNowTime();

    //TIMERPRINTF("Fast-path construct ID=%d, name=%s, descr=%s\n", id, name ? name : "(null)", desc ? desc : "(null)");
    id++;
    if((id > 0) && (id < __CUXTIMER_IDS))
        Start(id, name, desc, now);
    else
        invalid = 1;
}


//
// Destructor. This stops timing the current ID and backs off to the
// previous one automatically.
//
__device__ TIMER_NOINLINE cuxTimer::~cuxTimer()
{
    if(!invalid)
        Stop();
}


//
//  Start
//
//  Static member function which pushes the current ID and starts timing
//  with the new ID. The constructor uses this.
//
__device__ __forceinline__ void cuxTimer::Start(unsigned int id, const char *name, const char *descr, unsigned long long nowtime)
{
    // Now figure out how many threads have entered here
    int count = __popc(__ballot(1));

    // Only the master-thread of a warp need do this
    if(__masterthread())
    {
        ClockType now = (ClockType)nowtime;

        // If we weren't set up from the host, we need to do a lazy-init check
        if((_cuxtimer_constants.init == __CUXTIMERINIT_NOTINITIALISED) && (timerInitState != __CUXTIMERINIT_INITIALISED))
        {
            // Abort if we've failed before, or if Init fails
            if((timerInitState == __CUXTIMERINIT_FAILED) || (Init() == false))
            {
                TIMERPRINTF("cuxTimer: Timer init failed!\n");
                return;
            }
        }

        // If we have a function name, register it to this ID.
        unsigned int mask = 1 << (id & 31);
        if(name != NULL)
        {
            // Record in a bitmask those names which are used.
            // Copy the ID function name as soon as we realise we need to.
            if((atomicOr(&(globalNameMask[id >> 5]), mask) & mask) == 0)
                timerStrCpy(globalIDName[id], name, 255);
        }
        // If we have a description, register it to this ID.
        if(descr != NULL)
        {
            if((atomicOr(&(globalDescrMask[id >> 5]), mask) & mask) == 0)
                timerStrCpy(globalIDDescr[id], descr, 255);
        }

        int smid = __smid();
        int warpid = __warpid();
        //TIMERPRINTF("Inside cuxTimer::Start, stackOffset=%d\n", stackOffset);

        // When a new ID starts, the old one must be suspended (but it'll be
        // returned to later in Stop(), don't worry).
        timerAccum[stackOffset] += (now - currTime);

        // Now we push this onto the stack and start counting for the new ID
        timerPush(smid, warpid, currId, count);
        currId = id;
        //TIMERPRINTF("New ID is: %d @ time %lld, %d threads entering\n", currId, timerNowTime(), timerOccupancy[stackOffset]);
        currTime = timerNowTime();
    }
}


//
//  Stop
//
//  Static member function which pops the next ID from the stack and times
//  with it. The destructor uses this.
//
//  Because we can Stop() divergently from a warp (i.e. if(threadIdx.x < 10) return;)
//  we have the problem that the warp may not actually be stopped when Stop()
//  is called. This means we must track threads-in and threads-out, and only
//  count this position on the stack as stopped when all threads have exited.
//  We do, however, tally up the time spent PER-THREAD not per-warp, so we have
//  a bit of arithmetic to do.
//
__device__ __forceinline__ void cuxTimer::Stop()
{
    // First thing: record the time!
    ClockType now = timerNowTime();

    // Now figure out how many threads have entered here
    int count = __popc(__ballot(1));

    // Only the master-thread need do this
    if(__masterthread())
    {
        int warpid = __warpid();
        int smid = __smid();
        //TIMERPRINTF("Inside cuxTimer::Stop, stackOffset=%d: [%d,%d] (popc=%d)\n", stackOffset, timerPrevID[stackOffset], timerOccupancy[stackOffset], count);
        //TIMERPRINTF("Adding %llu to time[%d]=%llu\n", (now-currTime)*count, currId, timerTimes[currId]);
        
        // Count down the number of threads which have Stop()ped.
        // If it doesn't hit zero then don't pop. If it does, then do.
        if(timerOccupancy[stackOffset] != count)
        {
#ifndef __CUXTIMER_PER_WARP_STATS
            // To record true per-thread stats, we need to track each partial-warp exit. Slower, but more accurate.
            atomicAdd((AtomicType *)&(timerTimes[currId]), (AtomicType)(timerAccum[stackOffset] + (now - currTime)) * count);
            atomicMax((AtomicType *)&(timerMax[currId]), (AtomicType)(timerAccum[stackOffset] + (now - currTime)));
            atomicAdd(&(timerCount[currId]), timerOccupancy[stackOffset]);
#endif
            // Downcount the occupancy and do no more for now
            timerOccupancy[stackOffset] -= count;
            //TIMERPRINTF("Still remain %d threads to exit ID %d\n", timerOccupancy[stackOffset], currId);
        }
        else
        {
#ifdef __CUXTIMER_PER_WARP_STATS
            // This makes all stats record just 1 count for the whole warp.
            count = 1;
#endif
            // All threads are exiting, so record statistics and pop the stack.
            // For sm1.1, these must become 32-bit atomics
            atomicMax((AtomicType *)&(timerMax[currId]), (AtomicType)(timerAccum[stackOffset] + (now - currTime)));
            atomicAdd((AtomicType *)&(timerTimes[currId]), (AtomicType)(timerAccum[stackOffset] + (now - currTime)) * count);
            atomicAdd(&(timerCount[currId]), count);
            //TIMERPRINTF("Added time %lld to clock %d (now %lld).\n", (now-currTime)*count, currId, timerTimes[currId]);
            //TIMERPRINTF("Added count %d to clock %d (now %d).\n", count, currId, timerCount[currId]);

            // Pop the stack, and start timing for this element
            currId = timerPop(smid, warpid);
            //TIMERPRINTF("New ID is: %d @ time %lld\n", currId, timerNowTime());
            currTime = timerNowTime();
        }
    }
}

// Device-side setup and information query stuff.
// NOTE: For all of these, it is up to the user to call with a single
//       thread if desired.

//
//  Init
//
//  If you don't want to set everything up from the host, you can do it
//  from the device via this function. We basically just zero all the
//  memory to be safe, and mark the init status.
//
//  Note that we have to do some magic locking thing to get this to work,
//  so that if multiple warps try to call it at once we only do it one time.
//
//  Returns 0 on failure, 1 on success
//
__device__ __noinline__ int cuxTimer::Init()
{
    //TIMERPRINTF("Iniside init, state=%d\n", timerInitState);

    // If we've entered Init() already, then the state is set. If not, set it.
    if(timerInitState < __CUXTIMERINIT_INITIALISED)
    {
        // Only one thread can safely take a lock (this should be called by one thread anyway)
        if(__masterthread())
        {
            // Shift to "initisliastion in progress" state. If we're the first, do the init
            volatile unsigned int *state = (volatile unsigned int *)&timerInitState;
            if(atomicCAS((unsigned int *)state, __CUXTIMERINIT_NOTINITIALISED, __CUXTIMERINIT_INITIALISING) == __CUXTIMERINIT_NOTINITIALISED)
            {
                // Now clear everything
                //TIMERPRINTF("Doing the init\n");
                timerAutoDisplayCount = 0;      // Need to clear this counter here, where it's guarded as one-thread-only

                // We're the first thread to call init, so go and do it
                Reset();
                *state = __CUXTIMERINIT_INITIALISED;                 // Indicate we've initialised ourselves
            }
            else
            {
                //TIMERPRINTF("Waiting on init to complete\n");
                int downcount = 1000000;
                while((downcount--) && (*state == __CUXTIMERINIT_INITIALISING));        // Spin waiting for init to complete
                if(downcount < 1)
                    TIMERPRINTF("cuxTimer: Timeout waiting for init to complete!\n");
            }
        }
    }

    return (timerInitState == __CUXTIMERINIT_INITIALISED) ? true : false;
}


//
//  Reset
//
//  Clears the internal counter stuff for another run
//
__device__ __noinline__ void cuxTimer::Reset()
{
    // Clear the entire structure with just the single calling thread. We know
    // it's 4-aligned, at least, so we can do a small optimisation
    int *ptr = (int *)&(_cuxtimer_globals);
    int len = sizeof(_cuxtimer_globals) / sizeof(int);
    while(len--)
        *ptr++ = 0;

    __threadfence();
}


//
//  Display
//
//  Dumps the stats in the current buffer, displaying the *average* time
//  for each stat but also min/max/count. Uses printf, so only works on sm20
//
//  If "csv" is true, output is in CSV format. If "no_headings" is true,
//  there are not headers output at the top of the data.
//
__device__ __noinline__ void cuxTimer::Display(int csv, int show_headings)
{
#if __CUDA_ARCH__ >= 200
    unsigned int tot_count=0;
    unsigned long long tot_cycles=0, tot_max=0, tot_avg=0;

    // Display headings
    if(show_headings)
    {
        if(!csv)
        {
            printf("Timing Stats:\n");
            printf("%20s: %8s %12s %12s %12s", "Name", "Count", "Tot Cycles", "Max Cycles", "Avg Cycles");//, "Description");
        }
        else
        {
            printf("Name,Count,Tot Cycles,Max Cycles,Avg Cycles,Description");
        }
        printf("\n");
    }

    // Now just loop through all IDs, displaying those with data
    for(int id=0; id<__CUXTIMER_IDS; id++)
    {
        if(timerCount[id] > 0)
        {
            if(_timer_isNameSet(globalNameMask, id))
                printf((!csv) ? "%20s" : "\"%s\"", globalIDName[id]);
            else
                printf((!csv) ? "%20d" : "%d", id-1);

            if(!csv)
                printf(": %8d %12llu %12llu %12llu", timerCount[id], timerTimes[id], timerMax[id], timerTimes[id]/timerCount[id]);
            else
                printf(",%d,%llu,%llu,%llu", timerCount[id], timerTimes[id], timerMax[id], timerTimes[id]/timerCount[id]);

            if(_timer_isNameSet(globalDescrMask, id))
                printf((!csv) ? " %s" : ",\"%s\"", globalIDDescr[id]);
            else if(csv)
                printf(",none");
            printf("\n");

            tot_count += timerCount[id];
            tot_cycles += timerTimes[id];
            tot_max += timerMax[id];
            tot_avg += timerTimes[id]/timerCount[id];
        }
    }

    if(show_headings)
    {
        if(!csv)
        {
            printf("---------------------------------------------------------------------\n");
            printf("%20s: %8d %12llu %12llu %12llu\n", "TOTAL", tot_count, tot_cycles, tot_max, tot_avg);
        }
        else
        {
            printf("%s,%d,%llu,%llu,%llu,n/a", "TOTAL", tot_count, tot_cycles, tot_max, tot_avg);
            printf("\n");
        }
    }
#endif
}


//
//  class cuxTimerAutoDisplay
//
//  This is a convenience hack which counts who the last
//  warp to exit is, and automatically dumps the display
//  when it happens. Its for cases where you can't easily
//  trigger display from the host, or track last-man-out
//  from the device.
//
//  It works on the presumption that the hardware will
//  keep launching blocks of the grid, and so there'll
//  always be at least two warps active at any time
//  until the very last one exits. Dangerous assumption.
//
//  We also reset the counters on first-man-in, which is
//  maybe inconvenient - set "noreset = false" on construct
//  to avoid this behaviour.
//
__device__ TIMER_NOINLINE cuxTimerAutoDisplay::cuxTimerAutoDisplay(int unused, int csv, int show_headings, int reset) : auto_csv(csv), auto_show_headings(show_headings), auto_reset(reset)
{
    // We'll force this to only work without host-side initialisation.
    // It's not actually required, but it'll do for now.
    if(_cuxtimer_constants.init == __CUXTIMERINIT_NOTINITIALISED)
    {
        // Count our number of threads in. Must count threads not warps
        // to handle divergent exit.
        int count = __popc(__ballot(1));
        if(__masterthread())
        {
            // Begin by initialisation of the whole system.
            if(cuxTimer::Init())
                atomicAdd(&timerAutoDisplayCount, count);   // Only count up if init succeeded!
        }
    }
}

// The real "work" (such as it is) happens in the destructor. We use the
// count to determine if we're the last-man-out.
__device__ TIMER_NOINLINE cuxTimerAutoDisplay::~cuxTimerAutoDisplay()
{
    if(_cuxtimer_constants.init == __CUXTIMERINIT_NOTINITIALISED)
    {
        int count = __popc(__ballot(1));
        if(__masterthread())
        {
            // Downcount the number of threads. This handles divergent exit
            // from threads in a warp.
            if(atomicAdd(&timerAutoDisplayCount, (unsigned)(-count)) == count)
            {
                // Quick catch - if init failed then do nothing. We'll catch
                // it here because it's faster than on entry.
                if(timerInitState == __CUXTIMERINIT_INITIALISED)
                {
                    // Reset is risky - we don't lock so any new launch
                    // can be a race. To keep things as simple as possible,
                    // we'll force reset by forcing re-init.
                    // Note we assume there was no init from the host (otherwise
                    // why would you need autodisplay?). I can handle that case
                    // too, but I can't be bothered right now.
                    if(auto_reset)
                        timerInitState = __CUXTIMERINIT_NOTINITIALISED;

                    cuxTimer::Display(auto_csv, auto_show_headings);
                }
            }
        }
    }
}


#undef timerMax
#undef timerCount
#undef timerTimes

#undef timerPrevID
#undef timerOccupancy
#undef timerAccum
#undef stackOffset
#undef currId
#undef currTime

#undef globalNameMask
#undef globalIDName
#undef globalDescrMask
#undef globalIDDescr

#undef timerInitState
#undef timerAutoDisplayCount

#undef ClockType
#undef AtomicType
#undef TIMER_NOINLINE
#ifdef _CUXTIMER_DEBUG
#undef TIMERPRINTF
#endif

#endif  // __CUDA_ARCH__ > 0
#endif  // CUXTIMER_DISABLE

#endif  // CUXTIMER_CU

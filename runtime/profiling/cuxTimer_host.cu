//
//  cuxTimer_host.cu
//
//  Host-side functions for timing control. It offers two functions:
//      cuxTimerReset() - Resets the timer internal data structures.
//          This is only safe when there are no timer-using kernels
//          running on the machine.
//
//          Resetting everything from the host enables a moderate
//          performance increase on the device side, so is worth
//          doing if possible.
//
//      cuxTimerDisplay() - Outputs the timing data to the specified
//          FILE pointer (or screen if none). Display can be in CSV
//          format or pre-formatted for the screen. You may also
//          optionally disable headers and column names, in the event
//          that you call cuxTimerDisplay() repeatedly to accumulate
//          stats during the program run.
//
//          Note that cuxTimerDisplay does NOT reset the timing data
//          automatically - call cuxTimerReset for that.
//
//  This file automatically includes cuxTimer.h 
//

#ifndef CUXTIMER_HOST_CU
#define CUXTIMER_HOST_CU

#ifndef CUXTIMER_DISABLE
#include "cuxTimer.h"
#include "cuxTimer.cu"
#include "cuxTimer_private.h"

// Initialises things and clears all counters from the host side
static __host__ void cudaxTimerReset()
{
#ifdef TIMERDEBUG
    // If debugging, we need to init cuPrintf
    cudaPrintfInit();
#endif

    // Clear the data buffer on the device
    void *globalptr;
    cudaGetSymbolAddress(&globalptr, _cuxtimer_globals);
    cudaMemset(globalptr, 0, sizeof(_cuxtimer_globals));

    // Set up the constant info for the timer
    _timerConfig pc;
    pc.init = __CUXTIMERINIT_INITIALISED;
    cudaMemcpyToSymbol(_cuxtimer_constants, &pc, sizeof(pc));
}


//
//  cudaxTimerDisplay
//
//  Dumps out the timing data. It's pretty simple - we just dump it for all IDs
//  which were actually called.
//
//  We output to "fp". If "csv" is true, we output in CSV format.
//  If "show_headings" is true, we add column headings and other formatting.
//
static __host__ void cudaxTimerDisplay(FILE *fp, int csv, int show_headings)
{
#ifdef TIMERDEBUG
    // Dump printfs if we're debugging
    cudaPrintfDisplay(stdout, true);
#endif

    static _timerGlobals timerHostGlobals;                      // Static so it doesn't live on the stack
    _timerData &timerLocalData = timerHostGlobals.funcdata;
    _timerStrings &timerHostStrings = timerHostGlobals.strings;
    memset(&timerHostGlobals, 0, sizeof(timerHostGlobals));     // Clear our local buffer

    unsigned int tot_count=0;
    unsigned long long tot_cycles=0, tot_max=0, tot_avg=0;

    // Create local copy of the timing info and config
    cudaMemcpyFromSymbol(&timerHostGlobals, _cuxtimer_globals, sizeof(timerHostGlobals));
    
    // The config comes from either the constant bank (if host-side reset
    // was called) or from device memory. Determine which.
    _timerConfig pc;
    cudaMemcpyFromSymbol(&pc, _cuxtimer_constants, sizeof(pc));
    if(pc.init == __CUXTIMERINIT_NOTINITIALISED)
        cudaMemcpyFromSymbol(&pc, _cuxtimer_config, sizeof(pc));

    // Verify that things were initialised
    if(pc.init != __CUXTIMERINIT_INITIALISED)
    {
        fprintf(stderr, "cuxTimer data: %s\n", (pc.init == __CUXTIMERINIT_NOTINITIALISED) ? "NOT INITIALISED" : "DEVICE-INIT FAILED");
        return;
    }

    // Write out the headings
    if(show_headings)
    {
        if(!csv)
        {
            fprintf(fp, "Timing stats:\n");
            fprintf(fp, "%20s: %8s %12s %12s %12s", "Name", "Count", "Tot Cycles", "Max Cycles", "Avg Cycles", "Description");
        }
        else
        {
            fprintf(fp, "Name,Count,Tot Cycles,Max Cycles,Avg Cycles,Description");
        }
        fprintf(fp, "\n");
    }

    // Loop through all IDs, printing only those which have data
    for(int id=0; id<__CUXTIMER_IDS; id++)
    {
        if(timerLocalData.count[id] > 0)
        {
            // Print name - if it exists - or ID number if not.
            if(_timer_isNameSet(timerHostStrings.nameMask, id))
            {
                if(timerHostStrings.idNames[id][0] == '\0')
                    fprintf(fp, "Bad name for id %d", id);
                else
                    fprintf(fp, (!csv) ? "%20s" : "\"%s\"", timerHostStrings.idNames[id]);
            }
            else
                fprintf(fp, (!csv) ? "%20d" : "%d", id);

            // Print timing data
            if(!csv)
                fprintf(fp, ": %8d %12llu %12llu %12llu", timerLocalData.count[id], timerLocalData.idtime[id], timerLocalData.maximum[id], timerLocalData.idtime[id]/timerLocalData.count[id]);
            else
                fprintf(fp, ",%d,%llu,%llu,%llu", timerLocalData.count[id], timerLocalData.idtime[id], timerLocalData.maximum[id], timerLocalData.idtime[id]/timerLocalData.count[id]);

            // Print the description, if any
            if(_timer_isNameSet(timerHostStrings.descrMask, id))
            {
                if(timerHostStrings.idDescrs[id][0] == '\0')
                    fprintf(fp, "Bad description for id %d", id);
                else
                    fprintf(fp, (!csv) ? " %s" : ",\"%s\"", timerHostStrings.idDescrs[id]);
            }
            else if(csv)
                fprintf(fp, ",none");
            fprintf(fp, "\n");

            tot_count += timerLocalData.count[id];
            tot_cycles += timerLocalData.idtime[id];
            tot_max += timerLocalData.maximum[id];
            tot_avg += timerLocalData.idtime[id]/timerLocalData.count[id];
        }
    }

    // The "total" footer only comes out with headings turned on as well.
    if(show_headings)
    {
        if(!csv)
        {
            fprintf(fp, "---------------------------------------------------------------------\n");
            fprintf(fp, "%20s: %8d %12llu %12llu %12llu\n", "TOTAL", tot_count, tot_cycles, tot_max, tot_avg);
        }
        else
        {
            fprintf(fp, "%s,%d,%llu,%llu,%llu,n/a", "TOTAL", tot_count, tot_cycles, tot_max, tot_avg);
            fprintf(fp, "\n");
        }
    }
}

#endif  // CUXTIMER_DISABLE
#endif  // CUXTIMER_HOST_CU

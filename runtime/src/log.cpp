/*
 * log.cpp
 *
 *  Created on: Jun 8, 2012
 *      Author: jbedorf
 */


#include <stdarg.h>
#include <log.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

// ******************************************** //
// These functions add the process ID and total //
// number of processes to the output lines.     //
// Mainly for parallel debug purposes.          //
// ******************************************** //

#if ENABLE_LOG
  #ifdef USE_MPI
    extern bool ENABLE_RUNTIME_LOG;
    extern bool PREPEND_RANK;
    extern int PREPEND_RANK_PROCID;
    extern int PREPEND_RANK_NPROCS;

    const char prependPattern[] = {"[Proc: %d (%d)]\t"};
    
    char stderrBUFF[4096];

    //Standard out, note we write to a buffer to make
    //sure the whole line is output in one flush
    extern void prependrankLOG(const char *fmt, ...)
    {
           
           va_list ap;
           va_start(ap, fmt);

           sprintf(stderrBUFF, prependPattern, PREPEND_RANK_PROCID, PREPEND_RANK_NPROCS);
           int len = strlen(stderrBUFF);
           vsprintf(stderrBUFF+len, fmt, ap);
           va_end(ap);
           printf("%s",stderrBUFF);
    }

    extern void prependrankLOGF(const char *fmt, ...)
    {
//            char stderrBUFF[4096];
           va_list ap;
           va_start(ap, fmt);

           sprintf(stderrBUFF, prependPattern, PREPEND_RANK_PROCID, PREPEND_RANK_NPROCS);
           int len = strlen(stderrBUFF);
           vsprintf(stderrBUFF+len, fmt, ap);
           va_end(ap);
           fprintf(stderr, "%s", stderrBUFF);
    }
  #endif //USEMPI
#endif //ENABLE LOG




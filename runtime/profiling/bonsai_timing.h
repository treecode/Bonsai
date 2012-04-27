#ifndef _BONSAI_TIMING_H
#define _BONSAI_TIMING_H

#include <stdio.h>

#define CUXTIMER_DISABLE

#include "../profiling/cuxTimer_host.cu"

#ifdef CUXTIMER_DISABLE
#define PROF_HOOK(name)
#define CUXTIMER(...)
#else
#define PROF_HOOK(name) \
extern void name ## _init(); \
extern void name ## _display(FILE *fp=stdout, int csv=false, int show_headings=true);
#endif

PROF_HOOK(build_tree);
PROF_HOOK(compute_propertiesD);
PROF_HOOK(dev_approximate_gravity);
PROF_HOOK(parallel);
PROF_HOOK(sortKernels);
PROF_HOOK(timestep);

#undef PROF_HOOK

#ifdef CUXTIMER_DISABLE
#define PROF_MODULE(name)
#else
#define PROF_MODULE(name) \
    extern void name ## _init() { cudaxTimerReset(); } \
    extern void name ## _display(FILE *fp, int csv, int show_headings) { cudaxTimerDisplay(fp, csv, show_headings); }
#endif

#endif // _BONSAI_TIMING_H

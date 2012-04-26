#ifndef _BONSAI_TIMING_H
#define _BONSAI_TIMING_H

#include <stdio.h>

#define PROF_HOOK(name) \
extern void name ## _init(); \
extern void name ## _display(FILE *fp=stdout, int csv=false, int show_headings=true);

PROF_HOOK(build_tree);
PROF_HOOK(compute_propertiesD);
PROF_HOOK(dev_approximate_gravity);
PROF_HOOK(parallel);
PROF_HOOK(sortKernels);
PROF_HOOK(timestep);

#undef PROF_HOOK

#define PROF_MODULE(name) \
    extern void name ## _init() { cudaxTimerReset(); } \
    extern void name ## _display(FILE *fp, int csv, int show_headings) { cudaxTimerDisplay(fp, csv, show_headings); }

#endif // _BONSAI_TIMING_H

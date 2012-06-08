
#define ENABLE_LOG 1

#if ENABLE_LOG
  extern bool ENABLE_RUNTIME_LOG;
  extern bool PREPEND_RANK;
#endif

#if ENABLE_LOG
#ifdef WIN32
  #define LOG(fmt, ...) {if (ENABLE_RUNTIME_LOG) printf(fmt, __VA_ARGS__);}
#else



  #ifdef USE_MPI
        extern void prependrankLOG(const char *fmt, ...);
        #define LOG(...) {if (ENABLE_RUNTIME_LOG) if(PREPEND_RANK) prependrankLOG(__VA_ARGS__); else  printf(__VA_ARGS__);}
  #else
        #define LOG(...) {if (ENABLE_RUNTIME_LOG) printf(__VA_ARGS__);}
  #endif

  //
  //
#endif


//#define LOGF(file, fmt, ...) {if (ENABLE_RUNTIME_LOG) fprintf(file, fmt, __VA_ARGS__);}


  #ifdef USE_MPI
        extern void prependrankLOGF(const char *fmt, ...);
        #define LOGF(file, ...) {if (ENABLE_RUNTIME_LOG) if(PREPEND_RANK)  prependrankLOGF(__VA_ARGS__); else fprintf(file, __VA_ARGS__);}
  #else
        #define LOGF(file, fmt, ...) {if (ENABLE_RUNTIME_LOG) fprintf(file, fmt,__VA_ARGS__);}
  #endif


#else
  #define LOG(fmt, ...) ((void)0)
  #define LOGF(file, fmt, ...) ((void)0)
#endif







#if 0
  #define ENABLE_LOG 1

  #if ENABLE_LOG
  extern bool ENABLE_RUNTIME_LOG;
  #endif

  #if ENABLE_LOG
  #ifdef WIN32
    #define LOG(fmt, ...) {if (ENABLE_RUNTIME_LOG) printf(fmt, __VA_ARGS__);}
  #else
    #define LOG(...) {if (ENABLE_RUNTIME_LOG) printf(__VA_ARGS__);}
  #endif
  #define LOGF(file, fmt, ...) {if (ENABLE_RUNTIME_LOG) fprintf(file, fmt, __VA_ARGS__);}
  #else
  #define LOG(fmt, ...) ((void)0)
  #define LOGF(file, fmt, ...) ((void)0)
  #endif
#endif

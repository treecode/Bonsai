
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

#define ENABLE_LOG 1

#if ENABLE_LOG
#ifdef WIN32
  #define LOG(fmt, ...) printf(fmt, __VA_ARGS__)
#else
  #define LOG(...) printf(__VA_ARGS__)        
#endif
#define LOGF(file, fmt, ...) fprintf(file, fmt, __VA_ARGS__)
#else
#define LOG(fmt, ...) ((void)0)
#define LOGF(file, fmt, ...) ((void)0)
#endif

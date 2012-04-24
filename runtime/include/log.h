//#define ENABLE_LOG

#ifdef ENABLE_LOG
#define LOG(...) printf
#define LOGF(F, ...) fprintf(F, __VA_ARGS__)
#else
#define LOG(...) ((void)0)
#define LOGF(F, ...) ((void)0)
#endif

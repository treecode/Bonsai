#define ENABLE_LOG 0

#if ENABLE_LOG
#define LOG(fmt, ...) printf(fmt, __VA_ARGS__)
#define LOGF(file, fmt, ...) fprintf(file, fmt, __VA_ARGS__)
#else
#define LOG(fmt, ...) ((void)0)
#define LOGF(file, fmt, ...) ((void)0)
#endif

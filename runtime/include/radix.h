#pragma once
#include <omp.h>

#if 1 
template<int BITS>
struct Keys;

template<>
struct Keys<32>
{
  private:
    typedef unsigned int ulong;
    typedef unsigned int uint;
    ulong key;

  public:
    Keys() {}
    Keys(const uint x) : key(static_cast<ulong>(x)) {}

    uint get_uint(const int i) const
    {
      return (key >> (32*i)) & static_cast<ulong>(0xFFFFFFFF);
    }

    Keys operator<<(const int bits) const
    {
      return key << bits;
    }

    Keys operator>>(const int bits) const
    {
      return key >> bits;
    }


    operator uint() const {return get_uint(0);}

#if 1
    Keys(const uint4 value) : key(static_cast<ulong>(value.x)) {}
    uint4 get_uint4() const 
    {
      return (uint4){get_uint(0), 0,0,0};
    }
#endif
};

template<>
struct Keys<64>
{
  private:
    typedef unsigned long long ulong;
    typedef unsigned int uint;
    ulong key;

  public:
    Keys() {}
    Keys(const uint x) : key(static_cast<ulong>(x)) {}

    uint get_uint(const int i) const
    {
      return (key >> (32*i)) & static_cast<ulong>(0xFFFFFFFF);
    }

    Keys operator<<(const int bits) const
    {
      return key << bits;
    }

    Keys operator>>(const int bits) const
    {
      return key >> bits;
    }


    operator uint() const {return get_uint(0);}
#if 1
    Keys(const uint4 value) :
      key((static_cast<ulong>(value.x) << 32) | static_cast<ulong>(value.y)) {}
    uint4 get_uint4() const 
    {
      return (uint4){get_uint(1), get_uint(0),0,0};
    }
#endif
};

#ifdef __SIZEOF_INT128__
template<>
struct Keys<96>
{
  private:
    typedef unsigned __int128 ulong;
    typedef unsigned int uint;
    ulong key;

  public:
    Keys() {}
    Keys(const uint x) : key(static_cast<ulong>(x)) {}


    uint get_uint(const int i) const
    {
      return (key >> (32*i)) & static_cast<ulong>(0xFFFFFFFF);
    }

    Keys operator<<(const int bits) const
    {
      return key << bits;
    }

    Keys operator>>(const int bits) const
    {
      return key >> bits;
    }


    operator uint() const {return get_uint(0);}
#if 1
    Keys(const uint4 value) :
      key((static_cast<ulong>(value.x) << 64) | 
          (static_cast<ulong>(value.y) << 32) | static_cast<ulong>(value.z)) {}
    uint4 get_uint4() const 
    {
      return (uint4){get_uint(2), get_uint(1), get_uint(0),0};
    }
#endif
};
#endif

template<int BITS>
struct RadixSort
{
  typedef Keys<BITS> key_t;
  private:
  enum 
  {
    PAD = 1,
    numBits = 8,
    numBuckets = (1<<numBits),
    numBucketsPad = numBuckets * PAD
  };

  int count;
  int blockSize;
  int gridDim;
  int numBlocks;

  key_t *sorted;
  int *excScanBlockPtr, *countsBlockPtr;

  public:

  int get_numBits() const {return numBits;}

  RadixSort(const int _count) : count(_count)
  {
#pragma omp parallel
#pragma omp master
    gridDim = omp_get_num_threads();

    if (1)
    {
      blockSize = std::max((count/gridDim/64) & -64, 64);  /* sandy bridge */
    }
    else
    {
      blockSize = std::max((count/gridDim/4) & -64, 64);   /* xeonphi */
    }

    numBlocks  = (count + blockSize - 1) / blockSize;

    const int ntmp = numBlocks * numBucketsPad;
    posix_memalign((void**)&sorted, 64, count*sizeof(key_t));
    posix_memalign((void**)&excScanBlockPtr, 64, ntmp*sizeof(int));
    posix_memalign((void**)& countsBlockPtr, 64, ntmp*sizeof(int));

    int (*excScanBlock)[numBucketsPad] = (int (*)[numBucketsPad])excScanBlockPtr;
    int (* countsBlock)[numBucketsPad] = (int (*)[numBucketsPad]) countsBlockPtr;

#pragma omp parallel
    {
      const int blockIdx = omp_get_thread_num();
      for(int block = blockIdx; block < numBlocks; block += gridDim)
#pragma simd
        for (int i = 0; i < numBuckets; i++)
          countsBlock[block][i] = excScanBlock[block][i] = 0;
#pragma omp for
      for (int i = 0; i < count; i++)
        sorted[i] = 0;
    }
  } 

  ~RadixSort()
  {
    free(sorted);
    free(excScanBlockPtr);
    free(countsBlockPtr);
  }

  private:

  void countPass(
      const key_t* keys, 
      const   int bit,
      const int count,
      int* counts) 
  {
    // Compute the histogram of radix digits for this block only. This
    // corresponds exactly to the count kernel in the GPU implementation.
    const int mask = (1 << numBits) - 1;
#pragma simd
    for (int i = 0; i < numBuckets; i++)
      counts[i] = 0;
#if 1
    for(int i = 0; i < count; ++i) 
    {
      const int key = (keys[i] >> bit) & mask;
      counts[key]++;
    }
#endif
  }

  void sortPass(
      const key_t * keys,
      key_t * sorted, 
      int bit, 
      const int count,
      const int* digitOffsets,
      int* counts)
  {

    // Compute the histogram of radix digits for this block only. This
    // corresponds exactly to the count kernel in the GPU implementation.
    const int numBuckets = 1 << numBits;
    const int mask = numBuckets - 1;

#if 1
    for(int i = 0; i < count; i++)
    {
      // Extract the key 
      const int key = (keys[i]>> bit) & mask;
      const int rel = counts[key];
      const int scatter = rel + digitOffsets[key];

      sorted[scatter] = keys[i];

      counts[key] = 1 + rel;
    }
#endif
  }

  public:

  void sort(key_t *keys)
  {
    int  countsGlobal[numBuckets] __attribute__((aligned(64))) = {0};
    int excScanGlobal[numBuckets] __attribute__((aligned(64))) = {0};
    int excScanBlockL[numBuckets] __attribute__((aligned(64))) = {0};
    int  digitOffsets[numBuckets] __attribute__((aligned(64))) = {0};

    int (*excScanBlock)[numBucketsPad] = (int (*)[numBucketsPad])excScanBlockPtr;
    int (* countsBlock)[numBucketsPad] = (int (*)[numBucketsPad]) countsBlockPtr;


#if 0
#define PROFILE
#endif

#ifdef PROFILE
    double dt1, dt2, dt3, dt4, dt5;
    double t0,t1;
    dt1=dt2=dt3=dt4=dt5=0.0;

    const double tbeg = rtc();
#endif

#pragma omp parallel
    {
      const int blockIdx = omp_get_thread_num();

      for(int bit = 0; bit < BITS; bit += numBits)
      {

#ifdef PROFILE
#pragma omp master
        t0 = rtc();
#endif

        /* histogramming each of the block */
        for(int block = blockIdx; block < numBlocks; block += gridDim)
          countPass(
              keys + block*blockSize,
              bit,
              std::min(count - block*blockSize, blockSize),
              &countsBlock[block][0]);


#pragma omp barrier

#ifdef PROFILE
#pragma omp master
        { t1 = rtc(); dt1 += t1 - t0; t0 = t1; }
#endif


        /* compute global histogram */
        for (int digit = blockIdx; digit < numBuckets; digit += gridDim)
        {
          int sum = 0.0;
          for (int block = 0; block < numBlocks; block++)
            sum += countsBlock[block][digit];
          countsGlobal[digit] = sum;
        }

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
        { t1 = rtc(); dt2 += t1 - t0; t0 = t1; }
#endif

        /* exclusive scan on the histogram */
#pragma omp single
        for(int digit = 1; digit < numBuckets; digit++)
          excScanGlobal[digit] = excScanGlobal[digit - 1] + countsGlobal[digit - 1];

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
        { t1 = rtc(); dt3 += t1 - t0; t0 = t1; }
#endif

        /* computing offsets for each digit */
        for (int digit = blockIdx; digit < numBuckets; digit += gridDim)
        {
          int dgt =  digitOffsets[digit];
          for (int block = 0; block < numBlocks; block++)
          {
            excScanBlock[block][digit] = dgt + excScanGlobal[digit];
            dgt += countsBlock[block][digit];
          }
          digitOffsets[digit] = 0;
        }

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
        { t1 = rtc(); dt4 += t1 - t0; t0 = t1; }
#endif

        /* sorting */
        for(int block = blockIdx; block < numBlocks; block += gridDim)
        {
          int counts[numBuckets] = {0};

          const int keyIndex = block * blockSize;
          sortPass(
              keys + keyIndex, 
              sorted,
              bit, 
              std::min(count - keyIndex, blockSize), 
              &excScanBlock[block][0],
              &counts[0]);
        }

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
        { t1 = rtc(); dt5 += t1 - t0; t0 = t1; }
#endif

#pragma omp single
        {
#pragma simd
          for (int i = 0; i < numBuckets; i++)
            countsGlobal[i] = excScanGlobal[i]  = 0;
          std::swap(keys, sorted);
        }
      }
    }


#ifdef PROFILE
    const double tend = rtc();
    printf("dt1= %g \n", dt1);
    printf("dt2= %g \n", dt2);
    printf("dt3= %g \n", dt3);
    printf("dt4= %g \n", dt4);
    printf("dt5= %g \n", dt5);
    printf("dt = %g \n", tend-tbeg);
#endif
  }

};
#endif

#if 1
struct RadixSort64
{
  private:
    enum 
    {
      PAD = 1,
      numBits = 8,
      numBuckets = (1<<numBits),
      numBucketsPad = numBuckets * PAD
    };

    int count;
    int blockSize;
    int gridDim;
    int numBlocks;

    unsigned long long *sorted;
    int *excScanBlockPtr, *countsBlockPtr;

  public:

    int get_numBits() const {return numBits;}

    RadixSort64(const int _count) : count(_count)
  {
#pragma omp parallel
#pragma omp master
    gridDim = omp_get_num_threads();

    if (1)
    {
      blockSize = std::max((count/gridDim/64) & -64, 64);  /* sandy bridge */
    }
    else
    {
      blockSize = std::max((count/gridDim/4) & -64, 64);   /* xeonphi */
    }

    numBlocks  = (count + blockSize - 1) / blockSize;

    const int ntmp = numBlocks * numBucketsPad;
    posix_memalign((void**)&sorted, 64, count*sizeof(unsigned long long));
    posix_memalign((void**)&excScanBlockPtr, 64, ntmp*sizeof(int));
    posix_memalign((void**)& countsBlockPtr, 64, ntmp*sizeof(int));

    int (*excScanBlock)[numBucketsPad] = (int (*)[numBucketsPad])excScanBlockPtr;
    int (* countsBlock)[numBucketsPad] = (int (*)[numBucketsPad]) countsBlockPtr;

#pragma omp parallel
    {
      const int blockIdx = omp_get_thread_num();
      for(int block = blockIdx; block < numBlocks; block += gridDim)
#pragma simd
        for (int i = 0; i < numBuckets; i++)
          countsBlock[block][i] = excScanBlock[block][i] = 0;
#pragma omp for
      for (int i = 0; i < count; i++)
        sorted[i] = 0;
    }
  } 

    ~RadixSort64()
    {
      free(sorted);
      free(excScanBlockPtr);
      free(countsBlockPtr);
    }

  private:

    void countPass(
        const unsigned long long* keys, 
        const   int bit,
        const int count,
        int* counts) 
    {
      // Compute the histogram of radix digits for this block only. This
      // corresponds exactly to the count kernel in the GPU implementation.
      const unsigned long long mask = (1 << numBits) - 1;
#pragma simd
      for (int i = 0; i < numBuckets; i++)
        counts[i] = 0;
#if 1
      for(int i = 0; i < count; ++i) 
      {
        const int key = mask & (keys[i] >> bit);
        counts[key]++;
      }
#endif
    }

    void sortPass(
        const unsigned long long* keys,
        unsigned long long* sorted, 
        int bit, 
        const int count,
        const int* digitOffsets,
        int* counts)
    {

      // Compute the histogram of radix digits for this block only. This
      // corresponds exactly to the count kernel in the GPU implementation.
      const int numBuckets = 1<< numBits;
      const unsigned long long mask = numBuckets - 1;

#if 1
      for(int i = 0; i < count; i++)
      {
        // Extract the key 
        const int key = mask & (keys[i]>> bit);
        const int rel = counts[key];
        const int scatter = rel + digitOffsets[key];

        sorted[scatter] = keys[i];

        counts[key] = 1 + rel;
      }
#endif
    }

  public:

    void sort(unsigned long long *keys)
    {
      int  countsGlobal[numBuckets] __attribute__((aligned(64))) = {0};
      int excScanGlobal[numBuckets] __attribute__((aligned(64))) = {0};
      int excScanBlockL[numBuckets] __attribute__((aligned(64))) = {0};
      int  digitOffsets[numBuckets] __attribute__((aligned(64))) = {0};

      int (*excScanBlock)[numBucketsPad] = (int (*)[numBucketsPad])excScanBlockPtr;
      int (* countsBlock)[numBucketsPad] = (int (*)[numBucketsPad]) countsBlockPtr;


#if 0
#define PROFILE
#endif

#ifdef PROFILE
      double dt1, dt2, dt3, dt4, dt5;
      double t0,t1;
      dt1=dt2=dt3=dt4=dt5=0.0;

      const double tbeg = rtc();
#endif

#pragma omp parallel
      {
        const int blockIdx = omp_get_thread_num();

        for(int bit = 0; bit < 64; bit += numBits)
        {

#ifdef PROFILE
#pragma omp master
          t0 = rtc();
#endif

          /* histogramming each of the block */
          for(int block = blockIdx; block < numBlocks; block += gridDim)
            countPass(
                keys + block*blockSize,
                bit,
                std::min(count - block*blockSize, blockSize),
                &countsBlock[block][0]);


#pragma omp barrier

#ifdef PROFILE
#pragma omp master
          { t1 = rtc(); dt1 += t1 - t0; t0 = t1; }
#endif


          /* compute global histogram */
          for (int digit = blockIdx; digit < numBuckets; digit += gridDim)
          {
            int sum = 0.0;
            for (int block = 0; block < numBlocks; block++)
              sum += countsBlock[block][digit];
            countsGlobal[digit] = sum;
          }

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
          { t1 = rtc(); dt2 += t1 - t0; t0 = t1; }
#endif

          /* exclusive scan on the histogram */
#pragma omp single
            for(int digit = 1; digit < numBuckets; digit++)
                excScanGlobal[digit] = excScanGlobal[digit - 1] + countsGlobal[digit - 1];

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
          { t1 = rtc(); dt3 += t1 - t0; t0 = t1; }
#endif

          /* computing offsets for each digit */
          for (int digit = blockIdx; digit < numBuckets; digit += gridDim)
          {
            int dgt =  digitOffsets[digit];
            for (int block = 0; block < numBlocks; block++)
            {
              excScanBlock[block][digit] = dgt + excScanGlobal[digit];
              dgt += countsBlock[block][digit];
            }
            digitOffsets[digit] = 0;
          }

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
          { t1 = rtc(); dt4 += t1 - t0; t0 = t1; }
#endif

          /* sorting */
          for(int block = blockIdx; block < numBlocks; block += gridDim)
          {
            int counts[numBuckets] = {0};

            const int keyIndex = block * blockSize;
            sortPass(
                keys + keyIndex, 
                sorted,
                bit, 
                std::min(count - keyIndex, blockSize), 
                &excScanBlock[block][0],
                &counts[0]);
          }

#pragma omp barrier

#ifdef PROFILE
#pragma omp master
          { t1 = rtc(); dt5 += t1 - t0; t0 = t1; }
#endif

#pragma omp single
          {
#pragma simd
            for (int i = 0; i < numBuckets; i++)
              countsGlobal[i] = excScanGlobal[i]  = 0;
            std::swap(keys, sorted);
          }
        }
      }


#ifdef PROFILE
      const double tend = rtc();
      printf("dt1= %g \n", dt1);
      printf("dt2= %g \n", dt2);
      printf("dt3= %g \n", dt3);
      printf("dt4= %g \n", dt4);
      printf("dt5= %g \n", dt5);
      printf("dt = %g \n", tend-tbeg);
#endif
    }

};
#endif

#ifndef __SORT_H__
#define __SORT_H__

namespace b40c {
  namespace util {
    template <typename T1, typename T2> class DoubleBuffer;
  }
  namespace radix_sort {
    class Enactor;
  }
}

class Sort90
{
  b40c::util::DoubleBuffer<uint, uint> *double_buffer;
  b40c::radix_sort::Enactor *sort_enactor;
  
  bool selfAllocatedMemory;

public:
  Sort90(uint N);
  Sort90(uint N, void *generalBuffer);
  ~Sort90();
  
  // Back40 90-bit sorting: sorts the lower 30 bits in uint4's key
  void sort(my_dev::dev_mem<uint4> &srcKeys, 
            my_dev::dev_mem<uint4> &sortedKeys,
            int N);
};

#endif

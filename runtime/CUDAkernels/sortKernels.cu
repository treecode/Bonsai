#include "bonsai.h"
#include "../profiling/bonsai_timing.h"
PROF_MODULE(sortKernels);

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>



// Extract 32-bit word from uint4
template<int keyIdx>
struct ExtractBits: public thrust::unary_function<uint4, uint>
{
  __host__ __device__ __forceinline__ uint operator()(uint4 key) const
  {
    if      (keyIdx == 0) return key.x;
    else if (keyIdx == 1) return key.y;
    else                  return key.z;
  }
};

#define USE_CUB

#ifdef USE_CUB
	#include <cub/util_allocator.cuh>
	#include <cub/device/device_radix_sort.cuh>

  // Update permutation by sorting 32bit keys
  template<int keyIdx>
  void update_permutation_cub(int N,
                              my_dev::dev_mem<uint4>  &srcKeys,
                              my_dev::dev_mem<char>   &tempBuffer,
                              cub::DoubleBuffer<uint> &dValues,
                              cub::DoubleBuffer<uint> &dKeys)
  {
    // thrust ptr to thrust_permutation buffer
    thrust::device_ptr<uint> permutation = thrust::device_pointer_cast(dValues.d_buffers[dValues.selector]);
    thrust::device_ptr<uint> key32bit    = thrust::device_pointer_cast(dKeys.d_buffers[dKeys.selector]);

    // gather into temporary keys with the current reordering
    thrust::gather(permutation,
                   permutation + N,
                   thrust::make_transform_iterator(srcKeys.thrustPtr(), ExtractBits<keyIdx>()),
                   key32bit);

    //Retrieve number of bytes required for sorting and check if this fits in the pre-alloced buffer
    size_t  storageBytes  = 0;
    cub::DeviceRadixSort::SortPairs(NULL, storageBytes, dKeys, dValues, N, 0, 30);
    assert(storageBytes < tempBuffer.get_size());
    cub::DeviceRadixSort::SortPairs(tempBuffer.raw_p(), storageBytes, dKeys, dValues, N, 0, 30);
  }

  extern "C" void  cubSort(my_dev::dev_mem<uint4>  &srcKeys,
                           my_dev::dev_mem<uint>   &outPermutation,
                           my_dev::dev_mem<char>   &tempBuffer,
                           my_dev::dev_mem<uint>   &tempB,
                           my_dev::dev_mem<uint>   &tempC,
                           my_dev::dev_mem<uint>   &tempD,
                                               int N)
	{
		cub::DoubleBuffer<uint> dValues(outPermutation.raw_p(), tempB.raw_p());
		cub::DoubleBuffer<uint> dKeys  (tempC.raw_p(), tempD.raw_p());

		// initialize values (permutation) to [0, 1, 2, ... ,N-1]
    thrust::device_ptr<uint>  permutation = thrust::device_pointer_cast(dValues.d_buffers[dValues.selector]);
    thrust::sequence(permutation, permutation + N);

    // sort z, y, x careful: note 2, 1, 0 key word order, NOT 0, 1, 2.
    update_permutation_cub<2>(N, srcKeys, tempBuffer, dValues, dKeys);
    update_permutation_cub<1>(N, srcKeys, tempBuffer, dValues, dKeys);
    update_permutation_cub<0>(N, srcKeys, tempBuffer, dValues, dKeys);

    //Copy the final permutation in the permutation buffer, only if it's not there already
    if(thrust::device_pointer_cast(dValues.d_buffers[dValues.selector]).get() != outPermutation.raw_p())
    {
      outPermutation.copy_devonly(thrust::device_pointer_cast(dValues.d_buffers[dValues.selector]).get(), N);
    }
	}

#else
	//Use Thrust for sorting
	template <int keyIdx, typename KeyPtr, typename PermutationPtr, typename ExtractedPtr>
	void update_permutation_thrust(KeyPtr& keys, PermutationPtr& permutation, ExtractedPtr& temp, int N)
	{
    // permute the keys with the current reordering
    thrust::gather(permutation, permutation + N,
             thrust::make_transform_iterator(keys, ExtractBits<keyIdx>()), temp);
    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp, temp + N, permutation);
	}

	//Thrust does it's own memory allocation, hence fewer parameters. This increases the
	//risk of the GPU running out of memory.
	// TODO add our own allocator policy.
	extern "C" void thrustSort(my_dev::dev_mem<uint4> &srcKeys,
                             my_dev::dev_mem<uint>  &permutation_buffer,
                             my_dev::dev_mem<uint>  &temp_buffer,
                             int N)
	{
	  //Convert Bonsai ptr into thrst ptr
	  thrust::device_ptr<uint4> keys        = srcKeys.thrustPtr();
	  thrust::device_ptr<uint>  temp        = temp_buffer.thrustPtr();
	  thrust::device_ptr<uint>  permutation = permutation_buffer.thrustPtr();

	  // initialize permutation to [0, 1, 2, ... ,N-1]
	  thrust::sequence(permutation, permutation + N);

	  // sort z, y, x
	  update_permutation_thrust<2>(keys, permutation, temp, N);
	  update_permutation_thrust<1>(keys, permutation, temp, N);
	  update_permutation_thrust<0>(keys, permutation, temp, N);
	  // Note: permutation now maps unsorted keys to sorted order
	}

#endif

//Shuffle functions

extern "C" void thrustDataReorderU4(const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<uint4> &dIn, my_dev::dev_mem<uint4> &dOut) {
  thrust::gather(permutation.thrustPtr(), permutation.thrustPtr() + N, dIn.thrustPtr(), dOut.thrustPtr());
}
extern "C" void thrustDataReorderF4(const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<float4> &dIn, my_dev::dev_mem<float4> &dOut) {
  thrust::gather(permutation.thrustPtr(), permutation.thrustPtr() + N, dIn.thrustPtr(), dOut.thrustPtr());
}
extern "C" void thrustDataReorderF2(const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<float2> &dIn, my_dev::dev_mem<float2> &dOut) {
  thrust::gather(permutation.thrustPtr(), permutation.thrustPtr() + N, dIn.thrustPtr(), dOut.thrustPtr());
}
extern "C" void thrustDataReorderF1(const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<float> &dIn, my_dev::dev_mem<float> &dOut) {
  thrust::gather(permutation.thrustPtr(), permutation.thrustPtr() + N, dIn.thrustPtr(), dOut.thrustPtr());
}

typedef unsigned long long ullong; //ulonglong1
extern "C" void thrustDataReorderULL(const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<ullong> &dIn, my_dev::dev_mem<ullong> &dOut) {
  thrust::gather(permutation.thrustPtr(), permutation.thrustPtr() + N, dIn.thrustPtr(), dOut.thrustPtr());
}


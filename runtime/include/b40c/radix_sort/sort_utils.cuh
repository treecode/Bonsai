/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 *
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/******************************************************************************
 * Types and subroutines utilities that are common across all B40C LSB radix 
 * sorting kernels and host enactors  
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Bit-field extraction kernel subroutines
 ******************************************************************************/

/**
 * Bit extraction, specialized for non-64bit key types
 */
template <
	typename T,
	int BIT_OFFSET,
	int NUM_BITS,
	int LEFT_SHIFT>
struct Extract
{
	/**
	 * Super bitfield-extract (BFE, then left-shift).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		T source)
	{
		const T MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
		const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

		T bits = (source & MASK);
		if (SHIFT == 0) {
			return bits;
		} else {
			return util::MagnitudeShift<SHIFT>::Shift(bits);
		}
	}

	/**
	 * Super bitfield-extract (BFE, then left-shift, then add).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		T source,
		unsigned int addend)
	{
		const T MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
		const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

		T bits = (source & MASK);
		if (SHIFT == 0) {
			return bits + addend;
		} else {
			bits = (SHIFT > 0) ?
				(util::SHL_ADD(bits, SHIFT, addend)) :
				(util::SHR_ADD(bits, SHIFT * -1, addend));
			return bits;
		}
	}

};


/**
 * Bit extraction, specialized for 64bit key types
 */
template <
	int BIT_OFFSET,
	int NUM_BITS,
	int LEFT_SHIFT>
struct Extract<unsigned long long, BIT_OFFSET, NUM_BITS, LEFT_SHIFT>
{
	/**
	 * Super bitfield-extract (BFE, then left-shift).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		unsigned long long source)
	{
		const unsigned long long MASK = ((1ull << NUM_BITS) - 1) << BIT_OFFSET;
		const int SHIFT = LEFT_SHIFT - BIT_OFFSET;

		unsigned long long bits = (source & MASK);
		return util::MagnitudeShift<SHIFT>::Shift(bits);
	}

	/**
	 * Super bitfield-extract (BFE, then left-shift, then add).
	 */
	__device__ __forceinline__ static unsigned int SuperBFE(
		unsigned long long source,
		unsigned int addend)
	{
		return SuperBFE(source) + addend;
	}
};




/******************************************************************************
 * Traits for converting for converting signed and floating point types
 * to unsigned types suitable for radix sorting
 ******************************************************************************/

struct NopKeyConversion
{
	static const bool MustApply = false;		// We may early-exit this pass

	template <typename T>
	__device__ __host__ __forceinline__ static void Preprocess(T &key) {}

	template <typename T>
	__device__ __host__ __forceinline__ static void Postprocess(T &key) {}
};


template <typename UnsignedBits> 
struct UnsignedIntegerKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;
	
	static const bool MustApply = false;		// We may early-exit this pass

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key) {}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key) {}  
};


template <typename UnsignedBits> 
struct SignedIntegerKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MustApply = true;		// We must not early-exit this pass (conversion necessary)

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key)
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		converted_key ^= HIGH_BIT;
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key)  
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		converted_key ^= HIGH_BIT;	
	}
};


template <typename UnsignedBits> 
struct FloatingPointKeyConversion 
{
	typedef UnsignedBits ConvertedKeyType;

	static const bool MustApply = true;		// We must not early-exit this pass (conversion necessary)

	__device__ __host__ __forceinline__ static void Preprocess(UnsignedBits &converted_key)
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		UnsignedBits mask = (converted_key & HIGH_BIT) ? (UnsignedBits) -1 : HIGH_BIT;
		converted_key ^= mask;
	}

	__device__ __host__ __forceinline__ static void Postprocess(UnsignedBits &converted_key) 
	{
		const UnsignedBits HIGH_BIT = ((UnsignedBits) 0x1) << ((sizeof(UnsignedBits) * 8) - 1);
		UnsignedBits mask = (converted_key & HIGH_BIT) ? HIGH_BIT : (UnsignedBits) -1; 
		converted_key ^= mask;
    }
};




// Default unsigned types
template <typename T> struct KeyTraits : UnsignedIntegerKeyConversion<T> {};

// char
template <> struct KeyTraits<char> : SignedIntegerKeyConversion<unsigned char> {};

// signed char
template <> struct KeyTraits<signed char> : SignedIntegerKeyConversion<unsigned char> {};

// short
template <> struct KeyTraits<short> : SignedIntegerKeyConversion<unsigned short> {};

// int
template <> struct KeyTraits<int> : SignedIntegerKeyConversion<unsigned int> {};

// long
template <> struct KeyTraits<long> : SignedIntegerKeyConversion<unsigned long> {};

// long long
template <> struct KeyTraits<long long> : SignedIntegerKeyConversion<unsigned long long> {};

// float
template <> struct KeyTraits<float> : FloatingPointKeyConversion<unsigned int> {};

// double
template <> struct KeyTraits<double> : FloatingPointKeyConversion<unsigned long long> {};




} // namespace radix_sort
} // namespace b40c


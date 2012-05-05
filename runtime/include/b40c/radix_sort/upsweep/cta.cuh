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
 * CTA-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {

/**
 * Radix sort upsweep reduction CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	enum {
		MIN_CTA_OCCUPANCY  				= KernelPolicy::MIN_CTA_OCCUPANCY,
		CURRENT_BIT 					= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 					= KernelPolicy::CURRENT_PASS,

		RADIX_BITS						= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 					= 1 << RADIX_BITS,

		LOG_THREADS 					= KernelPolicy::LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_WARP_THREADS 				= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS					= 1 << LOG_WARP_THREADS,

		LOG_WARPS						= LOG_THREADS - LOG_WARP_THREADS,
		WARPS							= 1 << LOG_WARPS,

		LOG_LOAD_VEC_SIZE  				= KernelPolicy::LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= KernelPolicy::LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,


		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,


		// A shared-memory composite counter lane is a row of 32-bit words, one word per thread, each word a
		// composite of four 8-bit bin counters.  I.e., we need one lane for every four distribution bins.

		BYTES_PER_COUNTER				= sizeof(char),
		LOG_BYTES_PER_COUNTER			= util::Log2<BYTES_PER_COUNTER>::VALUE,

		PACKED_COUNTERS					= sizeof(int) / sizeof(char),
		LOG_PACKED_COUNTERS 			= util::Log2<PACKED_COUNTERS>::VALUE,

		LOG_COMPOSITE_LANES 			= CUB_MAX(0, RADIX_BITS - LOG_PACKED_COUNTERS),
		COMPOSITE_LANES 				= 1 << LOG_COMPOSITE_LANES,

		LOG_COMPOSITES_PER_LANE			= LOG_THREADS,				// Every thread contributes one partial for each lane
		COMPOSITES_PER_LANE 			= 1 << LOG_COMPOSITES_PER_LANE,

		// To prevent bin-counter overflow, we must partially-aggregate the
		// 8-bit composite counters back into SizeT-bit registers periodically.  Each lane
		// is assigned to a warp for aggregation.  Each lane is therefore equivalent to
		// four rows of SizeT-bit bin-counts, each the width of a warp.

		LOG_LANES_PER_WARP					= CUB_MAX(0, LOG_COMPOSITE_LANES - LOG_WARPS),
		LANES_PER_WARP 						= 1 << LOG_LANES_PER_WARP,

		LOG_COMPOSITES_PER_LANE_PER_THREAD 	= LOG_COMPOSITES_PER_LANE - LOG_WARP_THREADS,		// Number of partials per thread to aggregate
		COMPOSITES_PER_LANE_PER_THREAD 		= 1 << LOG_COMPOSITES_PER_LANE_PER_THREAD,

		AGGREGATED_ROWS						= RADIX_DIGITS,
		AGGREGATED_PARTIALS_PER_ROW 		= WARP_THREADS,
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIALS_PER_ROW + 1,

		// Unroll tiles in batches of X elements per thread (X = log(255) is maximum without risking overflow)
		LOG_UNROLL_COUNT 					= 6 - LOG_TILE_ELEMENTS_PER_THREAD,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};



	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			// Composite counter storage
			union {
				char counters[COMPOSITE_LANES][THREADS][4];
				int words[COMPOSITE_LANES][THREADS];
				int direct[COMPOSITE_LANES * THREADS];
			};

			// Final bin reduction storage
			SizeT aggregate[AGGREGATED_ROWS][PADDED_AGGREGATED_PARTIALS_PER_ROW];
		};
	};

	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 	&smem_storage;

	// Thread-local counters for periodically aggregating composite-counter lanes
	SizeT 			local_counts[LANES_PER_WARP][4];

	// Input and output device pointers
	KeyType			*d_in_keys;
	SizeT			*d_spine;

	int 			warp_id;
	int 			warp_idx;

	char 			*base;


	//---------------------------------------------------------------------
	// Helper structure for counter aggregation
	//---------------------------------------------------------------------

	/**
	 * Iterate next composite counter
	 */
	template <int WARP_LANE, int THREAD_COMPOSITE, int dummy = 0>
	struct Iterate
	{
		// ExtractComposites
		static __device__ __forceinline__ void ExtractComposites(Cta &cta)
		{
			const int LANE_OFFSET = WARP_LANE * WARPS * THREADS * 4;
			const int COMPOSITE_OFFSET = THREAD_COMPOSITE * CUB_WARP_THREADS(__CUB_CUDA_ARCH__) * 4;

			cta.local_counts[WARP_LANE][0] += *(cta.base + LANE_OFFSET + COMPOSITE_OFFSET + 0);
			cta.local_counts[WARP_LANE][1] += *(cta.base + LANE_OFFSET + COMPOSITE_OFFSET + 1);
			cta.local_counts[WARP_LANE][2] += *(cta.base + LANE_OFFSET + COMPOSITE_OFFSET + 2);
			cta.local_counts[WARP_LANE][3] += *(cta.base + LANE_OFFSET + COMPOSITE_OFFSET + 3);

			Iterate<WARP_LANE, THREAD_COMPOSITE + 1>::ExtractComposites(cta);
		}
	};

	/**
	 * Iterate next lane
	 */
	template <int WARP_LANE, int dummy>
	struct Iterate<WARP_LANE, COMPOSITES_PER_LANE_PER_THREAD, dummy>
	{
		// ExtractComposites
		static __device__ __forceinline__ void ExtractComposites(Cta &cta)
		{
			Iterate<WARP_LANE + 1, 0>::ExtractComposites(cta);
		}

		// ShareCounters
		static __device__ __forceinline__ void ShareCounters(Cta &cta)
		{
			int lane				= (WARP_LANE * WARPS) + cta.warp_id;
			int row 				= lane << 2;	// lane * 4;

			cta.smem_storage.aggregate[row + 0][cta.warp_idx] = cta.local_counts[WARP_LANE][0];
			cta.smem_storage.aggregate[row + 1][cta.warp_idx] = cta.local_counts[WARP_LANE][1];
			cta.smem_storage.aggregate[row + 2][cta.warp_idx] = cta.local_counts[WARP_LANE][2];
			cta.smem_storage.aggregate[row + 3][cta.warp_idx] = cta.local_counts[WARP_LANE][3];

			Iterate<WARP_LANE + 1, COMPOSITES_PER_LANE_PER_THREAD>::ShareCounters(cta);
		}

		// ResetCounters
		static __device__ __forceinline__ void ResetCounters(Cta &cta)
		{
			cta.local_counts[WARP_LANE][0] = 0;
			cta.local_counts[WARP_LANE][1] = 0;
			cta.local_counts[WARP_LANE][2] = 0;
			cta.local_counts[WARP_LANE][3] = 0;

			Iterate<WARP_LANE + 1, COMPOSITES_PER_LANE_PER_THREAD>::ResetCounters(cta);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LANES_PER_WARP, 0, dummy>
	{
		// ExtractComposites
		static __device__ __forceinline__ void ExtractComposites(Cta &cta) {}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LANES_PER_WARP, COMPOSITES_PER_LANE_PER_THREAD, dummy>
	{
		// ShareCounters
		static __device__ __forceinline__ void ShareCounters(Cta &cta) {}

		// ResetCounters
		static __device__ __forceinline__ void ResetCounters(Cta &cta) {}
	};


	//---------------------------------------------------------------------
	// Helper structure for tile unrolling
	//---------------------------------------------------------------------

	/**
	 * Unrolled tile processing
	 */
	struct UnrollTiles
	{
		// Recurse over counts
		template <int UNROLL_COUNT, int __dummy = 0>
		struct Iterate
		{
			static const int HALF = UNROLL_COUNT / 2;

			static __device__ __forceinline__ void ProcessTiles(
				Cta &cta, SizeT cta_offset)
			{
				Iterate<HALF>::ProcessTiles(cta, cta_offset);
				Iterate<HALF>::ProcessTiles(cta, cta_offset + (TILE_ELEMENTS * HALF));
			}
		};

		// Terminate (process one tile)
		template <int __dummy>
		struct Iterate<1, __dummy>
		{
			static __device__ __forceinline__ void ProcessTiles(
				Cta &cta, SizeT cta_offset)
			{
				cta.ProcessFullTile(cta_offset);
			}
		};
	};





	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			warp_id(threadIdx.x >> LOG_WARP_THREADS),
			warp_idx(util::LaneId())
	{
		base = (char *) (smem_storage.words[warp_id] + warp_idx);
	}


	/**
	 * Bucket a key into smem counters
	 */
	__device__ __forceinline__ void Bucket(KeyType key)
	{
		// Compute byte offset of smem counter.  Add in thread column.
		unsigned int offset = (threadIdx.x << (LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER));

		// Add in sub-counter offset
		offset = Extract<
			KeyType,
			CURRENT_BIT,
			LOG_PACKED_COUNTERS,
			LOG_BYTES_PER_COUNTER>::SuperBFE(
				key,
				offset);

		// Add in row offset
		offset = Extract<
			KeyType,
			CURRENT_BIT + LOG_PACKED_COUNTERS,
			LOG_COMPOSITE_LANES,
			LOG_THREADS + (LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER)>::SuperBFE(
				key,
				offset);

		((unsigned char *) smem_storage.counters)[offset]++;


	}


	template <int COUNT, int MAX>
	struct IterateKeys
	{
		static __device__ __forceinline__ void Bucket(
			Cta &cta, KeyType keys[LOADS_PER_TILE * LOAD_VEC_SIZE])
		{
			cta.Bucket(keys[COUNT]);
			IterateKeys<COUNT + 1, MAX>::Bucket(cta, keys);
		}
	};


	template <int MAX>
	struct IterateKeys<MAX, MAX>
	{
		static __device__ __forceinline__ void Bucket(
			Cta &cta, KeyType keys[LOADS_PER_TILE * LOAD_VEC_SIZE]) {}
	};


	/**
	 * Reset composite counters
	 */
	__device__ __forceinline__ void ResetCompositeCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < COMPOSITE_LANES; ++LANE) {
			smem_storage.words[LANE][threadIdx.x] = 0;
		}
	}


	/**
	 * Resets the aggregate counters
	 */
	__device__ __forceinline__ void ResetCounters()
	{
		Iterate<0, COMPOSITES_PER_LANE_PER_THREAD>::ResetCounters(*this);
	}


	/**
	 * Extracts and aggregates the shared-memory composite counters for each
	 * composite-counter lane owned by this warp
	 */
	__device__ __forceinline__ void ExtractComposites()
	{
		if (warp_id < COMPOSITE_LANES) {
			Iterate<0, 0>::ExtractComposites(*this);
		}
	}


	/**
	 * Places aggregate-counters into shared storage for final bin-wise reduction
	 */
	__device__ __forceinline__ void ShareCounters()
	{
		if (warp_id < COMPOSITE_LANES) {
			Iterate<0, COMPOSITES_PER_LANE_PER_THREAD>::ShareCounters(*this);
		}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{
		// Tile of keys
		KeyType keys[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				(KeyType (*)[LOAD_VEC_SIZE]) keys,
				d_in_keys,
				cta_offset);

		// Prevent bucketing from being hoisted (otherwise we don't get the desired outstanding loads)
		if (LOADS_PER_TILE > 1) __syncthreads();

		// Bucket tile of keys
		IterateKeys<0, LOADS_PER_TILE * LOAD_VEC_SIZE>::Bucket(
			*this,
			(KeyType *) keys);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		const SizeT &out_of_bounds)
	{
		// Process partial tile if necessary using single loads
		cta_offset += threadIdx.x;
		while (cta_offset < out_of_bounds) {

			// Load and bucket key
			KeyType key = d_in_keys[cta_offset];
			Bucket(key);
			cta_offset += THREADS;
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		ResetCounters();
		ResetCompositeCounters();


#if 1	// Use deep unrolling for better instruction efficiency

		// Unroll batches of full tiles
		const int UNROLLED_ELEMENTS = UNROLL_COUNT * TILE_ELEMENTS;
		while (cta_offset  + UNROLLED_ELEMENTS < work_limits.out_of_bounds) {

			UnrollTiles::template Iterate<UNROLL_COUNT>::ProcessTiles(
				*this,
				cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			ExtractComposites();

			__syncthreads();

			// Reset composite counters in lanes
			ResetCompositeCounters();
		}

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset) {
			ProcessFullTile(cta_offset);
			cta_offset += TILE_ELEMENTS;
		}

#else 	// Use shallow unrolling for faster compilation tiles

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset) {

			ProcessFullTile(cta_offset);
			cta_offset += TILE_ELEMENTS;

			const SizeT UNROLL_MASK = (UNROLL_COUNT - 1) << LOG_TILE_ELEMENTS;
			if ((cta_offset & UNROLL_MASK) == 0) {

				__syncthreads();

				// Aggregate back into local_count registers to prevent overflow
				ExtractComposites();

				__syncthreads();

				// Reset composite counters in lanes
				ResetCompositeCounters();
			}
		}
#endif

		// Process partial tile if necessary
		ProcessPartialTile(cta_offset, work_limits.out_of_bounds);

		__syncthreads();

		// Aggregate back into local_count registers
		ExtractComposites();

		__syncthreads();

		// Final raking reduction of counts by bin, output to spine.

		ShareCounters();

		__syncthreads();

		// Rake-reduce and write out the bin_count reductions
		if (threadIdx.x < RADIX_DIGITS) {

			SizeT bin_count = util::reduction::SerialReduce<AGGREGATED_PARTIALS_PER_ROW>::Invoke(
				smem_storage.aggregate[threadIdx.x]);

			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				bin_count,
				d_spine + spine_bin_offset);


		}
	}
};



} // namespace upsweep
} // namespace radix_sort
} // namespace b40c


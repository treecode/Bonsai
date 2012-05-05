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
 * CTA-processing functionality for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/tex_vector.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/tex_ref.cuh>

#include <b40c/radix_sort/spine/tex_ref.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Partitioning downsweep scan CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions and Constants
	//---------------------------------------------------------------------

	// Integer type for digit counters (to be packed in the RakingPartial type defined below)
	typedef unsigned short Counter;

	// Integer type for raking partials (packed counters).
	typedef typename util::If<
		(KernelPolicy::SMEM_8BYTE_BANKS),
		unsigned long long,
		unsigned int>::Type RakingPartial;

	enum {
		CURRENT_BIT 				= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 				= KernelPolicy::CURRENT_PASS,
		FLOP_TURN					= KernelPolicy::CURRENT_PASS & 0x1,					// (FLOP_TURN) ? (d_keys1 --> d_keys0) : (d_keys0 --> d_keys1)
		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,
		BANK_PADDING 				= 1,												// Whether or not to insert padding for exchanging keys

		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		LOG_THREADS 				= KernelPolicy::LOG_THREADS,
		THREADS						= 1 << LOG_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_THREAD_ELEMENTS 		= KernelPolicy::LOG_THREAD_ELEMENTS,
		THREAD_ELEMENTS				= 1 << LOG_THREAD_ELEMENTS,

		LOG_TILE_ELEMENTS			= LOG_THREADS + LOG_THREAD_ELEMENTS,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		PACKED_COUNTERS				= sizeof(RakingPartial) / sizeof(Counter),
		LOG_PACKED_COUNTERS			= util::Log2<PACKED_COUNTERS>::VALUE,

		LOG_SCAN_LANES				= CUB_MAX((RADIX_BITS - LOG_PACKED_COUNTERS), 0),				// Always at least one lane
		SCAN_LANES					= 1 << LOG_SCAN_LANES,

		LOG_SCAN_ELEMENTS			= LOG_SCAN_LANES + LOG_THREADS,
		SCAN_ELEMENTS				= 1 << LOG_SCAN_ELEMENTS,

		LOG_BASE_RAKING_SEG			= LOG_SCAN_ELEMENTS - LOG_THREADS,
		PADDED_RAKING_SEG			= (1 << LOG_BASE_RAKING_SEG) + 1,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		DIGITS_PER_SCATTER_PASS 	= THREADS / MEM_BANKS,
		SCATTER_PASSES 				= RADIX_DIGITS / DIGITS_PER_SCATTER_PASS,

		LOG_STORE_TXN_THREADS 		= LOG_MEM_BANKS,
		STORE_TXN_THREADS 			= 1 << LOG_STORE_TXN_THREADS,

		BYTES_PER_COUNTER			= sizeof(Counter),
		LOG_BYTES_PER_COUNTER		= util::Log2<BYTES_PER_COUNTER>::VALUE,

		ELEMENTS_PER_TEX			= Textures<
										KeyType,
										ValueType,
										THREAD_ELEMENTS>::ELEMENTS_PER_TEX,

		THREAD_TEX_LOADS	 		= THREAD_ELEMENTS / ELEMENTS_PER_TEX,

		TILE_TEX_LOADS				= THREADS * THREAD_TEX_LOADS,
	};

	static const util::io::ld::CacheModifier 	READ_MODIFIER 		= KernelPolicy::READ_MODIFIER;
	static const util::io::st::CacheModifier 	WRITE_MODIFIER 		= KernelPolicy::WRITE_MODIFIER;
	static const ScatterStrategy 				SCATTER_STRATEGY 	= KernelPolicy::SCATTER_STRATEGY;

	// Key texture type
	typedef typename Textures<
		KeyType,
		ValueType,
		THREAD_ELEMENTS>::KeyTexType KeyTexType;

	// Value texture type
	typedef typename Textures<
		KeyType,
		ValueType,
		THREAD_ELEMENTS>::ValueTexType ValueTexType;


	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		SizeT							tex_offset;
		SizeT							tex_offset_limit;

		bool 							non_trivial_pass;
		util::CtaWorkLimits<SizeT> 		work_limits;

		SizeT 							base_digit_offset[RADIX_DIGITS];

		// Storage for scanning local ranks
		volatile RakingPartial			warpscan[WARPS][WARP_THREADS * 3 / 2];

		struct {
			int4						align_padding;
			union {
				Counter					packed_counters[SCAN_LANES + 1][THREADS][PACKED_COUNTERS];
				RakingPartial			raking_grid[THREADS][PADDED_RAKING_SEG];
				KeyType 				key_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
				ValueType 				value_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
			};
		};
	};


	/**
	 * Tile state
	 */
	struct Tile
	{
		KeyType 			keys[THREAD_ELEMENTS];
		ValueType 			values[THREAD_ELEMENTS];
		Counter				thread_prefixes[THREAD_ELEMENTS];
		int 				ranks[THREAD_ELEMENTS];

		unsigned int 		counter_offsets[THREAD_ELEMENTS];

		SizeT				bin_offsets[THREAD_ELEMENTS];
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 						&smem_storage;

	KeyType								*d_keys0;
	KeyType								*d_keys1;

	ValueType							*d_values0;
	ValueType							*d_values1;

	RakingPartial						*raking_segment;
	Counter								*bin_counter;

	SizeT								my_base_digit_offset;

	int 								warp_id;
	volatile RakingPartial				*warpscan;


	//---------------------------------------------------------------------
	// Helper structure for tile elements iteration
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int VEC, int DUMMY = 0>
	struct IterateTileElements
	{
		// DecodeKeys
		static __device__ __forceinline__ void DecodeKeys(Cta &cta,	Tile &tile)
		{
			// Compute byte offset of smem counter.  Add in thread column.
			tile.counter_offsets[VEC] = (threadIdx.x << (LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER));

			// Add in sub-counter offset
			tile.counter_offsets[VEC] = Extract<
				KeyType,
				CURRENT_BIT + LOG_SCAN_LANES,
				LOG_PACKED_COUNTERS,
				LOG_BYTES_PER_COUNTER>::SuperBFE(
					tile.keys[VEC],
					tile.counter_offsets[VEC]);

			// Add in row offset
			tile.counter_offsets[VEC] = Extract<
				KeyType,
				CURRENT_BIT,
				LOG_SCAN_LANES,
				LOG_THREADS + LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER>::SuperBFE(
					tile.keys[VEC],
					tile.counter_offsets[VEC]);

			Counter* counter = (Counter*)
				(((unsigned char *) cta.smem_storage.packed_counters) + tile.counter_offsets[VEC]);

			// Load thread-exclusive prefix
			tile.thread_prefixes[VEC] = *counter;

			// Store inclusive prefix
			*counter = tile.thread_prefixes[VEC] + 1;

			// Next vector element
			IterateTileElements<VEC + 1>::DecodeKeys(cta, tile);
		}


		// ComputeLocalRanks
		static __device__ __forceinline__ void ComputeLocalRanks(Cta &cta, Tile &tile)
		{
			Counter* counter = (Counter*)
				(((unsigned char *) cta.smem_storage.packed_counters) + tile.counter_offsets[VEC]);

			// Add in CTA exclusive prefix
			tile.ranks[VEC] = tile.thread_prefixes[VEC] + *counter;

			// Next vector element
			IterateTileElements<VEC + 1>::ComputeLocalRanks(cta, tile);
		}


		// ScatterRanked
		template <typename T>
		static __device__ __forceinline__ void ScatterRanked(
			Cta &cta,
			Tile &tile,
			T items[THREAD_ELEMENTS])
		{
			int offset = (BANK_PADDING) ?
				util::SHR_ADD(tile.ranks[VEC], LOG_MEM_BANKS, tile.ranks[VEC]) :
				tile.ranks[VEC];

			((T*) cta.smem_storage.key_exchange)[offset] = items[VEC];

			// Next vector element
			IterateTileElements<VEC + 1>::ScatterRanked(cta, tile, items);
		}

		// GatherShared
		template <typename T>
		static __device__ __forceinline__ void GatherShared(
			Cta &cta,
			Tile &tile,
			T items[THREAD_ELEMENTS])
		{
			int gather_offset =
				threadIdx.x +
				(BANK_PADDING ?
					(threadIdx.x >> LOG_MEM_BANKS) :
					0) +
				(BANK_PADDING ?
					(VEC * THREADS) + ((VEC * THREADS) >> LOG_MEM_BANKS) :
					(VEC * THREADS));

			items[VEC] = ((T*) cta.smem_storage.key_exchange)[gather_offset];

			// Next vector element
			IterateTileElements<VEC + 1>::GatherShared(cta, tile, items);
		}

		// DecodeBinOffsets
		static __device__ __forceinline__ void DecodeBinOffsets(Cta &cta, Tile &tile)
		{
			// Decode address of bin-offset in smem
			unsigned int byte_offset = Extract<
				KeyType,
				CURRENT_BIT,
				RADIX_BITS,
				util::Log2<sizeof(SizeT)>::VALUE>::SuperBFE(
					tile.keys[VEC]);

			// Lookup global bin offset
			tile.bin_offsets[VEC] = *(SizeT *)(((char *) cta.smem_storage.base_digit_offset) + byte_offset);

			// Next vector element
			IterateTileElements<VEC + 1>::DecodeBinOffsets(cta, tile);
		}

		// ScatterGlobal
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			Cta &cta,
			Tile &tile,
			T items[THREAD_ELEMENTS],
			T *d_out,
			const SizeT &guarded_elements)
		{
			int tile_element = threadIdx.x + (VEC * THREADS);

			// Distribute if not out-of-bounds
			if ((guarded_elements >= TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				util::io::ModifiedStore<WRITE_MODIFIER>::St(
					items[VEC],
					d_out + threadIdx.x + (THREADS * VEC) + tile.bin_offsets[VEC]);
			}

			// Next vector element
			IterateTileElements<VEC + 1>::ScatterGlobal(cta, tile, items, d_out, guarded_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int DUMMY>
	struct IterateTileElements<THREAD_ELEMENTS, DUMMY>
	{
		// DecodeKeys
		static __device__ __forceinline__ void DecodeKeys(Cta &cta, Tile &tile) {}

		// ComputeLocalRanks
		static __device__ __forceinline__ void ComputeLocalRanks(Cta &cta, Tile &tile) {}

		// ScatterRanked
		template <typename T>
		static __device__ __forceinline__ void ScatterRanked(Cta &cta, Tile &tile, T items[THREAD_ELEMENTS]) {}

		// GatherShared
		template <typename T>
		static __device__ __forceinline__ void GatherShared(Cta &cta, Tile &tile, T items[THREAD_ELEMENTS]) {}

		// DecodeBinOffsets
		static __device__ __forceinline__ void DecodeBinOffsets(Cta &cta, Tile &tile) {}

		// ScatterGlobal
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(Cta &cta, Tile &tile, T items[THREAD_ELEMENTS], T *d_out, const SizeT &guarded_elements) {}
	};


	/**
	 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
	 * coalescing rules
	 */
	template <int PASS, int SCATTER_PASSES>
	struct AlignedScatter
	{
		template <typename T>
		static __device__ __forceinline__ void ScatterPass(
			Cta &cta,
			T *exchange,
			T *d_out,
			const SizeT &valid_elements)
		{
			int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
			int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;

			int my_digit = (PASS * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

			if (my_digit < RADIX_DIGITS) {

				int my_exclusive_scan = cta.smem_storage.warpscan[0][16 + my_digit - 1];
				int my_inclusive_scan = cta.smem_storage.warpscan[0][16 + my_digit];
				int my_digit_count = my_inclusive_scan - my_exclusive_scan;

				int my_carry = cta.smem_storage.base_digit_offset[my_digit] + my_exclusive_scan;
				int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

				while (my_aligned_offset < my_digit_count) {

					if ((my_aligned_offset >= 0) && (my_exclusive_scan + my_aligned_offset < valid_elements)) {

						T datum = exchange[my_exclusive_scan + my_aligned_offset];
						d_out[my_carry + my_aligned_offset] = datum;
					}
					my_aligned_offset += STORE_TXN_THREADS;
				}
			}

			AlignedScatter<PASS + 1, SCATTER_PASSES>::ScatterPass(
				cta,
				exchange,
				d_out,
				valid_elements);
		}
	};

	// Terminate
	template <int SCATTER_PASSES>
	struct AlignedScatter<SCATTER_PASSES, SCATTER_PASSES>
	{
		template <typename T>
		static __device__ __forceinline__ void ScatterPass(
			Cta &cta,
			T *exchange,
			T *d_out,
			const SizeT &valid_elements) {}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys0,
		KeyType 		*d_keys1,
		ValueType 		*d_values0,
		ValueType 		*d_values1,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_keys0(d_keys0),
			d_keys1(d_keys1),
			d_values0(d_values0),
			d_values1(d_values1),
			raking_segment(smem_storage.raking_grid[threadIdx.x])
	{
		int counter_lane = threadIdx.x & (SCAN_LANES - 1);
		int sub_counter = threadIdx.x >> (LOG_SCAN_LANES);
		bin_counter = &smem_storage.packed_counters[counter_lane][0][sub_counter];

		// Initialize warpscan identity regions
		warp_id = threadIdx.x >> 5;
		warpscan = &smem_storage.warpscan[warp_id][16 + (threadIdx.x & 31)];
		warpscan[-16] = 0;

		if ((THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS)) {

			// Read base_digit_offset in parallel
			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
			my_base_digit_offset = tex1Dfetch(spine::TexSpine<SizeT>::ref, spine_bin_offset);
		}
	}

	/**
	 * Load tile of keys
	 */
	__device__ __forceinline__ void LoadKeys(
		SizeT tex_offset,
		const SizeT &guarded_elements,
		Tile &tile)
	{
		if (guarded_elements >= TILE_ELEMENTS) {

			// Unguarded loads through tex
			KeyTexType *vectors = (KeyTexType *) tile.keys;

			#pragma unroll
			for (int PACK = 0; PACK < THREAD_TEX_LOADS; PACK++) {

				vectors[PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						TexKeys<KeyTexType>::ref1 :
						TexKeys<KeyTexType>::ref0,
					tex_offset + (threadIdx.x * THREAD_TEX_LOADS) + PACK);
			}

		} else {

			// Guarded loads with default assignment of -1 to out-of-bound keys
			util::io::LoadTile<
				0,									// log loads per tile
				LOG_THREAD_ELEMENTS,
				THREADS,
				READ_MODIFIER,
				false>::LoadValid(
					(KeyType (*)[THREAD_ELEMENTS]) tile.keys,
					(Cta::FLOP_TURN) ?
						d_keys1 :
						d_keys0,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements,
					KeyType(-1));
		}
	}

	/**
	 * Load tile of values
	 */
	__device__ __forceinline__ void LoadValues(
		SizeT tex_offset,
		const SizeT &guarded_elements,
		Tile &tile)
	{
		if (guarded_elements >= TILE_ELEMENTS) {

			// Unguarded loads through tex
			ValueTexType *vectors = (ValueTexType*) tile.values;

			#pragma unroll
			for (int PACK = 0; PACK < THREAD_TEX_LOADS; PACK++) {

				vectors[PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						TexValues<ValueTexType>::ref1 :
						TexValues<ValueTexType>::ref0,
					tex_offset + (threadIdx.x * THREAD_TEX_LOADS) + PACK);
			}

		} else {
			// Guarded loads with default assignment of -1 to out-of-bound values
			util::io::LoadTile<
				0,									// log loads per tile
				LOG_THREAD_ELEMENTS,
				THREADS,
				READ_MODIFIER,
				false>::LoadValid(
					(ValueType (*)[THREAD_ELEMENTS]) tile.values,
					(Cta::FLOP_TURN) ?
						d_values1 :
						d_values0,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements);
		}
	}


	/**
	 * Scan shared memory counters
	 */
	__device__ __forceinline__ void ScanCounters(Tile &tile)
	{
		// Upsweep reduce
		RakingPartial raking_partial = util::reduction::SerialReduce<PADDED_RAKING_SEG>::Invoke(
			raking_segment);

		// Warpscan
		RakingPartial partial = raking_partial;
		warpscan[0] = partial;

		warpscan[0] = partial =
			partial + warpscan[0 - 1];
		warpscan[0] = partial =
			partial + warpscan[0 - 2];
		warpscan[0] = partial =
			partial + warpscan[0 - 4];
		warpscan[0] = partial =
			partial + warpscan[0 - 8];
		warpscan[0] = partial =
			partial + warpscan[0 - 16];

		// Barrier
		__syncthreads();

		// Scan across warpscan totals
		RakingPartial warpscan_totals = 0;

		#pragma unroll
		for (int WARP = 0; WARP < WARPS; WARP++) {

			// Add totals from all previous warpscans into our partial
			RakingPartial warpscan_total = smem_storage.warpscan[WARP][(WARP_THREADS * 3 / 2) - 1];
			if (warp_id == WARP) {
				partial += warpscan_totals;
			}

			// Increment warpscan totals
			warpscan_totals += warpscan_total;
		}

		// Add lower totals from all warpscans into partial's upper
		#pragma unroll
		for (int PACKED = 1; PACKED < PACKED_COUNTERS; PACKED++) {
			partial += warpscan_totals << (16 * PACKED);
		}

		// Downsweep scan with exclusive partial
		RakingPartial exclusive_partial = partial - raking_partial;
		util::scan::SerialScan<PADDED_RAKING_SEG>::Invoke(
			raking_segment,
			exclusive_partial);
	}


	/**
	 * Truck along associated values.  (Specialized for key-value passes.)
	 */
	template <bool IS_KEYS_ONLY, int DUMMY = 0>
	struct TruckValues
	{
		static __device__ __forceinline__ void Invoke(
			SizeT tex_offset,
			const SizeT &guarded_elements,
			Cta &cta,
			Tile &tile)
		{
			// Load tile of values
			cta.LoadValues(tex_offset, guarded_elements, tile);

			__syncthreads();

			// Scatter values shared
			IterateTileElements<0>::ScatterRanked(cta, tile, tile.values);

			__syncthreads();

			// Gather values from shared memory and scatter to global
			if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

				// Use explicitly warp-aligned scattering of values from smem
				AlignedScatter<0, SCATTER_PASSES>::ScatterPass(
					cta,
					cta.smem_storage.value_exchange,
					(FLOP_TURN) ?
						cta.d_values0 :
						cta.d_values1,
						guarded_elements);

			} else {

				// Gather values shared
				IterateTileElements<0>::GatherShared(cta, tile, tile.values);

				// Scatter to global
				IterateTileElements<0>::ScatterGlobal(
					cta,
					tile,
					tile.values,
					(Cta::FLOP_TURN) ?
					cta.d_values0 :
					cta.d_values1,
					guarded_elements);
			}
		}
	};


	/**
	 * Truck along associated values.  (Specialized for keys-only passes.)
	 */
	template <int DUMMY>
	struct TruckValues<true, DUMMY>
	{
		static __device__ __forceinline__ void Invoke(
			SizeT tex_offset,
			const SizeT &guarded_elements,
			Cta &cta,
			Tile &tile)
		{
			// do nothing
		}
	};


	/**
	 * Gather keys from smem and scatter to global
	 */
	__device__ __forceinline__ void GatherScatterKeys(
		Tile &tile,
		const SizeT &guarded_elements)
	{
		if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

			// Use explicitly warp-aligned scattering of keys from smem
			AlignedScatter<0, SCATTER_PASSES>::ScatterPass(
				*this,
				smem_storage.key_exchange,
				(FLOP_TURN) ?
					d_keys0 :
					d_keys1,
					guarded_elements);

		} else {

			// Gather keys
			IterateTileElements<0>::GatherShared(*this, tile, tile.keys);

			// Decode global scatter offsets
			IterateTileElements<0>::DecodeBinOffsets(*this, tile);

			// Scatter to global
			IterateTileElements<0>::ScatterGlobal(
				*this,
				tile,
				tile.keys,
				(Cta::FLOP_TURN) ?
					d_keys0 :
					d_keys1,
				guarded_elements);
		}
	}


	/**
	 * Reset shared memory digit counters
	 */
	__device__ __forceinline__ void ResetCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < SCAN_LANES + 1; LANE++) {
			*((RakingPartial*) smem_storage.packed_counters[LANE][threadIdx.x]) = 0;
		}
	}


	/**
	 * Update global scatter offsets for each digit
	 */
	__device__ __forceinline__ void UpdateDigitScatterOffsets()
	{
		if ((THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS)) {

			Counter bin_inclusive = bin_counter[THREADS * PACKED_COUNTERS];
			smem_storage.warpscan[0][16 + threadIdx.x] = bin_inclusive;
			RakingPartial bin_exclusive = smem_storage.warpscan[0][16 + threadIdx.x - 1];

			my_base_digit_offset -= bin_exclusive;
			smem_storage.base_digit_offset[threadIdx.x] = my_base_digit_offset;
			my_base_digit_offset += bin_inclusive;
		}
	}


	/**
	 * Process tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT tex_offset,
		const SizeT &guarded_elements = TILE_ELEMENTS)
	{
		// State for the current tile
		Tile tile;

		// Load tile of keys
		LoadKeys(tex_offset, guarded_elements, tile);

		__syncthreads();

		// Reset shared memory digit counters
		ResetCounters();

		// Decode bins and update counters
		IterateTileElements<0>::DecodeKeys(*this, tile);

		__syncthreads();

		// Scan shared memory counters
		ScanCounters(tile);

		__syncthreads();

		// Update global scatter offsets for each digit
		UpdateDigitScatterOffsets();

		// Extract the local ranks of each key
		IterateTileElements<0>::ComputeLocalRanks(*this, tile);

		__syncthreads();

		// Scatter keys to shared memory in sorted order
		IterateTileElements<0>::ScatterRanked(*this, tile, tile.keys);

		__syncthreads();

		// Gather keys from shared memory and scatter to global
		GatherScatterKeys(tile, guarded_elements);

		// Truck along values (if applicable)
		TruckValues<KEYS_ONLY>::Invoke(tex_offset, guarded_elements, *this, tile);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT tex_offset = smem_storage.tex_offset;

		// Process full tiles of tile_elements
		while (tex_offset < smem_storage.tex_offset_limit) {
			ProcessTile(tex_offset);
			tex_offset += TILE_TEX_LOADS;
		}

		// Clean up last partial tile with guarded-io
		if (work_limits.guarded_elements) {
			ProcessTile(tex_offset, work_limits.guarded_elements);
		}
	}
};


} // namespace downsweep
} // namespace radix_sort
} // namespace b40c


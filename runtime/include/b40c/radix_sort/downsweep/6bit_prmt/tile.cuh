/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * Abstract tile-processing functionality for partitioning downsweep scan
 * kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace partition {
namespace downsweep {


/**
 * Templated texture reference for keys
 */
template <typename KeyType>
struct KeysTex
{
	static texture<KeyType, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename KeyType>
texture<KeyType, cudaTextureType1D, cudaReadModeElementType> KeysTex<KeyType>::ref;



/**
 * Tile
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedTile>
struct Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;

	typedef DerivedTile Dispatch;

	enum {
		LOAD_VEC_SIZE 				= KernelPolicy::LOAD_VEC_SIZE,
		LOADS_PER_TILE 				= KernelPolicy::LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD 	= KernelPolicy::TILE_ELEMENTS_PER_THREAD,

		LOG_SCAN_LANES				= KernelPolicy::LOG_SCAN_LANES_PER_TILE,
		SCAN_LANES					= KernelPolicy::SCAN_LANES_PER_TILE,

		LOG_PACKS_PER_LOAD			= KernelPolicy::LOG_LOAD_VEC_SIZE - KernelPolicy::LOG_PACK_SIZE,
		PACKS_PER_LOAD				= 1 << LOG_PACKS_PER_LOAD,

		LANE_ROWS_PER_LOAD 			= KernelPolicy::ByteGrid::ROWS_PER_LANE / KernelPolicy::LOADS_PER_TILE,
		LANE_STRIDE_PER_LOAD 		= KernelPolicy::ByteGrid::PADDED_PARTIALS_PER_ROW * LANE_ROWS_PER_LOAD,

		INVALID_BIN					= -1,

		LOG_RAKING_THREADS 			= KernelPolicy::ByteGrid::LOG_RAKING_THREADS,
		RAKING_THREADS 				= 1 << LOG_RAKING_THREADS,

		LOG_WARPSCAN_THREADS		= CUB_LOG_WARP_THREADS(CUDA_ARCH),
		WARPSCAN_THREADS 			= 1 << LOG_WARPSCAN_THREADS,

	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------


	// The keys (and values) this thread will read this tile
	KeyType 	keys[LOADS_PER_TILE][LOAD_VEC_SIZE];
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];

	// For each load:
	// 		counts_nibbles contains the bin counts within nibbles ordered right to left
	// 		bins_nibbles contains the bin for each key within nibbles ordered right to left
	// 		load_prefix_bytes contains the exclusive scan for each key within nibbles ordered right to left

	int 		bins_nibbles[(LOAD_VEC_SIZE + 7) / 8][LOADS_PER_TILE];

	int 		counts_nibbles[SCAN_LANES / 2][LOADS_PER_TILE];
	int			counts_bytes[SCAN_LANES][LOADS_PER_TILE];

	int 		load_prefix_bytes[(LOAD_VEC_SIZE + 3) / 4][LOADS_PER_TILE];

	int 		warpscan_shorts[LOADS_PER_TILE][4];

	int 		local_ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];		// The local rank of each key
	SizeT 		scatter_offsets[LOADS_PER_TILE][LOAD_VEC_SIZE];	// The global rank of each key

	int 		bins[TILE_ELEMENTS_PER_THREAD];


	//---------------------------------------------------------------------
	// Tile Methods
	//---------------------------------------------------------------------

	/**
	 * ExtractRanks
	 */
	template <int LOAD, int VEC, int REM = (VEC & 7)>
	struct ExtractRanks
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile, const bool shift_bytes) {}
	};


	/**
	 * ExtractRanks (VEC % 8 == 0)
	 */
	template <int LOAD, int VEC>
	struct ExtractRanks<LOAD, VEC, 0>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile, const bool shift_bytes)
		{
/*
			printf("\tTid(%d) Vec(%d) bins_nibbles(%08x)\n",
				threadIdx.x, VEC, tile->bins_nibbles[VEC / 8][LOAD]);
*/
			// Decode prefix bytes for first four keys
			tile->load_prefix_bytes[VEC / 4][LOAD] += util::PRMT(
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD],
				tile->bins_nibbles[VEC / 8][LOAD]);

			// Decode scan low and high packed words for first four keys
			int warpscan_prefix[2];
			warpscan_prefix[0] = util::PRMT(
				tile->warpscan_shorts[LOAD][0],
				tile->warpscan_shorts[LOAD][1],
				tile->bins_nibbles[VEC / 8][LOAD]);

			warpscan_prefix[1] = util::PRMT(
				tile->warpscan_shorts[LOAD][2],
				tile->warpscan_shorts[LOAD][3],
				tile->bins_nibbles[VEC / 8][LOAD]);

			// Low
			int packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x5140) +
				util::PRMT(								// Raking scan component (lower bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4140);

			packed_scatter = util::SHR_ADD(0xffe0ffe0 & packed_scatter, 5, packed_scatter);
			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 0] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 1] = packed_scatter >> 16;

			// High
			packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x7362) +
				util::PRMT(								// Raking scan component (upper bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4342);

			packed_scatter = util::SHR_ADD(0xffe0ffe0 & packed_scatter, 5, packed_scatter);
			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 2] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 3] = packed_scatter >> 16;

		}
	};


	/**
	 * ExtractRanks (VEC % 8 == 4)
	 */
	template <int LOAD, int VEC>
	struct ExtractRanks<LOAD, VEC, 4>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile, const bool shift_bytes)
		{
			int upper_bins_nibbles = tile->bins_nibbles[VEC / 8][LOAD] >> 16;

			// Decode prefix bytes for second four keys
			tile->load_prefix_bytes[VEC / 4][LOAD] += util::PRMT(
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD],
				upper_bins_nibbles);

			// Decode scan low and high packed words for second four keys
			int warpscan_prefix[2];
			warpscan_prefix[0] = util::PRMT(
				tile->warpscan_shorts[LOAD][0],
				tile->warpscan_shorts[LOAD][1],
				upper_bins_nibbles);

			warpscan_prefix[1] = util::PRMT(
				tile->warpscan_shorts[LOAD][2],
				tile->warpscan_shorts[LOAD][3],
				upper_bins_nibbles);

			// Low
			int packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x5140) +
				util::PRMT(								// Raking scan component (lower bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4140);

			packed_scatter = util::SHR_ADD(0xffe0ffe0 & packed_scatter, 5, packed_scatter);
			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 0] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 1] = packed_scatter >> 16;

			// High
			packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x7362) +
				util::PRMT(								// Raking scan component (upper bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4342);

			packed_scatter = util::SHR_ADD(0xffe0ffe0 & packed_scatter, 5, packed_scatter);
			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 2] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 3] = packed_scatter >> 16;
		}
	};



	//---------------------------------------------------------------------
	// IterateTileElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int LOAD, int VEC, int dummy = 0>
	struct IterateTileElements
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(
			Cta *cta,
			Tile *tile,
			const int CURRENT_BIT)
		{
			// Decode the bin for this key
			int bin = util::BFE(
				tile->keys[LOAD][VEC],
				CURRENT_BIT,
				KernelPolicy::LOG_SCAN_BINS);

			const int BITS_PER_NIBBLE = 4;
			int shift = bin * BITS_PER_NIBBLE;

			// Initialize exclusive scan bytes
			if (VEC == 0) {

				tile->load_prefix_bytes[VEC / 4][LOAD] = 0;

			} else {
				int prev_counts_nibbles = tile->counts_nibbles[0][LOAD] >> shift;

				if ((VEC & 3) == 0) {

					tile->load_prefix_bytes[VEC / 4][LOAD] = prev_counts_nibbles & 0xf;

				} else if ((VEC & 7) < 4) {

					util::BFI(
						tile->load_prefix_bytes[VEC / 4][LOAD],
						tile->load_prefix_bytes[VEC / 4][LOAD],
						prev_counts_nibbles,
						8 * (VEC & 7),
						BITS_PER_NIBBLE);

				} else {

					util::BFI(
						tile->load_prefix_bytes[VEC / 4][LOAD],
						tile->load_prefix_bytes[VEC / 4][LOAD],
						prev_counts_nibbles,
						8 * ((VEC & 7) - 4),
						BITS_PER_NIBBLE);
				}
			}

			// Initialize counts nibbles
			if (VEC == 0) {
				tile->counts_nibbles[0][LOAD] = 1 << shift;

			} else if (VEC == LOAD_VEC_SIZE - 1) {

				// last vector element
				if ((VEC & 15) == 15) {

					// Protect overflow: expand nibbles into bytes and then add
					util::NibblesToBytes(
						tile->counts_bytes[0][LOAD],
						tile->counts_bytes[1][LOAD],
						tile->counts_nibbles[0][LOAD]);

					shift = shift * 2;
					util::SHL_ADD(
						tile->counts_bytes[0][LOAD],
						1,
						shift,
						tile->counts_bytes[0][LOAD]);

					util::SHL_ADD(
						tile->counts_bytes[1][LOAD],
						1,
						shift - 32,
						tile->counts_bytes[1][LOAD]);

				} else {

					// Add nibble then expand into bytes
					util::SHL_ADD(
						tile->counts_nibbles[0][LOAD],
						1,
						shift,
						tile->counts_nibbles[0][LOAD]);

					util::NibblesToBytes(
						tile->counts_bytes[0][LOAD],
						tile->counts_bytes[1][LOAD],
						tile->counts_nibbles[0][LOAD]);
				}

			} else {
				util::SHL_ADD(
					tile->counts_nibbles[0][LOAD],
					1,
					shift,
					tile->counts_nibbles[0][LOAD]);
			}

			// Initialize bins nibbles
			if ((VEC & 7) == 0) {
				tile->bins_nibbles[VEC / 8][LOAD] = bin;

			} else {
				util::BFI(
					tile->bins_nibbles[VEC / 8][LOAD],
					tile->bins_nibbles[VEC / 8][LOAD],
					bin,
					4 * (VEC & 7),
					4);
			}

			// Next vector element
			IterateTileElements<LOAD, VEC + 1>::DecodeKeys(cta, tile, CURRENT_BIT);
		}

		// ComputeRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile, const bool shift_bytes)
		{
			if (VEC == 0) {

				const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

				// Extract prefix bytes from bytes raking grid
				tile->counts_bytes[0][LOAD] = cta->byte_grid_details.lane_partial[0][LANE_OFFSET];
				tile->counts_bytes[1][LOAD] = cta->byte_grid_details.lane_partial[1][LANE_OFFSET];

				// Extract warpscan shorts
				const int LOAD_RAKING_TID_OFFSET = (KernelPolicy::THREADS * LOAD) >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG;

				int base_raking_tid = threadIdx.x >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG;

				tile->warpscan_shorts[LOAD][0] = cta->smem_storage.short_prefixes[0][base_raking_tid + LOAD_RAKING_TID_OFFSET];
				tile->warpscan_shorts[LOAD][1] = cta->smem_storage.short_prefixes[1][base_raking_tid + LOAD_RAKING_TID_OFFSET];
				tile->warpscan_shorts[LOAD][2] = cta->smem_storage.short_prefixes[0][base_raking_tid + LOAD_RAKING_TID_OFFSET + (RAKING_THREADS / 2)];
				tile->warpscan_shorts[LOAD][3] = cta->smem_storage.short_prefixes[1][base_raking_tid + LOAD_RAKING_TID_OFFSET + (RAKING_THREADS / 2)];
			}

			ExtractRanks<LOAD, VEC>::Invoke(cta, tile, shift_bytes);
/*
			printf("tid(%d) vec(%d) key(%d) scatter(%d)\n",
				threadIdx.x,
				VEC,
				tile->keys[LOAD][VEC],
				tile->local_ranks[LOAD][VEC] / 4);
*/
			// Next vector element
			IterateTileElements<LOAD, VEC + 1>::ComputeRanks(cta, tile, shift_bytes);
		}
	};



	/**
	 * IterateTileElements next load
	 */
	template <int LOAD, int dummy>
	struct IterateTileElements<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(
			Cta *cta,
			Tile *tile,
			const int CURRENT_BIT)
		{
			const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

			// Place keys into raking grid
			cta->byte_grid_details.lane_partial[0][LANE_OFFSET] = tile->counts_bytes[0][LOAD];
			cta->byte_grid_details.lane_partial[1][LANE_OFFSET] = tile->counts_bytes[1][LOAD];
/*
			printf("Tid %u load %u:\t,"
				"load_prefix_bytes[0](%08x), "
				"load_prefix_bytes[1](%08x), "
				"counts_bytes[0](%08x), "
				"counts_bytes[1](%08x), "
				"\n",
				threadIdx.x, LOAD,
				tile->load_prefix_bytes[0][LOAD],
				tile->load_prefix_bytes[1][LOAD],
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD]);
*/
			// First vector element, next load
			IterateTileElements<LOAD + 1, 0>::DecodeKeys(cta, tile, CURRENT_BIT);
		}

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile, const bool shift_bytes)
		{
			// First vector element, next load
			IterateTileElements<LOAD + 1, 0>::ComputeRanks(cta, tile, shift_bytes);
		}

	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateTileElements<LOADS_PER_TILE, 0, dummy>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile, const int CURRENT_BIT) {}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile, const bool shift_bytes) {}
	};



	//---------------------------------------------------------------------
	// Tile Internal Methods
	//---------------------------------------------------------------------


	/**
	 * Scan Tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScanTile(Cta *cta, const int CURRENT_BIT, const bool shift_bytes)
	{

		// Decode bins and place keys into grid
		IterateTileElements<0, 0>::DecodeKeys(cta, this, CURRENT_BIT);

		__syncthreads();

		int tid = threadIdx.x & 31;
		int warp = threadIdx.x >> 5;
		volatile int *p = cta->smem_storage.short_prefixes[warp];
		volatile int *p2 = &cta->smem_storage.short_prefixes[warp][tid * 2];
		volatile int *warpscan = cta->smem_storage.warpscan[warp];

		// Use our raking threads to, in aggregate, scan the composite counter lanes
		if (threadIdx.x < RAKING_THREADS) {

/*
			if (threadIdx.x == 0) {
				printf("ByteGrid:\n");
				KernelPolicy::ByteGrid::Print();
				printf("\n");
			}
*/
			// Upsweep rake
			int partial_bytes = util::scan::SerialScan<KernelPolicy::ByteGrid::PARTIALS_PER_SEG>::Invoke(
				cta->byte_grid_details.raking_segment,
				0);
/*
			printf("\t\t\tRaking tid %d with partial_bytes (%08x)\n",
				threadIdx.x, partial_bytes);
*/
			// Extract bytes into shorts (first warp has 0-3, second has 4-7)
			p[tid] = util::PRMT(partial_bytes, 0, 0x4240);
			p[tid + CUB_WARP_THREADS(CUDA_ARCH)] = util::PRMT(partial_bytes, 0, 0x4341);

			int partial0 = *p2;
			int partial1 = *(p2 + 1);

			int partial = partial0 + partial1;

			warpscan[16 + tid] = partial;

			warpscan[16 + tid] = partial =
				partial + warpscan[16 + tid - 1];
			warpscan[16 + tid] = partial =
				partial + warpscan[16 + tid - 2];
			warpscan[16 + tid] = partial =
				partial + warpscan[16 + tid - 4];
			warpscan[16 + tid] = partial =
				partial + warpscan[16 + tid - 8];
			warpscan[16 + tid] = partial =
				partial + warpscan[16 + tid - 16];

			// Restricted barrier
			util::BAR(RAKING_THREADS);

			// Grab first warp total
			int total = cta->smem_storage.warpscan[0][16 + CUB_WARP_THREADS(CUDA_ARCH) - 1];
			if (threadIdx.x >= (RAKING_THREADS / 2)) {

				// Second warp adds halves from first warp total into partial
				int flip = util::PRMT(total, total, 0x1032);
				total += flip;
				partial += total;

				// Second warp replaces with second warp total
				total = cta->smem_storage.warpscan[1][16 + CUB_WARP_THREADS(CUDA_ARCH) - 1];
			}

			// Add lower into upper
			partial = util::SHL_ADD_C(total, 16, partial);

			int exclusive1 = partial - partial1;
			int exclusive0 = exclusive1 - partial0;

/*
			printf("\tRaking tid %d with inclusive_partial((%u,%u),(%u,%u)) and exclusive_partial((%u,%u),(%u,%u))\n",
				threadIdx.x,
				inclusive0 >> 16, inclusive0 & 0x0000ffff,
				inclusive1 >> 16, inclusive1 & 0x0000ffff,
				exclusive0 >> 16, exclusive0 & 0x0000ffff,
				exclusive1 >> 16, exclusive1 & 0x0000ffff);
*/
			// Trade
			// (0,2) .. (1,3)
			*p2 = exclusive0;
			*(p2 + 1) = exclusive1;

			// Interleave:
			// (0L, 1L, 2L, 3L)
			// (0H, 1H, 2H, 3H)
			int a = p[tid];								// 0,2
			int b = p[tid + 32];						// 1,3

			p[tid] =
				util::PRMT(a, b, 0x6240);
			p[tid + 32] =
				util::PRMT(a, b, 0x7351);
		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateTileElements<0, 0>::ComputeRanks(cta, this, shift_bytes);
	}



	//---------------------------------------------------------------------
	// IterateElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next tile element
	 */
	template <int ELEMENT, int dummy = 0>
	struct IterateElements
	{
		// GatherDecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherDecodeKeys(Cta *cta, Tile *tile)
		{
			const int LOAD_OFFSET = (ELEMENT * KernelPolicy::THREADS) + ((ELEMENT * KernelPolicy::THREADS) >> 5);

			KeyType *linear_keys = (KeyType *) tile->keys;

			linear_keys[ELEMENT] = cta->offset[LOAD_OFFSET];
			KeyType next_key = cta->next_offset[LOAD_OFFSET];

			tile->bins[ELEMENT] = util::BFE(
				linear_keys[ELEMENT],
				KernelPolicy::CURRENT_BIT,
				KernelPolicy::LOG_BINS);

			int2 item;	// (inclusive for bins[element], next bin)
			item.x = threadIdx.x + (ELEMENT * KernelPolicy::THREADS);
			item.y = ((ELEMENT == TILE_ELEMENTS_PER_THREAD - 1) && (threadIdx.x == KernelPolicy::THREADS - 1)) ?
				KernelPolicy::BINS :						// invalid bin
				util::BFE(
					next_key,
					KernelPolicy::CURRENT_BIT,
					KernelPolicy::LOG_BINS);

			if (tile->bins[ELEMENT] != item.y) {
				cta->smem_storage.bin_in_prefixes[tile->bins[ELEMENT]] = item;
			}

			IterateElements<ELEMENT + 1>::GatherDecodeKeys(cta, tile);
		}

		// ScatterKeysToGlobal
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterKeysToGlobal(
			Cta *cta,
			Tile *tile,
			const SizeT &guarded_elements)
		{
			KeyType *linear_keys = (KeyType *) tile->keys;

			int bin_carry = cta->smem_storage.bin_carry[tile->bins[ELEMENT]];
			int tile_element = threadIdx.x + (ELEMENT * KernelPolicy::THREADS);
/*
			printf("\tTid %d scattering key[%d](%d) with bin_carry(%d) to offset %d\n",
				threadIdx.x,
				ELEMENT,
				linear_keys[ELEMENT],
				bin_carry,
				threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + bin_carry);
*/
			if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					linear_keys[ELEMENT],
					cta->d_out_keys + threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + bin_carry);
			}

			IterateElements<ELEMENT + 1>::ScatterKeysToGlobal(cta, tile, guarded_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateElements<TILE_ELEMENTS_PER_THREAD, dummy>
	{
		// GatherDecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherDecodeKeys(
			Cta *cta, Tile *tile) {}

		// ScatterKeysToGlobal
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterKeysToGlobal(
			Cta *cta, Tile *tile, const SizeT &guarded_elements) {}
	};



	//---------------------------------------------------------------------
	// Partition/scattering specializations
	//---------------------------------------------------------------------


	/**
	 * Specialized for two-phase scatter, keys-only
	 */
	template <ScatterStrategy SCATTER_STRATEGY>
	struct PartitionTile
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT pack_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load keys
//			tile->LoadKeys(cta, cta_offset, guarded_elements);

			typedef typename util::VecType<KeyType, KernelPolicy::PACK_SIZE>::Type VectorType;
			VectorType (*vectors)[PACKS_PER_LOAD] = (VectorType (*)[PACKS_PER_LOAD]) tile->keys;

			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int PACK = 0; PACK < PACKS_PER_LOAD; PACK++) {

					vectors[LOAD][PACK] = tex1Dfetch(
						KeysTex<VectorType>::ref,
						pack_offset + (threadIdx.x * PACKS_PER_LOAD) + (LOAD * KernelPolicy::THREADS * PACKS_PER_LOAD) + PACK);
				}
			}

			// Scan tile
			tile->ScanTile(cta, KernelPolicy::CURRENT_BIT, true);

			__syncthreads();

			// Scatter keys to smem by local rank
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					char * ptr = (char *) cta->smem_storage.key_exchange;
					KeyType * ptr_key = (KeyType *)(ptr + tile->local_ranks[LOAD][VEC]);

					*ptr_key = tile->keys[LOAD][VEC];
				}
			}
/*
			__syncthreads();

			// Gather keys from smem (strided)
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					const int LOAD_IDX = LOAD * LOAD_VEC_SIZE * KernelPolicy::THREADS;
					const int LOAD_OFFSET = LOAD_IDX + (LOAD_IDX >> CUB_MAX(5, KernelPolicy::LOG_LOAD_VEC_SIZE));

					tile->keys[LOAD][VEC] = cta->smem_storage.key_exchange[
						(threadIdx.x * LOAD_VEC_SIZE) +
						((threadIdx.x * LOAD_VEC_SIZE) >> CUB_MAX(5, KernelPolicy::LOG_LOAD_VEC_SIZE)) +
						LOAD_OFFSET +
						VEC];
				}
			}

			__syncthreads();

			// Scan tile
			tile->ScanTile(cta, KernelPolicy::CURRENT_BIT + KernelPolicy::LOG_SCAN_BINS, true);

			__syncthreads();

			// Scatter keys to smem by local rank
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					char * ptr = (char *) cta->smem_storage.key_exchange;
					KeyType * ptr_key = (KeyType *)(ptr + tile->local_ranks[LOAD][VEC]);

					*ptr_key = tile->keys[LOAD][VEC];
				}
			}
*/
			__syncthreads();

			// Gather keys linearly from smem (also saves off bin in/exclusives)
			IterateElements<0>::GatherDecodeKeys(cta, tile);

			__syncthreads();

			if (threadIdx.x < KernelPolicy::BINS) {

				// Put exclusive count into corresponding bin
				int2 item = cta->smem_storage.bin_in_prefixes[threadIdx.x];
				int bin_inclusive = item.x + 1;
				cta->smem_storage.bin_ex_prefixes[item.y] = bin_inclusive;

				// Restricted barrier
				util::BAR(KernelPolicy::BINS);

				int bin_exclusive = cta->smem_storage.bin_ex_prefixes[threadIdx.x];

				cta->my_bin_carry -= bin_exclusive;
				cta->smem_storage.bin_carry[threadIdx.x] = cta->my_bin_carry;
				cta->my_bin_carry += bin_inclusive;

				item.x = -1;
				item.y = KernelPolicy::BINS;
				cta->smem_storage.bin_in_prefixes[threadIdx.x] = item;
/*
				printf("bin %d bin_inclusive %d bin_exclusive %d my_bin_carry %d\n",
					threadIdx.x, bin_inclusive, bin_exclusive, cta->my_bin_carry);
*/
			}

			__syncthreads();

			// Scatter keys to global bin partitions
			IterateElements<0>::ScatterKeysToGlobal(cta, tile, guarded_elements);

		}
	};





	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Loads, decodes, and scatters a tile into global partitions
	 */
	template <typename Cta>
	__device__ __forceinline__ void Partition(
		SizeT pack_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		PartitionTile<KernelPolicy::SCATTER_STRATEGY>::Invoke(
			pack_offset,
			guarded_elements,
			cta,
			(Dispatch *) this);

	}

};


} // namespace downsweep
} // namespace partition
} // namespace b40c


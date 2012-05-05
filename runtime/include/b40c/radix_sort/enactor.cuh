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
 * Radix sorting enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/spine.cuh>
#include <b40c/util/ping_pong_storage.cuh>
#include <b40c/util/numeric_traits.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/kernel_props.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/vector_types.cuh>
#include <b40c/util/error_utils.cuh>

#include <b40c/radix_sort/sort_utils.cuh>
#include <b40c/radix_sort/policy.cuh>
#include <b40c/radix_sort/upsweep/kernel_policy.cuh>
#include <b40c/radix_sort/upsweep/kernel.cuh>

#include <b40c/radix_sort/spine/kernel_policy.cuh>
#include <b40c/radix_sort/spine/kernel.cuh>
#include <b40c/radix_sort/spine/tex_ref.cuh>

#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/kernel.cuh>
#include <b40c/radix_sort/downsweep/tex_ref.cuh>

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Problem instance
 ******************************************************************************/

/**
 * Problem instance
 */
template <
	typename DoubleBuffer,
	typename SizeT>
struct ProblemInstance
{
	DoubleBuffer		&storage;
	SizeT				num_elements;

	util::Spine			&spine;
	int			 		max_grid_size;
	int 				ptx_arch;
	int 				sm_arch;
	int					sm_count;
	bool				debug;

	/**
	 * Constructor
	 */
	ProblemInstance(
		DoubleBuffer	&storage,
		SizeT			num_elements,
		util::Spine		&spine,
		int			 	max_grid_size,
		int 			ptx_arch,
		int 			sm_arch,
		int				sm_count,
		bool			debug) :
			spine(spine),
			storage(storage),
			num_elements(num_elements),
			max_grid_size(max_grid_size),
			ptx_arch(ptx_arch),
			sm_arch(sm_arch),
			sm_count(sm_count),
			debug(debug)
	{}
};


/******************************************************************************
 * Sorting pass
 ******************************************************************************/

/**
 * Sorting pass
 */
template <
	typename KeyType,
	typename ValueType,
	typename SizeT>
struct SortingPass
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	// Converted key type
	typedef typename KeyTraits<KeyType>::ConvertedKeyType ConvertedKeyType;

	// Kernel function types
	typedef void (*UpsweepKernelFunc)(SizeT*, ConvertedKeyType*, ConvertedKeyType*, util::CtaWorkDistribution<SizeT>);
	typedef void (*SpineKernelFunc)(SizeT*, SizeT*, int);
	typedef void (*DownsweepKernelFunc)(SizeT*, ConvertedKeyType*, ConvertedKeyType*, ValueType*, ValueType*, util::CtaWorkDistribution<SizeT>);

	// Texture binding function types
	typedef cudaError_t (*BindKeyTexFunc)(void *, void *, size_t);
	typedef cudaError_t (*BindValueTexFunc)(void *, void *, size_t);
	typedef cudaError_t (*BindTexSpineFunc)(void *, size_t);


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Dispatch
	 */
	template <typename ProblemInstance>
	static cudaError_t Dispatch(
		ProblemInstance							problem_instance,
		int 									radix_bits,
		util::KernelProps<UpsweepKernelFunc> 	&upsweep_props,
		util::KernelProps<SpineKernelFunc> 		&spine_props,
		util::KernelProps<DownsweepKernelFunc> 	&downsweep_props,
		BindKeyTexFunc 							bind_key_texture_func,
		BindValueTexFunc 						bind_value_texture_func,
		BindTexSpineFunc 						bind_spine_texture_func,
		int 									log_schedule_granularity,
		int										upsweep_tile_elements,
		int										spine_tile_elements,
		int										downsweep_tile_elements,
		bool									smem_8byte_banks,
		bool									unform_grid_size,
		bool									uniform_smem_allocation)
	{
		cudaError_t error = cudaSuccess;

		do {
			// Compute sweep grid size
			int schedule_granularity = 1 << log_schedule_granularity;
			int sweep_grid_size = downsweep_props.OversubscribedGridSize(
				schedule_granularity,
				problem_instance.num_elements,
				problem_instance.max_grid_size);

			// Compute spine elements (rounded up to nearest tile size)
			SizeT spine_elements = CUB_ROUND_UP_NEAREST(
				sweep_grid_size << radix_bits,
				spine_tile_elements);

			// Make sure our spine is big enough
			error = problem_instance.spine.Setup(sizeof(SizeT) * spine_elements);
			if (error) break;

			// Obtain a CTA work distribution
			util::CtaWorkDistribution<SizeT> work(
				problem_instance.num_elements,
				sweep_grid_size,
				log_schedule_granularity);

			if (problem_instance.debug) {
				work.Print();
			}

			// Bind key textures
			if (bind_key_texture_func != NULL) {
				error = bind_key_texture_func(
					problem_instance.storage.d_keys[problem_instance.storage.selector],
					problem_instance.storage.d_keys[problem_instance.storage.selector ^ 1],
					sizeof(ConvertedKeyType) * problem_instance.num_elements);
				if (error) break;
			}

			// Bind value textures
			if (bind_value_texture_func != NULL) {
				error = bind_value_texture_func(
					problem_instance.storage.d_values[problem_instance.storage.selector],
					problem_instance.storage.d_values[problem_instance.storage.selector ^ 1],
					sizeof(ValueType) * problem_instance.num_elements);
				if (error) break;
			}

			// Bind spine textures
			if (bind_spine_texture_func != NULL) {
				error = bind_spine_texture_func(
					problem_instance.spine(),
					sizeof(SizeT) * spine_elements);
				if (error) break;
			}

			// Operational details
			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{sweep_grid_size, 1, sweep_grid_size};

			// Grid size tuning
			if (unform_grid_size) {
				// Make sure that all kernels launch the same number of CTAs
				grid_size[1] = grid_size[0];
			}

			// Smem allocation tuning
			if (uniform_smem_allocation) {

				// Make sure all kernels have the same overall smem allocation
				int max_static_smem = CUB_MAX(
					upsweep_props.kernel_attrs.sharedSizeBytes,
					CUB_MAX(
						spine_props.kernel_attrs.sharedSizeBytes,
						downsweep_props.kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_props.kernel_attrs.sharedSizeBytes;

			} else {

				// Compute smem padding for upsweep to make upsweep occupancy a multiple of downsweep occupancy
				dynamic_smem[0] = upsweep_props.SmemPadding(downsweep_props.max_cta_occupancy);
			}

			if (problem_instance.debug) {
				printf(
					"Upsweep:   tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Spine:     tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Downsweep: tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n",
					upsweep_tile_elements, upsweep_props.max_cta_occupancy, grid_size[0], upsweep_props.threads, dynamic_smem[0],
					spine_tile_elements, spine_props.max_cta_occupancy, grid_size[1], spine_props.threads, dynamic_smem[1],
					downsweep_tile_elements, downsweep_props.max_cta_occupancy, grid_size[2], downsweep_props.threads, dynamic_smem[2]);
				fflush(stdout);
			}

			// Upsweep reduction into spine
			upsweep_props.kernel_func<<<grid_size[0], upsweep_props.threads, dynamic_smem[0]>>>(
				(SizeT*) problem_instance.spine(),
				(ConvertedKeyType *) problem_instance.storage.d_keys[problem_instance.storage.selector],
				(ConvertedKeyType *) problem_instance.storage.d_keys[problem_instance.storage.selector ^ 1],
				work);

			if (problem_instance.debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Upsweep kernel failed ", __FILE__, __LINE__)) break;
			}

			// Spine scan
			spine_props.kernel_func<<<grid_size[1], spine_props.threads, dynamic_smem[1]>>>(
				(SizeT*) problem_instance.spine(),
				(SizeT*) problem_instance.spine(),
				spine_elements);

			if (problem_instance.debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Spine kernel failed ", __FILE__, __LINE__)) break;
			}

			// Set shared mem bank mode
			enum cudaSharedMemConfig old_config;
			cudaDeviceGetSharedMemConfig(&old_config);
			cudaDeviceSetSharedMemConfig(smem_8byte_banks ?
				cudaSharedMemBankSizeEightByte :		// 64-bit bank mode
				cudaSharedMemBankSizeFourByte);			// 32-bit bank mode

			// Downsweep scan from spine
			downsweep_props.kernel_func<<<grid_size[2], downsweep_props.threads, dynamic_smem[2]>>>(
				(SizeT *) problem_instance.spine(),
				(ConvertedKeyType *) problem_instance.storage.d_keys[problem_instance.storage.selector],
				(ConvertedKeyType *) problem_instance.storage.d_keys[problem_instance.storage.selector ^ 1],
				problem_instance.storage.d_values[problem_instance.storage.selector],
				problem_instance.storage.d_values[problem_instance.storage.selector ^ 1],
				work);

			if (problem_instance.debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Downsweep kernel failed ", __FILE__, __LINE__)) break;
			}

			// Restore smem bank mode
			cudaDeviceSetSharedMemConfig(old_config);

		} while(0);

		return error;
	}


	/**
	 * Dispatch
	 */
	template <
		typename HostPassPolicy,
		typename DevicePassPolicy,
		typename ProblemInstance>
	static cudaError_t Dispatch(ProblemInstance &problem_instance)
	{
		typedef typename HostPassPolicy::UpsweepPolicy 		UpsweepPolicy;
		typedef typename HostPassPolicy::SpinePolicy 		SpinePolicy;
		typedef typename HostPassPolicy::DownsweepPolicy 	DownsweepPolicy;
		typedef typename HostPassPolicy::DispatchPolicy	 	DispatchPolicy;

		// Wrapper of downsweep texture types
		typedef downsweep::Textures<
			KeyType,
			ValueType,
			(1 << DownsweepPolicy::LOG_THREAD_ELEMENTS)> DownsweepTextures;

		// Downsweep key texture type
		typedef typename DownsweepTextures::KeyTexType KeyTexType;

		// Downsweep value texture type
		typedef typename DownsweepTextures::ValueTexType ValueTexType;

		// Upsweep kernel properties
		util::KernelProps<UpsweepKernelFunc> upsweep_props(
			upsweep::Kernel<typename DevicePassPolicy::UpsweepPolicy>,
			UpsweepPolicy::THREADS,
			problem_instance.sm_arch,
			problem_instance.sm_count);

		// Spine kernel properties
		util::KernelProps<SpineKernelFunc> spine_props(
			spine::Kernel<typename DevicePassPolicy::SpinePolicy>,
			SpinePolicy::THREADS,
			problem_instance.sm_arch,
			problem_instance.sm_count);

		// Downsweep kernel properties
		util::KernelProps<DownsweepKernelFunc> downsweep_props(
			downsweep::Kernel<typename DevicePassPolicy::DownsweepPolicy>,
			DownsweepPolicy::THREADS,
			problem_instance.sm_arch,
			problem_instance.sm_count);

		// Schedule granularity
		int log_schedule_granularity = CUB_MAX(
			int(UpsweepPolicy::LOG_TILE_ELEMENTS),
			int(DownsweepPolicy::LOG_TILE_ELEMENTS));

		// Texture binding for downsweep keys
		BindKeyTexFunc bind_key_texture_func =
			downsweep::TexKeys<KeyTexType>::BindTexture;

		// Texture binding for downsweep values
		BindValueTexFunc bind_value_texture_func =
			downsweep::TexValues<ValueTexType>::BindTexture;

		// Texture binding for spine
		BindTexSpineFunc bind_spine_texture_func =
			spine::TexSpine<SizeT>::BindTexture;

		return Dispatch(
			problem_instance,
			DispatchPolicy::RADIX_BITS,
			upsweep_props,
			spine_props,
			downsweep_props,
			bind_key_texture_func,
			bind_value_texture_func,
			bind_spine_texture_func,
			log_schedule_granularity,
			(1 << UpsweepPolicy::LOG_TILE_ELEMENTS),
			(1 << SpinePolicy::LOG_TILE_ELEMENTS),
			(1 << DownsweepPolicy::LOG_TILE_ELEMENTS),
			DownsweepPolicy::SMEM_8BYTE_BANKS,
			DispatchPolicy::UNIFORM_GRID_SIZE,
			DispatchPolicy::UNIFORM_SMEM_ALLOCATION);
	}


	/**
	 * Dispatch.  Custom tuning interface.
	 */
	template <
		typename PassPolicy,
		typename ProblemInstance>
	static cudaError_t Dispatch(ProblemInstance &problem_instance)
	{
		return Dispatch<PassPolicy, PassPolicy>(problem_instance);
	}


	//---------------------------------------------------------------------
	// Preconfigured pass dispatch
	//---------------------------------------------------------------------

	/**
	 * Specialized pass policies
	 */
	template <
		int TUNE_ARCH,
		int BITS_REMAINING,
		int CURRENT_BIT,
		int CURRENT_PASS>
	struct TunedPassPolicy;


	/**
	 * SM20
	 */
	template <int BITS_REMAINING, int CURRENT_BIT, int CURRENT_PASS>
	struct TunedPassPolicy<200, BITS_REMAINING, CURRENT_BIT, CURRENT_PASS>
	{
		enum {
			RADIX_BITS 		= CUB_MIN(BITS_REMAINING, ((BITS_REMAINING + 4) % 5 > 3) ? 5 : 4),
			KEYS_ONLY 		= util::Equals<ValueType, util::NullType>::VALUE,
			EARLY_EXIT 		= false,
			LARGE_DATA		= (sizeof(KeyType) > 4) || (sizeof(ValueType) > 4),
		};

		// Dispatch policy
		typedef radix_sort::DispatchPolicy <
			200,								// TUNE_ARCH
			RADIX_BITS,							// RADIX_BITS
			false, 								// UNIFORM_SMEM_ALLOCATION
			true> 								// UNIFORM_GRID_SIZE
				DispatchPolicy;

		// Upsweep kernel policy
		typedef upsweep::KernelPolicy<
			RADIX_BITS,						// RADIX_BITS
			CURRENT_BIT,					// CURRENT_BIT
			CURRENT_PASS,					// CURRENT_PASS
			8,								// MIN_CTA_OCCUPANCY	The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
			7,								// LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10
			LARGE_DATA ? 1 : 2,				// LOG_LOAD_VEC_SIZE	The vector-load size (log) for each load (log).  Valid range: 0-2
			1,								// LOG_LOADS_PER_TILE	The number of loads (log) per tile.  Valid range: 0-2
			b40c::util::io::ld::NONE,		// READ_MODIFIER		Load cache-modifier.  Valid values: NONE, ca, cg, cs
			b40c::util::io::st::NONE,		// WRITE_MODIFIER		Store cache-modifier.  Valid values: NONE, wb, cg, cs
			EARLY_EXIT>						// EARLY_EXIT			Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
				UpsweepPolicy;

		// Spine-scan kernel policy
		typedef spine::KernelPolicy<
			8,								// LOG_THREADS			The number of threads (log) to launch per CTA.  Valid range: 5-10
			2,								// LOG_LOAD_VEC_SIZE	The vector-load size (log) for each load (log).  Valid range: 0-2
			2,								// LOG_LOADS_PER_TILE	The number of loads (log) per tile.  Valid range: 0-2
			b40c::util::io::ld::NONE,		// READ_MODIFIER		Load cache-modifier.  Valid values: NONE, ca, cg, cs
			b40c::util::io::st::NONE>		// WRITE_MODIFIER		Store cache-modifier.  Valid values: NONE, wb, cg, cs
				SpinePolicy;

		// Downsweep kernel policy
		typedef typename util::If<
			(!LARGE_DATA),
			downsweep::KernelPolicy<
				RADIX_BITS,						// RADIX_BITS
				CURRENT_BIT,					// CURRENT_BIT
				CURRENT_PASS,					// CURRENT_PASS
				KEYS_ONLY ? 4 : 2,				// MIN_CTA_OCCUPANCY		The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
				KEYS_ONLY ? 7 : 8,				// LOG_THREADS				The number of threads (log) to launch per CTA.
				KEYS_ONLY ? 4 : 4,				// LOG_ELEMENTS_PER_TILE	The number of keys (log) per thread
				b40c::util::io::ld::NONE,		// READ_MODIFIER			Load cache-modifier.  Valid values: NONE, ca, cg, cs
				b40c::util::io::st::NONE,		// WRITE_MODIFIER			Store cache-modifier.  Valid values: NONE, wb, cg, cs
				downsweep::SCATTER_TWO_PHASE,	// SCATTER_STRATEGY			Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
				false,							// SMEM_8BYTE_BANKS
				EARLY_EXIT>,					// EARLY_EXIT				Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
			downsweep::KernelPolicy<
				RADIX_BITS,						// RADIX_BITS
				CURRENT_BIT,					// CURRENT_BIT
				CURRENT_PASS,					// CURRENT_PASS
				2,								// MIN_CTA_OCCUPANCY		The targeted SM occupancy to feed PTXAS in order to influence how it does register allocation
				8,								// LOG_THREADS				The number of threads (log) to launch per CTA.
				3,								// LOG_ELEMENTS_PER_TILE	The number of keys (log) per thread
				b40c::util::io::ld::NONE,		// READ_MODIFIER			Load cache-modifier.  Valid values: NONE, ca, cg, cs
				b40c::util::io::st::NONE,		// WRITE_MODIFIER			Store cache-modifier.  Valid values: NONE, wb, cg, cs
				downsweep::SCATTER_TWO_PHASE,	// SCATTER_STRATEGY			Whether or not to perform a two-phase scatter (scatter to smem first to recover some locality before scattering to global bins)
				false,							// SMEM_8BYTE_BANKS
				EARLY_EXIT> >::Type 			// EARLY_EXIT				Whether or not to early-terminate a sorting pass if we detect all keys have the same digit in that pass's digit place
					DownsweepPolicy;
	};





	/**
	 * Opaque pass policy
	 */
	template <int BITS_REMAINING, int CURRENT_BIT, int CURRENT_PASS>
	struct OpaquePassPolicy
	{
		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		static const int OPAQUE_ARCH = 200;
/*
			(__CUB_CUDA_ARCH__ >= 200) ?
				200 :
				(__CUB_CUDA_ARCH__ >= 130) ?
					130 :
					100;
*/
		typedef TunedPassPolicy<OPAQUE_ARCH, BITS_REMAINING, CURRENT_BIT, CURRENT_PASS> TunedPolicy;

		struct UpsweepPolicy : 		TunedPolicy::UpsweepPolicy {};
		struct SpinePolicy : 		TunedPolicy::SpinePolicy {};
		struct DownsweepPolicy : 	TunedPolicy::DownsweepPolicy {};
		struct DispatchPolicy : 	TunedPolicy::DispatchPolicy {};
	};


	/**
	 * Helper structure for iterating passes.
	 */
	template <int PTX_ARCH, int BITS_REMAINING, int CURRENT_BIT, int CURRENT_PASS>
	struct IteratePasses
	{
		// Dispatch pass
		template <typename ProblemInstance>
		static cudaError_t Dispatch(ProblemInstance &problem_instance)
		{
			typedef TunedPassPolicy<PTX_ARCH, BITS_REMAINING, CURRENT_BIT, CURRENT_PASS> TunedPolicy;

			typedef OpaquePassPolicy<BITS_REMAINING, CURRENT_BIT, CURRENT_PASS> OpaquePolicy;

			const int RADIX_BITS = TunedPolicy::DispatchPolicy::RADIX_BITS;

			cudaError_t error = cudaSuccess;
			do {
				if (problem_instance.debug) {
					printf("\nCurrent bit(%d), Pass(%d), Radix bits(%d), PTX arch(%d), SM arch(%d)\n",
						CURRENT_BIT, CURRENT_PASS, RADIX_BITS, PTX_ARCH, problem_instance.sm_arch);
					fflush(stdout);
				}

				// Dispatch current pass
				error = SortingPass::Dispatch<TunedPolicy, OpaquePolicy>(problem_instance);
				if (error) break;

				// Dispatch next pass
				error = IteratePasses<
					PTX_ARCH,
					BITS_REMAINING - RADIX_BITS,
					CURRENT_BIT + RADIX_BITS,
					CURRENT_PASS + 1>::Dispatch(problem_instance);
				if (error) break;

			} while (0);

			return error;
		}
	};


	/**
	 * Helper structure for iterating passes. (Termination)
	 */
	template <int PTX_ARCH, int CURRENT_BIT, int NUM_PASSES>
	struct IteratePasses<PTX_ARCH, 0, CURRENT_BIT, NUM_PASSES>
	{
		// Dispatch pass
		template <typename ProblemInstance>
		static cudaError_t Dispatch(ProblemInstance &problem_instance)
		{
			// We moved data between storage buffers at every pass
			problem_instance.storage.selector =
				(problem_instance.storage.selector + NUM_PASSES) & 0x1;

			return cudaSuccess;
		}
	};


	/**
	 * Dispatch
	 */
	template <
		int BITS_REMAINING,
		int CURRENT_BIT,
		typename ProblemInstance>
	static cudaError_t DispatchPasses(ProblemInstance &problem_instance)
	{
		if (problem_instance.ptx_arch >= 200) {

			return IteratePasses<200, BITS_REMAINING, CURRENT_BIT, 0>::Dispatch(problem_instance);
/*
		} else if (problem_instance.ptx_arch >= 130) {

			return IteratePasses<130, BITS_REMAINING, CURRENT_BIT, 0>::Dispatch(problem_instance);

		} else {

			return IteratePasses<100, BITS_REMAINING, CURRENT_BIT, 0>::Dispatch(problem_instance);
		}
*/
		} else {
			return cudaErrorNotYetImplemented;
		}
	}
};


/******************************************************************************
 * Radix sorting enactor class
 ******************************************************************************/

/**
 * Radix sorting enactor class
 */
class Enactor
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;

	// Device properties
	const util::CudaProperties cuda_props;

public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enact a sort.
	 *
	 * @param problem_storage
	 * 		Instance of b40c::util::PingPongStorage type describing the details of the
	 * 		problem to sort.
	 * @param num_elements
	 * 		The number of elements in problem_storage to sort (starting at offset 0)
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		int BITS_REMAINING,
		int CURRENT_BIT,
		typename DoubleBuffer>
	cudaError_t Sort(
		DoubleBuffer& 	problem_storage,
		int 			num_elements,
		int 			max_grid_size = 0,
		bool 			debug = false)
	{
		typedef typename DoubleBuffer::KeyType KeyType;
		typedef typename DoubleBuffer::ValueType ValueType;
		typedef SortingPass<KeyType, ValueType, int> SortingPass;

		// Create problem instance
		ProblemInstance<DoubleBuffer, int> problem_instance(
			problem_storage,
			num_elements,
			spine,
			max_grid_size,
			cuda_props.kernel_ptx_version,
			cuda_props.device_sm_version,
			cuda_props.device_props.multiProcessorCount,
			debug);

		// Dispatch sorting passes
		return SortingPass::template DispatchPasses<
			BITS_REMAINING,
			CURRENT_BIT>(problem_instance);
	}

};





} // namespace radix_sort
} // namespace b40c


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
 * Radix sort upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/radix_sort/upsweep/cta.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Radix sort upsweep reduction kernel entry point
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	SizeT 		*d_spine,
	KeyType 	*d_in_keys,
	KeyType 	*d_out_keys,
	util::CtaWorkDistribution<SizeT> work_decomposition)
{

	// CTA abstraction type
	typedef Cta<KernelPolicy, SizeT, KeyType> Cta;

	// Shared memory pool
	__shared__ typename Cta::SmemStorage smem_storage;
	
	// Determine where to read our input
	KeyType *d_keys = (KernelPolicy::CURRENT_PASS & 0x1) ?
		d_out_keys :
		d_in_keys;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.GetCtaWorkLimits(
		work_limits,
		KernelPolicy::LOG_TILE_ELEMENTS);

	Cta cta(smem_storage, d_keys, d_spine);
	cta.ProcessWorkRange(work_limits);
}


} // namespace upsweep
} // namespace radix_sort
} // namespace b40c


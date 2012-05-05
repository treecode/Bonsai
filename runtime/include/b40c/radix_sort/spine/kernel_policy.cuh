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
 * Configuration policy for radix sort spine scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace radix_sort {
namespace spine {


/**
 * Spine tuning policy.
 */
template <
	int 							_LOG_THREADS,
	int 							_LOG_LOAD_VEC_SIZE,
	int 							_LOG_LOADS_PER_TILE,
	util::io::ld::CacheModifier 	_READ_MODIFIER,
	util::io::st::CacheModifier 	_WRITE_MODIFIER>
struct KernelPolicy
{
	enum {
		LOG_THREADS 				= _LOG_THREADS,
		LOG_LOAD_VEC_SIZE  			= _LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE 			= _LOG_LOADS_PER_TILE,

		THREADS						= 1 << LOG_THREADS,
		LOG_TILE_ELEMENTS			= LOG_THREADS + LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
	};

	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;
};


} // namespace spine
} // namespace radix_sort
} // namespace b40c


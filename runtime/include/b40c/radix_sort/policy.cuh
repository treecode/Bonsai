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
 * Radix sort policy
 ******************************************************************************/

#pragma once

namespace b40c {
namespace radix_sort {


/******************************************************************************
 * Dispatch policy
 ******************************************************************************/

/**
 * Dispatch policy
 */
template <
	int			_TUNE_ARCH,
	int 		_RADIX_BITS,
	bool 		_UNIFORM_SMEM_ALLOCATION,
	bool 		_UNIFORM_GRID_SIZE>
struct DispatchPolicy
{
	enum {
		TUNE_ARCH					= _TUNE_ARCH,
		RADIX_BITS					= _RADIX_BITS,
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
	};
};


/******************************************************************************
 * Pass policy
 ******************************************************************************/

/**
 * Pass policy
 */
template <
	typename 	_UpsweepPolicy,
	typename 	_SpinePolicy,
	typename 	_DownsweepPolicy,
	typename 	_DispatchPolicy>
struct PassPolicy
{
	typedef _UpsweepPolicy			UpsweepPolicy;
	typedef _SpinePolicy 			SpinePolicy;
	typedef _DownsweepPolicy 		DownsweepPolicy;
	typedef _DispatchPolicy 		DispatchPolicy;
};


}// namespace radix_sort
}// namespace b40c


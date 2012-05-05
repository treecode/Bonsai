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
 * Texture references for spine kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace spine {

/**
 * Templated texture reference for spine
 */
template <typename SizeT>
struct TexSpine
{
	typedef texture<SizeT, cudaTextureType1D, cudaReadModeElementType> TexRef;

	static TexRef ref;

	/**
	 * Bind textures
	 */
	static cudaError_t BindTexture(void *d_spine, size_t bytes)
	{
		cudaError_t retval = cudaSuccess;
		do {
			cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<SizeT>();

			// Bind key texture ref0
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					ref,
					d_spine,
					tex_desc,
					bytes),
				"cudaBindTexture TexSpine failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}

};

// Reference definition
template <typename SizeT>
typename TexSpine<SizeT>::TexRef TexSpine<SizeT>::ref;






} // namespace spine
} // namespace radix_sort
} // namespace b40c


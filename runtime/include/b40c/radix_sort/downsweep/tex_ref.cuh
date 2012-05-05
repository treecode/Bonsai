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
 * Texture references for downsweep kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/tex_vector.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {

/******************************************************************************
 * Key textures
 ******************************************************************************/

/**
 * Templated texture reference for downsweep keys
 */
template <typename KeyVectorType>
struct TexKeys
{
	// Texture reference type
	typedef texture<KeyVectorType, cudaTextureType1D, cudaReadModeElementType> TexRef;

	static TexRef ref0;
	static TexRef ref1;

	/**
	 * Bind textures
	 */
	static cudaError_t BindTexture(
		void *d0,
		void *d1,
		size_t bytes)
	{
		cudaError_t retval = cudaSuccess;
		do {
			cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<KeyVectorType>();

			if (d0) {
				// Bind key texture ref0
				retval = cudaBindTexture(0, ref0, d0, tex_desc, bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
			if (d1) {
				// Bind key texture ref1
				retval = cudaBindTexture(0, ref1, d1, tex_desc, bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
		} while (0);

		return retval;
	}

};

// Texture reference definitions
template <typename KeyVectorType>
typename TexKeys<KeyVectorType>::TexRef TexKeys<KeyVectorType>::ref0;

template <typename KeyVectorType>
typename TexKeys<KeyVectorType>::TexRef TexKeys<KeyVectorType>::ref1;



/******************************************************************************
 * Value textures
 ******************************************************************************/

/**
 * Templated texture reference for downsweep values
 */
template <typename ValueVectorType>
struct TexValues
{
	// Texture reference type
	typedef texture<ValueVectorType, cudaTextureType1D, cudaReadModeElementType> TexRef;

	static TexRef ref0;
	static TexRef ref1;

	/**
	 * Bind textures
	 */
	static cudaError_t BindTexture(
		void *d0,
		void *d1,
		size_t bytes)
	{
		cudaError_t retval = cudaSuccess;
		do {
			cudaChannelFormatDesc tex_desc = cudaCreateChannelDesc<ValueVectorType>();

			if (d0) {
				// Bind key texture ref0
				retval = cudaBindTexture(0, ref0, d0, tex_desc, bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
			if (d1) {
				// Bind key texture ref1
				retval = cudaBindTexture(0, ref1, d1, tex_desc, bytes);
				if (retval = util::B40CPerror(retval, "cudaBindTexture failed", __FILE__, __LINE__)) break;
			}
		} while (0);

		return retval;
	}

};

// Texture reference definitions
template <typename ValueVectorType>
typename TexValues<ValueVectorType>::TexRef TexValues<ValueVectorType>::ref0;

template <typename ValueVectorType>
typename TexValues<ValueVectorType>::TexRef TexValues<ValueVectorType>::ref1;



/******************************************************************************
 * Texture types for downsweep kernel
 ******************************************************************************/

template <typename KeyType, typename ValueType, int THREAD_ELEMENTS>
struct Textures
{
	// Elements per texture load
	enum {
		KEY_ELEMENTS_PER_TEX		= util::TexVector<KeyType, THREAD_ELEMENTS>::ELEMENTS_PER_TEX,
		VALUE_ELEMENTS_PER_TEX		= util::TexVector<ValueType, THREAD_ELEMENTS>::ELEMENTS_PER_TEX,
		ELEMENTS_PER_TEX			= CUB_MIN(int(KEY_ELEMENTS_PER_TEX), int(VALUE_ELEMENTS_PER_TEX))
	};

	typedef typename util::TexVector<
		KeyType,
		ELEMENTS_PER_TEX>::VecType KeyTexType;

	// Texture binding for downsweep values
	typedef typename util::TexVector<
		ValueType,
		ELEMENTS_PER_TEX>::VecType ValueTexType;
};



} // namespace downsweep
} // namespace radix_sort
} // namespace b40c


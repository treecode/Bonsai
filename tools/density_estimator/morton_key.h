#ifndef __MORTON_KEY_H__
#define __MORTON_KEY_H__

#include <assert.h>

template <typename VEC, typename REAL>
struct morton_key{
	typedef unsigned long long key_t;
	key_t val;

	morton_key() : val(0) {}
	morton_key(const VEC &vec, const REAL &size){
		static key_t table[128] = {
			#include "key_table"
		};
		const REAL scale = (1<<20) / size;
		int xi = int(vec[0] * scale);
		int yi = int(vec[1] * scale);
		int zi = int(vec[2] * scale);
		assert((xi >> 21) == 0);
		assert((yi >> 21) == 0);
		assert((zi >> 21) == 0);
		key_t xkey = (table[xi&127]) | (table[(xi>>7)&127] << 21) | (table[(xi>>14)&127] << 42);
		key_t ykey = (table[yi&127]) | (table[(yi>>7)&127] << 21) | (table[(yi>>14)&127] << 42);
		key_t zkey = (table[zi&127]) | (table[(zi>>7)&127] << 21) | (table[(zi>>14)&127] << 42);
		val = (xkey<<2) | (ykey<<1) | zkey;
	}
};

#endif /* __MORTON_KEY_H__ */

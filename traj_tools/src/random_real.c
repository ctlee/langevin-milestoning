/*-
 * Copyright (c) 2014 Taylor R. Campbell
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Uniform random floats: How to generate a double-precision
 * floating-point numbers in [0, 1] uniformly at random given a uniform
 * random source of bits.
 *
 * See <http://mumble.net/~campbell/2014/04/28/uniform-random-float>
 * for explanation.
 *
 * Updated 2015-02-22 to replace ldexp(x, <constant>) by x * ldexp(1,
 * <constant>), since glibc and NetBSD libm both have slow software
 * bit-twiddling implementations of ldexp, but GCC can constant-fold
 * the latter.
 */

#include <math.h>
#include <stdint.h>
#include <pcg_variants.h>
#include "random_real.h"


/*
 * random_real_64: Pick an integer in {0, 1, ..., 2^64 - 1} uniformly
 * at random, convert it to double, and divide it by 2^64.  Values in
 * [2^-11, 1] are overrepresented, small exponents have low precision,
 * and exponents below -64 are not possible.
 */
double random_real_64(pcg64_random_t* rng){
    return pcg64_random_r(rng) * ldexp(1, -64);
}

/*
 * random_real_53: Pick an integer in {0, 1, ..., 2^53 - 1} uniformly
 * at random, convert it to double, and divide it by 2^53.  Many
 * possible outputs are not represented: 2^-54, 1, &c.  There are a
 * little under 2^62 floating-point values in [0, 1], but only 2^53
 * possible outputs here.
 */
double random_real_53(pcg64_random_t* rng){
    return (pcg64_random_r(rng) & ((1ull << 53) - 1)) * ldexp(1, -53);
}

#define	clz64	__builtin_clzll		/* XXX GCCism */

/*
 * random_real: Generate a stream of bits uniformly at random and
 * interpret it as the fractional part of the binary expansion of a
 * number in [0, 1], 0.00001010011111010100...; then round it.
 */
double random_real(pcg64_random_t* rng){
	int exponent = -64;
	uint64_t significand;
	unsigned shift;

	/*
	 * Read zeros into the exponent until we hit a one; the rest
	 * will go into the significand.
	 */
	while (__predict_false((significand = pcg64_random_r(rng)) == 0)) {
		exponent -= 64;
		/*
		 * If the exponent falls below -1074 = emin + 1 - p,
		 * the exponent of the smallest subnormal, we are
		 * guaranteed the result will be rounded to zero.  This
		 * case is so unlikely it will happen in realistic
		 * terms only if random64 is broken.
		 */
		if (__predict_false(exponent < -1074))
			return 0;
	}

	/*
	 * There is a 1 somewhere in significand, not necessarily in
	 * the most significant position.  If there are leading zeros,
	 * shift them into the exponent and refill the less-significant
	 * bits of the significand.  Can't predict one way or another
	 * whether there are leading zeros: there's a fifty-fifty
	 * chance, if random64 is uniformly distributed.
	 */
	shift = clz64(significand);
	if (shift != 0) {
		exponent -= shift;
		significand <<= shift;
		significand |= (pcg64_random_r(rng) >> (64 - shift));
	}

	/*
	 * Set the sticky bit, since there is almost surely another 1
	 * in the bit stream.  Otherwise, we might round what looks
	 * like a tie to even when, almost surely, were we to look
	 * further in the bit stream, there would be a 1 breaking the
	 * tie.
	 */
	significand |= 1;

	/*
	 * Finally, convert to double (rounding) and scale by
	 * 2^exponent.
	 */
	return ldexp((double)significand, exponent);
}

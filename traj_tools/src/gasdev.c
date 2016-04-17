/**
 * Generate gaussian random numbers using Box-Muller transform
 *    
 * Copyright 2016 Christopher T. Lee <ctlee@ucsd.edu>
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
**/

#include <math.h>
#include <pcg_variants.h>
#include "gasdev.h"
#include "random_real.h"

/**
 * Generate a set of gaussian random numbers with zero mean and 
 * unit variance. Implementation of the polar form of Box-Muller
 * transformation.
 * @param rng: a seeded pcg random number generator
 * @return: double
 */

double gasdev(pcg64_random_t* rng){
	static int iset = 0;
	static double gset;
	double fac, r, v1, v2;

	if  (iset == 0) {
		do {
		    // pick two numbers in a square extending from +/-1
			v1 = 2.0*random_real(rng)-1.0;
			v2 = 2.0*random_real(rng)-1.0;
			r = v1*v1 + v2*v2; // go to polar coords
		} while (r >= 1.0 || r == 0.0); 
        // Go until we're in the unit circle

		fac = sqrt(-2.0*log(r)/r); // Box-Muller transform
		gset = v1*fac;
		iset = 1;
		return v2*fac;
	} else {
		iset = 0;
		return gset;
	}
}

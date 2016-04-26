/* 
 * Langevin dynamics in a 2D PMF with viscosity in C.
 *    
 * Copyright 2016 Christopher T. lee <ctlee@ucsd.edu>
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
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pcg_variants.h>
#include "entropy.h"
#include "random_real.h"
#include "gasdev.h"


#pragma GCC push_options
#pragma GCC optimize ("O0")
int main(void){
  
    int rounds = 8192;

    pcg64_random_t rng;
    pcg128_t  seeds[2];
    entropy_getbytes((void *)seeds, sizeof(seeds));
    pcg64_srandom_r(&rng, seeds[0], seeds[1]);

    for (int i = 0; i < rounds; i++){
        printf("%d %e\n", i, gasdev(&rng));
        //printf("%d: 0x%016llx\n", i, (long long int) pcg64_random_r(&rng));
    }
    return 0;
}
#pragma GCC pop_options

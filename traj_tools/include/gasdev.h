/**
 * Langevin dynamics in 1D with PMF and viscosity.
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

#ifndef GASDEV_H
#define GASDEV_H

#include <pcg_variants.h>

#ifdef __cplusplus
extern "C" {
#endif

double gasdev(pcg64_random_t* rng);

#ifdef __cplusplus
}
#endif
#endif

# 
# Langevin dynamics in 1D with PMF and viscosity.
# 
# Copyright 2016 Christopher T. Lee <ctlee@ucsd.edu>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

TARGETS = randreal-demo pcg64-demo
LDLIBS += -lpcg_random -lm -L../../pcg-c/src
CFLAGS += -std=c99 -O3 -W -Wall -Werror -I../../pcg-c/include -I../include
# careful with optimization flags

all: $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

pcg64-demo: pcg64-demo.o entropy.o
randreal-demo: randreal-demo.o entropy.o random_real.o gasdev.o

randreal-demo.o: randreal-demo.c ../include/entropy.h
pcg64-demo.o: pcg64-demo.c ../include/entropy.h

entropy.o: entropy.c ../include/entropy.h ../include/pcg_spinlock.h
random_real.o: random_real.c ../include/random_real.h ../include/gasdev.h
gasdev.o: gasdev.c ../include/gasdev.h ../include/random_real.h

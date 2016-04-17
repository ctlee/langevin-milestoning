# 
# Langevin dynamics in a 2D PMF with viscosity in C.
# 
# Copyright 2016 Christopher T. lee <ctlee@ucsd.edu>
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

.PHONY: all
all: traj_tools.so
	cd src; $(MAKE)

traj_tools.so: src/traj_tools.c
	python setup.py build_ext --inplace

install: traj_tools.so all
	mkdir -p bin
	cp src/pcg64-demo bin
	cp src/randreal-demo bin
	cp traj_tools.so ..	

.PHONY: clean
clean:
	cd src; $(MAKE) clean
	rm -f *.o traj_tools.so
	rm -rf build

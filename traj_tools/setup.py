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

from distutils.core import setup, Extension
import numpy

# define the extension module
traj_tools = Extension('traj_tools', 
        define_macros = [('MAJOR_VERSION', '1'),
                        ('MINOR_VERSION', '0')],
        sources=['src/traj_tools.c', 'src/gasdev.c', 
                'src/random_real.c', 'src/entropy.c'], 
        include_dirs=[numpy.get_include(), 'include'],
        depends=['include/gasdev.h', 'include/random_real.h', 
                'include/entropy.h','include/pcg_spinlock.h'],
        extra_compile_args=['-std=c99'])

# run the setup
setup(name = "TrajTools",
        version = '1.0',
        description = 'Langevin dynamics in 1D with PMF and viscosity',
        author = 'Christopher T. Lee',
        author_email = 'ctlee@ucsd.edu',
        ext_modules=[traj_tools])

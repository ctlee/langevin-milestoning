# 1-D Langevin Dynamics 

This code implements Langevin dynamics in 1-D across user specified potential
of mean force and diffusivity profiles. 

We use this code for validating the new permeability equation for milestoning 
applications.  See our [Docs](http://langevin-milestoning.readthedocs.org/en/latest/).

## Building

The code is written in Python and C99-style C with no significant platform 
dependencies. On a Unix-style system (e.g., Linux, Mac OS X) you should be able
to just type

    make

The easiest way to get Python is through Conda.

## Dependencies

[PCG-Random website]: http://www.pcg-random.org
*  Included PCG-Random library ([PCG-Random website])
*  Python libraries:
  * numpy
  * scipy

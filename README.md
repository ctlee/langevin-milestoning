
# Simulating Bilayer Permeability

Permeability of drugs and other molecules across bilayers is important to many processes in biology.
Due to the rareness of passive permeation events, robust statistics of permeability are difficult to obtain using traditional Boltzmann sampling dynamics methods such as molecular dynamics.
Various enhanced sampling strategies have been developed to improve sampling and statistics.
Although these sampling methods vastly improve time to solution, they are still too slow to test many systems.
Here we have developed a 1D Langevin dynamics toy system to simulate the diffusion of a particle across a bilayer.
The presence of the bilayer is modeled by interactions of the particle with the local Potential of Mean Force (PMF) and Diffusivity (D) profiles.
PMF and D profiles are defined using a piecewise cubic hermite polynomials (PCHIP).
We use this code for validating the new permeability equation for milestoning applications described in the following paper.

L. W. Votapka+, C. T. Lee+, and R. E. Amaro, “Two Relations to Estimate Membrane Permeability Using Milestoning,” J. Phys. Chem. B, vol. 120, no. 33, pp. 8606–8616, Aug. 2016.


## Getting and Building

The code is written in Python and C99-style C with no significant platform dependencies.
On a Unix-style system (e.g., Linux, Mac OS X) after cloning this repository, the following lines will get the PCG submodule and make the C optimized integrator.

```bash
git submodule init
git submodule update
make all
```

The easiest way to get Python is through Conda.

## Dependencies

[PCG-Random website]: http://www.pcg-random.org
*  Included PCG-Random library ([PCG-Random website])
*  Python libraries:
  * numpy
  * scipy

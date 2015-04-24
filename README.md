spikerlib
=========

This is a collection of tools and metrics for analysing spike trains and data from simulations of neurons and neural networks.
The package is split into three submodules:

metrics
--
Spike train distance metrics. Includes code for calculating some of the most widely used spike time similarity metrics (Victor-Purpura, Kreuz SPIKE-distance, Florian modulus metric).

**Other implementations**:
  - Victor-Purpura distance code was adapted from the Matlab code found on the [Homepage of Thomas Kreuz](http://wwwold.fi.isc.cnr.it/users/thomas.kreuz/sourcecode.html).
  - The *official* implementations of the Kreuz SPIKE-distance metrics are part of the [PySpike project on GitHub](https://github.com/mariomulansky/PySpike).
  - The Modulus metric is available by the author on [GitHub](https://github.com/modulus-metric/modulus-metric).

variability
--
Variability metrics commonly used for measuring neural spiking variability (CV, CV2, LV, IR).

tools
--
General purpose tools.
Some of the functions in this submodule depend on Brian (http://briansimulator.org) to work, as they are used to generate inputs or run simulations using the simulator.


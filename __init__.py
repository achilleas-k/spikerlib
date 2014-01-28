"""
This is a collection of tools and metrics for analysing spike trains and data
from simulations of neurons and neural networks. The package is split into
three submodules:

    metrics     Spike train distance metrics. Includes code for calculating
                some of the most widely used spike time similarity metrics
                (Victor-Purpura, Kreuz SPIKE-distance, Florian modulus metric).

    variability Variability metrics commonly used for measuring neural spiking
                variability (CV, CV2, LV, IR).

    tools       General purpose tools. Some of the functions in this submodule
                depend on Brian (http://briansimulator.org) to work, as they
                are used to generate inputs or run simulations using the
                simulator.
"""
import metrics
import variability
import tools


# Binary Star Evolution with Nested Sampling

## Overview
This project implements **Binary Star Evolution (BSE)** simulations using the **Dynesty nested sampling sampler** to explore parameter spaces efficiently. The project is designed to analyze the evolution of binary star systems, optimizing physical parameters using Bayesian inference.

## Features
- Uses **BSE (Binary Star Evolution)** code for simulating stellar interactions.
- Implements **Dynesty** nested sampling for Bayesian parameter estimation.
- Provides **MCMC** and optimization techniques for fitting binary star models.
- Generates **corner plots** for posterior distributions.
- Parallel processing support to improve performance.

## Dependencies
To run the project, install the following dependencies:

```bash
pip install numpy scipy matplotlib emcee corner dynesty dill
```

## File Structure
- `compute_bse_nest.py`: Main script for running BSE and nested sampling.
- `corner2.py`: Utility for generating corner plots.
- `README.md`: This documentation.

## Usage
Run the main script:

```bash
python compute_bse_nest.py
```

This will execute the binary star evolution model with nested sampling and produce output files containing posterior samples and best-fit parameters.

## References
- Hurley, J. R., Pols, O. R., & Tout, C. A. (2002). Evolution of binary stars and the effect of tides on binary populations. *Monthly Notices of the Royal Astronomical Society, 329*(4), 897-928. [doi:10.1046/j.1365-8711.2002.05038.x](https://doi.org/10.1046/j.1365-8711.2002.05038.x)
- Speagle, J. S. (2020). DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences. *Monthly Notices of the Royal Astronomical Society, 493*(3), 3132-3158. [doi:10.1093/mnras/staa278](https://doi.org/10.1093/mnras/staa278)

## License
This project is open-source and available under the MIT License.

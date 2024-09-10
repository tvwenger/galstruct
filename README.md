# galstruct
#Helloworld
Modeling Galactic structure with simulation based inference.

## Installation

I recommend installing `galstruct` in a `conda` virtual environment.

```bash
conda create --name galstruct python==3.10 mkl-service
conda activate galstruct
```

Clone this repository, navigate to it, then install via

```bash
python -m pip install .
```

## Model

Currently only a logarithmic spiral model is implemented. The model parameters
are:

- `az0` (radians) Galactocentric azimuth at which the spiral has Galactocentric radius `Rref`
- `pitch` Spiral pitch angle
- `sigmaV` (km/s) intrinsic velocity scatter
- `sigma_arm_plane` (kpc) Spiral arm width in the Galactic disk
- `sigma_arm_height` (kpc) Spiral arm width perpendicular to the Galactic disk
- `R0` (kpc) Galactocentric radius of the Sun
- `Usun`, `Vsun`, `Wsun` (km/s) Solar motion components
- `Upec`, `Vpec` (km/s) Star forming region non-circular motion components
- `a2`, `a3` Parameters of the Persic et al. (1996) rotation curve
- `Zsun` (pc) Height of the Sun above the Galactic midplane
- `roll` (radians) Roll angle of the Galactic midplane
- `warp_amp` (1/kpc), `warp_off` (radians) Defines warp of Galactic plane such that `Z(R, az) = warp_amp*(R - R_ref)^2 * sin(az - warp_off)`

There are several useful utility functions for this model. See `tests/test_rotcurve.py` and `tests/test_model.py` for examples.

## Likelihood

We use simulation based inference to learn the likelihood function for this complicated
model and to generate the posteriors for the model parameters. See `galstruct/learn_likelihood.py`.

```
$ python galstruct/learn_likelihood.py --help
usage: learn_likelihood.py [-h] [-n NSIMS] [--density_estimator DENSITY_ESTIMATOR] [--features FEATURES]
                           [--layers LAYERS] [--sim_batch_size SIM_BATCH_SIZE]
                           [--training_batch_size TRAINING_BATCH_SIZE] [--Rmin RMIN] [--Rmax RMAX] [--Rref RREF]
                           [--prior PRIOR [PRIOR ...]] [--fixed FIXED [FIXED ...]] [--overwrite]
                           outfile

Train Neural Network for Spiral Model Likelihood

positional arguments:
  outfile               Where the neural network is stored (.pkl extension)

optional arguments:
  -h, --help            show this help message and exit
  -n NSIMS, --nsims NSIMS
                        Number of simulated observations (default: 1000)
  --density_estimator DENSITY_ESTIMATOR
                        Either maf (Masked Autoregressive Flow) or nsf (Neural Spline Flow) (default: maf)
  --features FEATURES   Number of neural spine flow hidden features (default: 50)
  --layers LAYERS       Number of neural spine flow transform layers (default: 5)
  --sim_batch_size SIM_BATCH_SIZE
                        Batch size for simulations (default: 1)
  --training_batch_size TRAINING_BATCH_SIZE
                        Batch size for training (default: 50)
  --Rmin RMIN           Minimum Galactocentric radius (kpc) (default: 3.0)
  --Rmax RMAX           Maximum Galactocentric radius (kpc) (default: 15.0)
  --Rref RREF           Reference Galactocentric radius (kpc) (default: 8.0)
  --prior PRIOR [PRIOR ...]
                        Priors on model parameters (e.g., --prior R0 normal 8.5 0.5 --prior az0 uniform 0.0 6.3 --prior
                        sigmaV halfnormal 10.0) (default: [['az0', 'uniform', 0.0, 6.3], ['pitch', 'uniform', 0.1, 0.4],
                        ['sigmaV', 'halfnormal', 10.0], ['sigma_arm_plane', 'halfnormal', 1.0], ['sigma_arm_height',
                        'halfnormal', 0.5], ['R0', 'normal', 8.5, 0.5], ['Usun', 'normal', 10.5, 1.0], ['Vsun', 'normal',
                        12.2, 10.0], ['Wsun', 'normal', 7.5, 2.5], ['Upec', 'normal', 5.8, 10.0], ['Vpec', 'normal',
                        -3.5, 10.0], ['a2', 'normal', 0.977, 0.01], ['a3', 'normal', 1.622, 0.01], ['Zsun', 'normal',
                        5.5, 10.0], ['roll', 'normal', 0.0, 0.05], ['warp_amp', 'halfnormal', 0.05], ['warp_off',
                        'normal', -0.5, 1.0]])
  --fixed FIXED [FIXED ...]
                        Fixed parameter names followed by their fixed value (e.g., --fixed R0 8.5 --fixed Usun 10.5)
                        (default: [])
  --overwrite           Overwrite existing outfile (default: False)
```

The learned likelihood can be plotted with
`galstruct/plot_likelihood.py`.  The program
`galstruct/train_models.py` is a wrapper that allows the user to
generate many neural networks with varying parameters.

## MCMC

The posterior can be sampled from the learned likelihood using
`galstruct/mcmc_posterior.py`

```
$ python galstruct/mcmc_posterior.py --help
WARNING (aesara.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
usage: mcmc_posterior.py [-h] [-n NITER] [--num_data NUM_DATA] [--spiral_params SPIRAL_PARAMS [SPIRAL_PARAMS ...]] [--grm_params GRM_PARAMS [GRM_PARAMS ...]]
                         [--warp_params WARP_PARAMS [WARP_PARAMS ...]] [--disk_params DISK_PARAMS [DISK_PARAMS ...]] [--Rmin RMIN] [--Rmax RMAX] [--Rref RREF]
                         [--num_spirals NUM_SPIRALS] [--prior PRIOR [PRIOR ...]] [-f FIXED [FIXED ...]] [-o OUTLIERS [OUTLIERS ...]] [--chains CHAINS] [--ntune NTUNE]
                         [--ninit NINIT] [--step_scale STEP_SCALE] [--target_accept TARGET_ACCEPT] [--overwrite]
                         dbfile outfile loglike_net

Sample Spiral Model Posterior using MCMC

positional arguments:
  dbfile                The HII region catalog database filename. If 'synthetic', generate synthetic data.
  outfile               Where the MCMC model and trace are stored (.pkl extension)
  loglike_net           Where the likelihood neural network is stored (.pkl extension)

options:
  -h, --help            show this help message and exit
  -n NITER, --niter NITER
                        Maximum number of MCMC iterations (default: 1000)
  --num_data NUM_DATA   Number of synthetic data to generate. (default: 2000)
  --spiral_params SPIRAL_PARAMS [SPIRAL_PARAMS ...]
                        Spiral parameters for synthetic data (default: [0.25, 0.25, 0.25, 0.25, 1.2566370614359172, 2.3911010752322315, 4.39822971502571,
                        5.532693728822025, 0.24434609527920614, 0.24434609527920614, 0.24434609527920614, 0.24434609527920614, 5.0, 0.5, 0.1])
  --grm_params GRM_PARAMS [GRM_PARAMS ...]
                        GRM parameters for synthetic data (default: [8.16643777, 10.4543041, 12.18499493, 7.71886874, 5.79095823, -3.39171583, 0.97757558, 1.62261724,
                        5.5, 0.0])
  --warp_params WARP_PARAMS [WARP_PARAMS ...]
                        Warp parameters for synthetic data (default: [0.02, -0.5])
  --disk_params DISK_PARAMS [DISK_PARAMS ...]
                        Exponential disk parameters for synthetic data (default: [35.0, 3.0, 2.5])
  --Rmin RMIN           Minimum Galactocentric radius (kpc) (default: 3.0)
  --Rmax RMAX           Maximum Galactocentric radius (kpc) (default: 15.0)
  --Rref RREF           Reference Galactocentric radius (kpc) (default: 8.0)
  --num_spirals NUM_SPIRALS
                        Number of spiral arms (default: 4)
  --prior PRIOR [PRIOR ...]
                        Priors on model parameters (e.g., --prior R0 normal 8.5 0.5 --prior az0 uniform 0.0 6.3 --prior sigmaV halfnormal 10.0) (default: [['q',
                        'dirichlet'], ['az0', 'uniform', 0.0, 1.5, 1.5, 3.2, 3.2, 4.7, 4.7, 6.3], ['pitch', 'uniform', 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4],
                        ['sigmaV', 'halfnormal', 10.0], ['sigma_arm_plane', 'halfnormal', 1.0], ['sigma_arm_height', 'halfnormal', 0.5], ['R0', 'normal', 8.5, 0.5],
                        ['Usun', 'normal', 10.5, 1.0], ['Vsun', 'normal', 12.2, 10.0], ['Wsun', 'normal', 7.5, 2.5], ['Upec', 'normal', 5.8, 10.0], ['Vpec', 'normal',
                        -3.5, 10.0], ['a2', 'normal', 0.977, 0.01], ['a3', 'normal', 1.622, 0.01], ['Zsun', 'normal', 5.5, 10.0], ['roll', 'normal', 0.0, 0.05],
                        ['warp_amp', 'halfnormal', 0.05], ['warp_off', 'normal', -0.5, 1.0]])
  -f FIXED [FIXED ...], --fixed FIXED [FIXED ...]
                        Fixed GRM parameter names followed by their fixed value. (default: [])
  -o OUTLIERS [OUTLIERS ...], --outliers OUTLIERS [OUTLIERS ...]
                        HII regions to exclude from analysis (default: None)
  --chains CHAINS       Number of Markov chains (default: 4)
  --ntune NTUNE         Number of MCMC tuning iterations (default: 1000)
  --ninit NINIT         Number of ADVI initialzation samples (default: 100000)
  --step_scale STEP_SCALE
                        Starting NUTS step_scale. (default: 0.25)
  --target_accept TARGET_ACCEPT
                        Desired acceptance rate. (default: 0.9)
  --overwrite           Overwrite existing outfile (default: False)
```

## License and Copyright

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Copyright(C) 2020-2022 by
Trey V. Wenger; tvwenger@gmail.com

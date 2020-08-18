#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
learn_likelihood.py

Use likelihood-free inference to train a neural network on the
likelihood function for a spiral model.

Copyright(C) 2020 by Trey Wenger <tvwenger@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Trey Wenger - August 2020
"""

import os
import argparse
import pickle

import torch

from sbi.utils import likelihood_nn
from sbi.inference import SNLE, prepare_for_sbi

import numpy as np

from model.simulator import simulator
from torch_prior import Prior

# set random seed
np.random.seed(1234)

# default parameter values
_NUM_SIMS = 1000
_DENSITY_ESTIMATOR = 'maf'
_HIDDEN_FEATURES = 50
_TRANSFORM_LAYERS = 5
_SIM_BATCH_SIZE = 1
_TRAINING_BATCH_SIZE = 50
_RMIN = 3.0
_RMAX = 15.0
_RREF = 8.0
_FIXED = {}
_OVERWRITE = False

def main(outfile, priors, num_sims=_NUM_SIMS,
         density_estimator=_DENSITY_ESTIMATOR,
         hidden_features=_HIDDEN_FEATURES,
         transform_layers=_TRANSFORM_LAYERS,
         sim_batch_size=_SIM_BATCH_SIZE,
         training_batch_size=_TRAINING_BATCH_SIZE,
         Rmin=_RMIN, Rmax=_RMAX, Rref=_RREF,
         fixed=_FIXED, overwrite=_OVERWRITE):
    """
    Train a neural network on the likelihood function for the spiral
    model.

    Inputs:
      outfile :: string
        Where the SBI posterior object is saved (.pkl extension)
      priors :: dictionary
        Priors for each paramter. The keys must be the parameter names:
            az0, pitch, sigmaV, sigma_arm_plane, sigma_arm_height,
            R0, Usun, Vsun, Upec, Vpec, a2, a3, Zsun, roll,
            warp_amp, warp_off
        The value of each key must be a list with one of the following
        formats.
          ['normal', mean, width]
          ['halfnormal', width]
          ['cauchy', mode, scale]
          ['halfcauchy', scale]
          ['uniform', lower, upper]
          ['fixed', value]
      density_estimator :: string
        Density estimator for neural network. Either 'maf' for
        Masked Autoregressive Flow or 'nsf' for Neural Spline Flow
      num_sims :: integer
        Number of simulations
      hidden_features :: integer
        Number of neural spline flow hidden features
      transform_layers :: integer
        Number of neural spline flow transform layers
      sim_batch_size, training_batch_size :: integers
        Batch sizes for simulations and training      
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      fixed :: dictionary
        Fixed GRM parameters (keys) and their fixed values.
      overwrite :: boolean
        If True, overwrite outfile if it exists.

    Returns: Nothing
    """
    #
    # Check that outfile does not already exist
    #
    if os.path.exists(outfile) and not overwrite:
        raise ValueError("{0} already exists!".format(outfile))
    #
    # Initialize priors and simulator
    #
    print("Using priors:")
    for key, item in priors.items():
        print("    ", key, item)
    if len(fixed) > 0:
        print("Fixed parameters:")
        for key, item in fixed.items():
            print("    ", key, item)
    prior = Prior(priors)
    sim = lambda theta: simulator(
        theta, Rmin=torch.tensor(Rmin), Rmax=torch.tensor(Rmax),
        Rref=torch.tensor(Rref), fixed=fixed)
    sim, prior = prepare_for_sbi(sim, prior)
    #
    # Learn likelihood
    #
    de = None
    if density_estimator == 'nsf':
        de = 'Neural Spline Flow'
    elif density_estimator == 'maf':
        de = 'Masked Autoregressive Flow'
    else:
        raise ValueError("Invalid density estimator: {0}".format(density_estimator))
    print("Learning likelihood with {0} density estimator,".format(de))
    print("{0} hidden features, and {1} transform layers.".format(
        hidden_features, transform_layers))
    density_estimator = likelihood_nn(
        model=density_estimator, hidden_features=hidden_features,
        num_transforms=transform_layers)
    inference = SNLE(
        sim, prior, density_estimator=density_estimator,
        simulation_batch_size=sim_batch_size, show_round_summary=True)
    print("Simulating with batch size: {0}".format(sim_batch_size))
    print("Training with batch size: {0}".format(training_batch_size))
    posterior = inference(
        num_sims, training_batch_size=training_batch_size)
    #
    # Save
    #
    print("Pickling results to {0}".format(outfile))
    with open(outfile, 'wb') as f:
        output = {
            'posterior': posterior, 'priors': priors, 'fixed': fixed}
        pickle.dump(output, f)
    print("Done!")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Train Neural Network for Spiral Model Likelihood",
        prog="train_likelihood.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PARSER.add_argument(
        "outfile", type=str,
        help="Where the neural network is stored (.pkl extension)")
    PARSER.add_argument(
        "-n", "--nsims", type=int, default=_NUM_SIMS,
        help="Number of simulated observations")
    PARSER.add_argument(
        "--density_estimator", type=str, default=_DENSITY_ESTIMATOR,
        help="Either maf (Masked Autoregressive Flow) or nsf (Neural Spline Flow)")
    PARSER.add_argument(
        "--features", type=int, default=_HIDDEN_FEATURES,
        help="Number of neural spine flow hidden features")
    PARSER.add_argument(
        "--layers", type=int, default=_TRANSFORM_LAYERS,
        help="Number of neural spine flow transform layers")
    PARSER.add_argument(
        "--sim_batch_size", type=int, default=_SIM_BATCH_SIZE,
        help="Batch size for simulations")
    PARSER.add_argument(
        "--training_batch_size", type=int, default=_TRAINING_BATCH_SIZE,
        help="Batch size for training")
    PARSER.add_argument(
        "--Rmin", type=float, default=_RMIN,
        help="Minimum Galactocentric radius (kpc)")
    PARSER.add_argument(
        "--Rmax", type=float, default=_RMAX,
        help="Maximum Galactocentric radius (kpc)")
    PARSER.add_argument(
        "--Rref", type=float, default=_RREF,
        help="Reference Galactocentric radius (kpc)")
    DEFAULT_AZ0 = ['uniform', 0.0, 6.3]
    PARSER.add_argument(
        "--paz0", nargs="+", default=DEFAULT_AZ0,
        help="Prior on spiral azimuth at refR (radians)")
    DEFAULT_PITCH = ['uniform', 0.1, 0.4]
    PARSER.add_argument(
        "--ppitch", nargs="+", default=DEFAULT_PITCH,
        help="Prior on spiral pitch angle")
    PARSER.add_argument(
        "--psigmaV", nargs="+", default=["halfnormal", 10.0],
        help="Prior on HII region streaming velocity (km/s)")
    PARSER.add_argument(
        "--psigma_arm_plane", nargs="+", default=["halfnormal", 1.0],
        help="Prior on arm width in the plane (kpc)")
    PARSER.add_argument(
        "--psigma_arm_height", nargs="+", default=["halfnormal", 0.5],
        help="Prior on arm width perpendicular to the plane (kpc)")
    PARSER.add_argument(
        "--pgrm", action='append', nargs="+", default=[],
        help="Priors on GRM parameters (like: R0 normal 8.5 0.5 Zsun normal 5.5 25.0)")
    PARSER.add_argument(
        "--pwarp_amp", nargs="+", default=["halfnormal", 0.05],
        help="Prior on warp mode amplitude (kpc-1)")
    PARSER.add_argument(
        "--pwarp_off", nargs="+", default=["normal", -0.5, 1.0],
        help="Prior on warp mode offset (radians)")
    PARSER.add_argument(
        "-f", "--fixed", nargs="+", default=[],
        help=("Fixed GRM parameter names followed by their fixed value."))
    PARSER.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outfile")
    ARGS = vars(PARSER.parse_args())
    #
    # Generate priors dictionary
    #
    PARAMS = [
        'az0', 'pitch', 'sigmaV', 'sigma_arm_plane', 'sigma_arm_height',
        'warp_amp', 'warp_off']
    PRIORS = {
        P: [ARGS['p'+P][0]] + [float(v) for v in ARGS['p'+P][1:]]
        for P in PARAMS}
    for PGRM in ARGS['pgrm']:
        PRIORS[PGRM[0]] = [PGRM[1]] + [float(v) for v in PGRM[2:]]
    FIXED = {}
    for FIX in range(len(ARGS['fixed'])//2):
        FIXED[ARGS['fixed'][2*FIX]] = float(ARGS['fixed'][2*FIX+1])
    main(ARGS['outfile'], PRIORS, num_sims=ARGS['nsims'],
         density_estimator=ARGS['density_estimator'],
         hidden_features=ARGS['features'], transform_layers=ARGS['layers'],
         sim_batch_size=ARGS['sim_batch_size'],
         training_batch_size=ARGS['training_batch_size'],
         Rmin=ARGS['Rmin'], Rmax=ARGS['Rmax'], Rref=ARGS['Rref'],
         fixed=FIXED, overwrite=ARGS['overwrite'])

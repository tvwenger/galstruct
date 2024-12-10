#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
learn_likelihood.py

Use likelihood-free inference to train a neural network on the
likelihood function for a spiral model.

Copyright(C) 2020-2022 by Trey Wenger <tvwenger@gmail.com>

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
Trey Wenger - May 2022 - Formatting
"""

import os
import argparse
import pickle
import numpy as np

import torch

import sbi.utils
from sbi.inference import NLE_A
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.neural_nets import likelihood_nn

from galstruct.model.simulator import simulator
from galstruct.torch_prior import Prior

# set random seed
np.random.seed(1234)

# default parameter values
_NUM_SIMS = 100000
_DENSITY_ESTIMATOR = "maf"
_HIDDEN_FEATURES = 50
_TRANSFORM_LAYERS = 5
_TRAINING_BATCH_SIZE = 50
_RMIN = 3.0
_RMAX = 15.0
_RREF = 8.0
_FIXED = {}
_OVERWRITE = False


def main(
    outfile,
    priors,
    num_sims=_NUM_SIMS,
    density_estimator=_DENSITY_ESTIMATOR,
    hidden_features=_HIDDEN_FEATURES,
    transform_layers=_TRANSFORM_LAYERS,
    training_batch_size=_TRAINING_BATCH_SIZE,
    Rmin=_RMIN,
    Rmax=_RMAX,
    Rref=_RREF,
    fixed=_FIXED,
    overwrite=_OVERWRITE,
):
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
      training_batch_size :: integer
        Batch size for training
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
    # Check that outfile does not already exist
    if os.path.exists(outfile) and not overwrite:
        raise ValueError("{0} already exists!".format(outfile))

    # Initialize priors and simulator
    print("Using priors:")
    for key, item in priors.items():
        print("    ", key, item)
    if len(fixed) > 0:
        print("Fixed parameters:")
        for key, item in fixed.items():
            print("    ", key, item)
    prior = Prior(priors)

    def sim(theta):
        return simulator(
            theta,
            Rmin=torch.tensor(Rmin),
            Rmax=torch.tensor(Rmax),
            Rref=torch.tensor(Rref),
            fixed=fixed,
            disk=None,
        )

    # prepare for SBI
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    sim = process_simulator(sim, prior, prior_returns_numpy)
    check_sbi_inputs(sim, prior)
    print(f"Learning {num_parameters} parameters")

    # Density estimator
    de = None
    if density_estimator == "nsf":
        de = "Neural Spline Flow"
    elif density_estimator == "maf":
        de = "Masked Autoregressive Flow"
    else:
        raise ValueError("Invalid density estimator: {0}".format(density_estimator))
    print("Learning likelihood with {0} density estimator".format(de))
    print(
        "{0} hidden features, and {1} transform layers.".format(
            hidden_features, transform_layers
        )
    )
    density_estimator_build_fun = likelihood_nn(
        model=density_estimator,
        hidden_features=hidden_features,
        num_transforms=transform_layers,
    )

    # Inference
    inference = NLE_A(
        prior=prior,
        density_estimator=density_estimator_build_fun,
    )

    # Sample prior
    print("Sampling prior...")
    theta = prior.sample((num_sims,))
    print(theta.shape)
    print(fixed)

    # Simulate
    print("Simulating...")
    x = sim(theta)
    isnan = torch.any(torch.isnan(x), axis=1)
    print(f"Dropping {isnan.sum()} simulations with NaNs")
    x = x[~isnan]
    theta = theta[~isnan]

    # Train
    print("Training with batch size: {0}".format(training_batch_size))
    print()
    density_estimator = inference.append_simulations(theta, x).train(
        training_batch_size=training_batch_size
    )
    posterior = inference.build_posterior(density_estimator)
    print()

    # Save
    print("Pickling results to {0}".format(outfile))
    with open(outfile, "wb") as f:
        output = {
            "posterior": posterior,
            "density_estimator": density_estimator,
            "priors": priors,
            "fixed": fixed,
        }
        pickle.dump(output, f)
    print("Done!")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Train Neural Network for Spiral Model Likelihood",
        prog="learn_likelihood.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "outfile", type=str, help="Where the neural network is stored (.pkl extension)"
    )
    PARSER.add_argument(
        "-n",
        "--nsims",
        type=int,
        default=_NUM_SIMS,
        help="Number of simulated observations",
    )
    PARSER.add_argument(
        "--density_estimator",
        type=str,
        default=_DENSITY_ESTIMATOR,
        help="Either maf (Masked Autoregressive Flow) or nsf (Neural Spline Flow)",
    )
    PARSER.add_argument(
        "--features",
        type=int,
        default=_HIDDEN_FEATURES,
        help="Number of neural spine flow hidden features",
    )
    PARSER.add_argument(
        "--layers",
        type=int,
        default=_TRANSFORM_LAYERS,
        help="Number of neural spine flow transform layers",
    )
    PARSER.add_argument(
        "--training_batch_size",
        type=int,
        default=_TRAINING_BATCH_SIZE,
        help="Batch size for training",
    )
    PARSER.add_argument(
        "--Rmin", type=float, default=_RMIN, help="Minimum Galactocentric radius (kpc)"
    )
    PARSER.add_argument(
        "--Rmax", type=float, default=_RMAX, help="Maximum Galactocentric radius (kpc)"
    )
    PARSER.add_argument(
        "--Rref",
        type=float,
        default=_RREF,
        help="Reference Galactocentric radius (kpc)",
    )
    DEFAULT_PRIORS = [
        ["az0", "uniform", 0.0, 6.3],
        ["pitch", "uniform", 0.1, 0.4],
        ["sigmaV", "halfnormal", 10.0],
        ["sigma_arm_plane", "halfnormal", 1.0],
        ["sigma_arm_height", "halfnormal", 0.5],
        ["R0", "normal", 8.5, 0.5],
        ["Usun", "normal", 10.5, 1.0],
        ["Vsun", "normal", 12.2, 10.0],
        ["Wsun", "normal", 7.5, 2.5],
        ["Upec", "normal", 5.8, 10.0],
        ["Vpec", "normal", -3.5, 10.0],
        ["a2", "normal", 0.977, 0.01],
        ["a3", "normal", 1.622, 0.01],
        ["Zsun", "normal", 5.5, 10.0],
        ["roll", "normal", 0.0, 0.05],
        ["warp_amp", "halfnormal", 0.05],
        ["warp_off", "normal", -0.5, 1.0],
    ]
    PARSER.add_argument(
        "--prior",
        action="append",
        nargs="+",
        default=DEFAULT_PRIORS,
        help=(
            "Priors on model parameters (e.g., --prior R0 normal 8.5 0.5 "
            + "--prior az0 uniform 0.0 6.3 --prior sigmaV halfnormal 10.0)"
        ),
    )

    PARSER.add_argument(
        "--fixed",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Fixed parameter names followed by their fixed value "
            + "(e.g., --fixed R0 8.5 --fixed Usun 10.5)"
        ),
    )
    PARSER.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outfile"
    )
    ARGS = vars(PARSER.parse_args())

    # Generate priors dictionary
    PARAMS = [
        "az0",
        "pitch",
        "sigmaV",
        "sigma_arm_plane",
        "sigma_arm_height",
        "R0",
        "Usun",
        "Vsun",
        "Wsun",
        "Upec",
        "Vpec",
        "a2",
        "a3",
        "Zsun",
        "roll",
        "warp_amp",
        "warp_off",
    ]
    PRIORS = {}
    FIXED = {}
    for PARAM in PARAMS:
        for FIX in ARGS["fixed"]:
            if FIX[0] == PARAM:
                FIXED[PARAM] = float(FIX[1])
        if PARAM in FIXED:
            continue
        FOUND = False
        for PRIOR in ARGS["prior"]:
            if PRIOR[0] == PARAM:
                PRIORS[PARAM] = [PRIOR[1]] + [float(v) for v in PRIOR[2:]]
                FOUND = True
        if not FOUND:
            for PRIOR in DEFAULT_PRIORS:
                if PRIOR[0] == PARAM:
                    PRIORS[PARAM] = [PRIOR[1]] + [float(v) for v in PRIOR[2:]]
    main(
        ARGS["outfile"],
        PRIORS,
        num_sims=ARGS["nsims"],
        density_estimator=ARGS["density_estimator"],
        hidden_features=ARGS["features"],
        transform_layers=ARGS["layers"],
        training_batch_size=ARGS["training_batch_size"],
        Rmin=ARGS["Rmin"],
        Rmax=ARGS["Rmax"],
        Rref=ARGS["Rref"],
        fixed=FIXED,
        overwrite=ARGS["overwrite"],
    )

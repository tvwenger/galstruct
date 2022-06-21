#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_models.py

Generate several neural networks with a variety of parameters.

Copyright(C) 2022 by Catie Terrey & Trey Wenger <tvwenger@gmail.com>

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

Catie Terrey & Trey Wenger - June 2022
"""

import argparse
import time
import itertools

from pathlib import Path

from galstruct import learn_likelihood, plot_likelihood

# default parameter values
_NUM_SIMS = [50000]
_DENSITY_ESTIMATOR = ["maf"]
_HIDDEN_FEATURES = [50]
_TRANSFORM_LAYERS = [5]
_SIM_BATCH_SIZE = [1]
_TRAINING_BATCH_SIZE = [50]
_RMIN = 3.0
_RMAX = 15.0
_RREF = 8.0

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


def main(
    outfile,
    density_estimators,
    nsims,
    features,
    layers,
    sim_batches,
    train_batches,
    priors={},
    Rmin=_RMIN,
    Rmax=_RMAX,
    Rref=_RREF,
    fixed={},
    plot=False,
    overwrite=False,
):
    """
    Generate several neural networks with a variety of parameters.

    Inputs:
      outfile :: string
          Output filename prefix. Files have names like:
          f"{outfile}_{density_estimator}_nsims={nsim}_feat={feat}_layers={layer}_sbs={sb}_tbs={tb}.pkl"
      density_estimators :: list of strings
          Density estimators to use
      nsims :: list of integers
          Number of simulations
      features :: list of integers
          Number of features
      layers :: list of integers
          Number of layers
      sim_batches :: list of integers
          Number of simulation batch sizes
      train_batches :: list of integers
          Number of training batch sizes
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
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      fixed :: dictionary
        Fixed GRM parameters (keys) and their fixed values.
      plot :: boolean
          If True, generate plot instead of network
      overwrite :: boolean
          If True, overwrite outfile

    Returns:
        Nothing
    """
    for de, nsim, nfeat, nlayer, sb, tb in itertools.product(
        density_estimators, nsims, features, layers, sim_batches, train_batches
    ):
        fname = f"{outfile}_{de}_nsims={nsim}_feat={nfeat}_layers={nlayer}_sbs={sb}_tbs={tb}.pkl"
        if plot:
            outdir = f"plots/{outfile}_{de}_nsims={nsim}_feat={nfeat}_layers={nlayer}_sbs={sb}_tbs={tb}/"
            Path(outdir).mkdir(parents=True, exist_ok=True)
            plot_likelihood.main(fname, outdir=outdir)
        else:
            tic = time.perf_counter()
            learn_likelihood.main(
                fname,
                priors,
                num_sims=nsim,
                density_estimator=de,
                hidden_features=nfeat,
                transform_layers=nlayer,
                sim_batch_size=sb,
                training_batch_size=tb,
                Rmin=Rmin,
                Rmax=Rmax,
                Rref=Rref,
                fixed=fixed,
                overwrite=overwrite,
            )
            toc = time.perf_counter()
            delta_t = toc - tic
            print(
                f"Training Complete. File exported to {fname} \n"
                + f"Time taken: {delta_t}"
            )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Train Neural Network for Spiral Model Likelihood",
        prog="train_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "outfile", type=str, help="Where the neural network is stored (.pkl extension)"
    )
    PARSER.add_argument(
        "-n",
        "--nsims",
        type=int,
        nargs="+",
        default=_NUM_SIMS,
        help="Number of simulated observations",
    )
    PARSER.add_argument(
        "--density_estimator",
        type=str,
        nargs="+",
        default=_DENSITY_ESTIMATOR,
        help="Either maf (Masked Autoregressive Flow) or nsf (Neural Spline Flow)",
    )
    PARSER.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=_HIDDEN_FEATURES,
        help="Number of neural spine flow hidden features",
    )
    PARSER.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=_TRANSFORM_LAYERS,
        help="Number of neural spine flow transform layers",
    )
    PARSER.add_argument(
        "--sim_batch_size",
        type=int,
        nargs="+",
        default=_SIM_BATCH_SIZE,
        help="Batch size for simulations",
    )
    PARSER.add_argument(
        "--training_batch_size",
        type=int,
        nargs="+",
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
        "--plot", default=False, help="Whether or not to generate all frames"
    )
    PARSER.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outfile"
    )
    ARGS = vars(PARSER.parse_args())

    # build priors dictionary
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
        ARGS["density_estimator"],
        ARGS["nsims"],
        ARGS["features"],
        ARGS["layers"],
        ARGS["sim_batch_size"],
        ARGS["training_batch_size"],
        priors=PRIORS,
        fixed=FIXED,
        plot=ARGS["plot"],
        overwrite=ARGS["overwrite"],
    )

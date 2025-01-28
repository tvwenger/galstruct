#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
mcmc_posterior.py

Generate spiral model posteriors by MCMC using trained likelihood
and real or synthetic data.

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
Trey Wenger - June 2022 - Updates for new pymc and aesara
"""

import os
import numpy as np
import argparse
import pickle
import dill
import pandas as pd

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import pytensor

pytensor.config.floatX = "float32"
# pytensor.config.warn_float64 = "raise"

import torch
import pymc as pm

from galstruct.model.simulator import simulator

from galstruct.likelihood_op import _params, LogLike, LogLikeCalc


# default values for synthetic spiral parameters
_spiral_params = [
    0.25,  # q
    0.25,
    0.25,
    0.25,
    np.deg2rad(72.0),  # az0
    np.deg2rad(137.0),
    np.deg2rad(252.0),
    np.deg2rad(317.0),
    np.deg2rad(14.0),  # pitch
    np.deg2rad(14.0),
    np.deg2rad(14.0),
    np.deg2rad(14.0),
    5.0,  # sigmaV
    0.5,  # sigma_arm_plane
    0.1,  # sigma_arm_height
]

# default values for GRM parameters
_grm_params = [
    8.16643777,  # R0
    10.4543041,  # Usun
    12.18499493,  # Vsun
    7.71886874,  # Wsun
    5.79095823,  # Upec
    -3.39171583,  # Vpec
    0.97757558,  # a2
    1.62261724,  # a3
    5.5,  # Zsun
    0.0,  # roll
]

# default values for warp parameters
_warp_params = [0.02, -0.5]  # warp_amp, warp_off

# default values for exponential disk parameters
_disk_params = [35.0, 3.0, 2.5]

# default parameter values
_NUM_DATA = 2000
_RMIN = 3.0
_RMAX = 15.0
_RREF = 8.0
_NUM_SPIRALS = 4
_NITER = 1_000
_NTUNE = 1_000
_NINIT = 100_000
_NUM_CHAINS = 8
_TARGET_ACCEPT = 0.9
_FIXED = {}
_OUTLIERS = None
_OVERWRITE = False


def main(
    datafile,
    outfile,
    priors,
    loglike_net,
    num_data=_NUM_DATA,
    spiral_params=_spiral_params,
    grm_params=_grm_params,
    warp_params=_warp_params,
    disk_params=_disk_params,
    Rmin=_RMIN,
    Rmax=_RMAX,
    Rref=_RREF,
    num_spirals=_NUM_SPIRALS,
    niter=_NITER,
    ntune=_NTUNE,
    ninit=_NINIT,
    num_chains=_NUM_CHAINS,
    target_accept=_TARGET_ACCEPT,
    fixed=_FIXED,
    outliers=_OUTLIERS,
    overwrite=_OVERWRITE,
    hiidb="/data/hii_v2_20201203.db",
):
    """
    Use MCMC to generate spiral model posteriors for real or
    synthetic HII region data.

    Inputs:
      datafile :: string
        HII region data file, or 'synthetic'
      outfile :: string
        Where the MCMC trace is saved (as Pickle)
      priors :: dictionary
        Priors for each paramter. The keys must be the parameter names:
            q, az0, pitch, sigmaV, sigma_arm_plane, sigma_arm_height,
            R0, Usun, Vsun, Upec, Vpec, a2, a3, Zsun, roll,
            warp_amp, warp_off
        The value of each key must be a list with one of the following
        formats. The values are repeated for each arm/warp mode.
          ['dirichlet']
          ['normal', mean1, width1, mean2, width2, ...]
          ['halfnormal', width1, width2, ...]
          ['cauchy', mean1, width1, mean2, width2, ...]
          ['halfcauchy', width1, width2, ...]
          ['uniform', lower1, upper1, lower2, upper2, ...]
          ['fixed', value1, value2, ...]
      loglike_net :: string
        Pickle file containing the SBI posterior and likelihood neural
        network
      num_data :: integer
        The number of synthetic data to generate
      spiral_params :: 1-D array of scalars
        The synthetic data spiral parameters
        [qs, az0s, pitchs, sigmaV, sigma_arm_plane, sigma_arm_height]
      grm_params :: 1-D array of scalars
        Synthetic GRM parameters
        [R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3, Zsun, roll]
      warp_params :: 1-D array of scalars
        Synthetic warp parameters [warp_amp, warp_off]
      disk_params :: 1-D array of scalars
        Synthetic exponential disk parameters [I2, Rs, Rc]
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      num_spirals :: integer
        The number of spirals
      niter :: integer
        Number of MCMC iterations per chain
      ntune :: integer
        Number of tuning/warm-up interations per chain
      ninit :: integer
        Number of ADVI initialization samples
      num_chains :: integer
        Number of Markov chains
      target_accept :: scalar
        Desired acceptance rate (0 to 1)
      fixed :: dictionary
        Fixed GRM parameters (keys) and their fixed values.
      outliers :: list of strings
        Remove sources with these gnames from the analysis
      overwrite :: boolean
        If True, overwrite outfile if it exists.

    Returns: Nothing
    """
    # Check that outfile does not already exist
    if os.path.exists(outfile) and not overwrite:
        raise ValueError("{0} already exists!".format(outfile))

    # Get data or generate synthetic data
    if datafile == "synthetic":
        q = spiral_params[:num_spirals]
        thetas = [
            torch.as_tensor(
                [spiral_params[num_spirals + i]]
                + [spiral_params[2 * num_spirals + i]]
                + spiral_params[3 * num_spirals :]
                + grm_params
                + warp_params
            )
            for i in range(num_spirals)
        ]
        data = torch.cat(
            tuple(
                simulator(
                    theta.expand(int(qi * num_data), -1),
                    Rmin=torch.tensor(Rmin),
                    Rmax=torch.tensor(Rmax),
                    Rref=torch.tensor(Rref),
                    disk=disk_params,
                )
                for theta, qi in zip(thetas, q)
            )
        ).float()
    else:
        print(f"Reading HII region data from {datafile}")
        data = pd.read_csv(datafile)

    # Get likelihood neural network object
    with open(loglike_net, "rb") as f:
        net = pickle.load(f)

    # Setup model
    with pm.Model() as model:
        # Get parameter priors
        determ = {}
        for param in priors:
            if param in fixed:
                continue
            num = 1
            shape = ()
            if param in ["q", "az0", "pitch"]:
                num = num_spirals
                shape = (num,)
            if priors[param][0] == "fixed":
                fixed[param] = np.array(priors[param][1:])
            elif priors[param][0] == "dirichlet":
                if num > 1:
                    determ[param] = pm.Dirichlet(
                        param,
                        a=np.ones(num).astype(np.float32),
                    )
                else:
                    fixed[param] = np.array([1.0])
            elif priors[param][0] == "uniform":
                lower = np.array(priors[param][1 : 2 * num + 1 : 2])
                upper = np.array(priors[param][2 : 2 * num + 1 : 2])
                if len(shape) == 0:
                    lower = lower[0]
                    upper = upper[0]
                determ[param] = pm.Uniform(
                    param,
                    lower=lower.astype(np.float32),
                    upper=upper.astype(np.float32),
                    shape=shape,
                )
            elif priors[param][0] == "normal":
                mean = np.array(priors[param][1 : 2 * num + 1 : 2])
                sigma = np.array(priors[param][2 : 2 * num + 1 : 2])
                if len(shape) == 0:
                    mean = mean[0]
                    sigma = sigma[0]
                determ[param] = pm.Normal(
                    param,
                    mu=mean.astype(np.float32),
                    sigma=sigma.astype(np.float32),
                    shape=shape,
                )
            elif priors[param][0] == "cauchy":
                alpha = np.array(priors[param][1 : 2 * num + 1 : 2])
                beta = np.array(priors[param][2 : 2 * num + 1 : 2])
                if len(shape) == 0:
                    alpha = alpha[0]
                    beta = beta[0]
                determ[param] = pm.Cauchy(
                    param,
                    alpha=alpha.astype(np.float32),
                    beta=beta.astype(np.float32),
                    shape=shape,
                )
            elif priors[param][0] == "halfnormal":
                sigma = np.array(priors[param][1 : num + 1])
                if len(shape) == 0:
                    sigma = sigma[0]
                determ[param] = pm.HalfNormal(
                    param,
                    sigma=sigma.astype(np.float32),
                    shape=shape,
                )
            elif priors[param][0] == "halfcauchy":
                beta = np.array(priors[param][1 : num + 1])
                if len(shape) == 0:
                    beta = beta[0]
                determ[param] = pm.HalfCauchy(
                    param,
                    beta=beta.astype(np.float32),
                    shape=shape,
                )
            else:
                raise ValueError("Invalid prior for {0}: {1}".format(param, priors[param][0]))

        # Pack model parameters
        theta = [determ[p] for p in _params if p not in fixed]

        # Create likelihood Operator
        loglike_calc = LogLikeCalc(net["density_estimator"], num_spirals, fixed)
        loglike_op = LogLike(loglike_calc, data)

        # Evalulate likelihood
        _ = pm.Potential("like", loglike_op(*theta)[0])

    # Run inference
    with model:
        trace = pm.sample(
            niter,
            init="auto",
            # init="advi",
            tune=ntune,
            n_init=ninit,
            cores=num_chains,
            chains=num_chains,
            target_accept=target_accept,
        )
        with open(outfile, "wb") as f:
            dill.dump({"data": data, "trace": trace}, f)
    print(pm.summary(trace).to_string())


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Sample Spiral Model Posterior using MCMC",
        prog="mcmc_posterior.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "dbfile",
        type=str,
        help="The HII region catalog database filename. If 'synthetic', generate synthetic data.",
    )
    PARSER.add_argument(
        "outfile",
        type=str,
        help="Where the MCMC model and trace are stored (.pkl extension)",
    )
    PARSER.add_argument(
        "loglike_net",
        type=str,
        help="Where the likelihood neural network is stored (.pkl extension)",
    )
    PARSER.add_argument(
        "-n",
        "--niter",
        type=int,
        default=_NITER,
        help="Maximum number of MCMC iterations",
    )
    PARSER.add_argument(
        "--num_data",
        type=int,
        default=_NUM_DATA,
        help="Number of synthetic data to generate.",
    )
    PARSER.add_argument(
        "--spiral_params",
        nargs="+",
        type=float,
        default=_spiral_params,
        help="Spiral parameters for synthetic data",
    )
    PARSER.add_argument(
        "--grm_params",
        nargs="+",
        type=float,
        default=_grm_params,
        help="GRM parameters for synthetic data",
    )
    PARSER.add_argument(
        "--warp_params",
        nargs="+",
        type=float,
        default=_warp_params,
        help="Warp parameters for synthetic data",
    )
    PARSER.add_argument(
        "--disk_params",
        nargs="+",
        type=float,
        default=_disk_params,
        help="Exponential disk parameters for synthetic data",
    )
    PARSER.add_argument("--Rmin", type=float, default=_RMIN, help="Minimum Galactocentric radius (kpc)")
    PARSER.add_argument("--Rmax", type=float, default=_RMAX, help="Maximum Galactocentric radius (kpc)")
    PARSER.add_argument(
        "--Rref",
        type=float,
        default=_RREF,
        help="Reference Galactocentric radius (kpc)",
    )
    PARSER.add_argument("--num_spirals", type=int, default=_NUM_SPIRALS, help="Number of spiral arms")
    DEFAULT_PRIORS = [
        ["q", "dirichlet"],
        ["az0", "uniform", 0.0, 1.5, 1.5, 3.2, 3.2, 4.7, 4.7, 6.3],
        ["pitch", "uniform", 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4],
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
        help=("Fixed GRM parameter names followed by their fixed value."),
    )
    PARSER.add_argument(
        "-o",
        "--outliers",
        nargs="+",
        default=_OUTLIERS,
        help="HII regions to exclude from analysis",
    )
    PARSER.add_argument("--chains", type=int, default=_NUM_CHAINS, help="Number of Markov chains")
    PARSER.add_argument("--ntune", type=int, default=_NTUNE, help="Number of MCMC tuning iterations")
    PARSER.add_argument("--ninit", type=int, default=_NINIT, help="Number of ADVI initialzation samples")
    PARSER.add_argument(
        "--target_accept",
        type=float,
        default=_TARGET_ACCEPT,
        help="Desired acceptance rate.",
    )
    PARSER.add_argument("--overwrite", action="store_true", help="Overwrite existing outfile")
    ARGS = vars(PARSER.parse_args())

    # Generate priors dictionary
    PARAMS = [
        "q",
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
                FIXED[PARAM] = [float(v) for v in FIX[1:]]
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
        ARGS["dbfile"],
        ARGS["outfile"],
        PRIORS,
        ARGS["loglike_net"],
        num_data=ARGS["num_data"],
        spiral_params=ARGS["spiral_params"],
        grm_params=ARGS["grm_params"],
        warp_params=ARGS["warp_params"],
        disk_params=ARGS["disk_params"],
        Rmin=ARGS["Rmin"],
        Rmax=ARGS["Rmax"],
        Rref=ARGS["Rref"],
        num_spirals=ARGS["num_spirals"],
        niter=ARGS["niter"],
        ntune=ARGS["ntune"],
        ninit=ARGS["ninit"],
        num_chains=ARGS["chains"],
        target_accept=ARGS["target_accept"],
        fixed=FIXED,
        outliers=ARGS["outliers"],
        overwrite=ARGS["overwrite"],
    )

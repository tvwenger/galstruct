#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_neural_nets.py

Test neural networks using simulated data.

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
Trey Wenger - December 2024 - updates
"""

import pickle
import argparse

import torch as tt
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp

from galstruct.torch_prior import Prior
from galstruct.model.simulator import simulator
from galstruct.model.likelihood import log_like

# parameter order for likelihood function
_params = [
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
    net_fname,
    num_sims=10000,
    Rmin=1.0,
    Rmax=25.0,
    Rref=8.0,
    fixed={},
    disk=[35.0, 4.5, 2.75],
):
    """
    Test neural network by computing the likelihood probability
    of simulated datasets drawn from the model to the likelihood
    predicted by the network.

    Inputs:
      net_fname :: string
        Directory containing neural networks
      num_sims :: integer
        Number of synthetic data to draw
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      fixed :: dictionary
        Fixed parameters
      disk :: list of scalars
        If None, do not apply an exponential disk
        Otherwise, contains the three exponential disk parameters
        [I2, Rs, Rc].

    Returns: Nothing
    """
    # Load neural network
    with open(net_fname, "rb") as f:
        net = pickle.load(f)

    # Generate prior
    prior = Prior(net["priors"])

    # Sample prior
    print("Sampling prior...")
    theta = prior.sample((num_sims,))

    # Simulate
    print("Simulating...")
    x = simulator(
        theta,
        Rmin=tt.tensor(Rmin),
        Rmax=tt.tensor(Rmax),
        Rref=tt.tensor(Rref),
        fixed=fixed,
        disk=disk,
    )
    isnan = tt.any(tt.isnan(x), axis=1)
    print(f"Dropping {isnan.sum()} simulations with NaNs")
    x = x[~isnan]
    theta = theta[~isnan]

    # pack fixed values into theta
    idx = 0
    all_theta = []
    for param in _params:
        if param in fixed:
            all_theta.append([fixed[param]] * num_sims)
        else:
            all_theta.append(list(theta[:, idx].numpy()))
            idx += 1
    all_theta = tt.tensor(np.array(all_theta))

    # Evaluate "true" log-prob
    logp_true = np.array(
        [
            log_like(
                dat[None, :],
                thet,
                Rmin=tt.tensor(Rmin),
                Rmax=tt.tensor(Rmax),
                Rref=tt.tensor(Rref),
                az_bins=10000,
            )
            .detach()
            .numpy()[0]
            for dat, thet in zip(x, all_theta.T)
        ]
    )

    # Evaluate network log-prob
    logp_net = net["density_estimator"].log_prob(x[None, :, :], theta)[0]
    logp_net = logp_net.detach().numpy()

    # drop nans
    print(f"{np.sum(np.isnan(logp_true))} NaN true log-prob")
    print(f"{np.sum(np.isnan(logp_net))} NaN network log-prob")
    isnan = np.isnan(logp_true) + np.isnan(logp_net)
    logp_true = logp_true[~isnan]
    logp_net = logp_net[~isnan]

    # offset median
    logp_true = logp_true - np.median(logp_true)
    logp_net = logp_net - np.median(logp_net)

    # Calculate statistic
    stat = anderson_ksamp([logp_true, logp_net])

    fig, axes = plt.subplots(
        2, layout="constrained", figsize=(6, 8), sharex=True, sharey=True
    )
    axes[0].ecdf(logp_true, color="k", label="True")
    axes[0].ecdf(logp_net, color="r", label="Network")
    axes[0].set_ylabel("CDF")
    axes[0].set_xlim(-10.0, 10.0)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(loc="upper left")
    axes[0].text(
        0.6,
        0.1,
        f"Simulated\n AD = {stat.statistic:.3f}\n p = {stat.pvalue:.3f}",
        ha="left",
        va="bottom",
        transform=axes[0].transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Now sample from neural net and compare
    print("Sampling...")
    x = net["density_estimator"].sample((1,), theta)[0]

    # Evaluate "true" log-prob
    logp_true = np.array(
        [
            log_like(
                dat[None, :],
                thet,
                Rmin=tt.tensor(Rmin),
                Rmax=tt.tensor(Rmax),
                Rref=tt.tensor(Rref),
                az_bins=10000,
            )
            .detach()
            .numpy()[0]
            for dat, thet in zip(x, all_theta.T)
        ]
    )

    # Evaluate network log-prob
    logp_net = net["density_estimator"].log_prob(x[None, :, :], theta)[0]
    logp_net = logp_net.detach().numpy()

    # drop nans
    print(f"{np.sum(np.isnan(logp_true))} NaN true log-prob")
    print(f"{np.sum(np.isnan(logp_net))} NaN network log-prob")
    isnan = np.isnan(logp_true) + np.isnan(logp_net)
    logp_true = logp_true[~isnan]
    logp_net = logp_net[~isnan]

    # offset medians
    logp_true = logp_true - np.median(logp_true)
    logp_net = logp_net - np.median(logp_net)

    # Calculate statistic
    stat = anderson_ksamp([logp_true, logp_net])

    # add to plot
    axes[1].ecdf(logp_true, color="k")
    axes[1].ecdf(logp_net, color="r")
    axes[1].set_xlabel(r"$\log L + C$")
    axes[1].set_ylabel("CDF")
    axes[1].text(
        0.6,
        0.1,
        f"Sampled\n AD = {stat.statistic:.3f}\n p = {stat.pvalue:.3f}",
        ha="left",
        va="bottom",
        transform=axes[1].transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    fname = net_fname.replace(".pkl", ".pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Check likelihood neural net against simulated data",
        prog="test_neural_net.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("net", type=str, help="Neural network pickle filename")
    PARSER.add_argument(
        "--outdir", type=str, default="frames", help="Directory where images are saved"
    )
    PARSER.add_argument(
        "--fixed",
        action="append",
        nargs=2,
        default=[],
        help=(
            "Fixed parameter names followed by their fixed value "
            + "(e.g., --fixed R0 8.5 --fixed Usun 10.5)"
        ),
    )
    ARGS = vars(PARSER.parse_args())
    FIXED = {}
    for PARAM in _params:
        for FIX in ARGS["fixed"]:
            if FIX[0] == PARAM:
                FIXED[PARAM] = float(FIX[1])
    main(ARGS["net"], fixed=FIXED)

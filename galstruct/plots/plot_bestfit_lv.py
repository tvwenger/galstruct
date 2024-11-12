#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_bestfit.py

Plots best fit model from MCMC simulation versus the default model. 

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
import matplotlib.pyplot as plt
import torch

from galstruct.model.simulator import simulator
from galstruct.model.likelihood import log_like
from galstruct.model.model import model
from galstruct.model.rotcurve import rotcurve_constants, calc_theta

# parameter order for likelihood function
_params = [
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


# values held as constant for each parameter
_THETA = [
    0.25, # pitch
    5.0, # sigmaV
    0.5,
    0.1,
    8.166,
    10.444,
    12.007,
    7.719,
    5.793,
    -3.562,
    0.978,
    1.623,
    5.5,
    0.0,
    0.0399,
    -0.5,
]
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
_default_q=[0.25,0.25,0.25,0.25]
_default_az0=[np.deg2rad(72.0), 
              np.deg2rad(137.0),
              np.deg2rad(252.0),
              np.deg2rad(317.0)]
_default_pitch=[np.deg2rad(14.0), 
                np.deg2rad(14.0),
                np.deg2rad(14.0),
                np.deg2rad(14.0)]
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

def main(
    mcmc_outfile,
    theta=_THETA,
    outdir="plots",
    Rmin=3.0,
    Rmax=15.0,
    Rref=8.0,
    num_data=1000
):
    """
    Inputs:
      mcmc_outfile : path to mcmc pickle output

    Returns: Nothing
    """
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # keep only free params for likelihood theta
    params={}
    for param, value in zip(_params, theta):
        params[param]=value
    
    az_bins = 100
    # Load neural network
    with open(mcmc_outfile, "rb") as f:
        trace = pickle.load(f)

    num_chains=trace['trace'].sample_stats.lp.shape[0]
    # find index of mcmc sample with best log-likelihood
    idx=np.argmax(np.concatenate(
        np.array([trace['trace'].sample_stats.lp.isel(chain=i).to_numpy() \
                  for i in range(num_chains)])))
    # find parameters at that index (averaged over chains)
    bf_params1 = np.array(trace['trace'].posterior.isel(draw=idx).mean().to_array())
    bf_params_az0s = np.array(trace['trace'].posterior.az0.isel(draw=idx).sum('chain')/num_chains)
    bf_params_pitch = np.array(trace['trace'].posterior.pitch.isel(draw=idx).sum('chain')/num_chains)
    bf_params_qs = np.array(trace['trace'].posterior.q.isel(draw=idx).sum('chain')/num_chains)
    bf_paramnames = np.array(list(trace['trace'].posterior.isel(draw=idx).mean().keys()))
    bf_params = dict(zip(bf_paramnames,bf_params1))

    fig, ax = plt.subplots(figsize=(8,12))    
    
    # Get synthetic data
    num_spirals = len(_default_az0)
    q = _spiral_params[:num_spirals]
    thetas = [
        torch.as_tensor(
            [_spiral_params[num_spirals + i]]
            + [_spiral_params[2 * num_spirals + i]]
            + _spiral_params[3 * num_spirals :]
            + _grm_params
            + _warp_params
        )
        for i in range(num_spirals)
    ]
    # Data formatted as (glong, glat, vlsr)
    data = torch.cat(
        tuple(
            simulator(
                theta.expand(int(qi * num_data), -1),
                Rmin=torch.tensor(Rmin),
                Rmax=torch.tensor(Rmax),
                Rref=torch.tensor(Rref),
                disk=_disk_params,
            )
            for theta, qi in zip(thetas, q)
        )
    ).float()
    
    ax.scatter(data[:,2],np.rad2deg(data[:,0]),alpha=0.3)
    # DEFAULT PARAMS
    for i in range(bf_params_az0s.size):
        
        min_az = _default_az0[i] - np.log(Rmax / Rref) / np.tan(_default_pitch[i])
        max_az = _default_az0[i] - np.log(Rmin / Rref) / np.tan(_default_pitch[i])
        az = np.linspace(min_az, max_az, az_bins)

        tilt = np.arcsin(params["Zsun"] / params["R0"] / 1000.0)
        cos_tilt, sin_tilt = np.cos(tilt), np.sin(tilt)
        cos_roll, sin_roll = np.cos(params["roll"]), np.sin(params["roll"])
        R0a22, lam, loglam, term1, term2 = rotcurve_constants(
            torch.tensor([params["R0"]]), torch.tensor([params["a2"]]), torch.tensor([params["a3"]])
        )
        theta0 = calc_theta(params["R0"], R0a22, lam, loglam, term1, term2)
        
        glong, glat, vlsr, dvlsr_ddist, dist = model(
            torch.tensor(az),
            _default_az0[i],
            _default_pitch[i],
            params["R0"],
            params["Usun"],
            params["Vsun"],
            params["Wsun"],
            params["Upec"],
            params["Vpec"],
            cos_tilt,
            sin_tilt,
            cos_roll,
            sin_roll,
            R0a22,
            lam,
            loglam,
            term1,
            term2,
            theta0,
            params["warp_amp"],
            params["warp_off"],
            Rref=torch.tensor([Rref]),
        )
        
        diff_glong = np.diff(glong.detach().numpy())
        diff_vlsr = np.diff(vlsr.detach().numpy())
        idx = np.where(diff_glong>=np.pi/2)[0]+1
        glong=np.insert(glong.detach().numpy(),idx,np.nan)
        vlsr=np.insert(vlsr.detach().numpy(),idx,np.nan)
        

        bf_min_az = bf_params_az0s[i]- np.log(Rmax / Rref) / np.tan(bf_params_pitch[i])
        bf_max_az = bf_params_az0s[i] - np.log(Rmin / Rref) / np.tan(bf_params_pitch[i])
        bf_az = np.linspace(bf_min_az, bf_max_az, az_bins)

        bf_tilt = np.arcsin(bf_params["Zsun"] / bf_params["R0"] / 1000.0)
        bf_cos_tilt, bf_sin_tilt = np.cos(bf_tilt), np.sin(bf_tilt)
        bf_cos_roll, bf_sin_roll = np.cos(bf_params["roll"]), np.sin(bf_params["roll"])
        bf_R0a22, bf_lam, bf_loglam, bf_term1, bf_term2 = rotcurve_constants(
            torch.tensor([bf_params["R0"]]), torch.tensor([bf_params["a2"]]), torch.tensor([bf_params["a3"]])
        )
        bf_theta0 = calc_theta(bf_params["R0"], bf_R0a22, bf_lam, bf_loglam, bf_term1, bf_term2)
        
        bf_glong, bf_glat, bf_vlsr, bf_dvlsr_ddist, bf_dist = model(
            torch.tensor(bf_az),
            bf_params_az0s[i],
            bf_params_pitch[i],
            bf_params["R0"],
            bf_params["Usun"],
            bf_params["Vsun"],
            bf_params["Wsun"],
            bf_params["Upec"],
            bf_params["Vpec"],
            bf_cos_tilt,
            bf_sin_tilt,
            bf_cos_roll,
            bf_sin_roll,
            bf_R0a22,
            bf_lam,
            bf_loglam,
            bf_term1,
            bf_term2,
            bf_theta0,
            bf_params["warp_amp"],
            bf_params["warp_off"],
            Rref=Rref,
        )
        
        
        bf_diff_glong = np.diff(bf_glong.detach().numpy())
        bf_diff_vlsr = np.diff(bf_vlsr.detach().numpy())
        bf_idx = np.where(bf_diff_glong>=np.pi/2)[0]+1
        bf_glong=np.insert(bf_glong.detach().numpy(),bf_idx,np.nan)
        bf_vlsr=np.insert(bf_vlsr.detach().numpy(),bf_idx,np.nan)
        
        ax.plot(vlsr,np.rad2deg(glong),
                linestyle='dashed',
                linewidth=2.5,
                label='Default Parameters, Arm {}'.format(i))
        ax.plot(bf_vlsr,np.rad2deg(bf_glong), 
                linestyle='dashed',
                linewidth=2.5,
                label='Best Fit Parameters, Arm {}'.format(i))
    ax.set_ylabel("Longitude (deg)")
    ax.set_xlabel("VLSR (km/s)")
    plt.legend()
    ax.grid(True)
    
    fname = os.path.join(outdir, "bestfit.png")
    fig.savefig(fname, dpi=80,bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Plot likelihood data",
        prog="plot_likelihoods.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("net", type=str, help="Neural network pickle filename")
    PARSER.add_argument(
        "--outdir", type=str, default="plots", help="Directory where images are saved"
    )

    ARGS = vars(PARSER.parse_args())
    main(ARGS["net"], outdir=ARGS["outdir"])

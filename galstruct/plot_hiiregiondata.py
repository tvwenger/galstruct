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
import sqlite3
from astropy.io import fits

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
    
    # Load in data from csv files
    Catalog = pandas.read_csv("data/Catalog.csv",header=None).to_numpy(str)
    CatalogGroups = pandas.read_csv('data/CatalogGroups.csv',header=None).to_numpy(str)
    Groups = pandas.read_csv("data/Groups.csv",header=None).to_numpy(str)
    
    # put the vlsrs into a dictionary (key = group ID number)
    vlsrs = Groups[:,2].astype(float)
    group_ids = Groups[:,0].astype(int)
    group_dict = dict(zip(group_ids,vlsrs))
    
    # put the longitudes into a dictionary (key = category ID number)
    glongs = Catalog[:,7].astype(float)
    cat_id = Catalog[:,0].astype(int)
    glong_dict = dict(zip(cat_id,glongs))
    
    # empty array to put data in
    data=np.zeros((2,Groups[:,2].shape[0]))
    
    # parts of data to cut out
    cut_glong = [-10,10]
    cut_vlsr = [-5,5]
    
    for j,k in CatalogGroups:
        j=int(j)
        k=int(k)
        gl = glong_dict[j]
        
        # convert longitude from [0,360] angle to [-180,180] angle
        if gl<180:
            # check if longitude meets cutoff, otherwise set equal to nan
            if gl > cut_glong[1]:
                data[0,k-1] = gl
            else:
                data[0,k-1] = np.nan
        elif gl>=180:
            if gl-360 < cut_glong[0]:
                data[0,k-1] = gl-360
            else:
                data[0,k-1] = np.nan
        
        vlsr = float(group_dict[k])
        # check if vlsr meets cutoff otherwise set equal to nan
        if cut_vlsr[0] < vlsr and cut_vlsr[1] > vlsr:
            data[1,k-1] = np.nan
        else:
            data[1,k-1] = vlsr

    # keep only free params for likelihood theta
    # params={}
    # for param, value in zip(_params, theta):
    #     params[param]=value
    
    #az_bins = 100
    
    fig, ax = plt.subplots(figsize=(8,12))    
    ax.scatter(data[1,:],data[0,:],alpha=0.3)
    ax.vlines(cut_vlsr,ymin=[-180,-180],ymax=[180,180])
    ax.hlines(cut_glong,xmin=[-120,-120],xmax=[120,120])
    
    # DEFAULT PARAMS
    # i=3
    # az0s=np.linspace(_default_az0[i]-0.5,_default_az0[i]+0.5,5)
    # for az0 in az0s:
    #     min_az = az0 - np.log(Rmax / Rref) / np.tan(_default_pitch[i])
    #     max_az = az0 - np.log(Rmin / Rref) / np.tan(_default_pitch[i])
    #     az = np.linspace(min_az, max_az, az_bins)

    #     tilt = np.arcsin(params["Zsun"] / params["R0"] / 1000.0)
    #     cos_tilt, sin_tilt = np.cos(tilt), np.sin(tilt)
    #     cos_roll, sin_roll = np.cos(params["roll"]), np.sin(params["roll"])
    #     R0a22, lam, loglam, term1, term2 = rotcurve_constants(
    #         torch.tensor([params["R0"]]), torch.tensor([params["a2"]]), torch.tensor([params["a3"]])
    #     )
    #     theta0 = calc_theta(params["R0"], R0a22, lam, loglam, term1, term2)
        
    #     glong, glat, vlsr, dvlsr_ddist, dist = model(
    #         torch.tensor(az),
    #         az0,
    #         _default_pitch[i],
    #         params["R0"],
    #         params["Usun"],
    #         params["Vsun"],
    #         params["Wsun"],
    #         params["Upec"],
    #         params["Vpec"],
    #         cos_tilt,
    #         sin_tilt,
    #         cos_roll,
    #         sin_roll,
    #         R0a22,
    #         lam,
    #         loglam,
    #         term1,
    #         term2,
    #         theta0,
    #         params["warp_amp"],
    #         params["warp_off"],
    #         Rref=torch.tensor([Rref]),
    #     )
        
    #     diff_glong = np.diff(glong.detach().numpy())
    #     diff_vlsr = np.diff(vlsr.detach().numpy())
    #     idx = np.where(diff_glong>=np.pi/2)[0]+1
    #     glong=np.insert(glong.detach().numpy(),idx,np.nan)
    #     vlsr=np.insert(vlsr.detach().numpy(),idx,np.nan)
        

    #     ax.plot(vlsr,np.rad2deg(glong),
    #             linestyle='dashed',
    #             linewidth=2.5,
    #             label='az0={}, Arm {}'.format(az0,i))

    ax.set_ylabel("Longitude (deg)")
    ax.set_xlabel("VLSR (km/s)")
    plt.legend()
    ax.grid(True)
    
    fname = os.path.join(outdir, "hii_region_data.png")
    fig.savefig(fname, dpi=80,bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Plot likelihood data",
        prog="plot_likelihoods.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "--outdir", type=str, default="plots", help="Directory where images are saved"
    )

    ARGS = vars(PARSER.parse_args())
    main(outdir=ARGS["outdir"])

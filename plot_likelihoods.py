#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_likelihoods.py

Generate plots of simulated data, data sampled from the learned
likelihood, and a grid of the learned likelihood.

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

import pickle
from torch_prior import Prior
import matplotlib.pyplot as plt
from model.simulator import simulator
import numpy as np
import torch

_THETA = [0.25, 5.0, 0.5, 0.1, 8.166, 10.444, 12.007, 7.719, 5.793,
          -3.562, 0.978, 1.623, 0.0399, -0.5]

def main(net_fname, num_data=500, theta=_THETA, outdir='frames',
         Rmin=3.0, Rmax=15.0, Rref=8.0):
    """
    Generate plots of simulated data, data sampled from the learned
    likelihood, and a grid of the learned likelihood for a given
    set of parameters and varying reference azimuth. The simulated
    data are saved with the prefix 'sim_', the sampled data with
    'sample_', and the grid with 'grid_'.

    Inputs:
      net_fname :: string
        Filename of the neural network pickle file
      num_data :: integer
        Number of simulated and sampled data to draw
      theta :: list of scalars
        Model parameters held fixed:
        pitch, sigmaV, sigma_arm_plane, sigma_arm_height, R0, Usun,
        Vsun, Wsun, Upec, Vpec, a2, a3, Zsun, roll, warp_amp, warp_off
      outdir :: string
        Directory where images are saved
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth

    Returns: Nothing
    """
    #
    # add az0 placeholder to theta. Get range of az0
    #
    theta = torch.tensor([0.0]+theta)
    az0s_deg = np.linspace(0.0, 359.0, 360)
    az0s = np.deg2rad(az0s_deg)
    #
    # Load neural network
    #
    with open(net_fname, 'rb') as f:
        net = pickle.load(f)
    #
    # Loop over azimuth and generate simulated data
    #
    for i, (az0_deg, az0) in enumerate(zip(az0s_deg, az0s)):
        theta[0] = az0
        data = simulator(
            theta.expand(num_data, -1),
            Rmin=torch.tensor(Rmin), Rmax=torch.tensor(Rmax),
            Rref=torch.tensor(Rref), disk=None, fixed=fixed)
        data = data.numpy()
        fig, ax = plt.subplots()
        cax = ax.scatter(
            data[:, 2], np.rad2deg(data[:, 0]), marker='.',
            c=np.rad2deg(data[:, 1]), vmin=-5.0, vmax=5.0)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_label('Latitude (deg)')
        ax.set_xlabel('VLSR (km/s)')
        ax.set_ylabel('Longitude (deg)')
        ax.set_xlim(-150.0, 150.0)
        ax.set_ylim(-180.0, 180.0)
        ax.set_title('Azimuth = {0:.1f} deg'.format(az0_deg))
        fig.tight_layout()
        fname = os.path.join(outdir, 'sim_{0:03d}.png'.format(i))
        fig.savefig(fname, dpi=100)
        plt.close(fig)
    #
    # Loop over azimuth and generate sampled data
    #
    for i, (az0_deg, az0) in enumerate(zip(az0s_deg, az0s)):
        theta[0] = az0
        data = net['posterior'].net.sample(num_data, context=theta[None])[0]
        data = data.detach().numpy()
        fig, ax = plt.subplots()
        cax = ax.scatter(
            data[:, 2], np.rad2deg(data[:, 0]), marker='.',
            c=np.rad2deg(data[:, 1]), vmin=-5.0, vmax=5.0)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_label('Latitude (deg)')
        ax.set_xlabel('VLSR (km/s)')
        ax.set_ylabel('Longitude (deg)')
        ax.set_xlim(-150.0, 150.0)
        ax.set_ylim(-180.0, 180.0)
        ax.set_title('Azimuth = {0:.1f} deg'.format(az0_deg))
        fig.tight_layout()
        fname = os.path.join(outdir, 'sample_{0:03d}.png'.format(i))
        fig.savefig(fname, dpi=100)
        plt.close(fig)
    #
    # Loop over azimuth and generate grid data
    #
    glong_axis = np.linspace(-np.pi, np.pi, 180)
    vlsr_axis = np.linspace(-150.0, 150.0, 150)
    glong_grid, vlsr_grid = np.meshgrid(glong_axis, vlsr_axis, indexing='ij')
    glong = glong_grid.flatten()
    vlsr = vlsr_grid.flatten()
    glat = np.zeros(len(glong))
    extent = [-150.0, 150.0, -180.0, 180.0]
    data = np.stack((glong, glat, vlsr)).T
    data = torch.tensor(data).float()
    for i, (az0_deg, az0) in enumerate(zip(az0s_deg, az0s)):
        theta[0] = az0
        logp = net['posterior'].net.log_prob(data, context=theta[None])
        logp = logp.detach().numpy()
        fig, ax = plt.subplots()
        cax = ax.imshow(
            logp, extent=extent, origin='lower', interpolation='none')
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_label('log $L$ ($b = 0^\circ$)')
        ax.set_xlabel('VLSR (km/s)')
        ax.set_ylabel('Longitude (deg)')
        ax.set_xlim(-150.0, 150.0)
        ax.set_ylim(-180.0, 180.0)
        ax.set_title('Azimuth = {0:.1f} deg'.format(az0_deg))
        fig.tight_layout()
        fname = os.path.join(outdir, 'grid_{0:03d}.png'.format(i))
        fig.savefig(fname, dpi=100)
        plt.close(fig)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Plot likelihood data",
        prog="plot_likelihoods.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PARSER.add_argument(
        "net", type=str,
        help="Neural network pickle filename")
    PARSER.add_arugment(
        "--outdir", type=str, default="frames",
        help="Directory where images are saved")
    ARGS = vars(PARSER.parse_args())
    main(ARGS['net'], outdir=ARGS['outdir'])

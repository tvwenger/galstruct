#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_likelihood_grad.py

Compute likelihood and likelihood gradient for a range of each model
parameter, while holding the other model parameters constant.

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

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

import torch
import theano
import theano.tensor as tt

from galstruct.mcmc_posterior_old import Loglike
from galstruct.model.simulator import simulator
from galstruct.torch_prior import Prior

# set random seed
np.random.seed(1234)

def main(net_fname, out_prefix='likelihood_'):
    """
    Plot likelihood and gradient for a range of parameters, holding
    the other parameters constant. Plots are saved to 
    out_prefix+'like.pdf' and out_prefix+'like_grad.pdf'
    
    Inputs:
      net_fname :: string
        File containing neural network

      out_prefix :: string
        Prefix for saved plots

    Returns:
      Nothing
    """
    #
    # Spiral parameters
    #
    num_spirals = 4
    q = np.array([0.3, 0.2, 0.25, 0.25])
    az0 = np.array([1.4, 2.2, 4.0, 5.5])
    pitch = np.array([0.244, 0.244, 0.244, 0.244])
    sigmaV = 5.0
    sigma_arm_plane = 0.5
    sigma_arm_height = 0.1
    #
    # GRM parameters
    #
    R0 = 8.16643777
    Usun = 10.4543041
    Vsun = 12.18499493
    Wsun = 7.71886874
    Upec = 5.79095823
    Vpec = -3.39171583
    a2 = 0.97757558
    a3 = 1.62261724
    Zsun = 5.5
    roll = 0.0
    #
    # Warp parameters
    #
    warp_amp = 0.02
    warp_off = -0.5
    #
    # Generate synthetic data
    #
    num_data = 2000
    Rmin = 3.0
    Rmax = 15.0
    Rref = 8.0
    thetas = [
        torch.as_tensor([
            az0[i], pitch[i], sigmaV, sigma_arm_plane, sigma_arm_height,
            R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3, Zsun, roll, warp_amp, warp_off])
        for i in range(num_spirals)]
    data = torch.cat(tuple(
        simulator(
            theta.expand(int(qi*num_data), -1),
            Rmin=torch.tensor(Rmin), Rmax=torch.tensor(Rmax),
            Rref=torch.tensor(Rref), disk=[35.0, 3.0, 2.5])
        for theta, qi in zip(thetas, q)))
    #
    # Model parameters
    #
    with open(net_fname, 'rb') as f:
        net = pickle.load(f)
        net = net['posterior'].net
    loglike = Loglike(net, data.float(), num_spirals)
    theta = tt.dvector('theta')
    calc_loglike = theano.function(
        [theta], loglike(theta))
    dloglike = tt.grad(loglike(theta), theta)
    calc_dloglike = theano.function([theta], dloglike)
    #
    # Ranges of free parameters
    #
    nominal_values = []
    ranges = []
    labels = []
    nominal_values += list(q)
    ranges += ["normalize" for _ in range(num_spirals)]
    labels += ['q{0}'.format(i) for i in range(num_spirals)]
    nominal_values += list(az0)
    ranges += [(-np.deg2rad(30.0), np.deg2rad(30.0)) for _ in range(num_spirals)]
    labels += ['az0{0}'.format(i) for i in range(num_spirals)]
    nominal_values += list(pitch)
    ranges += [(-np.deg2rad(5.0), np.deg2rad(5.0)) for _ in range(num_spirals)]
    labels += ['pitch{0}'.format(i) for i in range(num_spirals)]
    nominal_values += [sigmaV, sigma_arm_plane, sigma_arm_height]
    ranges += [(-5.0, 5.0), (-0.3, 0.5), (-0.05, 0.1)]
    labels += ['sigmaV', 'armplane', 'armheight']
    nominal_values += [
        R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3]
    ranges += [
        (-1.0, 1.0), (-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0),
        (-15.0, 15.0), (-15.0, 15.0), (-0.1, 0.1), (-0.1, 0.1)]
    labels += ['R0', 'Usun', 'Vsun', 'Wsun', 'Upec', 'Vpec', 'a2',
               'a3']
    nominal_values += [warp_amp, warp_off]
    ranges += [(-0.02, 0.05), (-0.5, 0.5)]
    labels += ['warpamp', 'warpoff']
    #
    # Loop over ranges, plot likelihood vs. parameter
    #
    fig_loglike, axes_loglike = plt.subplots(len(ranges), figsize=(8, 2*len(ranges)))
    fig_dloglike, axes_dloglike = plt.subplots(len(ranges), figsize=(8, 2*len(ranges)))
    num_points = 20
    for i, ran in enumerate(ranges):
        print(i, labels[i])
        values = nominal_values.copy()
        lnlike_vals = np.zeros(num_points)
        lnlike_ders = np.zeros(num_points)
        if ran == "normalize":
            r = np.linspace(0.1, 0.9, num_points)
        else:
            r = np.linspace(ran[0], ran[1], num_points)
        for j, update in enumerate(r):
            values[i] = nominal_values[i] + update
            # normalize q as necessary
            if i < num_spirals:
                # keep changed value the same
                values[i] = update
                # normalize to sum 1, but keep relative size of the
                # other arms the same
                # i.e. 0.3, 0.5, 0.2 --> 0.8, 0.5, 0.2 is update at myqi = 0
                #      0.8, 0.143, 0.057 keeps 1, 2 at same relative level
                total = 0
                for k in range(num_spirals):
                    if k == i:
                        continue
                    total += values[k]
                for k in range(num_spirals):
                    if k == i:
                        continue
                    values[k] = (1.0-update)*values[k]/total
            #
            # Evaluate likelihood and gradient
            #
            lnlike = calc_loglike(values)
            dlnlike = calc_dloglike(values)[i]
            #
            # Save. Get correct term from derivative
            #
            lnlike_vals[j] = float(lnlike)
            lnlike_ders[j] = float(dlnlike)
        #
        # Plot
        #
        if ranges[i] == 'normalize':
            xdata = r
            xtrue = nominal_values[i]
        else:
            xdata = nominal_values[i] + r
            xtrue = nominal_values[i]
        axes_loglike[i].plot(xdata, lnlike_vals, 'k-')
        axes_loglike[i].axvline(xtrue, color='b')
        axes_loglike[i].set_xlim(xdata.min(), xdata.max())
        axes_loglike[i].set_xlabel(labels[i])
        axes_loglike[i].set_ylabel('p')
        axes_dloglike[i].plot(xdata, lnlike_ders, 'k-')
        axes_dloglike[i].axvline(xtrue, color='b')
        axes_dloglike[i].set_xlim(xdata.min(), xdata.max())
        axes_dloglike[i].set_xlabel(labels[i])
        axes_dloglike[i].set_ylabel('dp')        
    fig_loglike.tight_layout()
    fig_loglike.savefig(out_prefix+'like.pdf')
    plt.close(fig_loglike)
    fig_dloglike.tight_layout()
    fig_dloglike.savefig(out_prefix+'like_grad.pdf')
    plt.close(fig_dloglike)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Plot likelihood and gradient",
        prog="test_likelihood_grad.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PARSER.add_argument(
        "net", type=str,
        help="Neural network pickle")
    PARSER.add_argument(
        "--prefix", type=str, default='likelihood_',
        help="Prefix for saved plots")
    ARGS = vars(PARSER.parse_args())
    main(ARGS['net'], out_prefix=ARGS['prefix'])

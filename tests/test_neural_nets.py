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
"""

import os
import pickle
import glob
import argparse

import torch
import numpy as np

from model.simulator import simulator
from torch_prior import Prior


def main(netdir, num_data=1000, num_sims=100, Rmin=3.0, Rmax=15.0,
         Rref=8.0, disk_params=[35.0, 3.0, 2.5]):
    """
    Test neural networks by computing the likelihood probability
    of simulated datasets drawn from the model, which should have
    a high likelihood, and random datasets, which should have a low
    likelihood.

    Inputs:
      netdir :: string
        Directory containing neural networks
      num_data :: integer
        Number of synthetic data to draw per simulation
      num_sims :: integer
        Number of simulations
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      disk_params :: 1-D array of scalars
        Synthetic exponential disk parameters [I2, Rs, Rc]

    Returns: Nothing
    """
    #
    # Get neural networks
    #
    nets = glob.glob(os.path.join(netdir, '*.pkl'))
    nets.sort()
    for net in nets:
        print("Network: {0}".format(net))
        #
        # Load neural net
        #
        with open(net, 'rb') as f:
            net_data = pickle.load(f)
        #
        # Generate prior
        #
        prior = Prior(net_data['priors'])
        #
        # Loop over simulated data sims
        #
        log_prob = 0.0
        for _ in range(num_sims):
            #
            # Get a prior sample and simulated data
            #
            theta = prior.sample()
            data = simulator(
                theta.expand(num_data, -1), Rmin=torch.tensor(Rmin),
                Rmax=torch.tensor(Rmax), Rref=torch.tensor(Rmax),
                disk=disk_params)
            #
            # Compute and add likelihood prob
            #
            lp = net_data['posterior'].net.log_prob(
                data, context=theta.expand(num_data, -1))
            log_prob += np.sum(lp.detach().numpy())
        #
        # Compute average
        #
        log_prob = log_prob / num_sims
        print("Average log_prob of simulated data: {0:.1f}".format(log_prob))
        #
        # Loop over random data sims
        #
        log_prob = 0.0
        for _ in range(num_sims):
            #
            # Get a prior sample and random data
            #
            theta = prior.sample()
            data = torch.tensor(
                np.stack([
                    np.random.uniform(-np.pi, np.pi, num_data),
                    np.random.uniform(-0.1, 0.1, num_data),
                    np.random.uniform(-150.0, 150.0, num_data)]).T)
            #
            # Compute and add likelihood prob
            #
            lp = net_data['posterior'].net.log_prob(
                data.float(), context=theta.expand(num_data, -1))
            log_prob += np.sum(lp.detach().numpy())
        #
        # Compute average
        #
        log_prob = log_prob / num_sims
        print("Average log_prob of random data: {0:.1f}".format(log_prob))
        print()

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Check likelihood neural nets against simulated and random data",
        prog="test_neural_nets.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PARSER.add_argument(
        "--dir", type=str, default="nets",
        help="Directory containing neural network pickles")
    ARGS = vars(PARSER.parse_args())
    main(ARGS['dir'])

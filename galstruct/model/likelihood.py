#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
likelihood.py

Calculate the likelihood of some data given a model by maximizing
the likelihood over a range of possible azimuths.

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

import torch
import torch.distributions as tdist
from galstruct.model.model import Model


def log_like(
    data,
    theta,
    Rmin=torch.tensor(3.0),
    Rmax=torch.tensor(15.0),
    Rref=torch.tensor(8.0),
    fixed={},
    az_bins=1000,
):
    """
    Calculate the log likelihood of some data given some model
    parameters.

    Inputs:
      data :: 2-D array of scalars (shape num_data, 3)
        The longitude, latitude, velocity data
      theta :: 1-D array of scalars
        The model parameters. At most it should include the following
        in this order. Any of parameters in the fixed dictionary
        are taken from there instead.
            az0 :: spiral reference azimuth (radians)
            pitch :: spiral pitch angle (radians)
            sigmaV :: spiral FWHM velocity width (km/s)
            sigma_arm_plane :: spiral FWHM physical width in the plane (kpc)
            sigma_arm_height :: spiral FWHM physical width perpendicular to plane (kpc)
            R0 :: Galactocentric radius of Sun (kpc)
            Usun, Vsun, Wsun :: Solar motion components (km/s)
            Upec, Vpec :: Peculiar motion components (km/s)
            a2, a3 :: Rotation curve parameters
            Zsun :: Sun's height above Galactic plane (pc)
            roll :: Galactic plane roll angle (radians)
            warp_amp :: Warp mode amplitude (kpc-1)
            warp_off :: Warp mode offset (radians)
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      fixed :: dictionary
        Fixed parameters
      az_bins :: integer
        Number of azimuth bins to maximize over

    Returns: log_prob
      log_prob :: torch.tensor
        Log likelihood of each data point
    """
    # Unpack spiral parameters
    idx = 0
    params = {}
    param_names = [
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
    for name in param_names:
        if name in fixed:
            params[name] = torch.as_tensor(fixed[name])
        else:
            params[name] = theta[idx]
            idx += 1

    # Get azimuth bins
    min_az = params["az0"] - torch.log(Rmax / Rref) / torch.tan(params["pitch"])
    max_az = params["az0"] - torch.log(Rmin / Rref) / torch.tan(params["pitch"])
    az = torch.linspace(min_az, max_az, az_bins)

    # Evaluate the model
    model = Model(**params)
    model_data, model_sigma2_data = model.model_spread(az)

    # Get likelihood (shape len(data), num_az)
    print(data.shape)
    glong_prob = tdist.Normal(model_data[0], torch.sqrt(model_sigma2_data[0]))
    glat_prob = tdist.Normal(model_data[1], torch.sqrt(model_sigma2_data[1]))
    vlsr_prob = tdist.Normal(model_data[2], torch.sqrt(model_sigma2_data[2]))
    log_prob = glong_prob.log_prob(data[:, 0, None])
    log_prob += glat_prob.log_prob(data[:, 1, None])
    log_prob += vlsr_prob.log_prob(data[:, 2, None])

    # Max along azimuth axis
    log_prob = torch.max(log_prob, axis=1)[0]
    return log_prob

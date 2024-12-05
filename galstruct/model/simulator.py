#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulator.py

Generate simulated HII region data from the model with given
parameters.

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

import torch as tt
from galstruct.model.model import Model


def simulator(
    theta,
    Rmin=tt.tensor(3.0),
    Rmax=tt.tensor(15.0),
    Rref=tt.tensor(8.0),
    fixed={},
    disk=None,
):
    """
    Generate simulated HII region longitude, latitude, and velocity
    data from the model with some given parameters.

    Inputs:
      theta :: 2-D array of scalars (shape batch_size, num_parameters)
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
      disk :: list of scalars
        If None, do not apply an exponential disk
        Otherwise, contains the three exponential disk parameters
        [I2, Rs, Rc].

    Returns: [glong, glat, vlsr]
      [glong, glat, vlsr] :: tt.tensor
        glong, glat :: scalars (radians)
          Simulated Galactic longitude and latitude
        vlsr :: scalar (km/s)
          Simulated LSR velocity
    """
    theta = tt.atleast_2d(theta)
    num_data = theta.shape[0]

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
            params[name] = tt.as_tensor([fixed[name]] * num_data)
        else:
            params[name] = theta[:, idx]
            idx += 1

    # Get azimuth range, pick a random azimuth
    min_az = params["az0"] - tt.log(Rmax / Rref) / tt.tan(params["pitch"])
    max_az = params["az0"] - tt.log(Rmin / Rref) / tt.tan(params["pitch"])

    if disk is not None:
        # apply exponential disk
        spiral_az = tt.stack(tuple(tt.linspace(mina, maxa, 1000) for mina, maxa in zip(min_az, max_az)))
        I2, Rs, Rc = disk
        spiral_R = Rref * tt.exp((params["az0"][:, None] - spiral_az) * tt.tan(params["pitch"][:, None]))
        prob = tt.exp(-spiral_R / Rs) / (1.0 + I2 * tt.exp(-spiral_R / Rc))
        prob = prob / tt.sum(prob, axis=1, keepdims=True)
        idx = prob.multinomial(num_samples=1)
        az = tt.gather(spiral_az, 1, idx)[:, 0]
    else:
        az = (max_az - min_az) * tt.rand(num_data) + min_az

    # Get model longitude, latitude, velocity
    model = Model(
        # these are physical parameters of the model
        az0=params["az0"],
        R0=params["R0"],
        a2=params["a2"],
        a3=params["a3"],
        Rref=Rref,
        # these are the Sun's kinematic parameters in the model
        Usun=params["Usun"],
        Vsun=params["Vsun"],
        Wsun=params["Wsun"],
        Zsun=params["Zsun"],
        Upec=params["Upec"],
        Vpec=params["Vpec"],
        # angular parameters dictating the shape of the spiral arms
        pitch=params["pitch"],
        roll=params["roll"],
        warp_amp=params["warp_amp"],
        warp_off=params["warp_off"],
        # parameters defining the spread in the spiral arms
        sigma_arm_height=params["sigma_arm_height"],
        sigma_arm_plane=params["sigma_arm_plane"],
        sigmaV=params["sigmaV"],
    )

    # this is a call to our Model, it will return out the final glong, glat, and vlsr for a generated HII region
    data, _ = model.model_spread(az)

    return data

#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
model.py

Compute the separation between a log spiral at a given azimuth and
a given longitude, latitude, velocity point.

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

from . import rotcurve
from .model import transforms


def model_vlsr(
    dist,
    cos_glong,
    sin_glong,
    cos_glat,
    sin_glat,
    R0,
    Usun,
    Vsun,
    Wsun,
    Upec,
    Vpec,
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
):
    """
    Derive the model-predicted IAU-LSR velocity at a given position.

    Inputs:
      dist :: scalar (kpc)
        Distance
      cos_glong, sin_glong, cos_glat, sin_glat :: scalars
        Consine and sine of Galactic longitude and latitude
      R0 :: scalar (kpc)
        Solar Galactocentric radius
      Usun, Vsun, Wsun :: scalars (km/s)
        Solar peculiar motion components
      Upec, Vpec :: scalars (km/s)
        Peculiar motion components
      cos_tilt, sin_tilt, cos_roll, sin_roll :: scalars
        Cosine and sine of Galactic plane tilt and roll angles
      R0a22, lam, loglam, term1, term2 :: scalars
        The rotation curve constants
      theta0 :: scalar (km/s)
        Rotation speed at R0

    Returns: vlsr
      vlsr :: scalar (km/s)
        IAU-LSR velocity
    """
    # Convert distance to R, azimuth
    midplane_dist = dist * cos_glat
    R = tt.sqrt(R0 ** 2.0 + midplane_dist ** 2.0 - 2.0 * R0 * midplane_dist * cos_glong)
    cos_az = (R0 - midplane_dist * cos_glong) / R
    sin_az = midplane_dist * sin_glong / R

    # Calculate rotation speed at R
    theta = rotcurve.calc_theta(R, R0a22, lam, loglam, term1, term2)
    vR = -Upec
    vAz = theta + Vpec

    # Convert velocities to barycentric Cartesian frame
    vXg, vYg = transforms.v_gcencyl_to_gcencar(vR, vAz, cos_az, sin_az)
    vXb, vYb, vZb = transforms.v_gcencar_to_barycar(
        vXg, vYg, Usun, Vsun, Wsun, theta0, cos_tilt, sin_tilt, cos_roll, sin_roll
    )

    # Convert velocities to IAU-LSR radial velocity
    vlsr = transforms.calc_vlsr(cos_glong, sin_glong, cos_glat, sin_glat, vXb, vYb, vZb)
    return vlsr


def model(
    az,
    az0,
    tan_pitch,
    R0,
    Usun,
    Vsun,
    Wsun,
    Upec,
    Vpec,
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
    warp_amp,
    warp_off,
    Rref=8.0,
):
    """
    Derive the model-predicted Galactic longitude, latitude,
    IAU-LSR velocity, velocity gradient w.r.t. distance, and distance
    of a spiral at a given azimuth.

    Inputs:
      az :: scalar (radians)
        Galactocentric azimuth
      az0 :: scalar (radians)
        Reference azimuth at Rref
      tan_pitch :: scalar
        Tangent of pitch angle
      R0 :: scalar (kpc)
        Galactocentric radius of the Sun
      Usun, Vsun, Wsun :: scalars (km/s)
        Solar peculiar motion components
      Upec, Vpec :: scalars (km/s)
        Peculiar motion components
      cos_tilt, sin_tilt, cos_roll, sin_roll :: scalars
        Cosine and sine of Galactic plane tilt and roll angles
      R0a22, lam, loglam, term1, term2 :: scalars
        The rotation curve constants
      theta0 :: scalar (km/s)
        Rotation speed at R0
      warp_amp, warp_off :: scalars (kpc-1, rad)
        Warp is defined as Z(R, az) = warp_amp*(R - R_ref)^2 * sin(az - warp_off)
      Rref :: scalar (kpc)
        Reference Galactocentric radius

    Returns: glong, glat, vlsr, dvlsr_ddist, dist
      glong, glat :: scalars (radians)
        Galactic longitude and latitude
      vlsr :: scalar (km/s)
        IAU-LSR velocity
      dvlsr_ddist :: scalar
        Partial derivative of IAU-LSR velocity w.r.t. distance
      dist :: scalar (kpc)
        Distance
    """
    R = Rref * tt.exp((az0 - az) * tan_pitch)
    cos_az = tt.cos(az)
    sin_az = tt.sin(az)

    # Apply Galactic warp
    Zg = warp_amp * (R - Rref) ** 2.0 * tt.sin(az - warp_off)
    Zg[R < Rref] = 0.0

    # Convert positions to Galactocentric Cartesian frame
    Xg, Yg = transforms.gcencyl_to_gcencar(R, cos_az, sin_az)

    # Convert positions to the barycentric Cartesian frame
    Xb, Yb, Zb = transforms.gcencar_to_barycar(
        Xg, Yg, Zg, R0, cos_tilt, sin_tilt, cos_roll, sin_roll
    )

    # Convert positions to Galactic longitude and latitude
    cos_glong, sin_glong, cos_glat, sin_glat, dist = transforms.barycar_to_galactic(
        Xb, Yb, Zb
    )
    glong = tt.atan2(sin_glong, cos_glong)
    glat = tt.asin(sin_glat)

    # Get IAU-LSR velocity and derivative w.r.t. distance
    dist.requires_grad = True
    vlsr = model_vlsr(
        dist,
        cos_glong,
        sin_glong,
        cos_glat,
        sin_glat,
        R0,
        Usun,
        Vsun,
        Wsun,
        Upec,
        Vpec,
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
    )
    vlsr.sum().backward()
    dvlsr_ddist = dist.grad
    return glong, glat, vlsr, dvlsr_ddist, dist

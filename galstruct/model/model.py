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
import numpy as np

from galstruct.model import rotcurve, transforms
from galstruct.model.utils import (
    calc_spiral_angle,
    calc_sigma2_glong,
    calc_sigma2_glat,
)


class Model:
    """
    This class generates a model galaxy, with default parameters provided.
    """

    # init helps us to define some default model parameters
    def __init__(
        self,
        # these are physical parameters of the model
        az0=tt.tensor(np.pi),
        R0=tt.tensor(8.5),
        a2=tt.tensor(0.95),
        a3=tt.tensor(1.65),
        Rref=tt.tensor(8.0),
        # these are the Sun's kinematic parameters in the model
        Usun=tt.tensor(0.0),
        Vsun=tt.tensor(0.0),
        Wsun=tt.tensor(0.0),
        Zsun=tt.tensor(0.0),
        Upec=tt.tensor(0.0),
        Vpec=tt.tensor(0.0),
        # angular parameters dictating the shape of the spiral arms
        pitch=tt.tensor(0.0),
        roll=tt.tensor(0.0),
        warp_amp=tt.tensor([0.0]),
        warp_off=tt.tensor([0.0]),
        # parameters defining the spread in the spiral arms
        sigma_arm_height=tt.tensor(0.0),
        sigma_arm_plane=tt.tensor(0.0),
        sigmaV=tt.tensor(0.0),
    ):

        self.az0 = az0
        self.R0 = R0
        self.a2 = a2  # a2 and a3 are needed for the rotcurve_constraints function to obtain R0a22, lam, loglam, term1, and term2
        self.a3 = a3
        self.Rref = Rref

        self.Usun = Usun
        self.Vsun = Vsun
        self.Wsun = Wsun
        self.Zsun = Zsun
        self.Upec = Upec
        self.Vpec = Vpec

        self.pitch = pitch
        self.roll = roll
        self.warp_amp = warp_amp
        self.warp_off = warp_off

        self.sigma_arm_height = sigma_arm_height
        self.sigma_arm_plane = sigma_arm_plane
        self.sigmaV = sigmaV

        # here we use some of the above defined model parameters to perform necessary calculations
        self.tilt = tt.asin(self.Zsun / self.R0 / 1000.0)
        self.cos_tilt, self.sin_tilt = tt.cos(self.tilt), tt.sin(self.tilt)
        self.cos_roll, self.sin_roll = tt.cos(self.roll), tt.sin(self.roll)
        self.R0a22, self.lam, self.loglam, self.term1, self.term2 = rotcurve.rotcurve_constants(
            self.R0, self.a2, self.a3
        )
        self.theta0 = rotcurve.calc_theta(self.R0, self.R0a22, self.lam, self.loglam, self.term1, self.term2)

    def model_vlsr(
        self,
        dist,
        cos_glong,
        sin_glong,
        cos_glat,
        sin_glat,
    ):
        """
        Derive the model-predicted IAU-LSR velocity at a given position.

        Inputs:
          dist :: scalar (kpc)
            Distance
          cos_glong, sin_glong, cos_glat, sin_glat :: scalars
            Consine and sine of Galactic longitude and latitude

        Returns: vlsr
          vlsr :: scalar (km/s)
            IAU-LSR velocity
        """
        # Convert distance to R, azimuth
        midplane_dist = dist * cos_glat
        R = tt.sqrt(self.R0**2.0 + midplane_dist**2.0 - 2.0 * self.R0 * midplane_dist * cos_glong)
        cos_az = (self.R0 - midplane_dist * cos_glong) / R
        sin_az = midplane_dist * sin_glong / R

        # Calculate rotation speed at R
        theta = rotcurve.calc_theta(R, self.R0a22, self.lam, self.loglam, self.term1, self.term2)
        vR = -self.Upec
        vAz = theta + self.Vpec

        # Convert velocities to barycentric Cartesian frame
        vXg, vYg = transforms.v_gcencyl_to_gcencar(vR, vAz, cos_az, sin_az)
        vXb, vYb, vZb = transforms.v_gcencar_to_barycar(
            vXg,
            vYg,
            self.Usun,
            self.Vsun,
            self.Wsun,
            self.theta0,
            self.cos_tilt,
            self.sin_tilt,
            self.cos_roll,
            self.sin_roll,
        )

        # Convert velocities to IAU-LSR radial velocity
        vlsr = transforms.calc_vlsr(cos_glong, sin_glong, cos_glat, sin_glat, vXb, vYb, vZb)
        return vlsr

    def model(self, az):
        """
        Derive the model-predicted Galactic longitude, latitude,
        IAU-LSR velocity, velocity gradient w.r.t. distance, and distance
        of a spiral at a given azimuth.

        Inputs:
          az :: scalar (radians)
            Galactocentric azimuth

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
        R = self.Rref * tt.exp((self.az0 - az) * tt.tan(self.pitch))
        cos_az = tt.cos(az)
        sin_az = tt.sin(az)

        # Apply Galactic warp
        Zg = self.warp_amp * (R - self.Rref) ** 2.0 * tt.sin(az - self.warp_off)
        Zg[R < self.Rref] = 0.0

        # Convert positions to Galactocentric Cartesian frame
        Xg, Yg = transforms.gcencyl_to_gcencar(R, cos_az, sin_az)

        # Convert positions to the barycentric Cartesian frame
        Xb, Yb, Zb = transforms.gcencar_to_barycar(
            Xg, Yg, Zg, self.R0, self.cos_tilt, self.sin_tilt, self.cos_roll, self.sin_roll
        )

        # Convert positions to Galactic longitude and latitude
        cos_glong, sin_glong, cos_glat, sin_glat, dist = transforms.barycar_to_galactic(Xb, Yb, Zb)

        glong = tt.atan2(sin_glong, cos_glong)
        glat = tt.asin(sin_glat)

        # Get IAU-LSR velocity and derivative w.r.t. distance
        vlsr = self.model_vlsr(
            dist,
            cos_glong,
            sin_glong,
            cos_glat,
            sin_glat,
        )
        return glong, glat, vlsr, dist

    def model_spread(self, az):
        """
        This adds 3D spread to the spiral arm's structure using the following parameters:

        sigmaV :: spiral FWHM velocity width (km/s)
        sigma_arm_plane :: spiral FWHM physical width in the plane (kpc)
        sigma_arm_height :: spiral FWHM physical width perpendicular to plane (kpc)

        """
        glong, glat, vlsr, dist = self.model(az)

        angle = calc_spiral_angle(az, dist, self.pitch, self.R0)
        sigma2_glat = calc_sigma2_glat(dist, self.sigma_arm_height)
        sigma2_glong = calc_sigma2_glong(dist, angle, self.sigma_arm_plane)
        glong = glong + tt.randn_like(glong) * tt.sqrt(sigma2_glong)
        glat = glat + tt.randn_like(glat) * tt.sqrt(sigma2_glat)
        vlsr = vlsr + tt.randn_like(vlsr) * self.sigmaV

        return (
            tt.stack((glong, glat, vlsr)).T.detach(),
            tt.stack((sigma2_glong, sigma2_glat)).T.detach(),
        )

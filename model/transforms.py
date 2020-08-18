#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
transforms.py

Utility functions for converting positions and velocities between
reference frames.

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

import torch as tt

# IAU-defined solar motion parameters (km/s)
_USTD = 10.27
_VSTD = 15.32
_WSTD = 7.74

def gcencyl_to_gcencar(R, cos_az, sin_az):
    """
    Convert Galactocentric cylindrical positions to the Galactocentric
    Cartesian frame.

    Inputs:
      R :: scalar (kpc)
        Galactocentric radius
      cos_az, sin_az :: scalars
        Cosine and sine of Galactocentric azimuth

    Returns: (Xg, Yg)
      Xg, Yg :: scalars (kpc)
        Galactocentric Cartesian positions
    """
    Xg = -R*cos_az
    Yg = R*sin_az
    return (Xg, Yg)

def v_gcencyl_to_gcencar(vR, vAz, cos_az, sin_az):
    """
    Convert Galactocentric cylindrical velocities to the
    Galactocentric Cartesian frame.

    Inputs:
      vR, vAz :: scalars (km/s)
        Galactocentric cylindrical velocities
      cos_az, sin_az :: scalars
        Cosine and sine of Galactocentric azimuth

    Returns: (vXg, vYg)
      vXg, vYg :: scalars (km/s)
        Galactocentric Cartesian velocities
    """
    vXg = -vR*cos_az + sin_az*vAz
    vYg = vR*sin_az + cos_az*vAz
    return vXg, vYg

def gcencar_to_barycar(Xg, Yg, Zg, R0, cos_tilt, sin_tilt, cos_roll,
                       sin_roll):
    """
    Convert Galactocentric Cartesian positions to the barycentric
    Cartesian frame.

    Inputs:
      Xg, Yg, Zg :: scalars (kpc)
        Galactocentric Cartesian position
      cos_tilt, sin_tilt :: scalars
        Cosine and sine of the tilt angle
      cos_roll, sin_roll :: scalars
        Cosine and sine of the roll angle

    Returns: (Xb, Yb, Zb)
      Xb, Yb, Zb :: scalars (kpc)
        Barycentric Cartesian position
    """
    Xb = Xg*cos_tilt - Zg*sin_tilt + R0
    Zb = Xg*sin_tilt + Zg*cos_tilt
    Yb = Yg*cos_roll + Zb*sin_roll
    Zb = -Yg*sin_roll + Zb*cos_roll
    return (Xb, Yb, Zb)

def v_gcencar_to_barycar(vXg, vYg, Usun, Vsun, Wsun, theta0,
                         cos_tilt, sin_tilt, cos_roll, sin_roll):
    """
    Convert Galactocentric Cartesian velocities to the barycentric
    Cartesian frame.

    Inputs:
      vXg, vYg :: scalars (km/s)
        Galactocentric Cartesian velocities
      Usun, Vsun, Wsun :: scalars (km/s)
        Solar non-circular motion components
      theta0 :: scalar (km/s)
        Rotation curve speed at R0
      cos_tilt, sin_tilt :: scalars
        cosine and sine of the tilt angle
      cos_roll, sin_roll :: scalars
        cosine and sine of the roll angle

    Returns: (vXb, vYb, vZb)
      vXb, vYb, vZb :: scalars (km/s)
        Barycentric Cartesian velocities
    """
    vXg = vXg - Usun
    vYg = vYg - (Vsun + theta0)
    vZg = -Wsun
    vXb = vXg*cos_tilt - vZg*sin_tilt;
    vZb = vXg*sin_tilt + vZg*cos_tilt;
    vYb = vYg*cos_roll + vZb*sin_roll;
    vZb = -vYg*sin_roll + vZb*cos_roll;
    return vXb, vYb, vZb

def barycar_to_galactic(Xb, Yb, Zb):
    """
    Convert barycentric Cartesian positions to Galactic longitude
    and latitude.

    Inputs:
      Xb, Yb, Zb :: scalars (kpc)
        Barycentric Cartesian position

    Returns: (cos_glong, sin_glong, cos_glat, sin_glat, dist)
      cos_glong, sin_glong, cos_glat, sin_glat :: scalars
        Cosine and sine of Galactic longitude and latitude
      dist :: scalar (kpc)
        Distance
    """
    midplane_dist = tt.sqrt(Xb**2.0 + Yb**2.0)
    dist = tt.sqrt(Xb**2.0 + Yb**2.0 + Zb**2.0)
    cos_glong = Xb/midplane_dist
    sin_glong = Yb/midplane_dist
    cos_glat = midplane_dist/dist
    sin_glat = Zb/dist
    return (cos_glong, sin_glong, cos_glat, sin_glat, dist)

def calc_vlsr(cos_glong, sin_glong, cos_glat, sin_glat, vXb, vYb, vZb):
    """
    Convert barycentric Cartesian velocities to IAU-LSR velocity.

    Inputs:
      cos_glong, sin_glong :: scalars
        Cosine and sine of longitude
      cos_glat, sin_glat :: scalars
        Cosine and sine of latitude
      vXb, vYb, vZb :: scalars (km/s)
        Barycentric Cartesian velocities

    Returns: vlsr
      vlsr :: scalar (km/s)
        IAU-LSR velocity
    """
    vlsr = ((cos_glong*(vXb + _USTD) + sin_glong*(vYb + _VSTD))*cos_glat +
            (vZb + _WSTD)*sin_glat)
    return vlsr

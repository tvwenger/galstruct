#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py

Other general utility functions.

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


def calc_spiral_angle(az, dist, pitch, R0):
    """
    Calculate the angle between a line-of-sight and a log spiral.

    Inputs:
      az :: scalar (radians)
        Galactocentric azimuth
      dist :: scalar (kpc)
        Distance
      pitch :: scalar (radians)
        Pitch angle
      R0 :: scalar (kpc)
        Solar Galactocentric radius

    Returns: angle
      angle :: scalar (radians)
        The angle between the line-of-sight and the spiral
    """
    return pitch + np.pi / 2.0 - tt.asin(R0 * tt.sin(az) / dist)


def calc_sigma2_glat(dist, sigma_arm):
    """
    Calculate the width of a spiral in the latitude direction.

    Inputs:
      dist :: scalar (kpc)
        Distance
      sigma_arm :: scalar (kpc)
        Arm width

    Returns: sigma2_glat
      sigma2_glat :: scalar (radians^2)
        The width^2 in the latitude direction
    """
    return tt.atan(sigma_arm / dist) ** 2.0


def calc_sigma2_glong(dist, angle, sigma_arm):
    """
    Calculate the width of a spiral in the longitude direction.

    Inputs:
      dist :: scalar (kpc)
        Distance
      angle :: scalar (radians)
        Angle between line-of-sight and spiral
      sigma_arm :: scalar (kpc)
        Arm width

    Returns: sigma2_glong
      sigma2_glong :: scalar (radians^2)
        The width^2 in the longitude direction
    """
    return (tt.atan(sigma_arm / dist) * tt.cos(angle)) ** 2.0

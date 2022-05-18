#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
rotcurve.py

Utilities functions for the Persic+1996 rotation curve.

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


def rotcurve_constants(R0, a2, a3):
    """
    Return the Persic+1996 Universal rotation curve constants.

    Inputs:
      R0 :: scalar (kpc)
        Solar Galactocentric radius
      a2, a3 :: scalars
        Persic+1996 rotation curve parameters

    Returns: (R0a22, lam, loglam, term1, term2)
      R0a22, lam, loglam, term1, term2 :: scalars
        Constants of the Persic+1996 rotation curve
    """
    R0a22 = (R0 * a2) ** 2.0
    lam = (a3 / 1.5) ** 5.0
    loglam = tt.log10(lam)
    term1 = 200.0 * lam ** 0.41
    term2 = 0.8 + 0.49 * loglam + 0.75 * tt.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4)
    term2 = tt.sqrt(term2)
    return R0a22, lam, loglam, term1, term2


def calc_theta(R, R0a22, lam, loglam, term1, term2):
    """
    Derive the Persic+1996 Universal rotation curve speed at a
    given Galactocentric radius.

    Inputs:
      R :: scalar (kpc)
        Galactocentric radius
      R0a22 :: scalar
        The term (R0^2 * a2^2)
      lam, loglam :: scalars
        The lambda and log10(lambda) parameters.
      term1, term2 :: scalar
        The first two terms of the Persic+1996 equation

    Returns: theta
      theta :: scalar (km/s)
        Rotation speed at R
    """
    rho2 = R ** 2.0 / R0a22
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho2 ** 0.61 / (rho2 + 0.61) ** 1.43
    term4 = 1.6 * tt.exp(-0.4 * lam) * rho2 / (rho2 + 2.25 * lam ** 0.4)
    theta = term1 / term2 * tt.sqrt(term3 + term4)
    return theta

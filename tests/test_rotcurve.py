#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_rotcurve.py

Test functionality of rotcurve.py

Copyright(C) 2022 by Trey Wenger <tvwenger@gmail.com>

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

Trey Wenger - May 2022
"""

import torch as tt
import numpy as np
import matplotlib.pyplot as plt

from galstruct.model import rotcurve


def test_rotcurve():
    """
    Test functionality of rotcurve.py

    Inputs: Nothing

    Returns: Nothing
    """
    # Plot rotation curve for reasonable values of parameters
    R0 = tt.tensor(8.5)  # kpc
    a2 = tt.tensor(0.977)
    a3 = tt.tensor(1.624)
    R0a22, lam, loglam, term1, term2 = rotcurve.rotcurve_constants(R0, a2, a3)
    R = tt.tensor(np.linspace(0.0, 20.0, 1000))
    theta = rotcurve.calc_theta(R, R0a22, lam, loglam, term1, term2)

    fig, ax = plt.subplots()
    ax.plot(R.detach().numpy(), theta.detach().numpy(), "k-")
    ax.set_xlabel("Galactocentric Radius (kpc)")
    ax.set_ylabel(r"$\Theta(R)$ (km s$^{-1}$)")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    test_rotcurve()

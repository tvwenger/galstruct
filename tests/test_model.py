#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_model.py

Test functionality of model.py

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

from galstruct.model import rotcurve, model


def test_model_vlsr():
    """
    Test functionality of model_vlsr()

    Inputs: Nothing

    Returns: Nothing
    """
    # Plot LSR vs. distance for a line of sight
    glong = tt.tensor(np.deg2rad(42.0))  # radians
    cos_glong, sin_glong = tt.cos(glong), tt.sin(glong)
    glat = tt.tensor(np.deg2rad(1.0))  # radians
    cos_glat, sin_glat = tt.cos(glat), tt.sin(glat)

    R0 = tt.tensor(8.5)  # kpc
    a2 = tt.tensor(0.977)
    a3 = tt.tensor(1.624)
    R0a22, lam, loglam, term1, term2 = rotcurve.rotcurve_constants(R0, a2, a3)
    theta0 = rotcurve.calc_theta(R0, R0a22, lam, loglam, term1, term2)

    Usun = tt.tensor(10.5)  # km/s
    Vsun = tt.tensor(12.2)  # km/s
    Wsun = tt.tensor(7.7)  # km/s
    Upec = tt.tensor(5.8)  # km/s
    Vpec = tt.tensor(-3.4)  # km/s

    Zsun = tt.tensor(5.5)  # pc
    roll = tt.tensor(np.deg2rad(-1.0))  # radians
    tilt = tt.asin(Zsun / R0 / 1000.0)  # radians
    cos_tilt, sin_tilt = tt.cos(tilt), tt.sin(tilt)
    cos_roll, sin_roll = tt.cos(roll), tt.sin(roll)

    dist = tt.tensor(np.linspace(0.0, 20.0, 1000))
    vlsr = model.model_vlsr(
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

    fig, ax = plt.subplots()
    ax.plot(dist.detach().numpy(), vlsr.detach().numpy(), "k-")
    ax.set_xlabel("Distance (kpc)")
    ax.set_ylabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def test_model():
    """
    Test functionality of model()

    Inputs: Nothing

    Returns: Nothing
    """
    # Plot l-v diagram of spiral
    R0 = tt.tensor(8.5)  # kpc
    a2 = tt.tensor(0.977)
    a3 = tt.tensor(1.624)
    R0a22, lam, loglam, term1, term2 = rotcurve.rotcurve_constants(R0, a2, a3)
    theta0 = rotcurve.calc_theta(R0, R0a22, lam, loglam, term1, term2)

    Usun = tt.tensor(10.5)  # km/s
    Vsun = tt.tensor(12.2)  # km/s
    Wsun = tt.tensor(7.7)  # km/s
    Upec = tt.tensor(5.8)  # km/s
    Vpec = tt.tensor(-3.4)  # km/s

    Zsun = tt.tensor(5.5)  # pc
    roll = tt.tensor(np.deg2rad(-1.0))  # radians
    tilt = tt.asin(Zsun / R0 / 1000.0)  # radians
    cos_tilt, sin_tilt = tt.cos(tilt), tt.sin(tilt)
    cos_roll, sin_roll = tt.cos(roll), tt.sin(roll)

    az0 = tt.tensor(np.deg2rad(30.0))  # radians
    pitch = tt.tensor(np.deg2rad(10.0))  # radians
    tan_pitch = tt.tan(pitch)
    warp_amp = tt.tensor(0.02)  # kpc
    warp_off = tt.tensor(-0.5)  # radians
    Rref = 8.0  # kpc

    az = tt.tensor(np.linspace(-2.0 * np.pi, 4.0 * np.pi, 1000))
    glong, glat, vlsr, dvlsr_ddist, dist = model.model(
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
        Rref=Rref,
    )

    fig, ax = plt.subplots()
    cax = ax.scatter(
        vlsr.detach().numpy(),
        np.rad2deg(glong.detach().numpy()),
        c=np.rad2deg(glat.detach().numpy()),
        marker="o",
        s=dist.detach().numpy(),
        alpha=0.5,
    )
    fig.colorbar(cax, label="Galactic Latitude (deg)")
    ax.legend(
        *cax.legend_elements(prop="sizes", num=5, color=cax.cmap(0.7), fmt="{x:.1f} kpc"),
        loc="upper right",
        title="Distance"
    )
    ax.set_xlabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
    ax.set_ylabel("Galactic Longitude (deg)")
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    test_model_vlsr()
    test_model()

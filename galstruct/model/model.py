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

#import some necessary python stuff
import torch as tt
import numpy as np
import matplotlib.pyplot as plt

#updated script import statements, these pull from files in the same directory
#from . import rotcurve
#from . import transforms
import rotcurve
import transforms 

class Model:
    """
    This class generates a model galaxy, with default parameters provided.
    """     

    #init helps us to define some default model parameters
    def __init__(self,
                az0 = tt.tensor(np.pi), 
                tan_pitch = tt.tensor(np.tan(np.deg2rad(14))), 
                R0 = tt.tensor(8.5), 
                Usun = tt.tensor(0.0), 
                Vsun = tt.tensor(0.0), 
                Wsun = tt.tensor(0.0), 
                Upec = tt.tensor(0.0), 
                Vpec = tt.tensor(0.0), 
                sin_tilt = tt.tensor(0.0), 
                cos_tilt = tt.tensor(1.0), 
                cos_roll = tt.tensor(1.0), 
                sin_roll = tt.tensor(0.0), 
                a2 = tt.tensor(0.95), 
                a3= tt.tensor(1.65), 
                theta0 = tt.tensor(220.0), 
                warp_amp = tt.tensor([0.0]), 
                warp_off = tt.tensor([0.0]), 
                Rref = tt.tensor(8.0)):                    
        self.az0 = az0
        self.tan_pitch = tan_pitch
        self.R0 = R0
        self.Usun = self.Vsun = self.Wsun = Usun = Vsun = Wsun
        self.Upec = self.Vpec = Upec = Vpec
        self.sin_tilt = sin_tilt
        self.cos_tilt = cos_tilt
        self.cos_roll= cos_roll
        self.sin_roll = sin_roll
        self.a2 = a2 #a2 and a3 are needed for the rotcurve_constraints function to obtain R0a22, lam, loglam, term1, and term2
        self.a3 = a3
        self.R0a22, self.lam, self.loglam, self.term1, self.term2 = rotcurve.rotcurve_constants(self.R0, self.a2, self.a3)
        self.theta0 = theta0
        self.warp_amp = warp_amp
        self.warp_off = warp_off
        self.Rref = Rref
    
    def model_vlsr(self,
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
        R = tt.sqrt(self.R0 ** 2.0 + midplane_dist ** 2.0 - 2.0 * self.R0 * midplane_dist * cos_glong)
        cos_az = (self.R0 - midplane_dist * cos_glong) / R
        sin_az = midplane_dist * sin_glong / R
    
        # Calculate rotation speed at R
        theta = rotcurve.calc_theta(R, self.R0a22, self.lam, self.loglam, self.term1, self.term2)
        vR = -self.Upec
        vAz = theta + self.Vpec
    
        # Convert velocities to barycentric Cartesian frame
        vXg, vYg = transforms.v_gcencyl_to_gcencar(vR, vAz, cos_az, sin_az)
        vXb, vYb, vZb = transforms.v_gcencar_to_barycar(
            vXg, vYg, self.Usun, self.Vsun, self.Wsun, self.theta0, self.cos_tilt, self.sin_tilt, self.cos_roll, self.sin_roll
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
        R = self.Rref * tt.exp((self.az0 - az) * self.tan_pitch)
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
        cos_glong, sin_glong, cos_glat, sin_glat, dist = transforms.barycar_to_galactic(
            Xb, Yb, Zb
        )
        
        glong = tt.atan2(sin_glong,cos_glong)
        glat = tt.asin(sin_glat)
    
        # Get IAU-LSR velocity and derivative w.r.t. distance
        dist.requires_grad = True
        vlsr = self.model_vlsr(
            dist,
            cos_glong,
            sin_glong,
            cos_glat,
            sin_glat,
        )
        vlsr.sum().backward()
        dvlsr_ddist = dist.grad
        return glong, glat, vlsr, dvlsr_ddist, dist

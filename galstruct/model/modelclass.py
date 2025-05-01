import torch as tt
import numpy as np
import matplotlib.pyplot as plt

from galstruct.model import rotcurve, transforms
from galstruct.model.utils import (
    calc_spiral_angle,
    calc_sigma2_glong,
    calc_sigma2_glat,
)

class BaseModel:
    """
    This class BaseModel generates an exponential model of a galaxy. 
    It also provides a baseline structure for the spiral model of a galaxy

    Inputs
        sigmaV :: spiral FWHM velocity width (km/s)
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
        I2, Rs, Rc :: Disk parameters
    """

    # init helps us to define some default model parameters
    def __init__(
        self,
        # these are physical parameters of the model
        R0=tt.tensor(8.5),
        a2=tt.tensor(0.95),
        a3=tt.tensor(1.65),
        Rmin=tt.tensor(3.0),
        Rmax=tt.tensor(15.0),
        Rref=tt.tensor(8.0),
        # these are the Sun's kinematic parameters in the model
        Usun=tt.tensor(0.0),
        Vsun=tt.tensor(0.0),
        Wsun=tt.tensor(0.0),
        Zsun=tt.tensor(0.0),
        Upec=tt.tensor(0.0),
        Vpec=tt.tensor(0.0),
        # angular parameters dictating the shape of the spiral arms
        roll=tt.tensor(0.0),
        warp_amp=tt.tensor(0.0),
        warp_off=tt.tensor(0.0),
        # parameters defining the spread in the spiral arms
        sigma_arm_height=tt.tensor(0.0),
        sigmaV=tt.tensor(0.0),

        #parameters defining the shape of the exponential disk, from Trey's thesis
        I2 = 35,
        Rs = 4.5,
        Rc = 2.75,
    ):
        self.R0 = R0
        self.a2 = a2  # a2 and a3 are needed for the rotcurve_constraints function to obtain R0a22, lam, loglam, term1, and term2
        self.a3 = a3
        self.Rref = Rref
        self.Rmin = Rmin
        self.Rmax = Rmax

        self.Usun = Usun
        self.Vsun = Vsun
        self.Wsun = Wsun
        self.Zsun = Zsun
        self.Upec = Upec
        self.Vpec = Vpec

        self.roll = roll
        self.warp_amp = warp_amp
        self.warp_off = warp_off

        self.sigma_arm_height = sigma_arm_height
        self.sigmaV = sigmaV

        self.I2 = I2
        self.Rs = Rs
        self.Rc = Rc
        
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

    def model(self, az, R):

        """
        Derive the model-predicted Galactic longitude, latitude,
        IAU-LSR velocity, velocity gradient w.r.t. distance, and distance
        of a spiral at a given azimuth.

        Inputs:
          az :: scalar (radians)
            Galactocentric azimuth
          R :: scalar 
            radius measured from center of galaxy

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

    def model_spread(self, az, R):

        """
        This adds 3D spread to the spiral arm's structure using the following parameters:

        sigmaV :: spiral FWHM velocity width (km/s)
        sigma_arm_height :: spiral FWHM physical width perpendicular to plane (kpc)
        """
        
        glong, glat, vlsr, dist = self.model(az, R)

        sigma2_glong = tt.zeros_like(glong)
        sigma2_glat = calc_sigma2_glat(dist, self.sigma_arm_height)
        glong = glong + tt.randn_like(glong) * tt.sqrt(sigma2_glong)
        glat = glat + tt.randn_like(glat) * tt.sqrt(sigma2_glat)
        vlsr = vlsr + tt.randn_like(vlsr) * self.sigmaV

        return glong, glat, vlsr, dist, sigma2_glong, sigma2_glat

    def simulate(self, num_sims):    

        """
        Generate simulated HII region longitude, latitude, and velocity
        data from the model
        """
        
        R = tt.stack(tuple(tt.linspace(self.Rmin, self.Rmax, 1000) for _ in range(num_sims)))
        
        prob = tt.exp(-R / self.Rs) / (1.0 + self.I2 * tt.exp(-R / self.Rc))
        prob = prob / tt.sum(prob, axis=1, keepdims=True) 
        
        idx = prob.multinomial(num_samples=1)
        R_sel = tt.gather(R, 1, idx)[:,0]
        az = tt.tensor([np.random.uniform(0.0, 2.0*np.pi, size=(num_sims))])[0]
    
        glong, glat, vlsr, dist, sigma2_glong, sigma2_glat = self.model_spread(az, R)
    
        return glong, glat, vlsr, dist


class SpiralModel(BaseModel):

    """
    This class builds off of the BaseModel structure, to create a galactic spiral model

    Inputs
    az0 :: spiral reference azimuth (radians)
    pitch :: spiral pitch angle (radians)
    sigma_arm_plane :: spiral FWHM physical width in the plane (kpc)
    """
    
    def __init__(
        self,
        az0=tt.tensor(np.pi),
        pitch=tt.tensor(np.deg2rad(14.0)),
        sigma_arm_plane=tt.tensor(0.0),
        **kwargs,
        ):
        
        super().__init__(**kwargs)
        self.az0 = az0
        self.pitch = pitch
        self.sigma_arm_plane = sigma_arm_plane
    
    def model(self, az, R=None):

        """
        defines radius (R) based upon azimuth (az) input and calls 
        BaseModel model function to return:

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
        
        if R is None:
            R = self.Rref * tt.exp((self.az0 - az) * tt.tan(self.pitch))
            
        return super().model(az, R)
    
    def model_spread(self, az):

        """
        adds spread to the spiral model via

        sigmaV :: spiral FWHM velocity width (km/s)
        sigma_arm_height :: spiral FWHM physical width perpendicular to plane (kpc)
        sigma_arm_plane :: spread in the plane of the spiral arm 
        """
        
        R = self.Rref * tt.exp((self.az0 - az) * tt.tan(self.pitch))
        glong, glat, vlsr, dist, sigma2_glong, sigma2_glat = super().model_spread(az, R)
        
        # update sigma2_glong
        angle = calc_spiral_angle(az, dist, self.pitch, self.R0)
        sigma2_glong = calc_sigma2_glong(dist, angle, self.sigma_arm_plane)
        
        return glong, glat, vlsr, dist, sigma2_glong, sigma2_glat
    
    def simulate(self, num_sims):

        """
        Generate simulated HII region longitude, latitude, and velocity
        data from the model
        """
        
        # Get azimuth range, pick a random azimuth
        min_az = self.az0 - tt.log(self.Rmax / self.Rref) / tt.tan(self.pitch)
        max_az = self.az0 - tt.log(self.Rmin / self.Rref) / tt.tan(self.pitch)
        
        az = (max_az - min_az) * tt.rand(num_sims) + min_az
        
        glong, glat, vlsr, dist, sigma2_glong, sigma2_glat = self.model_spread(az)
        
        return glong, glat, vlsr, dist
#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
mcmc_posterior.py

Generate spiral model posteriors by MCMC using trained likelihood
and real or synthetic data.

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

import os
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import argparse
import sqlite3
import pickle
import dill

import torch
import pymc3 as pm
import theano.tensor as tt

import numpy as np
np.random.seed(1234)

from model.simulator import simulator
from torch_prior import Prior

# default values for synthetic spiral parameters
_spiral_params = [0.25, 0.25, 0.25, 0.25,
                  np.deg2rad(72.0), np.deg2rad(137.0),
                  np.deg2rad(252.0), np.deg2rad(317.0),
                  np.deg2rad(14.0), np.deg2rad(14.0),
                  np.deg2rad(14.0), np.deg2rad(14.0),
                  5.0, 0.5, 0.1]
# default values for GRM parameters
_grm_params = [8.16643777, 10.4543041, 12.18499493, 7.71886874, 5.79095823,
               -3.39171583, 0.97757558, 1.62261724, 5.5, 0.0]
# default values for warp parameters
_warp_params = [0.02, -0.5]
# default values for exponential disk parameters
_disk_params = [35.0, 3.0, 2.5]

# set random seed
np.random.seed(1234)

# default parameter values
_NUM_DATA = 2000
_RMIN = 3.0
_RMAX = 15.0
_RREF = 8.0
_NUM_SPIRALS = 4
_NITER = 1000
_NTUNE = 1000
_NINIT = 100000
_NUM_CHAINS = 4
_STEP_SCALE = 0.25
_TARGET_ACCEPT = 0.9
_FIXED = {}
_OUTLIERS = None
_OVERWRITE = False

class Loglike(tt.Op):
    """
    Theano.Operator handling likelihood and gradient calculations
    from neural net.
    """
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, loglike, data, num_spirals):
        """
        Initialize a new Loglike object.

        Inputs:
          loglike :: neural networks object
            Neural network
          data :: 2-D torch.tensor
            Data (shape N x 3)
          num_spirals :: integer
            Number of spirals

        Returns: loglike
          loglike :: A new Loglike instance
        """
        self.loglike = loglike
        self.data = data
        self.num_spirals = num_spirals
        self.loglike_grad = LoglikeGrad(
            self.loglike, self.data, self.num_spirals)

    def perform(self, node, inputs, outputs):
        """
        Handle the calling of this Operator.

        Inputs:
          theta :: theano.vector
            Parameters in this order:
            q, az0, pitch, other parameters expected by likelihood
        """
        theta, = inputs
        q = torch.as_tensor(theta[:self.num_spirals]).float()
        az0 = torch.as_tensor(theta[self.num_spirals:2*self.num_spirals]).float()
        pitch = torch.as_tensor(theta[2*self.num_spirals:3*self.num_spirals]).float()
        params = torch.as_tensor(theta[3*self.num_spirals:]).float()
        # stack parameters for each likelihood evaluation
        thetas = [
            torch.cat((
                az0[i].reshape(1), pitch[i].reshape(1), params))
            for i in range(self.num_spirals)]
        # evaluate loglike for each spiral and data point.
        logp = torch.stack([
            self.loglike.log_prob(
                self.data, th.expand(len(self.data), -1))
            for th in thetas])
        # catch nans
        logp[torch.isnan(logp)] = -np.inf
        # marginalize over spiral
        logp = torch.logsumexp(logp + torch.log(q[..., None]), 0)
        # sum over data
        logp = torch.sum(logp)
        outputs[0][0] = logp.detach().double().numpy()

    def grad(self, inputs, grad_outputs):
        """
        Handle the gradient of this Operator by invoking the
        LoglikeGrad operator
        """
        theta, = inputs
        grads = self.loglike_grad(theta)
        return [grad_outputs[0]*grads]

class LoglikeGrad(tt.Op):
    """
    Theano.Operator handling gradient calculations from neural net.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, data, num_spirals):
        """
        Initialize a new LoglikeGrad object.

        Inputs:
          loglike :: likelihood neural networks
            Neural network
          data :: 2-D torch.tensor
            Data (shape N x 3)
          num_spirals :: integer
            Number of spirals

        Returns: loglike
          loglike :: A new Loglike instance
        """
        self.loglike = loglike
        self.data = data
        self.num_spirals = num_spirals

    def perform(self, node, inputs, outputs):
        """
        Handle the calling of this Operator.
        """
        theta, = inputs
        q = torch.as_tensor(theta[:self.num_spirals]).float()
        q.requires_grad = True
        az0 = torch.as_tensor(theta[self.num_spirals:2*self.num_spirals]).float()
        az0.requires_grad = True
        pitch = torch.as_tensor(theta[2*self.num_spirals:3*self.num_spirals]).float()
        pitch.requires_grad = True
        params = torch.as_tensor(theta[3*self.num_spirals:]).float()
        params.requires_grad = True
        # stack parameters for each likelihood evaluation
        thetas = [
            torch.cat((
                az0[i].reshape(1), pitch[i].reshape(1), params))
            for i in range(self.num_spirals)]
        # evaluate loglike for each spiral and data point.
        logp = torch.stack([
            self.loglike.log_prob(
                self.data, th.expand(len(self.data), -1))
            for th in thetas])
        # catch nans
        logp[torch.isnan(logp)] = -np.inf
        # marginalize over spiral
        logp = torch.logsumexp(logp + torch.log(q[..., None]), 0)
        # sum over data
        logp = torch.sum(logp)
        logp.backward()
        grads = np.concatenate((
            q.grad.detach().double().numpy(),
            az0.grad.detach().double().numpy(),
            pitch.grad.detach().double().numpy(),
            params.grad.detach().double().numpy()))
        outputs[0][0] = grads

def main(db, outfile, priors, loglike_net, grmfile, num_data=_NUM_DATA,
         spiral_params=_spiral_params, grm_params=_grm_params,
         warp_params=_warp_params, disk_params=_disk_params,
         Rmin=_RMIN, Rmax=_RMAX, Rref=_RREF, num_spirals=_NUM_SPIRALS,
         niter=_NITER, ntune=_NTUNE, ninit=_NINIT, num_chains=_NUM_CHAINS,
         step_scale=_STEP_SCALE, target_accept=_TARGET_ACCEPT,
         fixed=_FIXED, outliers=_OUTLIERS, overwrite=_OVERWRITE):
    """
    Use MCMC to generate spiral model posteriors for real or
    synthetic HII region data.

    Inputs:
      db :: string
        HII region database, or 'synthetic'
      outfile :: string
        Where the MCMC trace is saved (as Pickle)
      priors :: dictionary
        Priors for each paramter. The keys must be the parameter names:
            q, az0, pitch, sigmaV, sigma_arm_plane, sigma_arm_height,
            R0, Usun, Vsun, Upec, Vpec, a2, a3, Zsun, roll,
            warp_amp, warp_off
        The value of each key must be a list with one of the following
        formats. The values are repeated for each arm/warp mode.
          ['dirichlet']
          ['normal', mean1, width1, mean2, width2, ...]
          ['halfnormal', width1, width2, ...]
          ['cauchy', mean1, width1, mean2, width2, ...]
          ['halfcauchy', width1, width2, ...]
          ['uniform', lower1, upper1, lower2, upper2, ...]
          ['fixed', value1, value2, ...]
      loglike_net :: string
        Pickle file containing the SBI posterior and likelihood neural
        network
      grmfile :: string
        File containing GRM KDEs
      num_data :: integer
        The number of synthetic data to generate
      spiral_params :: 1-D array of scalars
        The synthetic data spiral parameters
        [qs, az0s, pitchs, sigmaV, sigma_arm_plane, sigma_arm_height]
      grm_params :: 1-D array of scalars
        Synthetic GRM parameters
        [R0, Usun, Vsun, Wsun, Upec, Vpec, a2, a3, Zsun, roll]
      warp_params :: 1-D array of scalars
        Synthetic warp parameters [warp_amp, warp_off]
      disk_params :: 1-D array of scalars
        Synthetic exponential disk parameters [I2, Rs, Rc]
      Rmin, Rmax :: scalars (kpc)
        The minimum and maximum radii of the spirals
      Rref :: scalar (kpc)
        The radius where the arm crosses the reference azimuth
      num_spirals :: integer
        The number of spirals
      niter :: integer
        Number of MCMC iterations per chain
      ntune :: integer
        Number of tuning/warm-up interations per chain
      ninit :: integer
        Number of ADVI initialization samples
      num_chains :: integer
        Number of Markov chains
      step_scale :: scalar
        Starting NUTS step_scale. Starting step_size is:
        step_scale / ndim**0.25
      target_accept :: scalar
        Desired acceptance rate (0 to 1)
      fixed :: dictionary
        Fixed GRM parameters (keys) and their fixed values.
      outliers :: list of strings
        Remove sources with these gnames from the analysis
      overwrite :: boolean
        If True, overwrite outfile if it exists.

    Returns: Nothing
    """
    #
    # Check that outfile does not already exist
    #
    if os.path.exists(outfile) and not overwrite:
        raise ValueError("{0} already exists!".format(outfile))
    #
    # Get the HII region data
    #
    if db == 'synthetic':
        q = spiral_params[:num_spirals]
        thetas = [
            torch.as_tensor(
                [spiral_params[num_spirals+i]]+
                [spiral_params[2*num_spirals+i]]+
                spiral_params[3*num_spirals:]+grm_params+warp_params)
            for i in range(num_spirals)]
        data = torch.cat(tuple(
            simulator(
                theta.expand(int(qi*num_data), -1),
                Rmin=torch.tensor(Rmin), Rmax=torch.tensor(Rmax),
                Rref=torch.tensor(Rref), disk=_disk_params)
            for theta, qi in zip(thetas, q)))
    else:
        raise NotImplementedError("synthetic data only")
    #
    # Get GRM KDEs
    #
    print("Reading GRM data from {0}".format(grmfile))
    with open(grmfile, 'rb') as f:
        grm_kdes = pickle.load(f)
    grm_kde_params = [key for key in grm_kdes.keys() if key != 'full']
    # approximate GRM KDEs as Gaussians
    grm_samples = grm_kdes['full'].resample(10000)
    grm_mus = np.mean(grm_samples, axis=1)
    grm_cov = np.cov(grm_samples, rowvar=1)
    grm_std = np.sqrt(np.diag(grm_cov))
    for param, mu, std in zip(grm_kde_params, grm_mus, grm_std):
        priors[param] = ['normal', mu, std]
    #
    # Get SBI posterior and likelihood neural network
    #
    with open(loglike_net, 'rb') as f:
        net = pickle.load(f)
    #
    # Create likelihood Op
    #
    loglike = Loglike(
        net['posterior'].net, data.float(), num_spirals)
    #
    # Setup model
    #
    with pm.Model() as model:
        #
        # Get parameter priors
        #
        determ = {}
        for param in priors:
            if param in fixed:
                continue
            num = 1
            shape = ()
            if param in ['q', 'az0', 'pitch']:
                num = num_spirals
                shape = (num,)
            if priors[param][0] == 'fixed':
                determ[param] = np.array(priors[param][1:])
            elif priors[param][0] == 'dirichlet':
                if num > 1:
                    determ[param] = pm.Dirichlet(
                        param, a=np.ones(num),
                        testval=np.ones(num)/num_spirals)
                else:
                    determ[param] = np.array([1.0])
            elif priors[param][0] == 'uniform':
                lower = np.array(priors[param][1:2*num+1:2])
                upper = np.array(priors[param][2:2*num+1:2])
                if len(shape) == 0:
                    lower = lower[0]
                    upper = upper[0]
                determ[param] = pm.Uniform(
                    param, lower=lower, upper=upper, shape=shape,
                    testval=(upper-lower)/2.0 + lower)
            elif priors[param][0] == 'normal':
                mean = np.array(priors[param][1:2*num+1:2])
                sigma = np.array(priors[param][2:2*num+1:2])
                if len(shape) == 0:
                    mean = mean[0]
                    sigma = sigma[0]
                determ[param] = pm.Normal(
                    param, mu=mean, sigma=sigma, shape=shape,
                    testval=mean)
            elif priors[param][0] == 'cauchy':
                alpha = np.array(priors[param][1:2*num+1:2])
                beta = np.array(priors[param][2:2*num+1:2])
                if len(shape) == 0:
                    alpha = alpha[0]
                    beta = beta[0]
                determ[param] = pm.Cauchy(
                    param, alpha=alpha, beta=beta, shape=shape,
                    testval=alpha)
            elif priors[param][0] == 'halfnormal':
                sigma = np.array(priors[param][1:num+1])
                if len(shape) == 0:
                    sigma = sigma[0]
                determ[param] = pm.HalfNormal(
                    param, sigma=sigma, shape=shape, testval=sigma)
            elif priors[param][0] == 'halfcauchy':
                beta = np.array(priors[param][1:num+1])
                if len(shape) == 0:
                    beta = beta[0]
                determ[param] = pm.HalfCauchy(
                    param, beta=beta, shape=shape, testval=beta)
            else:
                raise ValueError(
                    "Invalid prior for {0}: {1}".format(param, priors[param][0]))
        #
        # Add fixed parameters
        #
        for param, value in fixed.items():
           determ[param] = value
        #
        # Pack model parameters expected by likelihood net
        #
        theta = []
        params = [
            'sigmaV', 'sigma_arm_plane', 'sigma_arm_height', 'R0',
            'Usun', 'Vsun', 'Wsun', 'Upec', 'Vpec', 'a2', 'a3',
            'Zsun', 'roll', 'warp_amp', 'warp_off']
        for p in params:
            if p in net['priors']:
                theta += [determ[p]]
        theta = tt.as_tensor_variable(theta)
        theta = tt.concatenate([
            determ['q'], determ['az0'], determ['pitch'], theta])
        #
        # Evalulate likelihood
        #
        like = pm.DensityDist(
            'like', loglike, observed=theta)
    #
    # Run inference
    #
    with model:
        trace = pm.sample(
            niter, init='advi', tune=ntune, n_init=ninit,
            cores=num_chains, chains=num_chains,
            target_accept=target_accept, step_scale=step_scale)
        with open(outfile, 'wb') as f:
            dill.dump({"data": data, "model": model, "trace": trace}, f)
    print(pm.summary(trace).to_string())

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Sample Spiral Model Posterior using MCMC",
        prog="mcmc_posterior.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PARSER.add_argument(
        "dbfile", type=str,
        help="The HII region catalog database filename. If 'synthetic', generate synthetic data.")
    PARSER.add_argument(
        "outfile", type=str,
        help="Where the MCMC model and trace are stored (.pkl extension)")
    PARSER.add_argument(
        "loglike_net", type=str,
        help="Where the likelihood neural network is stored(.pkl extension)")
    PARSER.add_argument(
        "grmfile", type=str,
        help="Where the GRM KDEs are stored (.pkl extension)")
    PARSER.add_argument(
        "-n", "--niter", type=int, default=_NITER,
        help="Maximum number of MCMC iterations")
    PARSER.add_argument(
        "--num_data", type=int, default=_NUM_DATA,
        help="Number of synthetic data to generate.")
    PARSER.add_argument(
        "--spiral_params", nargs="+", type=float, default=_spiral_params,
        help="Spiral parameters for synthetic data")
    PARSER.add_argument(
        "--grm_params", nargs="+", type=float, default=_grm_params,
        help="GRM parameters for synthetic data")
    PARSER.add_argument(
        "--warp_params", nargs="+", type=float, default=_warp_params,
        help="Warp parameters for synthetic data")
    PARSER.add_argument(
        "--disk_params", nargs="+", type=float, default=_disk_params,
        help="Exponential disk parameters for synthetic data")
    PARSER.add_argument(
        "--Rmin", type=float, default=_RMIN,
        help="Minimum Galactocentric radius (kpc)")
    PARSER.add_argument(
        "--Rmax", type=float, default=_RMAX,
        help="Maximum Galactocentric radius (kpc)")
    PARSER.add_argument(
        "--Rref", type=float, default=_RREF,
        help="Reference Galactocentric radius (kpc)")
    PARSER.add_argument(
        "--num_spirals", type=int, default=_NUM_SPIRALS,
        help="Number of spiral arms")
    PARSER.add_argument(
        "--pq", nargs="+", default=['dirichlet'],
        help="Prior on HII region arm assignment")
    DEFAULT_AZ0 = ['uniform', 0.0, 1.5, 1.5, 3.2, 3.2, 4.7, 4.7, 6.3]
    PARSER.add_argument(
        "--paz0", nargs="+", default=DEFAULT_AZ0,
        help="Prior on spiral azimuths at refR (radians)")
    DEFAULT_PITCH = ['uniform', 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]
    PARSER.add_argument(
        "--ppitch", nargs="+", default=DEFAULT_PITCH,
        help="Prior on spiral pitch angles")
    PARSER.add_argument(
        "--psigmaV", nargs="+", default=["halfnormal", 10.0],
        help="Prior on HII region streaming velocity (km/s)")
    PARSER.add_argument(
        "--psigma_arm_plane", nargs="+", default=["halfnormal", 1.0],
        help="Prior on arm width in the plane (kpc)")
    PARSER.add_argument(
        "--psigma_arm_height", nargs="+", default=["halfnormal", 0.5],
        help="Prior on arm width perpendicular to the plane (kpc)")
    PARSER.add_argument(
        "--pgrm", action='append', nargs="+", default=[],
        help="Priors on GRM parameters (like: R0 normal 8.5 0.5 Zsun normal 5.5 25.0)")
    PARSER.add_argument(
        "--pwarp_amp", nargs="+", default=["halfnormal", 0.05],
        help="Prior on warp mode amplitude (kpc-1)")
    PARSER.add_argument(
        "--pwarp_off", nargs="+", default=["normal", -0.5, 1.0],
        help="Prior on warp mode offset (radians)")
    PARSER.add_argument(
        "-f", "--fixed", nargs="+", default=[],
        help=("Fixed GRM parameter names followed by their fixed value."))
    PARSER.add_argument(
        "-o", "--outliers", nargs="+", default=_OUTLIERS,
        help="HII regions to exclude from analysis")
    PARSER.add_argument(
        "--chains", type=int, default=_NUM_CHAINS,
        help="Number of Markov chains")
    PARSER.add_argument(
        "--ntune", type=int, default=_NTUNE,
        help="Number of MCMC tuning iterations")
    PARSER.add_argument(
        "--ninit", type=int, default=_NINIT,
        help="Number of ADVI initialzation samples")
    PARSER.add_argument(
        "--step_scale", type=float, default=_STEP_SCALE,
        help="Starting NUTS step_scale.")
    PARSER.add_argument(
        "--target_accept", type=float, default=_TARGET_ACCEPT,
        help="Desired acceptance rate.")
    PARSER.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outfile")
    ARGS = vars(PARSER.parse_args())
    #
    # Generate priors dictionary
    #
    PARAMS = [
        'q', 'az0', 'pitch', 'sigmaV', 'sigma_arm_plane',
        'sigma_arm_height', 'warp_amp', 'warp_off']
    PRIORS = {
        P: [ARGS['p'+P][0]] + [float(v) for v in ARGS['p'+P][1:]]
        for P in PARAMS}
    for PGRM in ARGS['pgrm']:
        PRIORS[PGRM[0]] = [PGRM[1]] + [float(v) for v in PGRM[2:]]
    FIXED = {}
    for FIX in range(len(ARGS['fixed'])//2):
        FIXED[ARGS['fixed'][2*FIX]] = float(ARGS['fixed'][2*FIX+1])
    main(ARGS['dbfile'], ARGS['outfile'], PRIORS, ARGS['loglike_net'],
         ARGS['grmfile'], num_data=ARGS['num_data'],
         spiral_params=ARGS['spiral_params'], grm_params=ARGS['grm_params'],
         warp_params=ARGS['warp_params'], disk_params=ARGS['disk_params'],
         Rmin=ARGS['Rmin'], Rmax=ARGS['Rmax'], Rref=ARGS['Rref'],
         num_spirals=ARGS['num_spirals'], niter=ARGS['niter'],
         ntune=ARGS['ntune'], ninit=ARGS['ninit'], num_chains=ARGS['chains'],
         step_scale=ARGS['step_scale'], target_accept=ARGS['target_accept'],
         fixed=FIXED, outliers=ARGS['outliers'], overwrite=ARGS['overwrite'])

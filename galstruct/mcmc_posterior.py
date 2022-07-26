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
Trey Wenger - June 2022 - Updates for new pymc and aesara
"""

import os
import multiprocessing
import numpy as np
import argparse
import pickle
import dill
import sqlite3

import aesara
import aesara.tensor as at

import torch
import pymc as pm

from galstruct.model.simulator import simulator

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

aesara.config.floatX = "float32"

np.random.seed(1234)

# parameter order for likelihood function
_params = [
    "q",
    "az0",
    "pitch",
    "sigmaV",
    "sigma_arm_plane",
    "sigma_arm_height",
    "R0",
    "Usun",
    "Vsun",
    "Wsun",
    "Upec",
    "Vpec",
    "a2",
    "a3",
    "Zsun",
    "roll",
    "warp_amp",
    "warp_off",
]
# default values for synthetic spiral parameters
_spiral_params = [
    0.25,  # q
    0.25,
    0.25,
    0.25,
    np.deg2rad(72.0),  # az0
    np.deg2rad(137.0),
    np.deg2rad(252.0),
    np.deg2rad(317.0),
    np.deg2rad(14.0),  # pitch
    np.deg2rad(14.0),
    np.deg2rad(14.0),
    np.deg2rad(14.0),
    5.0,  # sigmaV
    0.5,  # sigma_arm_plane
    0.1,  # sigma_arm_height
]
# default values for GRM parameters
_grm_params = [
    8.16643777,  # R0
    10.4543041,  # Usun
    12.18499493,  # Vsun
    7.71886874,  # Wsun
    5.79095823,  # Upec
    -3.39171583,  # Vpec
    0.97757558,  # a2
    1.62261724,  # a3
    5.5,  # Zsun
    0.0,  # roll
]
# default values for warp parameters
_warp_params = [0.02, -0.5]  # warp_amp, warp_off
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


class LogLikeCalc(torch.nn.Module):
    def __init__(self, density_estimator, num_spirals, fixed):
        super(LogLikeCalc, self).__init__()
        self.density_estimator = density_estimator
        self.num_spirals = num_spirals
        self.free_params = []
        for p in _params:
            if p in fixed:
                setattr(self, p, torch.nn.Parameter(torch.tensor(fixed[p])))
            else:
                if p in ["q", "az0", "pitch"]:
                    setattr(
                        self, p, torch.nn.Parameter(torch.tensor([0.0] * num_spirals))
                    )
                else:
                    setattr(self, p, torch.nn.Parameter(torch.tensor(0.0)))
                self.free_params.append(getattr(self, p))

    def forward(self, data):
        # parameters that do not depend on spiral
        other_params = [getattr(self, p).reshape(1) for p in _params[3:]]
        # stack parameters for each likelihood evaluation
        thetas = [
            torch.cat(
                (
                    self.az0[i].reshape(1),
                    self.pitch[i].reshape(1),
                    *other_params,
                )
            )
            for i in range(self.num_spirals)
        ]
        # evaluate loglike for each spiral and data point.
        logp = torch.stack(
            [
                self.density_estimator.log_prob(
                    data, context=theta.expand(data.shape[0], -1)
                )
                for theta in thetas
            ]
        )
        # catch nans
        logp[torch.isnan(logp)] = -np.inf
        # marginalize over spiral
        logp = torch.logsumexp(logp + torch.log(self.q[..., None]), 0)
        # sum over data
        return torch.sum(logp).float()


class LogLike(at.Op):
    def __init__(self, loglike_calc, data):
        self.loglike_calc = loglike_calc
        self.data = data

    def make_node(self, *args):
        return aesara.graph.basic.Apply(
            self, args, [at.fscalar().type()] + [a.type() for a in args]
        )

    def perform(self, node, inputs, outputs):
        for param, value in zip(self.loglike_calc.free_params, inputs):
            param.data = torch.tensor(value)
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        # evaluate likelihood and gradients
        loglike = self.loglike_calc(self.data)
        loglike.backward()

        outputs[0][0] = loglike.detach().numpy()
        for i, param in enumerate(self.loglike_calc.free_params):
            outputs[i + 1][0] = param.grad.detach().numpy()

    def grad(self, inputs, gradients):
        return [gradients[0] * d for d in self(*inputs)[1:]]


def main(
    db,
    outfile,
    priors,
    loglike_net,
    num_data=_NUM_DATA,
    spiral_params=_spiral_params,
    grm_params=_grm_params,
    warp_params=_warp_params,
    disk_params=_disk_params,
    Rmin=_RMIN,
    Rmax=_RMAX,
    Rref=_RREF,
    num_spirals=_NUM_SPIRALS,
    niter=_NITER,
    ntune=_NTUNE,
    ninit=_NINIT,
    num_chains=_NUM_CHAINS,
    step_scale=_STEP_SCALE,
    target_accept=_TARGET_ACCEPT,
    fixed=_FIXED,
    outliers=_OUTLIERS,
    overwrite=_OVERWRITE,
    hiidb='/data/hii_v2_20201203.db'
):
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
    # Check that outfile does not already exist
    if os.path.exists(outfile) and not overwrite:
        raise ValueError("{0} already exists!".format(outfile))

    # Get data or generate synthetic data
    if db == "synthetic":
        q = spiral_params[:num_spirals]
        thetas = [
            torch.as_tensor(
                [spiral_params[num_spirals + i]]
                + [spiral_params[2 * num_spirals + i]]
                + spiral_params[3 * num_spirals :]
                + grm_params
                + warp_params
            )
            for i in range(num_spirals)
        ]
        data = torch.cat(
            tuple(
                simulator(
                    theta.expand(int(qi * num_data), -1),
                    Rmin=torch.tensor(Rmin),
                    Rmax=torch.tensor(Rmax),
                    Rref=torch.tensor(Rref),
                    disk=disk_params,
                )
                for theta, qi in zip(thetas, q)
            )
        ).float()
    else:
        print("Opening HII Region data from {}".format(db))
        
        with sqlite3.connect(db) as conn:
            cur = conn.cursor()
            cur.execute('PRAGMA foreign_keys = ON')
            #
            # Get previously-known HII Regions
            # (i.e. not GBT HRDS and not SHRDS)
            #
            cur.execute('''
            SELECT cat.gname,cat.kdar,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near,dis.tangent FROM Detections det
            INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
            INNER JOIN Catalog cat on catdet.catalog_id = cat.id
            INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id 
            WHERE det.vlsr IS NOT NULL AND det.source = "WISE Catalog" AND cat.kdar IS NOT NULL
            AND NOT INSTR(det.author, "Anderson") AND NOT INSTR(det.author, "Brown") AND NOT INSTR(det.author, "Wenger")
            AND dis.Rgal IS NOT NULL
            GROUP BY cat.gname, det.component
            ''')
            prehrds = np.array(cur.fetchall(),
                            dtype=[('gname', 'U15'), ('kdar','U1'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8'),('near','f8'),('tangent','f8')])
            print("{0} Pre-HRDS Detections".format(len(prehrds)))
            print(
                "{0} Pre-HRDS Detections with unique GName".format(len(np.unique(prehrds['gname']))))
            print()
            #
            # Get HII regions discovered by HRDS
            #
            cur.execute('''
            SELECT cat.gname,cat.kdar,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near,dis.tangent FROM Detections det
            INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
            INNER JOIN Catalog cat on catdet.catalog_id = cat.id
            INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id 
            WHERE det.vlsr IS NOT NULL AND det.source = 'WISE Catalog' AND INSTR(det.author, "Anderson")
            AND dis.Rgal IS NOT NULL AND cat.kdar IS NOT NULL
            GROUP BY cat.gname, det.component
            ''')
            hrds = np.array(cur.fetchall(),
                            dtype=[('gname', 'U15'), ('kdar','U1'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'), ('far','f8'),('near','f8'),('tangent','f8')])
            # remove any sources in previously-known
            good = np.array([gname not in prehrds['gname']
                            for gname in hrds['gname']])
            hrds = hrds[good]
            print("{0} HRDS Detections".format(len(hrds)))
            print("{0} HRDS Detections with unique GName".format(
                len(np.unique(hrds['gname']))))
            print()
            #
            # Get HII regions discovered by SHRDS Full Catalog
            # Limit to stacked detection with highest line_snr
            #
            cur.execute('''
            SELECT cat.gname,cat.kdar,det.glong,det.glat,det.vlsr,cat.radius,det.author,dis.Rgal,dis.far,dis.near,dis.tangent FROM Detections det 
            INNER JOIN CatalogDetections catdet on catdet.detection_id = det.id 
            INNER JOIN Catalog cat on catdet.catalog_id = cat.id
            INNER JOIN Distances_Reid2019 dis on dis.catalog_id = cat.id
            WHERE det.vlsr IS NOT NULL AND 
            ((det.source="SHRDS Full Catalog" AND det.lines="H88-H112") OR (det.source="SHRDS Pilot" AND det.lines="HS"))
            AND dis.Rgal IS NOT NULL AND cat.kdar IS NOT NULL
            GROUP BY cat.gname, det.component HAVING MAX(det.line_snr) ORDER BY cat.gname
            ''')
            shrds_full = np.array(cur.fetchall(),
                                dtype=[('gname', 'U15'), ('kdar','U1'), ('glong', 'f8'), ('glat', 'f8'), ('vlsr', 'f8'), ('radius', 'f8'), ('author', 'U100'), ('Rgal', 'f8'),('far','f8'),('near','f8'),('tangent','f8')])
            # remove any sources in previously-known or GBT HRDS
            good = np.array([(gname not in prehrds['gname']) and (gname not in hrds['gname'])
                            for gname in shrds_full['gname']])
            shrds_full = shrds_full[good]
            print("{0} SHRDS Full Catalog Detections".format(len(shrds_full)))
            print("{0} SHRDS Full Catalog Detections with unique GName".format(
                len(np.unique(shrds_full['gname']))))
            print()
            print("{0} Total Detections".format(
                len(prehrds)+len(hrds)+len(shrds_full)))
            print("{0} Total Detections with unique GName".format(len(np.unique(
                np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname']))))))
            print()
            #
            # Get all WISE Catalog objects
            #
            cur.execute('''
            SELECT cat.gname, cat.catalog  FROM Catalog cat
            ''')
            wise = np.array(cur.fetchall(),
                            dtype=[('gname', 'U15'), ('catalog', 'U1')])
            #
            # Get all continuum detections without RRL detections
            #
            cur.execute('''
            SELECT cat.gname, det.cont, det.vlsr, COALESCE(det.line_snr, 1.0) AS snr FROM Detections det
            INNER JOIN CatalogDetections catdet ON catdet.detection_id = det.id
            INNER JOIN Catalog cat ON catdet.catalog_id = cat.id
            WHERE det.source = 'SHRDS Full Catalog' AND det.lines = 'H88-H112'
            AND cat.catalog = 'Q'
            GROUP BY cat.gname HAVING MAX(snr)
            ''')
            wise_quiet = np.array(cur.fetchall(),
                                dtype=[('gname', 'U15'), ('cont', 'f8'), ('vlsr', 'f8'), ('snr', 'f8')])
            #
            # Count known
            #
            not_quiet = np.sum(np.isnan(wise_quiet['vlsr']))
            print("SHRDS found continuum emission but no RRL emission toward {0} sources".format(
                not_quiet))
            known = np.sum(wise['catalog'] == 'K')+len(set(wise['gname'][wise['catalog'] != 'K']
                                                        ).intersection(np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname']))))
            candidate = np.sum(wise['catalog'] == 'C') - len(set(wise['gname'][wise['catalog'] == 'C']).intersection(
                np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname'])))) + not_quiet
            quiet = np.sum(wise['catalog'] == 'Q') - len(set(wise['gname'][wise['catalog'] == 'Q']).intersection(
                np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname'])))) - not_quiet
            group = np.sum(wise['catalog'] == 'G') - len(set(wise['gname'][wise['catalog'] == 'G']
                                                            ).intersection(np.concatenate((prehrds['gname'], hrds['gname'], shrds_full['gname']))))
            print("Now WISE Catalog containers:")
            print("{0} known".format(known))
            print("{0} candidate".format(candidate))
            print("{0} quiet".format(quiet))
            print("{0} group".format(group))
            glongs = torch.cat([torch.tensor(prehrds["glong"]),torch.tensor(hrds['glong']),torch.tensor(shrds_full['glong'])])
            glats = torch.cat([torch.tensor(prehrds["glat"]),torch.tensor(hrds['glat']),torch.tensor(shrds_full['glat'])])
            vlsrs = torch.cat([torch.tensor(prehrds["vlsr"]),torch.tensor(hrds['vlsr']),torch.tensor(shrds_full['vlsr'])])
            data= torch.stack((glongs, glats, vlsrs)).T.float()
            
    # Get likelihood neural network object
    with open(loglike_net, "rb") as f:
        net = pickle.load(f)

    # Setup model
    with pm.Model() as model:
        # Get parameter priors
        determ = {}
        for param in priors:
            if param in fixed:
                continue
            num = 1
            shape = ()
            if param in ["q", "az0", "pitch"]:
                num = num_spirals
                shape = (num,)
            if priors[param][0] == "fixed":
                fixed[param] = np.array(priors[param][1:])
            elif priors[param][0] == "dirichlet":
                if num > 1:
                    determ[param] = pm.Dirichlet(
                        param,
                        a=np.ones(num),
                    )
                else:
                    fixed[param] = np.array([1.0])
            elif priors[param][0] == "uniform":
                lower = np.array(priors[param][1 : 2 * num + 1 : 2])
                upper = np.array(priors[param][2 : 2 * num + 1 : 2])
                if len(shape) == 0:
                    lower = lower[0]
                    upper = upper[0]
                determ[param] = pm.Uniform(
                    param,
                    lower=lower,
                    upper=upper,
                    shape=shape,
                )
            elif priors[param][0] == "normal":
                mean = np.array(priors[param][1 : 2 * num + 1 : 2])
                sigma = np.array(priors[param][2 : 2 * num + 1 : 2])
                if len(shape) == 0:
                    mean = mean[0]
                    sigma = sigma[0]
                determ[param] = pm.Normal(
                    param,
                    mu=mean,
                    sigma=sigma,
                    shape=shape,
                )
            elif priors[param][0] == "cauchy":
                alpha = np.array(priors[param][1 : 2 * num + 1 : 2])
                beta = np.array(priors[param][2 : 2 * num + 1 : 2])
                if len(shape) == 0:
                    alpha = alpha[0]
                    beta = beta[0]
                determ[param] = pm.Cauchy(
                    param,
                    alpha=alpha,
                    beta=beta,
                    shape=shape,
                )
            elif priors[param][0] == "halfnormal":
                sigma = np.array(priors[param][1 : num + 1])
                if len(shape) == 0:
                    sigma = sigma[0]
                determ[param] = pm.HalfNormal(
                    param,
                    sigma=sigma,
                    shape=shape,
                )
            elif priors[param][0] == "halfcauchy":
                beta = np.array(priors[param][1 : num + 1])
                if len(shape) == 0:
                    beta = beta[0]
                determ[param] = pm.HalfCauchy(
                    param,
                    beta=beta,
                    shape=shape,
                )
            else:
                raise ValueError(
                    "Invalid prior for {0}: {1}".format(param, priors[param][0])
                )

        # Pack model parameters
        theta = [determ[p] for p in _params if p not in fixed]

        # Create likelihood Operator
        loglike_calc = LogLikeCalc(net["density_estimator"], num_spirals, fixed)
        loglike_op = LogLike(loglike_calc, data)

        # Evalulate likelihood
        like = pm.Potential("like", loglike_op(*theta)[0])

    # Run inference
    with model:
        trace = pm.sample(
            niter,
            init="advi",
            tune=ntune,
            n_init=ninit,
            cores=num_chains,
            chains=num_chains,
            target_accept=target_accept,
        )
        with open(outfile, "wb") as f:
            dill.dump({"data": data, "trace": trace}, f)
            # following fails due to issue with dill in python >3.8
            # dill.dump({"model": model}, f)
    print(pm.summary(trace).to_string())


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Sample Spiral Model Posterior using MCMC",
        prog="mcmc_posterior.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "dbfile",
        type=str,
        help="The HII region catalog database filename. If 'synthetic', generate synthetic data.",
        default="data/hii_v2_20201203.db"
    )
    PARSER.add_argument(
        "outfile",
        type=str,
        help="Where the MCMC model and trace are stored (.pkl extension)",
    )
    PARSER.add_argument(
        "loglike_net",
        type=str,
        help="Where the likelihood neural network is stored (.pkl extension)",
    )
    PARSER.add_argument(
        "-n",
        "--niter",
        type=int,
        default=_NITER,
        help="Maximum number of MCMC iterations",
    )
    PARSER.add_argument(
        "--num_data",
        type=int,
        default=_NUM_DATA,
        help="Number of synthetic data to generate.",
    )
    PARSER.add_argument(
        "--spiral_params",
        nargs="+",
        type=float,
        default=_spiral_params,
        help="Spiral parameters for synthetic data",
    )
    PARSER.add_argument(
        "--grm_params",
        nargs="+",
        type=float,
        default=_grm_params,
        help="GRM parameters for synthetic data",
    )
    PARSER.add_argument(
        "--warp_params",
        nargs="+",
        type=float,
        default=_warp_params,
        help="Warp parameters for synthetic data",
    )
    PARSER.add_argument(
        "--disk_params",
        nargs="+",
        type=float,
        default=_disk_params,
        help="Exponential disk parameters for synthetic data",
    )
    PARSER.add_argument(
        "--Rmin", type=float, default=_RMIN, help="Minimum Galactocentric radius (kpc)"
    )
    PARSER.add_argument(
        "--Rmax", type=float, default=_RMAX, help="Maximum Galactocentric radius (kpc)"
    )
    PARSER.add_argument(
        "--Rref",
        type=float,
        default=_RREF,
        help="Reference Galactocentric radius (kpc)",
    )
    PARSER.add_argument(
        "--num_spirals", type=int, default=_NUM_SPIRALS, help="Number of spiral arms"
    )
    DEFAULT_PRIORS = [
        ["q", "dirichlet"],
        ["az0", "uniform", 0.0, 1.5, 1.5, 3.2, 3.2, 4.7, 4.7, 6.3],
        ["pitch", "uniform", 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4],
        ["sigmaV", "halfnormal", 10.0],
        ["sigma_arm_plane", "halfnormal", 1.0],
        ["sigma_arm_height", "halfnormal", 0.5],
        ["R0", "normal", 8.5, 0.5],
        ["Usun", "normal", 10.5, 1.0],
        ["Vsun", "normal", 12.2, 10.0],
        ["Wsun", "normal", 7.5, 2.5],
        ["Upec", "normal", 5.8, 10.0],
        ["Vpec", "normal", -3.5, 10.0],
        ["a2", "normal", 0.977, 0.01],
        ["a3", "normal", 1.622, 0.01],
        ["Zsun", "normal", 5.5, 10.0],
        ["roll", "normal", 0.0, 0.05],
        ["warp_amp", "halfnormal", 0.05],
        ["warp_off", "normal", -0.5, 1.0],
    ]
    PARSER.add_argument(
        "--prior",
        action="append",
        nargs="+",
        default=DEFAULT_PRIORS,
        help=(
            "Priors on model parameters (e.g., --prior R0 normal 8.5 0.5 "
            + "--prior az0 uniform 0.0 6.3 --prior sigmaV halfnormal 10.0)"
        ),
    )
    PARSER.add_argument(
        "-f",
        "--fixed",
        nargs="+",
        default=[],
        help=("Fixed GRM parameter names followed by their fixed value."),
    )
    PARSER.add_argument(
        "-o",
        "--outliers",
        nargs="+",
        default=_OUTLIERS,
        help="HII regions to exclude from analysis",
    )
    PARSER.add_argument(
        "--chains", type=int, default=_NUM_CHAINS, help="Number of Markov chains"
    )
    PARSER.add_argument(
        "--ntune", type=int, default=_NTUNE, help="Number of MCMC tuning iterations"
    )
    PARSER.add_argument(
        "--ninit", type=int, default=_NINIT, help="Number of ADVI initialzation samples"
    )
    PARSER.add_argument(
        "--step_scale",
        type=float,
        default=_STEP_SCALE,
        help="Starting NUTS step_scale.",
    )
    PARSER.add_argument(
        "--target_accept",
        type=float,
        default=_TARGET_ACCEPT,
        help="Desired acceptance rate.",
    )
    PARSER.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outfile"
    )
    ARGS = vars(PARSER.parse_args())

    # Generate priors dictionary
    PARAMS = [
        "q",
        "az0",
        "pitch",
        "sigmaV",
        "sigma_arm_plane",
        "sigma_arm_height",
        "R0",
        "Usun",
        "Vsun",
        "Wsun",
        "Upec",
        "Vpec",
        "a2",
        "a3",
        "Zsun",
        "roll",
        "warp_amp",
        "warp_off",
    ]
    PRIORS = {}
    FIXED = {}
    for PARAM in PARAMS:
        for FIX in ARGS["fixed"]:
            if FIX[0] == PARAM:
                FIXED[PARAM] = [float(v) for v in FIX[1:]]
        if PARAM in FIXED:
            continue
        FOUND = False
        for PRIOR in ARGS["prior"]:
            if PRIOR[0] == PARAM:
                PRIORS[PARAM] = [PRIOR[1]] + [float(v) for v in PRIOR[2:]]
                FOUND = True
        if not FOUND:
            for PRIOR in DEFAULT_PRIORS:
                if PRIOR[0] == PARAM:
                    PRIORS[PARAM] = [PRIOR[1]] + [float(v) for v in PRIOR[2:]]

    main(
        ARGS["dbfile"],
        ARGS["outfile"],
        PRIORS,
        ARGS["loglike_net"],
        num_data=ARGS["num_data"],
        spiral_params=ARGS["spiral_params"],
        grm_params=ARGS["grm_params"],
        warp_params=ARGS["warp_params"],
        disk_params=ARGS["disk_params"],
        Rmin=ARGS["Rmin"],
        Rmax=ARGS["Rmax"],
        Rref=ARGS["Rref"],
        num_spirals=ARGS["num_spirals"],
        niter=ARGS["niter"],
        ntune=ARGS["ntune"],
        ninit=ARGS["ninit"],
        num_chains=ARGS["chains"],
        step_scale=ARGS["step_scale"],
        target_accept=ARGS["target_accept"],
        fixed=FIXED,
        outliers=ARGS["outliers"],
        overwrite=ARGS["overwrite"],
    )

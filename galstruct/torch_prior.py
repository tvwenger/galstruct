#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
torch_prior.py

Define a complicated prior object for use with pyTorch and SBI.

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
import torch.distributions as dist


class Prior:
    """
    Defines a Prior object, which returns the total log prior
    probability and samples from the prior.
    """

    def __init__(self, priors):
        """
        Initialize a new Prior object.

        Inputs:
          priors :: dictionary
            Priors for each paramter. The keys must be the parameter names:
              az0, pitch, sigmaV, sigma_arm_plane, sigma_arm_height,
              R0, Usun, Vsun, Upec, Vpec, a2, a3, Zsun, roll,
              warp_amp, warp_off
            The value of each key must be a list with one of the following
            formats.
              ['normal', mean, width]
              ['halfnormal', width]
              ['cauchy', mode, scale]
              ['halfcauchy', scale]
              ['uniform', lower, upper]
              ['fixed', value]

        Returns: prior
          prior :: a new Prior object
        """
        self.priors = []
        param_names = [
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
        for name in param_names:
            if name not in priors:
                raise ValueError(f"Invalid parameter name {name}")
            if priors[name][0] == "uniform":
                self.priors.append(dist.Uniform(priors[name][1], priors[name][2]))
            elif priors[name][0] == "normal":
                self.priors.append(dist.Normal(priors[name][1], priors[name][2]))
            elif priors[name][0] == "cauchy":
                self.priors.append(dist.Cauchy(priors[name][1], priors[name][2]))
            elif priors[name][0] == "halfnormal":
                self.priors.append(dist.HalfNormal(priors[name][1]))
            elif priors[name][0] == "halfcauchy":
                self.priors.append(dist.HalfCauchy(priors[name][1]))
            else:
                raise ValueError(
                    "Invalid prior type {0} for {1}".format(priors[name][0], name)
                )

    def sample(self, sample_shape=()):
        """
        Return a sample from the priors.

        Inputs:
          sample_shape :: tuple
            Output sample shape (num_samples, parameter_dim)

        Returns: samp
          samp :: 1-D torch.tensor
            A sample of the parameters
        """
        if len(sample_shape) == 0:
            samp = tt.empty(1, len(self.priors))
        else:
            samp = tt.empty(sample_shape[0], len(self.priors))
        for i in range(samp.shape[0]):
            samp[i] = tt.tensor([p.rsample() for p in self.priors])
        if len(sample_shape) == 0:
            return samp[0]
        return samp

    def log_prob(self, value):
        """
        Return the log probability of the priors evaluated at a given
        position.

        Inputs:
          value :: torch.tensor
            Parameter position(s)

        Returns: log_p
          log_p :: scalar
            Log probability
        """
        if value.ndim > 1:
            log_p = tt.empty(value.shape[0])
            for i, val in enumerate(value):
                log_p[i] = tt.sum(
                    tt.tensor([p.log_prob(v) for p, v in zip(self.priors, val)])
                )
        else:
            log_p = tt.sum(
                tt.tensor([p.log_prob(v) for p, v in zip(self.priors, value)])
            )
        return log_p

import torch
import numpy as np
import pytensor
import pytensor.tensor as pt


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
                    setattr(self, p, torch.nn.Parameter(torch.tensor([0.0] * num_spirals)))
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
        logp = torch.stack([self.density_estimator.log_prob(data[:, None, :], theta[None, :]) for theta in thetas])[
            :, :, 0
        ]

        # catch nans
        logp[torch.isnan(logp)] = -np.inf

        # marginalize over spiral
        logp = torch.logsumexp(logp + torch.log(self.q[..., None]), 0)

        # sum over data
        return torch.sum(logp)


class LogLike(pt.Op):
    def __init__(self, loglike_calc, data):
        self.loglike_calc = loglike_calc
        self.data = data

    def make_node(self, *args):
        return pytensor.graph.basic.Apply(self, args, [pt.fscalar().type()] + [a.type() for a in args])

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

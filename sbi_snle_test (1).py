import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from sbi import utils
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
import pickle
import numpy as np
import matplotlib.pyplot as pl
import corner
import pymc as pm
# theano is now aesara
import aesara
aesara.config.floatX = "float32"
import aesara.tensor as at

import multiprocessing
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

def simulator(theta):
    # simulate some y data for random values of x
    slope, sigma, intercept = theta
    x = np.random.rand()
    return [x, intercept + slope * x + np.random.randn()*sigma]

class LoglikeCalc(torch.nn.Module):
    def __init__(self, density_estimator, params, num_components):
        super(LoglikeCalc, self).__init__()
        self.params = []
        for p in params:
            if p in ["q", "slope", "sigma"]:
                setattr(self, p, torch.nn.Parameter(
                    torch.tensor([0.0]*num_components)))
            else:
                setattr(self, p, torch.nn.Parameter(
                    torch.tensor(0.0)))
            self.params.append(getattr(self, p))
        self.density_estimator = density_estimator
        self.num_components = num_components
        # fix q if necessary
        if num_components == 1:
            self.q = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, data):
        contexts = [
            torch.cat([
                self.slope[i].reshape(1), self.sigma[i].reshape(1),
                self.intercept.reshape(1)])
            for i in range(self.num_components)]
        logp = torch.stack([
            self.density_estimator.log_prob(
                data, context=context.expand(data.shape[0], -1))
            for context in contexts])
        # marginalize over component
        logp = torch.logsumexp(logp + torch.log(self.q[..., None]), 0)
        # product over data
        logp = torch.sum(logp)
        return logp.float()
        

class LoglikeOp(at.Op):
    def __init__(self, loglike_calc, data):
        self.loglike_calc = loglike_calc
        self.data = data

    def make_node(self, *args):
        return aesara.graph.basic.Apply(
            self, args, [at.fscalar().type()] + [a.type() for a in args]
        )

    def perform(self, node, inputs, outputs):
        # set RV values and reset grad if necessary
        for param, value in zip(self.loglike_calc.params, inputs):
            param.data = torch.tensor(value)
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        # evaluate likelihood and gradients
        loglike = self.loglike_calc(self.data)
        loglike.backward()

        outputs[0][0] = loglike.detach().numpy()
        for i, param in enumerate(self.loglike_calc.params):
            outputs[i+1][0] = param.grad.numpy()

    def grad(self, inputs, gradients):
        return [gradients[0] * d for d in self(*inputs)[1:]]

def learn_likelihood(model_path):
    # define parameter prior
    prior = utils.BoxUniform(low=[-2, 0.01, 0.0], high=[0, 0.5, 10.0])

    # prepare model for training
    sim,prior = prepare_for_sbi(simulator,prior)

    density_estimator = "maf" #what does this stand for?
    hidden_features = 20 # how wide to make the neural network (number of dots per line)
    transform_layers = 5 # how deep to make the neural network (number of lines with dots)
    num_sims = 50000 # number of simulations to train on
    sim_batch_size = 1 # batch size for simulating (one point at a time)
    training_batch_size = 50 # batch size for training

    # build the density estimator (this will be trained to learn the likelihood function)
    density_estimator_build_fun = utils.likelihood_nn(
        model=density_estimator,
        hidden_features=hidden_features,
        num_transforms=transform_layers
    )

    # set up SNLE system with priors and likelihood estimator info
    inference = SNLE(prior=prior,density_estimator=density_estimator_build_fun)

    # simulate data for use in training
    theta, x = simulate_for_sbi(
        sim,
        proposal=prior,
        num_simulations=num_sims,
        simulation_batch_size=sim_batch_size,
    )

    # use data to train the neural network (calculate the likelihood function)
    density_estimator = inference.append_simulations(theta,x).train(
        training_batch_size=training_batch_size
    )

    # calculate posterior distribution from likelihood function
    posterior=inference.build_posterior(density_estimator)

    # Save model to a pickle file
    print("Pickling results to {0}".format(model_path))
    with open(model_path, "wb") as f:
        output = {
            "posterior": posterior,
            "density_estimator": density_estimator,
            "priors": prior,
        }
        pickle.dump(output, f)
    print("Done!")

def mcmc_posterior(model_path):
    # Get likelihood neural network object from that pickle file
    with open(model_path, "rb") as f:
        net = pickle.load(f)

    # generate synthetic observations
    num_components = 2
    num_sims = 1000
    if num_components == 1:
        q_true = [1.0]
        slope_true = [-1.5]
        sigma_true = [0.10]
        intercept_true = 2.0
    elif num_components == 2:
        q_true = [0.75, 0.25]
        slope_true = [-1.5, -0.5]
        sigma_true = [0.05, 0.10]
        intercept_true = 2.0
    else:
        raise ValueError("only 1 or 2 components supported")
    x = np.random.uniform(0.1, 0.9, num_sims)
    y = np.ones(num_sims)*np.nan
    start = 0
    end = None
    for i in range(len(q_true)):
        num = int(q_true[i] * num_sims)
        if i == len(q_true)-1:
            num = num_sims - start
        end = start + num
        y[start:end] = (
            intercept_true + slope_true[i]*x[start:end] +
            np.random.randn(num)*sigma_true[i])
        start = end
    data = np.array([x, y]).T.astype(np.float32)

    if num_components > 1:
        params = ["q", "slope", "sigma", "intercept"]
    else:
        params = ["slope", "sigma", "intercept"]
    loglike_calc = LoglikeCalc(
        net["density_estimator"], params, num_components)
    loglike_op = LoglikeOp(loglike_calc, data)
    
    linear_model = pm.Model()
    with linear_model:
        if num_components > 1:
            q = pm.Dirichlet(
                "q", a=np.ones(num_components),
                shape=(num_components,))
            slope = pm.Uniform(
                "slope", lower=[-2.0, -1.0], upper=[-1.0, 0.0],
                shape=(num_components,))
        else:
            slope = pm.Uniform(
                "slope", lower=-2.0, upper=0.0,
                shape=(num_components,))

        sigma = pm.Uniform(
            "sigma", lower=0.01, upper=0.5, shape=(num_components,))
        intercept = pm.Uniform(
            "intercept", lower=0.0, upper=10.0)
        if num_components > 1:
            theta = [q, slope, sigma, intercept]
        else:
            theta = [slope, sigma, intercept]
        like = pm.Potential("like", loglike_op(*theta)[0])

    num_samples = 1000
    num_chains = 8
    target_accept = 0.9
    with linear_model:
        trace = pm.sample(
            num_samples,
            cores=num_chains,
            chains=num_chains,
            target_accept=target_accept,
            return_inferencedata=True,
        )
    
    # summary and plot
    print(pm.summary(trace))
    fig = corner.corner(
        trace, quantiles=[0.16, 0.5, 0.84],
        show_titles=True)
    pl.savefig("snle_line_cornerplot.pdf")

if __name__ == "__main__":
    model_path="data/snle_line_model.pkl" #where to save the neural network
    if not os.path.exists(model_path):
        learn_likelihood(model_path)
    mcmc_posterior(model_path)

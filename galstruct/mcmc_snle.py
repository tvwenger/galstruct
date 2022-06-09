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
    slope, sigma = theta
    x = np.random.rand()
    return [x, slope * x + np.random.randn()*sigma]

class LoglikeCalc(torch.nn.Module):
    def __init__(self, density_estimator):
        super(LoglikeCalc, self).__init__()
        self.slope = torch.nn.Parameter(torch.tensor(0.0))
        self.sigma = torch.nn.Parameter(torch.tensor(0.0))
        self.density_estimator = density_estimator

    def forward(self, data):
        context = torch.cat([self.slope.reshape(1), self.sigma.reshape(1)])
        loglike = self.density_estimator.log_prob(
            data, context=context.expand(data.shape[0], -1))
        loglike = torch.sum(loglike)
        return loglike.float()
        

class LoglikeOp(at.Op):
    itypes = [at.fscalar, at.fscalar]
    otypes = [at.fscalar, at.fscalar, at.fscalar]
    
    def __init__(self, loglike_calc, params, data):
        self.loglike_calc = loglike_calc
        self.params = params
        self.data = data

    # def make_node(self, *inputs):
    #     return aesara.graph.Apply(
    #         self, inputs, [at.fscalar().type(), at.fscalar().type(), at.fscalar().type()])    

    def perform(self, node, inputs, outputs):
        # set RV values and reset grad if necessary
        for param, value in zip(self.params, inputs):
            param.data = torch.tensor(value)
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        # evaluate likelihood and gradients
        loglike = self.loglike_calc(self.data)
        loglike.backward()

        outputs[0][0] = loglike.detach().numpy()
        for i, param in enumerate(self.params):
            outputs[i+1][0] = param.grad.numpy()

    def grad(self, inputs, gradients):
        return [gradients[0] * d for d in self(*inputs)[1:]]

def learn_likelihood(model_path):
    # define parameter prior
    prior = utils.BoxUniform(low=[-2, 0.1], high=[0, 0.5])

    # prepare model for training
    sim,prior = prepare_for_sbi(simulator,prior)

    density_estimator = "maf" #what does this stand for?
    hidden_features = 20 # how wide to make the neural network (number of dots per line)
    transform_layers = 5 # how deep to make the neural network (number of lines with dots)
    num_sims = 2000 # number of simulations to train on
    sim_batch_size = 1 # batch size for simulating (one point at a time)
    training_batch_size=50 # batch size for training
    fixed={} #???

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
            "fixed": fixed,
        }
        pickle.dump(output, f)
    print("Done!")

def mcmc_posterior(model_path):
    # Get likelihood neural network object from that pickle file
    with open(model_path, "rb") as f:
        net = pickle.load(f)

    # generate 100 observations                     
    slope_true = -1.0
    sigma_true = 0.25
    x = np.linspace(0.1, 0.9, 1000)
    y = slope_true * x + np.random.randn(len(x))*sigma_true
    data = np.array([x, y]).T.astype(np.float32)

    loglike_calc = LoglikeCalc(net["density_estimator"])
    params = [loglike_calc.slope, loglike_calc.sigma]
    loglike_op = LoglikeOp(loglike_calc, params, data)
    
    linear_model = pm.Model()
    with linear_model:
        slope = pm.Uniform("slope",lower=-2.0,upper=0.0)
        sigma = pm.Uniform("sigma",lower=0.1,upper=0.5)
        like = pm.Potential("like", loglike_op(slope, sigma)[0])

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
        trace, labels=["slope", r"$\sigma$"],
        truths=[slope_true, sigma_true], quantiles=[0.16, 0.5, 0.84],
        show_titles=True)
    pl.savefig("snle_line_cornerplot.pdf")

if __name__ == "__main__":
    model_path="data/snle_line_model.pkl" #where to save the neural network
    if not os.path.exists(model_path):
        learn_likelihood(model_path)
    mcmc_posterior(model_path)

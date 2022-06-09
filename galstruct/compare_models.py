import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributions as dis

from pathlib import Path

from sbi.utils import likelihood_nn
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi

from galstruct import learn_likelihood, plot_likelihood
from galstruct.model.simulator import simulator
from galstruct.model.likelihood import log_like

# default parameter values
_NUM_SIMS = [50000]
_DENSITY_ESTIMATOR = ["maf"]
_HIDDEN_FEATURES = [50]
_TRANSFORM_LAYERS = [5]
_SIM_BATCH_SIZE = [1]
_TRAINING_BATCH_SIZE = [50]
_RMIN = 3.0
_RMAX = 15.0
_RREF = 8.0
_FIXED = {}
_OVERWRITE = False
_THETA = [
    0.25,
    5.0,
    0.5,
    0.1,
    8.166,
    10.444,
    12.007,
    7.719,
    5.793,
    -3.562,
    0.978,
    1.623,
    5.5,
    0.0,
    0.0399,
    -0.5,
]
DEFAULT_PRIORS = [
    ["az0", "uniform", 0.0, 6.3],
    ["pitch", "uniform", 0.1, 0.4],
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
# Generate priors dictionary
PARAMS = [
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

def main(
    netpaths,
    num_data=500,
    num_param_samples=50,
    Rmin=3.0,
    Rmax=15.0,
    Rref=8.0,
    net_params={"type":"N/A", "list":[None]}
):
    """
    Calculates summary statistics for a collection of neural networks.
    
    Inputs:
        netpaths :: list of strings
            A list of paths to network .pkl files
        num_data :: int
            How many data points to use per parameter set
        num_param_samples :: int
            How many pairs of parameter sets to compare per neural network
        Rmin, Rmax :: float (kpc)
            The minimum and maximum radii of the spirals
        Rref :: float (kpc)
            The radius where the arm crosses the reference azimuth
        net_params :: {string, list of floats}

    Returns: Nothing
    """

    thetas1 = torch.zeros([len(_THETA),num_param_samples])
    thetas2 = torch.zeros([len(_THETA),num_param_samples])
    az0s_deg = np.linspace(0.0, 359.0, 360)
    az0s = np.deg2rad(az0s_deg)

    # data grid
    glong_axis = np.linspace(-np.pi, np.pi, 180)
    vlsr_axis = np.linspace(-150.0, 150.0, 150)
    glong_grid, vlsr_grid = np.meshgrid(glong_axis, vlsr_axis, indexing="ij")
    glong = glong_grid.flatten()
    vlsr = vlsr_grid.flatten()
    glat = np.zeros(len(glong))
    extent = [-150.0, 150.0, -180.0, 180.0]
    grid = np.stack((glong, glat, vlsr)).T
    grid = torch.tensor(grid).float()

    for i,p in enumerate(DEFAULT_PRIORS):
        if p[1]=="uniform":
            prior=dis.uniform.Uniform(torch.tensor([p[2]]),torch.tensor([p[3]]))
            thetas1[i,:] = prior.sample(num_param_samples)
            thetas2[i,:] = prior.sample(num_param_samples)
        elif p[1]=="normal":
            prior=dis.normal.Normal(torch.tensor([p[2]]),torch.tensor([p[3]]))
            thetas1[i,:] = prior.sample(num_param_samples)
            thetas2[i,:] = prior.sample(num_param_samples)
        elif p[1]=="halfnormal":
            prior=dis.half_normal.HalfNormal(torch.tensor([p[2]]))
            thetas1[i,:] = prior.sample(num_param_samples)
            thetas2[i,:] = prior.sample(num_param_samples)

    logpRatios        = np.empty((netpaths,2*num_param_samples),dtype=float)
    logpRatio_means   = np.empty((netpaths,1),dtype=float)
    logpRatio_stds    = np.empty((netpaths,1),dtype=float)
    logpRatio_medians = np.empty((netpaths,1),dtype=float)
    for k, netpath in enumerate(netpaths):
        for j in range(num_param_samples):
            data1 = simulator(
                thetas1[:,j].expand(num_data, -1),
                Rmin=torch.tensor(Rmin),
                Rmax=torch.tensor(Rmax),
                Rref=torch.tensor(Rref),
            )
            data2 = simulator(
                thetas2[:,j].expand(num_data, -1),
                Rmin=torch.tensor(Rmin),
                Rmax=torch.tensor(Rmax),
                Rref=torch.tensor(Rref),
            )
            with open(netpath, "rb") as f:
                net = pickle.load(f)
            # Grid learned likelihood data
            logpA1 = net["density_estimator"].log_prob(
                    data1, context=thetas1[:,j].expand(len(grid), -1)
                )
            logpA1 = logpA1.detach().numpy()
            logpA1 = logpA1.reshape(glong_grid.shape)
            
            logpA2 = net["density_estimator"].log_prob(
                    data1, context=thetas2[:,j].expand(len(grid), -1)
                )
            logpA2 = logpA2.detach().numpy()
            logpA2 = logpA2.reshape(glong_grid.shape)
            
            logpB1 = net["density_estimator"].log_prob(
                    data2, context=thetas1[:,j].expand(len(grid), -1)
                )
            logpB1 = logpB1.detach().numpy()
            logpB1 = logpB1.reshape(glong_grid.shape)
            
            logpB2 = net["density_estimator"].log_prob(
                    data2, context=thetas2[:,j].expand(len(grid), -1)
                )
            logpB2 = logpB2.detach().numpy()
            logpB2 = logpB2.reshape(glong_grid.shape)
            logpRatios[k,2*j]   = logpB1/logpB2
            logpRatios[k,2*j+1] = logpA1/logpA2
        logpRatio_means[k]   = np.mean(logpRatios[k,:])
        logpRatio_stds[k]    = np.std(logpRatios[k,:])
        logpRatio_medians[k] = np.median(logpRatios[k,:])
    
    fig1,axes1=plt.subplots(3,1,sharex=True)
    axes1[0].hist(logpRatio_means)
    axes1[1].hist(logpRatio_stds)
    axes1[2].hist(logpRatio_medians)
    plt.savefig("network_comparison_summarystats.pdf",dpi=300)

    if not(net_params["list"]==None):
        fig2,axes2=plt.subplots(3,1,sharex=True)
        axes2[0].plot(net_params,logpRatio_means)
        axes2[1].plot(net_params,logpRatio_stds)
        axes2[2].plot(net_params,logpRatio_medians)
        

if __name__=="__main__":
    PARSER = argparse.ArgumentParser(
        description="Train Neural Network for Spiral Model Likelihood",
        prog="compare_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "netlist",
        nargs="+",
        type=str,
        help="list of paths to neural networks"
    )
    PARSER.add_argument(
        "--ndata",
        type=int,
        help="number of data points to use per parameter set"
    )
    PARSER.add_argument(
        "--nparams",
        type=int,
        help="number of pairs of parameter sets to compare"
    )
    PARSER.add_argument(
        "--netparams",
        type=float,
        nargs="+",
        help="quantitative descriptors for each neural network"
    )
    PARSER.add_argument(
        "--netparamtype",
        type=str,
        help="description of above inputs"
    )
    ARGS=PARSER.parse_args()
    if not(ARGS["netparams"] == None) and not(len(ARGS["netparams"])==len(ARGS["netlist"])) :
        raise argparse.ArgumentError("Length of descriptor list must match length of path list")
             
import glob, os
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
    outpath,
    num_data,
    num_param_samples,
    Rmin=3.0,
    Rmax=15.0,
    Rref=8.0,
    net_params=None
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

    # Initialize empty containers for the parameter sets
    # rows = samples of each parameter, columns = sampled parameter set
    thetas = torch.zeros([len(DEFAULT_PRIORS),num_param_samples])


    # Generate samples of each parameter using the prior distributions
    for i,p in enumerate(DEFAULT_PRIORS):
        if p[1]=="uniform":
            # Uniform distribution - {"name", "uniform", lower=p[2], upper=p[3]}
            prior=dis.uniform.Uniform(torch.tensor([p[2]]),torch.tensor([p[3]]))
            thetas[i,:] = prior.sample((num_param_samples,1))[:,0,0]

        elif p[1]=="normal":
            # Normal distribution - {"name", "normal", mu=p[2], sigma=p[3]}
            prior=dis.normal.Normal(torch.tensor([p[2]]),torch.tensor([p[3]]))
            thetas[i,:] = prior.sample((num_param_samples,1))[:,0,0]

        elif p[1]=="halfnormal":
            # Half-Normal distribution - {"name", "halfnormal", sigma=p[2]}
            prior=dis.half_normal.HalfNormal(torch.tensor([p[2]]))
            thetas[i,:] = prior.sample((num_param_samples,1))[:,0,0]
        
        print("Parameter [{}/{}]\r".format(i+1,len(DEFAULT_PRIORS)),end='')
    print('\n')

    # Initialize empty arrays to store the stats and output data

    logp1_means  = []
    logp2_means  = []
    logp_stds    = []
    logp_medians = []
    logp_stats   = []
    # Calculate stats for each neural network

    # loop through pairs of neural networks
    for i in range(len(netpaths)):
        for j in range(i+1,len(netpaths)):
        # Get log-likelihoods for each parameter set
            logp1s=[]
            logp2s=[]
            for k in range(num_param_samples):
                print("i={}, j={}, k={}\r".format(i,j,k),end='')
                # Generate simulated dataset for this particular parameter set
                data = simulator(
                    thetas[:,k].expand(num_data, -1),
                    Rmin=torch.tensor(Rmin),
                    Rmax=torch.tensor(Rmax),
                    Rref=torch.tensor(Rref),
                )

                # load neural network
                with open(netpaths[i], "rb") as f:
                    net1 = pickle.load(f)
                with open(netpaths[j], "rb") as f:
                    net2 = pickle.load(f)
                  
                logp1s.append(np.nansum(net1["density_estimator"].log_prob(
                        data, context=thetas[:,k].expand(len(data), -1)
                    ).detach().numpy())  )     
                logp2s.append(np.nansum(net2["density_estimator"].log_prob(
                        data, context=thetas[:,j].expand(len(data), -1)
                    ).detach().numpy()))       
            logp1s=np.array(logp1s)
            logp2s=np.array(logp2s) 
                
            # Calculate summary statistics for each neural network
            logp1_means.append(np.mean(logp1s))
            logp2_means.append(np.mean(logp2s))
            logp_stds.append(np.std(logp1s-logp2s))
            logp_medians.append(np.median(logp1s-logp2s))
            logp_stats.append({"path1":netpaths[i], "path2":netpaths[j], "mean":np.abs(logp1_means[-1]-logp2_means[-1]),\
                "std":logp_stds[-1],"median":logp_medians[-1]})

    # Write stats to file
    with open(outpath+"/network_comparison_summarystats_2.txt",'w') as f:
        for i in range(len(logp_stats)):
            f.write(str(logp_stats[i])+"\n\n")
    
    # Plot histogram of summary statistics
    fig1,axes1=plt.subplots(3,1)
    axes1[0].hist(np.abs(np.array(logp1_means)-np.array(logp2_means)))
    axes1[0].set_xlabel(r"Mean score")
    axes1[1].hist(logp_stds)
    axes1[1].set_xlabel(r"Standard deviation of scores")
    axes1[2].hist(logp_medians)
    axes1[2].set_xlabel(r"Median score")
    plt.title("Summary Statistics")
    fig1.tight_layout()
    fig1.savefig(outpath+"/"+"network_comparison_summarystats2.pdf",dpi=300,bbox_inches='tight')
    plt.close()

    if not(net_params==None):
        # Plot statistics against network parameters
        fig2,axes2=plt.subplots(3,1,sharex=True)
        axes2[0].plot(net_params[1],logp_means)
        axes2[1].plot(net_params[1],logp_stds)
        axes2[2].plot(net_params[1],logp_medians)
        fig2.tight_layout()
        fig2.savefig(outpath+"network_comparison_summarystats22.pdf",dpi=300,bbox_inches='tight')
        plt.close()

if __name__=="__main__":
    PARSER = argparse.ArgumentParser(
        description="Compare Neural Networks for Spiral Model Likelihood",
        prog="compare_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "inpath",
        type=str,
        help="Path to folder of neural networks"
    )
    PARSER.add_argument(
        "outpath",
        type=str,
        help="Path to output data and plots"
    )
    PARSER.add_argument(
        "--ndata",
        type=int,
        default=1000,
        help="Number of data points to use per parameter set"
    )
    PARSER.add_argument(
        "--nparams",
        type=int,
        default=100,
        help="Number of pairs of parameter sets to compare"
    )
    PARSER.add_argument(
        "--netparams",
        type=float,
        nargs="+",
        help="Quantitative descriptors for each neural network. \
            Used to plot neural network statistics vs parameters"
    )
    PARSER.add_argument(
        "--netparams_name",
        type=str,
        help="Name of quantitative descriptor above (eg. features)"
    )
    ARGS=PARSER.parse_args()

    netlist= glob.glob(ARGS.inpath+'/*.pkl')

    if not(ARGS.netparams == None) and not(len(ARGS.netparams)==len(netlist)) :
        raise argparse.ArgumentError("Length of descriptor list must match length of path list")

    main(
        netlist,
        ARGS.outpath,
        num_data=ARGS.ndata,
        num_param_samples=ARGS.nparams,
        net_params=ARGS.netparams
    )
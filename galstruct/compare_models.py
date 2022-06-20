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
    varied_param=None,
    hyperparams=[]
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
    thetas1 = torch.zeros([len(DEFAULT_PRIORS),num_param_samples])
    thetas2 = torch.zeros([len(DEFAULT_PRIORS),num_param_samples])

    # Generate samples of each parameter using the prior distributions
    for i,p in enumerate(DEFAULT_PRIORS):
        if p[1]=="uniform":
            # Uniform distribution - {"name", "uniform", lower=p[2], upper=p[3]}
            prior=dis.uniform.Uniform(torch.tensor([p[2]]),torch.tensor([p[3]]))
            thetas1[i,:] = prior.sample((num_param_samples,1))[:,0,0]
            thetas2[i,:] = prior.sample((num_param_samples,1))[:,0,0]
        elif p[1]=="normal":
            # Normal distribution - {"name", "normal", mu=p[2], sigma=p[3]}
            prior=dis.normal.Normal(torch.tensor([p[2]]),torch.tensor([p[3]]))
            thetas1[i,:] = prior.sample((num_param_samples,1))[:,0,0]
            thetas2[i,:] = prior.sample((num_param_samples,1))[:,0,0]
        elif p[1]=="halfnormal":
            # Half-Normal distribution - {"name", "halfnormal", sigma=p[2]}
            prior=dis.half_normal.HalfNormal(torch.tensor([p[2]]))
            thetas1[i,:] = prior.sample((num_param_samples,1))[:,0,0]
            thetas2[i,:] = prior.sample((num_param_samples,1))[:,0,0]
        print("Parameter [{}/{}]".format(i+1,len(DEFAULT_PRIORS)))

    # Initialize empty arrays to store the stats and output data
    logpRatios        = np.empty((len(netpaths),2*num_param_samples),dtype=float)
    logpRatio_means   = np.empty((len(netpaths),1),dtype=float)
    logpRatio_stds    = np.empty((len(netpaths),1),dtype=float)
    logpRatio_medians = np.empty((len(netpaths),1),dtype=float)
    logp_stats        = []
    # Calculate stats for each neural network
 
    for k, netpath in enumerate(netpaths):
        # Get log-likelihoods for each parameter set
        logp1=[]
        logp2=[]
        for j in range(num_param_samples):
            if (j+1)%100==0:
                print("Iteration [{}/{},{}/{}]".format(k+1,len(netpaths),j+1,num_param_samples))
            # Generate simulated datasets for this particular parameter set
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

            # load neural network
            with open(netpath, "rb") as f:
                net = pickle.load(f)

            # calculate logL(data1|theta1),logL(data1|theta2),
            #           logL(data2|theta1),logL(data2|theta2)    
            logpA1 = np.nansum(net["density_estimator"].log_prob(
                    data1, context=thetas1[:,j].expand(len(data1), -1)
                ).detach().numpy())       
            logpA2 = np.nansum(net["density_estimator"].log_prob(
                    data1, context=thetas2[:,j].expand(len(data1), -1)
                ).detach().numpy())       
            logpB1 = np.nansum(net["density_estimator"].log_prob(
                    data2, context=thetas1[:,j].expand(len(data2), -1)
                ).detach().numpy())           
            logpB2 = np.nansum(net["density_estimator"].log_prob(
                    data2, context=thetas2[:,j].expand(len(data2), -1)
                ).detach().numpy())
            # Collect the ratios of the log likelihoods    
            logpRatios[k,2*j]   = logpB2-logpB1
            logpRatios[k,2*j+1] = logpA1-logpA2
            logp1.append(logpA1)
            logp1.append(logpB2)
            logp2.append(logpB1)
            logp2.append(logpA2)
        # Calculate summary statistics for each neural network
        logpRatio_means[k]   = np.mean(logpRatios[k,:])
        logpRatio_stds[k]    = np.std(logpRatios[k,:])
        logpRatio_medians[k] = np.median(logpRatios[k,:])
        print([logpRatio_means[k],logpRatio_stds[k],logpRatio_medians[k]])
        logp_stats.append({"path":netpath, "p1": np.mean(logp1), "p1":np.mean(logp2), "mean diff":logpRatio_means[k],\
            "std":logpRatio_stds[k],"median":logpRatio_medians[k]})

    # Write stats to file
    with open(outpath+"/network_comparison_summarystats.txt",'w') as f:
        for i in range(len(logp_stats)):
            f.write(str(logp_stats[i])+"\n\n")
    
    # Plot histogram of summary statistics
    fig1,axes1=plt.subplots(3,1)
    axes1[0].hist(logpRatio_means)
    axes1[0].set_xlabel(r"$\langle\log L(A|\alpha)-\log L(A|\beta)\rangle$")
    axes1[1].hist(logpRatio_stds)
    axes1[1].set_xlabel(r"$\sqrt{\rm{Var}(\log L(A|\alpha)-\log L(A|\beta))}$")
    axes1[2].hist(logpRatio_medians)
    axes1[2].set_xlabel(r"$\rm{median}(\log L(A|\alpha)-\log L(A|\beta))$")
    plt.title("Summary Statistics")
    fig1.tight_layout()
    fig1.savefig(outpath+"/"+"network_comparison_summarystats.pdf",dpi=300,bbox_inches='tight')
    plt.close()

    if not(varied_param==None):
        # Plot statistics against network parameters
        fig2,axes2=plt.subplots(3,1,sharex=True)
        axes2[0].scatter(hyperparams,logpRatio_means)
        axes2[0].set_xlabel(r"$\langle\log L(A|\alpha)-\log L(A|\beta)\rangle$")
        axes2[1].scatter(hyperparams,logpRatio_stds)
        axes2[1].set_xlabel(r"$\sqrt{\rm{Var}(\log L(A|\alpha)-\log L(A|\beta))}$")
        axes2[2].scatter(hyperparams,logpRatio_medians)
        axes2[2].set_xlabel(r"$\rm{median}(\log L(A|\alpha)-\log L(A|\beta))$")
        plt.title(varied_param)
        fig2.tight_layout()
        fig2.savefig(outpath+"network_comparison_summarystats_vs_params.pdf",dpi=300,bbox_inches='tight')
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
        default=100,
        help="Number of data points to use per parameter set"
    )
    PARSER.add_argument(
        "--nparams",
        type=int,
        default=1000,
        help="Number of pairs of parameter sets to compare"
    )
    PARSER.add_argument(
        "--variedparam",
        type=str,
        help="Quantitative descriptor for each neural network. \
            Used to plot neural network statistics vs parameters"
    )
    ARGS=PARSER.parse_args()

    # Get list of file names
    netlist= glob.glob(ARGS.inpath+'/*.pkl')

    # Empty list
    hyperparams=[]
    for fn in netlist:
        # Eliminate folder names
        fn=fn.rsplit('\\',1)[-1]
        # Find where the parameter is listed in the filepath
        idx    = fn.find(ARGS.variedparam)
        # Get length of descriptor part of the string
        length = len(ARGS.variedparam+'=')
        # Find where the number ends
        idx2   = fn.find("_",idx)
        # Get the numerical part of the filepath, convert to float, and save to list
        hyperparams.append(float(fn[idx+length:idx2]))

    # Sorts the file paths and parameters so that they can be plotted properly
    hyperparams, netlist= [list(t) for t in zip(*sorted(zip(hyperparams,netlist)))]

    main(
        netlist,
        ARGS.outpath,
        num_data=ARGS.ndata,
        num_param_samples=ARGS.nparams,
        varied_param=ARGS.variedparam,
        hyperparams=hyperparams
    )
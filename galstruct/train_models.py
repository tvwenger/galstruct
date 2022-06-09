import os
import argparse
import pickle
import numpy as np

import torch

from pathlib import Path

from sbi.utils import likelihood_nn
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi

from galstruct import learn_likelihood, plot_likelihood

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

if __name__=="__main__":
    PARSER = argparse.ArgumentParser(
        description="Train Neural Network for Spiral Model Likelihood",
        prog="train_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "outfile", type=str, help="Where the neural network is stored (.pkl extension)"
    )
    PARSER.add_argument(
        "-n",
        "--nsims",
        type=int,
        nargs="+",
        default=_NUM_SIMS,
        help="Number of simulated observations",
    )
    PARSER.add_argument(
        "--density_estimator",
        type=str,
        nargs="+",
        default=_DENSITY_ESTIMATOR,
        help="Either maf (Masked Autoregressive Flow) or nsf (Neural Spline Flow)",
    )
    PARSER.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=_HIDDEN_FEATURES,
        help="Number of neural spine flow hidden features",
    )
    PARSER.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=_TRANSFORM_LAYERS,
        help="Number of neural spine flow transform layers",
    )
    PARSER.add_argument(
        "--sim_batch_size",
        type=int,
        nargs="+",
        default=_SIM_BATCH_SIZE,
        help="Batch size for simulations",
    )
    PARSER.add_argument(
        "--training_batch_size",
        type=int,
        nargs="+",
        default=_TRAINING_BATCH_SIZE,
        help="Batch size for training",
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
        "--fixed",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Fixed parameter names followed by their fixed value "
            + "(e.g., --fixed R0 8.5 --fixed Usun 10.5)"
        ),
    )
    PARSER.add_argument(
        "--plot",
        default=False,
        help="Whether or not to generate all frames"
    )
    PARSER.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outfile"
    )
    ARGS = vars(PARSER.parse_args())

    PRIORS = {}
    FIXED = {}
    for PARAM in PARAMS:
        for FIX in ARGS["fixed"]:
            if FIX[0] == PARAM:
                FIXED[PARAM] = float(FIX[1])
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
    for DE in ARGS["density_estimator"]:
        for n_sims in ARGS["nsims"]:
            for feat in ARGS["features"]:
                for layers in ARGS["layers"]:
                    for sbs in ARGS["sim_batch_size"]:
                        for tbs in ARGS["training_batch_size"]:
                            if not(ARGS["plot"]):
                                learn_likelihood.main(
                                    ARGS["outfile"]+"_"+DE+"_nsims="+str(n_sims)+"_feat="+str(feat)+"_layers="+str(layers)\
                                        +"_sbs="+str(sbs)+"_tbs="+str(tbs)+".pkl",
                                    PRIORS,
                                    num_sims=n_sims,
                                    density_estimator=DE,
                                    hidden_features=feat,
                                    transform_layers=layers,
                                    sim_batch_size=sbs,
                                    training_batch_size=tbs,
                                    Rmin=ARGS["Rmin"],
                                    Rmax=ARGS["Rmax"],
                                    Rref=ARGS["Rref"],
                                    fixed=FIXED,
                                    overwrite=ARGS["overwrite"],
                                )
                            else:
                                outdir="plots/"+ARGS["outfile"]+"_"+DE+"_nsims="+str(n_sims)+"_feat="+str(feat)+"_layers="+str(layers)\
                                        +"_sbs="+str(sbs)+"_tbs="+str(tbs)+"/"
                                Path(outdir).mkdir(parents=True,exist_ok=True)
                                plot_likelihood.main(
                                    ARGS["outfile"]+"_"+DE+"_nsims="+str(n_sims)+"_feat="+str(feat)+"_layers="+str(layers)\
                                        +"_sbs="+str(sbs)+"_tbs="+str(tbs)+".pkl",
                                    outdir=outdir
                                )
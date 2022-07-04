import argparse
import os
import dill as pickle
import numpy as np
import matplotlib.pyplot as pl
import argparse 
import corner

if __name__=="__main__":
    PARSER = argparse.ArgumentParser(
        description="Cornerplot for Spiral Model Likelihood",
        prog="cornerplot.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument(
        "inpath", type=str, help="Path to trace file (.pkl extension)"
    )
    PARSER.add_argument(
        "outpath", type=str, help="Path to output plot to"
    )

    args = PARSER.parse_args()
    print("Loading {}".format(args.inpath))
    with open(args.inpath,"rb") as file:
        data = pickle.load(file)
        print(data)
    trace=data['trace']
    fig = corner.corner(
        trace, quantiles=[0.16, 0.5, 0.84],
        show_titles=True)
    pl.savefig(args.outpath+'.png')
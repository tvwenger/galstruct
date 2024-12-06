#!/bin/bash

for fname in *.pkl; do
    echo ${fname}
    outdir=`basename -s .pkl ${fname}`
    mkdir -p ${outdir}
    python ../galstruct/plots/plot_likelihood_varyaz0.py \
	   --fixed Zsun 5.5 \
	   --fixed roll 0.0 \
	   --outdir ${outdir} \
	   ${fname}
    python ../tests/test_neural_net.py \
	   --fixed Zsun 5.5 \
	   --fixed roll 0.0 \
	   ${fname}
done    

#!/bin/bash

# python /workdir/reco_scripts/mr_direct_recon.py /input /output
python stgeorges_pipeline.py
nii2dcm -d MR /output/nii/solution.nii /output/
rm -rf /output/nii/
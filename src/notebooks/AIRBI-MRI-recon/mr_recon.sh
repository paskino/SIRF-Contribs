#!/bin/bash

python stgeorges_pipeline.py
nii2dcm -d MR /output/nii/solution.nii /output/
for i in $(find /output/nii/ -type f -name '*.nii'); do 
  echo $i ;
  nii2dcm -d MR $i $(dirname $i)
  rm $i
done
# rm -rf /output/nii/
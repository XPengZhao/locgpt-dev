#!/bin/bash

# Loop over the specific datasets
for i in 1
do
   # Zero pad the dataset number
   num=$(printf "%01d" $i)

   # Construct the yaml file name
   yaml_file="variant-s10-enc$num-450K.yaml"

   # Run the training command
   python runner.py --mode train --gpu 0 --config conf/variant/$yaml_file
   python runner.py --mode test --gpu 0 --config conf/variant/$yaml_file
done

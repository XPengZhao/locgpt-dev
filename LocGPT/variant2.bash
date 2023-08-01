#!/bin/bash

# Loop over the specific datasets
for i in 6 8
do
   # Zero pad the dataset number
   num=$(printf "%01d" $i)

   # Construct the yaml file name
   yaml_file="variant-s10-enc$num.yaml"

   # Run the training command
   python runner.py --mode train --gpu 2 --config conf/variant/$yaml_file
   python runner.py --mode test --gpu 2 --config conf/variant/$yaml_file
done

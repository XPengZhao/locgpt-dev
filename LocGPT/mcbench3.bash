#!/bin/bash

# Loop over the datasets
for i in {21..28}
do
   # Zero pad the dataset number
   num=$(printf "%02d" $i)

   # Construct the yaml file name
   yaml_file="mcbench-s$num.yaml"

   # Run the training command
   python runner.py --mode train --gpu 1 --config conf/$yaml_file
   python runner.py --mode test --gpu 1 --config conf/$yaml_file
done

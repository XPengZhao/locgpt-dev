#!/bin/bash

# Loop over the datasets
for i in {6..10}
do
   # Skip 25 and 27
   # if [ "$i" -eq 01 ] || [ "$i" -eq 02 ]
   # then
   #    continue
   # fi

   # Zero pad the dataset number
   num=$(printf "%02d" $i)

   # Construct the yaml file name
   yaml_file="mcbench-s$num.yaml"

   # Run the training command
   python runner.py --mode train --gpu 0 --config conf/mcbench/$yaml_file
   python runner.py --mode test --gpu 0 --config conf/mcbench/$yaml_file
done

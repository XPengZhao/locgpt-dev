#!/bin/bash

# Loop over the datasets 40 41 42 44 45
for i in 40 41 42 44 45
do
   # Skip 25 and 27
   # if [ "$i" -eq 01 ] || [ "$i" -eq 02 ]
   # then
   #    continue
   # fi

   # Zero pad the dataset number
   num=$(printf "%02d" $i)

   # Construct the yaml file name
   yaml_file="baseline-s$num-40.yaml"

   # Run the training command
   python runner.py --mode train --gpu 2 --config conf/fine-tune/$yaml_file
#    python runner.py --mode test --gpu 0 --config conf/fine-tune/$yaml_file
done
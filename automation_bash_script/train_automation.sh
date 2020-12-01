#!/bin/bash

seeds=$1
cuda_visible_devices=$2
config_file=$3
device=$4
wrappder_mode=$5

for seed in $seeds;
do
  echo $seed
  echo $cuda_visible_devices
  echo $config_file
  echo $device
  echo $wrappder_mode
  python train.py -s $seed -c $cuda_visible_devices -cf $config_file -d $device -w $wrappder_mode
done
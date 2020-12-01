#!/bin/bash

seeds=$1
cuda_visible_devices=$2
config_file=$3
device=$4

for seed in $seeds;
do
  echo $seed
  echo $cuda_visible_devices
  echo $config_file
  echo $device
  python train.py -s $seed -c $cuda_visible_devices -cf $config_file -d $device
done
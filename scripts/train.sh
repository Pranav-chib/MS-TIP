#!/bin/bash
echo "Start training task queues"

# Hyperparameters
dataset_array=("eth" "hotel" "univ" "zara1" "zara2")
device_id_array=(0 1 2 3 4)
prefix="end_point_"
suffix="_experiment"
nans=0.1
lr=1e-4

# Arguments
while getopts p:s:d:i:n: flag
do
  case "${flag}" in
    p) prefix=${OPTARG};;
    s) suffix=${OPTARG};;
    d) dataset_array=(${OPTARG});;
    i) device_id_array=(${OPTARG});;
    n) nans=(${OPTARG});;
    l) lr=(${OPTARG});;
    *) echo "usage: $0 [-p PREFIX] [-s SUFFIX] [-d \"eth hotel univ zara1 zara2\"] [-i \"0 1 2 3 4\"] -n [<float value between 0 and 1>]" >&2
      exit 1 ;;
  esac
done

if [ ${#dataset_array[@]} -ne ${#device_id_array[@]} ]
then
    printf "Arrays must all be same length. "
    printf "len(dataset_array)=${#dataset_array[@]} and len(device_id_array)=${#device_id_array[@]}\n"
    exit 1
fi

# Signal handler
PID_array=()

sighdl ()
{
  echo "Kill training processes"
  for (( i=0; i<${#dataset_array[@]}; i++ ))
  do
    kill ${PID_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM

# Start training tasks
for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  printf "Training ${dataset_array[$i]}"
  CUDA_VISIBLE_DEVICES=${device_id_array[$i]} python3 train.py \
  --dataset "${dataset_array[$i]}" --nans "${nans}" --saits_lr "${lr}" --tag "${prefix}""${dataset_array[$i]}""${suffix}" &
  PID_array[$i]=$!
  printf " job ${#PID_array[@]} pid ${PID_array[$i]}\n"
done

for (( i=0; i<${#dataset_array[@]}; i++ ))
do
  wait ${PID_array[$i]}
done

echo "Done."

#!/bin/bash

# set path to dataset here
lm="deberta-large"
portion="all"
window="nlu"
task="entity"
eval_set="test"
num_gpus=1
visible=0

CUDA_VISIBLE_DEVICES=${visible} python3 -m torch.distributed.launch \
       --nproc_per_node ${num_gpus} baseline.py \
       --params_file baseline/configs/params-${lm}.json \
       --dataroot data/${portion}/${task}/${window} \
       --exp_name ${portion}-${lm}-${window}-${task} \
       --eval_dataset ${eval_set}

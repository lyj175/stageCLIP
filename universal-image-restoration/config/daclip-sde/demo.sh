#!/bin/bash

#############################################################
### training ###

# for single GPU
#python train.py -opt=options/train.yml

# for multiple GPUs
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=6552 train.py -opt=options/train.yml --launcher pytorch
torchrun --nproc_per_node=2 --master_port=6553 train.py -opt=options/train.yml --launcher pytorch
# 多机多卡
#torchrun --nproc_per_node=2 --node_rank=0 --master_addr="10.42.0.1" --master_port=6553 train.py -opt=options/train.yml --launcher pytorch



#############################################################

### testing ###
# python test.py -opt=options/test.yml

#############################################################

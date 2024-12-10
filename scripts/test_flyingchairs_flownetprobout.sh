#!/bin/bash


# meta
CHECKPOINT="/home/deu/Models/lightprobnets/flownet_probout.ckpt"  # set this to your trained model file
FLYINGCHAIRS_HOME="/home/deu/Datasets/FlyingChairs_release/data/"
MODEL=FlowNetProbOut

# validate clean configuration
PREFIX="validate-chairs"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--loss_with_auc=True \
--validation_dataset=FlyingChairsValid  \
--validation_dataset_root=$FLYINGCHAIRS_HOME \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"
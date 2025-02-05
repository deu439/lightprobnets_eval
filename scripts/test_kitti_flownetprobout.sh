#!/bin/bash


# meta
CHECKPOINT="/home/deu/Models/lightprobnets/flownet_probout.ckpt"  # set this to your trained model file
MODEL=FlowNetProbOut

# validate clean configuration
PREFIX="validate-kitti-2012"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
python ../main.py \
--batch_size=1 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--loss_with_auc=True \
--validation_dataset=KittiTrain  \
--validation_dataset_root="/home/deu/Datasets/KITTI_2012" \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"

# validate clean configuration
PREFIX="validate-kitti-2015"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
python ../main.py \
--batch_size=1 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--loss_with_auc=True \
--validation_dataset=KittiTrain  \
--validation_dataset_root="/home/deu/Datasets/KITTI_2015" \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"
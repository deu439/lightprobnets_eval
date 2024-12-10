#!/bin/bash


# meta
CHECKPOINT="/home/deu/Models/lightprobnets/flownet_probout.ckpt"  # set this to your trained model file
SINTEL_HOME="/home/deu/Datasets/MPI_Sintel"
MODEL=FlowNetProbOut

# validate clean configuration
PREFIX="validate-sintel-clean"
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
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"

# validate final configuration
PREFIX="validate-sintel-final"
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--loss_with_auc=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingFinalFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"
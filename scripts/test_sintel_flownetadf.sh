#!/bin/bash

# dataset, output, model, (optional) checkpoint
CHECKPOINT="/home/deu/Models/lightprobnets/flownet_adf.ckpt"  # set this to your trained model file
SINTEL_HOME="/home/deu/Datasets/MPI_Sintel"
MODEL=FlowNetADF

# validate clean configuration
PREFIX="validate-sintel-clean"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX-clean"
python ../main.py \
--batch_size=4 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--model_noise_variance=1e-3 \
--model_min_variance=1e-4 \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--loss_with_auc=True \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"

# validate final configuration
PREFIX="validate-sintel-final"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX-final"
python ../main.py \
--batch_size=4 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--loss_with_auc=True \
--model=$MODEL \
--model_noise_variance=1e-3 \
--model_min_variance=1e-4 \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingFinalFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe, auc]" \
--validation_keys_minimize="[True, False]"

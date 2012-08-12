#!/bin/bash

# Usage: train-all.sh [NUM_PROTOTYPES]
# By default, 10 prototypes will be used.
# Train one model for each of the classes of "dog", "leash", and "pedestrian".
# Training data should be stored in "GLIMPSE_HOME/training-crops".
# Trained models will be stored in "GLIMPSE_HOME/models".
# To set the model parameters, overwrite the contents of "GLIMPSE_HOME/rc/model.params".
#
# Author: Mick Thomure
# Date: 8/12/2012

# Training image directory names.
MODELS="dogs leashes pedestrians"
# Directory in which to store trained models.
MODEL_DIR=models
# Path to model parameters file.
PARAMS=rc/model.params
# Path to training image directories.
TRAIN_DIR=training-crops

if [[ "$#" < 1 ]]; then
  NUM_PROTOTYPES=10
else
  NUM_PROTOTYPES=$1
fi

# Change to GLIMPSE_HOME directory.
cd $(dirname $0)/..

# Make sure model directory exists.
mkdir -p $MODEL_DIR

# Add Glimpse to python path.
export PYTHONPATH=.

for F in $MODELS; do
  echo "Training class:" $F
  python bin/train-model.py $MODEL_DIR $F $TRAIN_DIR/$F $TRAIN_DIR/dist_$F $PARAMS $NUM_PROTOTYPES
done

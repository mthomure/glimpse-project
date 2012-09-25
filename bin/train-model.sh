#!/bin/bash -l

# Note: "-l" flag on bash command needed to work around weird matlab
# environment.

# Usage: train-model.sh MODEL POS_DIR NEG_DIR [NUM_PROTOTYPES]
# where MODEL is the name of a model class (e.g., "dog", "leash", etc), while
# POS_DIR and NEG_DIR give the locations of positive and negative training 
# images (respectively). Finally, NUM_PROTOTYPES optionally sets the number of
# S2 prototypes for the learned model (default is 200).
#
# Author: Mick Thomure
# Date: 9/11/2012

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 MODEL POS_DIR NEG_DIR [NUM_PROTOTYPES]" 1>&2
  exit -1
fi

MODEL=$1
POS_DIR=$2
NEG_DIR=$3
shift 3

if [[ "$#" < 1 ]]; then
  NUM_PROTOTYPES=200
else
  NUM_PROTOTYPES=$1
fi

# Add Glimpse to the python path.
GLIMPSE_HOME=$(dirname $0)/..
export PYTHONPATH=$GLIMPSE_HOME:$PYTHONPATH

PARAMS=$GLIMPSE_HOME/rc/model.params
python "$GLIMPSE_HOME/bin/train-model.py" "$GLIMPSE_HOME/models" $MODEL $POS_DIR $NEG_DIR $PARAMS $NUM_PROTOTYPES

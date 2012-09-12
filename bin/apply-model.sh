#!/bin/bash

# Usage: apply-model.sh MODEL INPUT_1 [... INPUT_N]
# where MODEL is the name of a model class (e.g., "dog", "leash", etc)
# and each INPUT can be an image file (e.g., "cat1.jpg") or data in
# matlab format. In the latter case, the image data should be in a
# variable named "IMAGE", "image", "img", "im", or "data", and should be
# given as a 2D array in the format (height, width) with pixel values in
# the range [0,1]. The given model is applied to each input, and the
# classifier's decision value is written to standard output on a
# separate line. The applied model is read from GLIMPSE_HOME/models.
#
# Author: Mick Thomure
# Date: 8/12/2012

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 MODEL INPUT" 1>&2
  exit -1
fi

MODEL=$1
shift

# Add Glimpse to the python path.
GLIMPSE_HOME=$(dirname $0)/..
export PYTHONPATH=$GLIMPSE_HOME:$PYTHONPATH

python "$GLIMPSE_HOME/bin/apply-model.py" "$GLIMPSE_HOME/models" $MODEL "$@"

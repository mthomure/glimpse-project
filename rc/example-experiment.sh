#!/bin/bash

# A helper script to set options and run an experiment including imprinting
# prototypes, transforming the images, and running an SVM over the results.
#  1) Copy this script to a new directory
#  2) Uncomment and set CORPUS_DIR and either NUM_PROTOTYPES or PROTOTYPE_FILE
#  2) Set other options as necessary
#  3) Execute the copied script
# Results will be written to the current directory.

# Path to image corpus. Path should contain one or more sub-directories, each
# labelled by the image class.
CORPUS_DIR=

# Source of S2 prototypes. Set either NUM_PROTOTYPES (to imprint) or
# PROTOTYPE_FILE (to use existing prototypes).
NUM_PROTOTYPES=
PROTOTYPE_FILE=

# Whether to run an SVM on the feature vectors. Set to "1" to enable.
SVM=1

# Whether to report cross-validated SVM accuracy. Set to "1" to enable.
SVM_CROSSVAL=0

# The options to use during transformation.
cat > options.py <<EOF
EOF


#### ADVANCED OPTIONS ##########################################################

# Location to store results. Default is the current directory.
RESULT_DIR=.

# Options for imprint-random-prototypes.py
IMPRINT_ARGS="-o options.py"

# Options for transform command
TRANSFORM_ARGS="-o options.py -l it -s feature-vector"

# Options for svm -- this selects learning of an unbiased classifier by the
# SVM-LIGHT package.
SVM_ARGS="-u -t LIBSVM"


#### DO NOT EDIT BELOW THIS LINE ###############################################

export NUM_PROTOTYPES PROTOTYPE_FILE SVM_CROSSVAL SVM
export IMPRINT_ARGS TRANSFORM_ARGS SVM_ARGS PROPAGATE_ARGS NUM_PROCS
exp-run "$CORPUS_DIR" "$RESULT_DIR"

#!/bin/bash
# Helper script to run avocado_train.py with proper GPU configuration

# Set Theano flags for GPU usage
export THEANO_FLAGS="device=cuda,floatX=float32,optimizer=fast_run,gpuarray.preallocate=0.8,dnn.enabled=True"
export KERAS_BACKEND=theano

# Run the training script with any passed arguments
python /project/deeprna_data/corgi-reproduction/code/avocado_validation.py "$@"
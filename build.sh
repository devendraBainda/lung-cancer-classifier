#!/usr/bin/env bash
# Tell TensorFlow to disable optimizations
pip install --upgrade pip
pip install -r requirements.txt
# Explicitly disable optimizations
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
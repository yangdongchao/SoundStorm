#!/bin/bash
export MAIN_ROOT=`realpath ${PWD}/../../`

export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}
MODEL=s2
export BIN_DIR=${MAIN_ROOT}/soundstorm/${MODEL}/exps

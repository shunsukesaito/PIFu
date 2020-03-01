#!/usr/bin/env bash
set -ex

TEST_FOLDER_PATH=$1
VOL_RES=$2
VOL_LOAD_SIZE=$3
TEX_LOAD_SIZE=$4

# command
./scripts/eval_default.sh $TEST_FOLDER_PATH $VOL_RES $VOL_LOAD_SIZE --only_sdf temp.npy
./scripts/eval_default.sh $TEST_FOLDER_PATH $VOL_RES $TEX_LOAD_SIZE --load_sdf output.npy

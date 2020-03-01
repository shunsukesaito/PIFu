#!/usr/bin/env bash
set -ex

VOL_RES=256
LOAD_SIZE=512
TEST_FOLDER_PATH='./sample_images'

# command
./scripts/eval_default.sh $TEST_FOLDER_PATH $VOL_RES $LOAD_SIZE
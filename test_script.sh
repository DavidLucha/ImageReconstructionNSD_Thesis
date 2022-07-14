#!/bin/bash

RUN_NAME=$(date +%Y%m%d-%H%M%S)
# 20220712_172256
# Don't need the network train anymore. Just carry the RUN_NAME var throughout, and make the strings in Python. TODO: THIS <<<

# Study 1
# Pretrain one network. Used in both GOD and NSD.
python pretrain.py --epochs 20 --dataset GOD --run_name $RUN_NAME
echo Done!


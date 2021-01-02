#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python main.py -d duke -a resnet50 -b 64 -j 4 --epochs 100 --log logs/duke/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ../datasets/DukeMTMC-reID/

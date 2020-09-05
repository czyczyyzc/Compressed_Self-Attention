#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python main.py -d cuhk -a resnet50 -b 64 -j 4 --epochs 200 --log logs/cuhk/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 80 --data-dir ~/project/ziyechen/datasets/CUHK03/detected --evaluate

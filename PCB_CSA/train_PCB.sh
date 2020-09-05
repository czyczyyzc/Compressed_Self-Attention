#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py -d market -a resnet50 -b 64 -j 4 --epochs 100 --log logs/market/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/datasets/Market-1501/
CUDA_VISIBLE_DEVICES=0 python main.py -d duke   -a resnet50 -b 64 -j 4 --epochs 100 --log logs/duke/PCB/   --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/datasets/DukeMTMC-reID/
CUDA_VISIBLE_DEVICES=0 python main.py -d cuhk   -a resnet50 -b 64 -j 4 --epochs 200 --log logs/cuhk/PCB/   --combine-trainval --feature 256 --height 384 --width 128 --step-size 80 --data-dir ~/datasets/CUHK03/detected/

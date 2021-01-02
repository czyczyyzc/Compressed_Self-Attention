#!/bin/bash

#SBATCH -A dubo
#SBATCH -J mywork
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -o out_scream.out
#SBATCH -t 30-00:00:00    


module load nvidia/cuda/10.1
source activate base
# watch -n 1 nvidia-smi
# python main.py -d market -a resnet50 -b 64 -j 4 --epochs 100 --log logs/market/PCB/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/project/ziyechen/datasets/Market-1501/
# python main.py -d duke   -a resnet50 -b 64 -j 4 --epochs 100 --log logs/duke/PCB/   --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/project/ziyechen/datasets/DukeMTMC-reID/
python main.py -d cuhk   -a resnet50 -b 64 -j 4 --epochs 200 --log logs/cuhk/PCB/   --combine-trainval --feature 256 --height 384 --width 128 --step-size 80 --data-dir ~/project/ziyechen/datasets/CUHK03/detected/


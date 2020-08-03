#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=1 python3 video_classification.py --pos_dir /data/anie/AutoGrade/rec_vidoes_toy_programs/pos_train \
  --neg_dir /data/anie/AutoGrade/rec_vidoes_toy_programs/neg_train \
  --max_frames 3000 --frameskip 5 --device 0 --batch_size 10 --epochs 5
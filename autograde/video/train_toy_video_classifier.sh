#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=1 python3 video_classification.py --pos_dir /data/anie/AutoGrade/rec_vidoes_toy_programs/pos_train \
  --neg_dir /data/anie/AutoGrade/rec_vidoes_toy_programs/neg_train \
  --frameskip 5 --max_frames 200 --device 0 --batch_size 4 --epochs 7 --output_dir ./output/video/r3d_18_eval_on_test_skip_5_max_200
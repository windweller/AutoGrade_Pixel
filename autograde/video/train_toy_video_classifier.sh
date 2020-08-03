#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=1 python3 video_classification.py --pos_dir /home/anie/AutoGrade/video/random_agent_videos_max_frame_${VARIABLE}_prefix_0 \
  --neg_dir /home/anie/AutoGrade/video/random_agent_broken_videos_max_frame_${VARIABLE}_prefix_0 \
  --max_frames 3000 --frameskip 5 --device 0 --batch_size 10 --epochs 1
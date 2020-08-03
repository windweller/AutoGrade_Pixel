# README

`envs`: contains implementations of Games only in PyGame/Pymunk, no OpenAI dependencies.

`rl_envs`: augment the enviroments in `envs`. For RL training purpose, we import from this folder.

`train`: put RL-specific training in there

`video`: put video-related training in there

`grading`: Preprocessing student programs, running to evaluate RL agent and Video classifier, tally result.

# Installation

`pip install -r requirements.txt`

If encounter OpenCV error:

`sudo apt install libsm6 libxext6 libxrender-dev`

Then:

`pip install -e .`

Run:

`python autograde/train/train_pixel_agent.py`

# Tensorboard

`tensorboard --logdir ./tensorboard_first_test_log/`

# Caution

1. GrayScale + FrameStack is needed for non-recurrent policy (it squashes RGB 3 channels into 1 channel, and framestack extends that dimension to N).
There currently is a weird problem in Grayscale where the output image is skewed.
2. ReSize wrapper is enough for recurrent policy.
3. For this environment, you need to add `gym.wrappers.TimLimit()`, otherwise it won't train.
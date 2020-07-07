"""
Turn human recording into playable video
"""

import numpy as np
import sys
from gym.wrappers.monitoring.video_recorder import ImageEncoder

def convert_np_to_video(frames_name, output_video_name):
    assert '.mp4' in output_video_name, "The format is mp4"

    frames = np.load(frames_name)
    frame_shape = (400, 400, 3)
    frames_per_sec = 50
    ime = ImageEncoder(output_video_name, frame_shape, frames_per_sec)

    for f in frames:
        ime.capture_frame(f)

    ime.close()

if __name__ == '__main__':
    pass
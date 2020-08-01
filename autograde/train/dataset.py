"""
We load in trained RL agents and programs
"""

import os


class RolloutGenerator(object):
    def __init__(self, program_dir):
        os.listdir(program_dir)

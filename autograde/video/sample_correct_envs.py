"""
We want to evaluate our agent's performance on correct environments from:
Standard, Head (N=51), Body (N=100), Tail (N=100)

This program does sampling, and also creating folders in "envs/bounce_programs/"

"""

import shutil
import os
from os.path import join as pjoin

root_dir = "/home/aimingnie/AutoGrade/autograde/envs/bounce_programs/"

def create_standard_folders():
    new_folder_name = "env_eval_standard/"
    os.makedirs(pjoin(root_dir, new_folder_name), exist_ok=True)

def create_head_folders():
    new_folder_name = "env_eval_head/"
    os.makedirs(pjoin(root_dir, new_folder_name), exist_ok=True)

    target_dir = pjoin(root_dir, '100_uniq_programs_skip_0', 'correct')
    dest_dir = pjoin(root_dir, new_folder_name)
    for f_n in os.listdir(target_dir):
        shutil.copy2(pjoin(target_dir, f_n), dest_dir)

def create_body_folders(cutoff=100):
    new_folder_name = "env_eval_body/"
    os.makedirs(pjoin(root_dir, new_folder_name), exist_ok=True)

    folders = ['1000_uniq_programs_skip_0',
                '2000_uniq_programs_skip_1000',
                '3000_uniq_programs_skip_2000',
                '4000_uniq_programs_skip_3000',
                '5000_uniq_programs_skip_4000',
                '6000_uniq_programs_skip_5000',
                '7000_uniq_programs_skip_6000',
                '8000_uniq_programs_skip_7000',
                '8359_uniq_programs_skip_8000']

    data = []
    for folder in folders:
        target_dir = pjoin(root_dir, folder, 'correct')
        dest_dir = pjoin(root_dir, new_folder_name)
        for f_n in os.listdir(target_dir):
            data.append((pjoin(target_dir, f_n), dest_dir))
            # shutil.copy2(pjoin(target_dir, f_n), dest_dir)

    import random
    random.seed(12354)
    random.shuffle(data)

    data = data[:100]
    for tup in data:
        target_dir, dest_dir = tup
        shutil.copy2(target_dir, dest_dir)

def create_tail_folders(cutoff=100):
    new_folder_name = "env_eval_tail/"
    os.makedirs(pjoin(root_dir, new_folder_name), exist_ok=True)

    folders = ['500_sampled_tail_uniq_programs']
    data = []
    for folder in folders:
        target_dir = pjoin(root_dir, folder, 'correct')
        dest_dir = pjoin(root_dir, new_folder_name)
        for f_n in os.listdir(target_dir):
            data.append((pjoin(target_dir, f_n), dest_dir))
    
    import random
    random.seed(12354)
    random.shuffle(data)

    data = data[:100]
    for tup in data:
        target_dir, dest_dir = tup
        shutil.copy2(target_dir, dest_dir)

if __name__ == "__main__":
    pass
    # create_standard_folders()
    # create_head_folders()
    # create_body_folders()
    # create_tail_folders()
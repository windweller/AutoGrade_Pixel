# Does some manual labor in moving files from one to another
# specifically, move the first 100 files 

import os
import shutil

def move_head_programs():
    # move head (100) out of the 1000 folders, because we evaluate them seperately
    all_files = os.listdir("./eval_reward_value_stats_n1000_skip0")
    head_file_name_patterns = ["srcID_{}_program".format(i) for i in range(100)]

    files_we_need = []
    for f_n in all_files:
        for pattern in head_file_name_patterns:
            if pattern in f_n:
                files_we_need.append(f_n)

    os.makedirs("./eval_reward_value_stats_n100_skip0", exist_ok=True)
    old_folder = "./eval_reward_value_stats_n1000_skip0/"
    new_folder = "./eval_reward_value_stats_n100_skip0/"

    for f_n in files_we_need:
        shutil.move(old_folder + f_n, new_folder + f_n)

    print("Done!")

def isolate_tail_500():
    # before we have tail 1000
    # now we just do 500
    # 250 correct, 250 broken
    all_files = os.listdir("./eval_reward_value_stats_1000_sampled_tail")
    correct_src_IDs = set()
    broken_src_IDs = set()
    for f_n in all_files:
        name = f_n.replace("_program_info.json", "").replace("_program_rewards.txt", "")
        if 'broken' in name:
            broken_src_IDs.add(name)
        else:
            correct_src_IDs.add(name)
    
    correct_src_IDs = list(correct_src_IDs)[:250]
    broken_src_IDs = list(broken_src_IDs)[:250]

    head_file_name_patterns = correct_src_IDs + broken_src_IDs

    files_we_need = []
    for f_n in all_files:
        for pattern in head_file_name_patterns:
            if pattern in f_n:
                files_we_need.append(f_n)

    os.makedirs("./eval_reward_value_stats_500_sampled_tail", exist_ok=True)
    old_folder = "./eval_reward_value_stats_1000_sampled_tail/"
    new_folder = "./eval_reward_value_stats_500_sampled_tail/"

    for f_n in files_we_need:
        shutil.copyfile(old_folder + f_n, new_folder + f_n)

    print("Done!")

if __name__ == "__main__":
    pass
    # move_head_programs()
    isolate_tail_500()
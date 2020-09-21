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

if __name__ == "__main__":
    move_head_programs()
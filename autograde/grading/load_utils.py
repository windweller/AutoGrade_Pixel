import os
import json
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

def extract_reference_program_data_from_dir(correct_ref_dir, broken_ref_dir):
    correct_program_to_rewards = {}
    broken_program_to_rewards = {}

    correct_program_to_info = {}
    broken_program_to_info = {}
    
    correct_rewards, broken_rewards = [], []
    
    # correct reference files
    for file_name in os.listdir(correct_ref_dir):
        # we need to get rid of a bunch of speed variations...
        
        if '.txt' in file_name:
            f = open(correct_ref_dir + "{}".format(file_name))
            text = f.readlines()[0]
            rewards = [float(r) for r in text.split(',')]
            correct_rewards.extend(rewards)
            correct_program_to_rewards[file_name.lstrip("correct_").rstrip("_program_rewards.txt")] = rewards
        else:
            f = open(correct_ref_dir + "{}".format(file_name))
            info_dic = json.load(f)
            program_name = file_name.lstrip("correct_").rstrip("_program_info.json")
            correct_program_to_info[program_name] = {'step_values': info_dic['step_values'],
                                                         'step_rewards': info_dic['step_rewards']}
    # broken reference files
    for file_name in os.listdir(broken_ref_dir):
        if '.txt' in file_name:
            f = open(broken_ref_dir + "{}".format(file_name))
            text = f.readlines()[0]
            rewards = [float(r) for r in text.split(',')]
            broken_rewards.extend(rewards)
            broken_program_to_rewards[file_name.lstrip("broken_").rstrip("_program_rewards.txt")] = rewards
        else:
            f = open(broken_ref_dir + "{}".format(file_name))
            info_dic = json.load(f)
            program_name = file_name.lstrip("broken_").rstrip("_program_info.json")
            broken_program_to_info[program_name] = {'step_values': info_dic['step_values'],
                                                         'step_rewards': info_dic['step_rewards']}
    
    return correct_program_to_rewards, broken_program_to_rewards, correct_program_to_info, broken_program_to_info, correct_rewards, broken_rewards

def extract_reference_program_data_filtered():
    # this we filtered out bad speed combos
    return extract_reference_program_data_from_dir("./reference_eval_reward_value_stats_correct_programs_8_theme_15_speed/",
                        "./reference_eval_reward_value_stats_broken_10_programs_2rounds/")

def extract_reference_program_data():
    # this we did not filter out bad speed combinations
    return extract_reference_program_data_from_dir("./reference_eval_reward_value_stats_correct_programs_8_theme_24_speed/",
                        "./reference_eval_reward_value_stats_broken_10_programs/")

def visualize_distributions():
    correct_program_to_rewards, broken_program_to_rewards, correct_program_to_info, broken_program_to_info, correct_rewards, broken_rewards = extract_reference_program_data()
    df = pd.DataFrame(data={'Collected Reward': correct_rewards + broken_rewards, "Reference Program Type": ['Correct'] * len(correct_rewards) + ['Incorrect'] * len(broken_rewards)})
    sns.displot(data=df, x="Collected Reward", hue="Reference Program Type", kind='kde', fill=True)

    correct_program_to_rewards, broken_program_to_rewards, correct_program_to_info, broken_program_to_info, correct_rewards, broken_rewards = extract_reference_program_data_filtered()
    df = pd.DataFrame(data={'Collected Reward': correct_rewards + broken_rewards, "Reference Program Type": ['Correct'] * len(correct_rewards) + ['Incorrect'] * len(broken_rewards)})
    sns.displot(data=df, x="Collected Reward", hue="Reference Program Type", kind='kde', fill=True)

if __name__ == "__main__":
    print("here")
    # extract_reference_program_data_filtered()
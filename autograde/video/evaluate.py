"""
Will load in external videos like the 100 unique programs
and run them through the video pipeline to generate a label
"""
import os
import csv
from collections import defaultdict

import torch
from tqdm import tqdm
import random
import glob
import shutil
import torchvision
from autograde.video.video_classification import AnomalyDataset, run_epoch, DataLoader, bootstrap


def evaluate_on_unique_100():
    # build the dataset
    # get dataloader
    # run and save predictions.csv

    root_dir = "/home/anie/AutoGrade/rec_vidoes_100_uniq_programs_n10_800"

    d = AnomalyDataset(pos_dir=root_dir + "/pos_eval",
                       neg_dir=root_dir + "/neg_eval",
                       maxframe=200,
                       frameskip=5,
                       no_normalization=True,
                       resize_width=100,
                       resize_height=100,
                       train_num_cap=-1,
                       parallel_gen=True)

    device = torch.device('cuda')

    dataloader = DataLoader(d,
                            batch_size=4, num_workers=4, shuffle=False,
                            pin_memory=(device.type == "cuda"), drop_last=True)

    model = torchvision.models.video.__dict__['r3d_18'](pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load model
    # checkpoint = torch.load("/home/anie/AutoGrade/output/video/r3d_18_top_10_programs_n20_480_temp/best.pt")
    checkpoint = torch.load("./autograde/video/output/video/r3d_18_top_10_programs_n20_480/best.pt")
    model.load_state_dict(checkpoint['state_dict'])

    loss, yhat, y = run_epoch(model, dataloader, "test", None, device, save_all=True,
                              blocks=200)

    print("Test (single-crop): {:.3f} ({:.3f} - {:.3f})\n".format(*bootstrap(yhat, y)))

    with open(os.path.join(root_dir, "predictions.csv"), "w") as g:
        for (filename, gold, pred) in zip(d.fnames, y, yhat):
            g.write("{},{},{:.4f}\n".format(filename, gold, pred))


def analyze_prediction_csv():
    file_path = "../../rec_vidoes_100_uniq_programs_n10_1000/predictions.csv"
    golds, preds = [], []
    with open(file_path) as f:
        reader = csv.reader(f)
        for line in reader:
            _, gold, pred = line
            golds.append(float(gold))
            preds.append(float(pred))

    import sklearn

    print(sklearn.metrics.accuracy_score(golds, preds))

    grouped_by_program = defaultdict(list)
    with open(file_path) as f:
        reader = csv.reader(f)
        for line in reader:
            filename, gold, pred = line
            gold, pred = float(gold), float(pred)
            program_name = filename.split("/")[-1].split('_program')[0].lstrip('ppo2-')
            grouped_by_program[program_name].append(gold == pred)

    cnt = 0
    for program_name, pred_list in grouped_by_program.items():
        print(program_name, sum(pred_list) / len(pred_list))
        if sum(pred_list) / len(pred_list) > 0.5:
            cnt += 1

    print("Total correct program number: {}".format(cnt))


import numpy as np


def detailed_analyze_prediction_csv():
    file_path = "../../rec_vidoes_100_uniq_programs_n10_800/predictions.csv"
    correct_program_to_rewards = defaultdict(list)
    broken_program_to_rewards = defaultdict(list)
    golds, preds = [], []
    with open(file_path) as f:
        reader = csv.reader(f)
        for line in reader:
            filename, gold, pred = line
            golds.append(float(gold))
            preds.append(float(pred))
            program_name = filename.split('/')[-1].lstrip('ppo2-').split('-')[0].rstrip('_program')
            # 1 is broken, 0 is correct (in reward analysis you swapped it)
            # print(program_name, gold)
            if float(gold) == 1:
                broken_program_to_rewards[program_name].append(float(pred))
            else:
                correct_program_to_rewards[program_name].append(float(gold))

    import sklearn

    print("instance level")
    print(sklearn.metrics.classification_report(golds, preds))
    print(sklearn.metrics.accuracy_score(golds, preds))

    golds, preds = [], []
    for program_name, instance_preds in correct_program_to_rewards.items():
        golds.append(1)  # we swap here :)
        preds.append(np.mean(instance_preds) > 0.5)

    for program_name, instance_preds in broken_program_to_rewards.items():
        golds.append(0)  # we swap here :)
        preds.append(np.mean(instance_preds) > 0.5)

    print("program level")
    print(sklearn.metrics.classification_report(golds, preds))
    print(sklearn.metrics.accuracy_score(golds, preds))


def move_files():
    # don't worry about this function
    # it does a one-time file transfer

    source_dir = "/home/anie/AutoGrade/rec_vidoes_top_10_programs_n20_480_mixed_theme/pos_train"
    dest_dir = "/home/anie/AutoGrade/rec_videos_top_10_programs_n20_480_temp/pos_train/"
    files = [f for f in glob.glob(source_dir + "/*.mp4", recursive=False)]
    random.shuffle(files)
    sampled_files = files[:240]

    print(sum([1 if 'correct_sample_with_theme' in f else 0 for f in sampled_files]))
    for f in tqdm(sampled_files):
        shutil.copy(f, dest_dir)

    print("finished")


# python autograde/video/evaluate.pyr3d_18_top_10_programs_n20_480_temp
if __name__ == '__main__':
    pass
    # evaluate_on_unique_100()  # 30 minutes
    # analyze_prediction_csv()
    detailed_analyze_prediction_csv()

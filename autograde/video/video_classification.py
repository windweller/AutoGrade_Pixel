"""
A simple video classification training

https://github.com/pytorch/vision/releases/tag/v0.4.0

Right now we are not cutting to video clips...

from EchoNet
(clips, frames, channel, height, width)
"""

import pathlib
import torch
import torchvision
from torch.utils.data import Dataset as tDataset
from torch.utils.data import Subset, DataLoader
import os
import numpy as np
from tqdm import tqdm
import sklearn
import glob
import random
from autograde.video.video_utils import loadvideo, get_mean_and_std, bootstrap
import argparse

random.seed(5512106458)


class AnomalyDataset(tDataset):
    def __init__(self, pos_dir=None,
                 neg_dir=None,
                 neg_prefix="broken_videos",
                 no_normalization=True,
                 mean=0., std=1.,
                 train_num_cap=-1,
                 maxframe=100,
                 frameskip=1,
                 crops=1,
                 shuffling_indices=None,
                 resize_width=100,
                 resize_height=100):

        self.mean = mean
        self.std = std
        self.no_normalization = no_normalization

        self.neg_prefix = neg_prefix

        self.frameskip = frameskip  # this is "period" on EchoNet
        self.maxframe = maxframe  # this is "length" on EchoNet

        self.crops = crops

        self.resize_width = resize_width
        self.resize_height = resize_height

        self.pos_fnames = [f for f in glob.glob(pos_dir + "/*.mp4", recursive=False)]
        self.neg_fnames = [f for f in glob.glob(neg_dir + "/*.mp4", recursive=False)]

        if train_num_cap != -1:
            assert train_num_cap % 2 == 0
            self.pos_fnames = self.pos_fnames[:int(train_num_cap / 2)]
            self.neg_fnames = self.neg_fnames[:int(train_num_cap / 2)]

        # no shuffle here, but we shuffle outside
        # hopefully that's enough!
        self.fnames = self.pos_fnames + self.neg_fnames
        if shuffling_indices is not None:
            self.fnames = np.array(self.fnames)[shuffling_indices].tolist()
        # random.shuffle(self.fnames)  # we can still take difference easily

    def shuffle(self, shuffling_indices):
        self.fnames = np.array(self.fnames)[shuffling_indices].tolist()

    def apply_normalization(self, mean_vec, std_vec):
        self.mean = mean_vec
        self.std = std_vec
        self.no_normalization = False

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        video = loadvideo(self.fnames[index], resize_width=self.resize_width, resize_height=self.resize_height)

        # Apply normalization
        assert (type(self.mean) == type(self.std))
        if not self.no_normalization:
            if isinstance(self.mean, int) or isinstance(self.mean, float):
                video = (video - self.mean) / self.std
            else:
                video = (video - self.mean.reshape(3, 1, 1, 1)) / self.std.reshape(3, 1, 1, 1)

        # (3, 483, 400, 400)
        c, f, h, w = video.shape
        if self.maxframe == None:
            length = f // self.frameskip
        else:
            length = self.maxframe

        if f < length * self.frameskip:
            # Pad video with frames filled with zeros if too short
            video = np.concatenate((video, np.zeros((c, length * self.frameskip - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        if self.crops == "all":
            start = np.arange(f - (length - 1) * self.frameskip)
        else:
            start = np.random.choice(f - (length - 1) * self.frameskip, self.crops)

        target = 0 if self.neg_prefix in self.fnames[index] else 1
        target = np.array(target, dtype=np.float32)

        # Select random crops
        video = tuple(video[:, s + self.frameskip * np.arange(length), :, :] for s in start)
        if self.crops == 1:
            video = video[0]
        else:
            video = np.stack(video)

        return video, target

def run(num_epochs=45,
        output=None):
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")

        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch \n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                loss, yhat, y = run_epoch(model, dataloaders[phase], phase, optim, device)
                f.write("{},{},{},{}\n".format(epoch, phase, loss, sklearn.metrics.accuracy_score(yhat, y)))
                f.flush()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'frameskip': args.frameskip,
                'maxframe': args.max_frames,
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["best_loss"], checkpoint["epoch"]))

        # test
        loss, yhat, y = run_epoch(model, test_dataloader_single_crop, "test", None, device, save_all=True,
                                                      blocks=200)

        f.write("Test (single-crop): {:.3f} ({:.3f} - {:.3f})\n".format(*bootstrap(yhat, y)))

        # loss, yhat, y = run_epoch(model, test_dataloader_all_crop, "test", None, device, save_all=True,
        #                                               blocks=50)
        # f.write("Test (all crops): {:.3f} ({:.3f} - {:.3f})\n".format(
        #     *bootstrap(np.array(list(map(lambda x: x.mean(), yhat))), y)))
        f.flush()

        with open(os.path.join(output, "predictions.csv"), "w") as g:
            for (filename, gold, pred) in zip(d.fnames, y, yhat):
                g.write("{},{},{:.4f}\n".format(filename, gold, pred))


def run_epoch(model, dataloader, phase, optim, device, save_all=False, blocks=None):
    criterion = torch.nn.BCEWithLogitsLoss()

    runningloss = 0.0

    model.train(phase == 'train')

    counter = 0
    summer = 0
    summer_squared = 0

    yhat = []
    y = []

    with torch.set_grad_enabled(phase == 'train'):
        with tqdm(total=len(dataloader)) as pbar:
            for (i, (X, outcome)) in enumerate(dataloader):
                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                # it's not "exact" averaging...
                # just merging clips and batch
                average = (len(X.shape) == 6)
                if average:
                    batch, n_crops, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                summer += outcome.sum()
                summer_squared += (outcome ** 2).sum()

                if blocks is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + blocks), ...]) for j in range(0, X.shape[0], blocks)])

                if save_all:
                    pred = (torch.sigmoid(outputs.view(-1)) > 0.5).to('cpu').detach().numpy().astype(float)
                    yhat.append(pred)

                if average:
                    outputs = outputs.view(batch, n_crops, -1).mean(1)

                if not save_all:
                    pred = (torch.sigmoid(outputs.view(-1)) > 0.5).to('cpu').detach().numpy().astype(float)
                    yhat.append(pred)

                loss = criterion(outputs.view(-1), outcome)

                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # TODO: X.size(0) is wrong when averaging
                runningloss += loss.item() * X.size(0)
                counter += X.size(0)

                epoch_loss = runningloss / counter

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(epoch_loss, loss.item(),
                                                                       summer_squared / counter - (
                                                                                   summer / counter) ** 2))
                pbar.update()

    # if not save_all:
    yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return epoch_loss, yhat, y


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_dir", type=str, help="")
    parser.add_argument("--neg_dir", type=str, help="")
    parser.add_argument("--modelname", type=str, default="r3d_18")
    parser.add_argument("--output_dir", type=str, default=None, help="r3d_18")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--train_portion", type=float, default=0.8, help="how many videos to train, rest to test")
    parser.add_argument("--max_frames", type=int, default=100, help="how many steps to take in the environment")
    parser.add_argument("--train_num_cap", type=int, default=-1, help="only train on < train_num_cap examples")
    parser.add_argument("--frameskip", type=int, default=5, help="how many steps to take in the environment")
    parser.add_argument("--device", type=int, default=0, help="which GPU to use")
    parser.add_argument("--batch_size", type=int, default=20, help="which GPU to use")
    parser.add_argument("--epochs", type=int, default=45, help="which GPU to use")
    parser.add_argument("--resize_width", type=int, default=100, help="resize video")
    parser.add_argument("--resize_height", type=int, default=100, help="resize video")

    args = parser.parse_args()

    d = AnomalyDataset(pos_dir=args.pos_dir,
                       neg_dir=args.neg_dir,
                       maxframe=args.max_frames,
                       frameskip=args.frameskip,
                       no_normalization=True,
                       resize_width=args.resize_width,
                       resize_height=args.resize_height,
                       train_num_cap=args.train_num_cap)

    mean_vec, std_vec = get_mean_and_std(d, samples=30)
    # shape: mean_vec (3)
    d.apply_normalization(mean_vec, std_vec)

    # shuffle once here
    shuffling_indices = list(range(len(d)))
    random.shuffle(shuffling_indices)
    d.shuffle(shuffling_indices)

    if args.output_dir is None:
        output = os.path.join("output", "video",
                              "{}_{}_{}_{}".format(args.modelname, args.max_frames, args.frameskip, "pretrained" if args.pretrained else "random"))
    else:
        output = args.output_dir

    if args.device >= 0:
        device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')  # device = -1

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    model = torchvision.models.video.__dict__[args.modelname](pretrained=args.pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    # model.fc.bias.data[0] = 55.6  # TODO: set mean properly for corresponding task

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    # technically, they have a function called random_split()
    # but we can do it ourselves...

    # also here is another solution:
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

    n_train_data = int(len(d) * args.train_portion)
    indices = list(range(len(d)))
    random.shuffle(indices)

    n_val_data = int((len(d) - n_train_data) / 2)

    train_indices = indices[:n_train_data]
    val_indices = indices[n_train_data:n_train_data + n_val_data]
    test_indices = indices[n_train_data + n_val_data:]

    train_dataset = Subset(d, train_indices)
    val_dataset = Subset(d, val_indices)

    train_dataloader = DataLoader(train_dataset,
                                   batch_size=args.batch_size, num_workers=8, shuffle=True,
                                   pin_memory=(device.type=="cuda"), drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=args.batch_size, num_workers=8, shuffle=True,
                                 pin_memory=(device.type=="cuda"))

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # d_test = AnomalyDataset(pos_dir="random_agent_videos_max_frame_100_prefix_0",
    #                    neg_dir="/home/anie/AutoGrade/video/random_agent_broken_videos_max_frame_100_prefix_0",
    #                    maxframe=args.max_frames,
    #                    frameskip=args.frameskip,
    #                    crops="all",
    #                    shuffling_indices=shuffling_indices,
    #                    resize_width=args.resize_width,
    #                    resize_height=args.resize_height,
    #                         train_num_cap=args.train_num_cap)
    # d_test.apply_normalization(mean_vec, std_vec)

    test_dataset = Subset(d, test_indices)

    test_dataloader_single_crop = DataLoader(test_dataset,
                                 batch_size=args.batch_size, num_workers=8, shuffle=True,
                                 pin_memory=(device.type=="cuda"))

    # test_dataset_all = Subset(d_test, test_indices)
    # test_dataloader_all_crop = DataLoader(test_dataset_all,
    #                              batch_size=args.batch_size, num_workers=8, shuffle=True,
    #                              pin_memory=(device.type=="cuda"))

    run(num_epochs=args.epochs, output=output)


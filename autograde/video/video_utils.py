import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics


def get_mean_and_std(dataset, samples=10):
    # todo: bigger batch (use collate_fn), subsample to avoid loading things post break
    if len(dataset) > samples:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), samples, replace=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

    n = 0
    mean = 0.
    std = 0.
    for (i, (x, t)) in enumerate(tqdm(dataloader)):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        mean += torch.sum(x, dim=1).numpy()
        std += torch.sum(x ** 2, dim=1).numpy()
    mean /= n
    std = np.sqrt(std / n - mean ** 2)

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std


def loadvideo(filename, resize_width=None, resize_height=None):
    if not os.path.exists(filename):
        raise FileNotFoundError()
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # empty numpy array of appropriate length, fill in when possible from front
    if resize_width is None:
        video = np.zeros((frame_count, frame_width, frame_height, 3), np.float32)
    else:
        video = np.zeros((frame_count, resize_width, resize_height, 3), np.float32)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_width is not None and resize_height is not None:
            frame = cv2.resize(frame, (resize_width, resize_height))

        video[count] = frame

    video = video.transpose((3, 0, 1, 2))

    return video


def bootstrap(y, yhat, eval_func=metrics.accuracy_score, samples=10000):
    bootstraps = []
    for i in range(samples):
        ind = np.random.choice(len(y), len(y))
        bootstraps.append(eval_func(yhat[ind], y[ind]))
    bootstraps = sorted(bootstraps)

    return eval_func(yhat, y), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


# Based on https://nipunbatra.github.io/blog/2014/latexify.html
def latexify():
    import matplotlib
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              # 'text.usetex': True,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)

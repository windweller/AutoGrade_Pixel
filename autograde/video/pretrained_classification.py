# We load in pretrained CNN weights
# and train with skip frame

import gym
import pathlib
import torch
import torchvision
from torch.utils.data import Dataset as tDataset
from torch.utils.data import Subset, DataLoader
import os
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm
import sklearn
import glob
import random
from autograde.video.video_utils import loadvideo, get_mean_and_std, bootstrap
from autograde.video.video_classification import AnomalyDataset
import argparse

from stable_baselines import PPO2
import torch.nn as nn

random.seed(5512106458)


def verify_dataset():
    d = AnomalyDataset(pos_dir='./rec_vidoes_toy_programs/pos_train',
                       neg_dir='./rec_vidoes_toy_programs/neg_train',
                       maxframe=200,
                       frameskip=5,
                       no_normalization=True,
                       resize_width=100,
                       resize_height=100,
                       train_num_cap=-1)

    shuffling_indices = list(range(len(d)))
    random.shuffle(shuffling_indices)
    d.shuffle(shuffling_indices)

    # split train/val
    n_train_data = int(len(d) * 0.8)
    n_val_data = int((len(d) - n_train_data) / 2)
    indices = list(range(len(d)))

    train_indices = indices[:n_train_data]
    val_indices = indices[n_train_data:n_train_data + n_val_data]
    test_indices = indices[n_train_data + n_val_data:]

    train_dataset = Subset(d, train_indices)
    val_dataset = Subset(d, val_indices)
    d_test = Subset(d, test_indices)

    device = torch.device('cpu')

    dataloader = DataLoader(train_dataset,
                            batch_size=4, num_workers=4, shuffle=False,
                            pin_memory=(device.type == "cpu"), drop_last=True)

    for (i, (X, outcome, fnames)) in enumerate(dataloader):
        print(outcome)
        print(fnames)

    test_dataloader_single_crop = DataLoader(d_test,
                                             batch_size=4, num_workers=4, shuffle=False,
                                             pin_memory=(device.type == "cuda"))

    # d_test = AnomalyDataset(pos_dir=None,
    #                         neg_dir="./rec_vidoes_toy_programs/neg_test",
    #                         maxframe=200,
    #                         frameskip=5,
    #                         no_normalization=True,
    #                         resize_width=100,
    #                         resize_height=100,
    #                         train_num_cap=-1)
    #
    # print(d_test.fnames)
    #
    # test_loader = DataLoader(d_test, batch_size=4, num_workers=4, shuffle=False,
    #                       pin_memory=(device.type == "cpu"), drop_last=True)

    for (i, (X, outcome, fnames)) in enumerate(test_dataloader_single_crop):
        print(outcome)
        print(fnames)




# class PyTorchCnnPolicy(nn.Module):
#     def __init__(self):
#         super(PyTorchCnnPolicy, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=8, stride=4, padding=0, bias=True)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=0, bias=True)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
#         self.fc1 = nn.Linear(3136, 512)
#         self.fc2 = nn.Linear(512, 4)
#         self.relu = nn.ReLU()
#         self.out_activ = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         # shape before flattening
#         # tf: (?, 7, 7, 64)
#         # pytorch: [1, 64, 7, 7]
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = self.out_activ(x)
#         return x


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# def nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#
#     :param scaled_images: (TensorFlow Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (TensorFlow Tensor) The CNN output layer
#     """
#     activ = tf.nn.relu
#     layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
#     layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_3 = conv_to_fc(layer_3)
#     return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


# TODO: some quick rework shall resolve this
def copy_cnn_weights(baselines_model):
    # https://github.com/hill-a/stable-baselines/issues/372

    # Box(84, 84, 3)
    # now we get channel-first!
    obs_space = gym.spaces.Box(high=255, low=0, shape=(3, 84, 84), dtype='uint8')

    torch_cnn = NatureCNN(obs_space)
    model_params = baselines_model.get_parameters()
    # Get only the policy parameters
    policy_keys = [key for key in model_params.keys() if "fc" in key or "c" in key]
    policy_params = [model_params[key] for key in policy_keys]

    for (th_key, pytorch_param), key, policy_param in zip(torch_cnn.named_parameters(), policy_keys, policy_params):
        param = policy_param.copy()
        # Copy parameters from stable baselines model to pytorch model

        # Conv layer
        if len(param.shape) == 4:
            # https://gist.github.com/chirag1992m/4c1f2cb27d7c138a4dc76aeddfe940c2
            # Tensorflow 2D Convolutional layer: height * width * input channels * output channels
            # PyTorch 2D Convolutional layer: output channels * input channels * height * width
            param = np.transpose(param, (3, 2, 0, 1))

        # weight of fully connected layer
        if len(param.shape) == 2:
            param = param.T

        # bias
        if 'b' in key:
            param = param.squeeze()

        param = torch.from_numpy(param)
        pytorch_param.data.copy_(param.data.clone())

    return torch_cnn

# https://colab.research.google.com/drive/1XwCWeZPnogjz7SLW2kLFXEJGmynQPI-4#scrollTo=eXlo2wl5tFQP
def build_model_with_pretrained(rl_model_path):
    rl_model = PPO2.load(rl_model_path)
    # params = rl_model.get_parameters()
    # Wow, so this outputs actual parameters in numpy!!!

    # odict_keys(['model/c1/w:0', 'model/c1/b:0', 'model/c2/w:0', 'model/c2/b:0', 'model/c3/w:0',
    # 'model/c3/b:0', 'model/fc1/w:0', 'model/fc1/b:0', 'model/lstm1/wx:0', 'model/lstm1/wh:0',
    # 'model/lstm1/b:0', 'model/vf/w:0', 'model/vf/b:0', 'model/pi/w:0', 'model/pi/b:0', 'model/q/w:0', 'model/q/b:0'])
    # for key in params.keys():
    #     print(key, params[key].shape)

    torch_cnn = copy_cnn_weights(rl_model)
    print(torch_cnn)

    # TODO: 1. Add an LSTM on top of this....hmmmm
    # TODO: 2. Add this into a training loop
    # TODO: 3. Train it?


if __name__ == '__main__':
    # verify_dataset()

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    build_model_with_pretrained(model_file)

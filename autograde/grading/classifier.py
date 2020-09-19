import sklearn
from sklearn.neighbors import KernelDensity

import numpy as np

from load_utils import extract_reference_program_data_from_dir, extract_evaluation_program_data

class FeatureExtractor(object):
    @staticmethod
    def process_single(*argv):
        # pass in a list of features from a datapoint
        # the feature extractor will map it to a feature
        # since each program we have, have 8 observations, we need something to aggregate
        raise NotImplementedError

class Trainable(object):
    def __init__(self):
        self.trained = False

    def train(self, parameter_list):
        # some feature extractors need to be trained
        raise NotImplementedError

class TotalRewardFeatureExtractor(FeatureExtractor):
    @staticmethod
    def process_single(data):
        # data: [time_steps, observation_N], Numpy Array
        # we always take in trajectory data; for a single point
        data = data.squeeze()
        total_reward = data.sum(axis=0)
        return total_reward

class TotalRewardClassifier(object):
    def __init__(self, train_correct_folder="./reference_eval_reward_value_stats_correct_programs_8_theme_15_speed/", 
                train_broken_folder="./reference_eval_reward_value_stats_broken_10_programs_2rounds/", cls_type=sklearn.linear_model.LogisticRegression):
        # we want a simple linear decision rules (optimal decision rules based on the feature)
        # this way, things are explanable (feedback can be provided)
        # the trajectory is also stored in there
        
        self.correct_program_to_rewards, self.broken_program_to_rewards \
        , self.correct_program_to_info, self.broken_program_to_info \
        , self.correct_rewards, self.broken_rewards = extract_reference_program_data_from_dir(train_correct_folder, train_broken_folder)
        
        self.cls = cls_type()
        
        # total reward
        self.X, self.y = self.prepare_data_from_json(self.correct_program_to_info, self.broken_program_to_info)
    
    def prepare_data_from_json(self, correct_program_to_info, broken_program_to_info, return_numpy=True):
        # correct = 1, incorrect = 0
        # if return numpy, we return numpy array, otherwise we return a list
        X, y = [], []
        for _, data in correct_program_to_info.items():
            # data['step_rewards']: (1, 1000, 8) Shape
            np_data = np.array(data['step_rewards'])
            for round_idx in range(np_data.shape[0]):
                raw_x = np_data[round_idx, :, :].T
                for obs_index in range(raw_x.shape[0]):
                    X.append(TotalRewardFeatureExtractor.process_single(raw_x[obs_index, :]))
                    y.append(1)
        
        for _, data in broken_program_to_info.items():
            # data['step_rewards']: (1, 1000, 8) Shape
            np_data = np.array(data['step_rewards'])
            for round_idx in range(np_data.shape[0]):
                raw_x = np_data[round_idx, :, :].T
                for obs_index in range(raw_x.shape[0]):
                    X.append(TotalRewardFeatureExtractor.process_single(raw_x[obs_index, :]))
                    y.append(0)

        if return_numpy:
            return np.array(X).reshape(-1, 1), np.array(y)
        else:
            return X, y
    
    def negative_filter(self, X, y):
        new_X, new_y = [], []
        for x, yi in zip(X[:, 0].tolist(), y.tolist()):
            if x <= 0 and yi == 1:
                continue  # for correct program with negative observed reward, we skip
            else:
                new_X.append(x)
                new_y.append(yi)
        new_X, new_y = np.array(new_X).reshape(-1, 1), np.array(new_y)
        return new_X, new_y
    
    def train(self, verbose=True, filter_negative=False):
        # Note that training performance is NOT indicative of evaluation performance
        # because our training set is NOT i.i.d sampled from actual distribution
        if filter_negative:
            X, y = self.negative_filter(self.X, self.y)
        else:
            X, y = self.X, self.y
                        
        self.cls = self.cls.fit(X, y)
        if verbose:
            y_hat = self.cls.predict(X)
            print(sklearn.metrics.classification_report(y, y_hat))
        
    def evaluate(self, folder_dirs):
        assert type(folder_dirs) == list
        X, y = [], []
        for folder_dir in folder_dirs:
            _, _, correct_program_to_info, broken_program_to_info, _, _ = extract_evaluation_program_data(folder_dir)
            new_X, new_y = self.prepare_data_from_json(correct_program_to_info, broken_program_to_info)
            X.extend(new_X)
            y.extend(new_y)
            
        X, y = np.array(X).reshape(-1, 1), np.array(y)
        
        y_hat = self.cls.predict(X)
        print(sklearn.metrics.classification_report(y, y_hat))

class AnomalyFeatureClassifier(object):
    # Will write this one later...
    def __init__(self, train_correct_folder="./reference_eval_reward_value_stats_correct_programs_8_theme_15_speed/", 
                train_broken_folder="./reference_eval_reward_value_stats_broken_10_programs_2rounds/"):
        # we want a simple linear decision rules (optimal decision rules based on the feature)
        # this way, things are explanable (feedback can be provided)
        # the trajectory is also stored in there
        
        self.correct_program_to_rewards, self.broken_program_to_rewards \
        , self.correct_program_to_info, self.broken_program_to_info \
        , self.correct_rewards, self.broken_rewards = extract_reference_program_data_from_dir(train_correct_folder, train_broken_folder)

        pass

    def train(self):
        # we only train on training files
        pass

    def evaluate(self, parameter_list):
        # we can evaluate on any folder
        # it's this folder that needs to load in tons of trajectories
        pass

def run_reward_total():
    pass

if __name__ == "__main__":
    print("here")
    # extract_reference_program_data_filtered()
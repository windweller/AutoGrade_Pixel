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

class TrajectoryDataLoader(object):
    pass

class DataLoader(object):
    def prepare_flat_data_from_json(self, correct_program_to_info, broken_program_to_info, return_numpy,
                                    feature_extractor):
        # This is used for training
        # correct = 1, incorrect = 0
        # if return numpy, we return numpy array, otherwise we return a list
        # TODO: extend this to return expected reward as well
        X, y = [], []
        for _, data in correct_program_to_info.items():
            # data['step_rewards']: (1, 1000, 8) Shape
            np_data = np.array(data['step_rewards'])
            for round_idx in range(np_data.shape[0]):
                raw_x = np_data[round_idx, :, :].T
                for obs_index in range(raw_x.shape[0]):
                    X.append(feature_extractor.process_single(raw_x[obs_index, :]))
                    y.append(1)

        for _, data in broken_program_to_info.items():
            # data['step_rewards']: (1, 1000, 8) Shape
            np_data = np.array(data['step_rewards'])
            for round_idx in range(np_data.shape[0]):
                raw_x = np_data[round_idx, :, :].T
                for obs_index in range(raw_x.shape[0]):
                    X.append(feature_extractor.process_single(raw_x[obs_index, :]))
                    y.append(0)

        if return_numpy:
            # this reshape is for sklearn
            return np.array(X).reshape(-1, 1), np.array(y)
        else:
            return X, y

    def prepare_program_level_data_from_json(self, correct_program_to_info, broken_program_to_info,
                                             feature_extractor, return_numpy):
        # return will be Numpy, shape: [N_observation, Traj of observed reward]
        # not sure if we want to add feature extractor here or elsewhere
        Xs, ys = [], []
        for _, data in correct_program_to_info.items():
            step_rewards = np.array(data['step_rewards'])  # target
            
            step_rewards = np.array(step_rewards).transpose(0, 2, 1) # self.flatten_rounds(np.array(step_rewards))
            # (repeated_times, 8_runs, 1000_traj)
            
            for i in range(step_rewards.shape[0]):
                X, y = [], []
                for round_idx in range(step_rewards.shape[1]):
                    # slide window on both, and take the last element of the target
                    X.append(feature_extractor.process_single(step_rewards[i, round_idx, :]))
                    y.append(1)

                Xs.append(X)
                ys.append(y)

        for _, data in broken_program_to_info.items():
            step_rewards = np.array(data['step_rewards'])  # target
            
            step_rewards = np.array(step_rewards).transpose(0, 2, 1) # self.flatten_rounds(np.array(step_rewards))
            
            for i in range(step_rewards.shape[0]):
                X, y = [], []
                for round_idx in range(step_rewards.shape[1]):
                    # slide window on both, and take the last element of the target
                    X.append(feature_extractor.process_single(step_rewards[i, round_idx, :]))
                    y.append(0)
                
                Xs.append(X)
                ys.append(y)
            
        if return_numpy:
            return np.array(Xs, dtype=float), np.array(ys, dtype=int)

        return Xs, ys


class TotalRewardFeatureExtractor(FeatureExtractor):
    @staticmethod
    def process_single(data):
        # data: [time_steps, observation_N], Numpy Array
        # we always take in trajectory data; for a single point
        data = data.squeeze()
        total_reward = data.sum(axis=0)
        return total_reward


class TotalRewardClassifier(DataLoader):
    def __init__(self, train_correct_folder="./reference_eval_reward_value_stats_correct_programs_8_theme_15_speed/",
                 train_broken_folder="./reference_eval_reward_value_stats_broken_10_programs_2rounds/",
                 cls_type=sklearn.linear_model.LogisticRegression):
        # we want a simple linear decision rules (optimal decision rules based on the feature)
        # this way, things are explanable (feedback can be provided)
        # the trajectory is also stored in there

        self.correct_program_to_rewards, self.broken_program_to_rewards \
            , self.correct_program_to_info, self.broken_program_to_info \
            , self.correct_rewards, self.broken_rewards = extract_reference_program_data_from_dir(train_correct_folder,
                                                                                                  train_broken_folder)

        self.cls = cls_type()

        # total reward
        self.X, self.y = self.prepare_flat_data_from_json(self.correct_program_to_info, self.broken_program_to_info)

    def prepare_flat_data_from_json(self, correct_program_to_info, broken_program_to_info, return_numpy=True,
                                    feature_extractor=TotalRewardFeatureExtractor):
        return super(TotalRewardClassifier, self).prepare_flat_data_from_json(correct_program_to_info,
                                                                              broken_program_to_info,
                                                                              return_numpy=return_numpy,
                                                                              feature_extractor=TotalRewardFeatureExtractor)

    def prepare_program_level_data_from_json(self, correct_program_to_info, broken_program_to_info,
                                             feature_extractor=TotalRewardFeatureExtractor, return_numpy=True):
        # return will be Numpy, shape: [N_observation, Traj of observed reward]
        # not sure if we want to add feature extractor here or elsewhere
        X, y = super(TotalRewardClassifier, self).prepare_program_level_data_from_json(correct_program_to_info,
                                                                                       broken_program_to_info,
                                                                                       feature_extractor=TotalRewardFeatureExtractor,
                                                                                       return_numpy=return_numpy)

        y = y.mean(axis=-1)
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
            print(sklearn.metrics.classification_report(y, y_hat, digits=3))

    def evaluate(self, folder_dirs, verbose=True):
        assert type(folder_dirs) == list
        X, y = [], []
        for folder_dir in folder_dirs:
            _, _, correct_program_to_info, broken_program_to_info, _, _ = extract_evaluation_program_data(folder_dir)
            new_X, new_y = self.prepare_flat_data_from_json(correct_program_to_info, broken_program_to_info)
            X.extend(new_X)
            y.extend(new_y)

        X, y = np.array(X).reshape(-1, 1), np.array(y)

        y_hat = self.cls.predict(X)
        if verbose:
            print(sklearn.metrics.classification_report(y, y_hat, digits=3))

    def evaluate_program_level(self, folder_dirs, verbose=True):
        assert type(folder_dirs) == list
        X, y = [], []
        for folder_dir in folder_dirs:
            _, _, correct_program_to_info, broken_program_to_info, _, _ = extract_evaluation_program_data(folder_dir)
            # new preprocess that maps to X: [N, N_observ], with a loop run each [N_observ, 1], and average result
            new_X, new_y = self.prepare_program_level_data_from_json(correct_program_to_info, broken_program_to_info,
                                                                     return_numpy=True)
            # (1000, 8)
            X.append(new_X)
            y.extend(new_y.tolist())

        X = np.vstack(X)
        y = np.array(y)
        # Now it's (8000, 8)
        # we run classifier over

        N, t = X.shape[0], X.shape[1]

        X_feat = X.reshape(-1, 1)
        y_hat = self.cls.predict(X_feat)
        y_result = y_hat.reshape(N, t)

        # still just compute mean here
        y_pred = y_result.mean(axis=1) > 0.5
        print(sklearn.metrics.classification_report(y, y_pred, digits=3))


# 10 minutes!!! Don't spend more time than that
class GenerativeRewardClassifier(DataLoader):
    def __init__(self, train_correct_folder="./reference_eval_reward_value_stats_correct_programs_8_theme_15_speed/",
                 train_broken_folder="./reference_eval_reward_value_stats_broken_10_programs_2rounds/",
                 bandwidth=0.5):
        # we want a simple linear decision rules (optimal decision rules based on the feature)
        # this way, things are explanable (feedback can be provided)
        # the trajectory is also stored in there
        
        # for now we don't treat this as a feature...even though we could

        self.correct_program_to_rewards, self.broken_program_to_rewards \
            , self.correct_program_to_info, self.broken_program_to_info \
            , self.correct_rewards, self.broken_rewards = extract_reference_program_data_from_dir(train_correct_folder,
                                                                                                  train_broken_folder)

        self.density = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        self.cls = sklearn.linear_model.LogisticRegression()

    def train(self, filter_negative=False, filter_threshold=0, verbose=True):
        # we only fit on correct rewards
        filtered_rewards = self.correct_rewards
        if filter_negative:
            filtered_rewards = [r for r in self.correct_rewards if r > filter_threshold]
            
        filtered_rewards = np.array(filtered_rewards).reshape(-1, 1)
        self.density.fit(filtered_rewards)
        print(filtered_rewards)
        
        # then we find the optimal threshold
        # we implicitly use a linear classifier for this...
        # we return the threshold and set internally
        
        X, y = self.prepare_flat_data_from_json(self.correct_program_to_info, self.broken_program_to_info,
                                               feature_extractor=TotalRewardFeatureExtractor, return_numpy=True)
        
        feat_X = self.density.score_samples(X)
        self.cls.fit(feat_X.reshape(-1, 1), y)
        
        if verbose:
            y_hat = self.cls.predict(feat_X.reshape(-1, 1))
            print(sklearn.metrics.classification_report(y, y_hat, digits=3))
            print("Boundary threshold:", self.cls.coef_[0][0])
            
        self.threshold = self.cls.coef_[0][0]
            
    def evaluate(self, folder_dirs, verbose=True):
        assert type(folder_dirs) == list
        X, y = [], []
        for folder_dir in folder_dirs:
            _, _, correct_program_to_info, broken_program_to_info, _, _ = extract_evaluation_program_data(folder_dir)
            new_X, new_y = self.prepare_flat_data_from_json(correct_program_to_info, broken_program_to_info,
                                                           feature_extractor=TotalRewardFeatureExtractor, return_numpy=False)
            X.extend(new_X)
            y.extend(new_y)

        X, y = np.array(X).reshape(-1, 1), np.array(y)

        X_feat = self.density.score_samples(X)
        y_hat = self.cls.predict(X_feat.reshape(-1, 1))
        
        if verbose:
            print(sklearn.metrics.classification_report(y, y_hat, digits=3))

    def evaluate_program_level(self, folder_dirs, verbose=True):
        assert type(folder_dirs) == list
        X, y = [], []
        for folder_dir in folder_dirs:
            _, _, correct_program_to_info, broken_program_to_info, _, _ = extract_evaluation_program_data(folder_dir)
            # new preprocess that maps to X: [N, N_observ], with a loop run each [N_observ, 1], and average result
            new_X, new_y = self.prepare_program_level_data_from_json(correct_program_to_info, broken_program_to_info,
                                                                     return_numpy=True, feature_extractor=TotalRewardFeatureExtractor)
            # (1000, 8)
            X.append(new_X)
            y.extend(new_y.tolist())

        X = np.vstack(X)
        y = np.array(y).mean(axis=-1)
        # Now it's (8000, 8)
        # we run classifier over

        N, t = X.shape[0], X.shape[1]

        X_feat = X.reshape(-1, 1)
        X_feat = self.density.score_samples(X_feat)
        y_hat = self.cls.predict(X_feat.reshape(-1, 1))
        y_result = y_hat.reshape(N, t)

        # still just compute mean here
        y_pred = y_result.mean(axis=1) > 0.5
        print(sklearn.metrics.classification_report(y, y_pred, digits=3))


class KLDistanceClassifier(DataLoader):
    # distance to "correct" programs (determine the "optimal" distance)
    pass


class MarkovTrajMLP(DataLoader):
    def __init__(self, order_n=5,
                train_correct_folder="./reference_eval_reward_value_stats_correct_programs_8_theme_15_speed/",
                train_broken_folder="./reference_eval_reward_value_stats_broken_10_programs_2rounds/"):

        self.correct_program_to_rewards, self.broken_program_to_rewards \
            , self.correct_program_to_info, self.broken_program_to_info \
            , self.correct_rewards, self.broken_rewards = extract_reference_program_data_from_dir(train_correct_folder,
                                                                                                  train_broken_folder)

import sklearn.neural_network

# this is the best network (on par w/ random forest)
def build_mlp():
    # turn off L2 regularization
    # L2 focuses on every feature...which is not what we want.
    return sklearn.neural_network.MLPClassifier(activation='tanh', hidden_layer_sizes=(100,), max_iter=10000, alpha=0)


if __name__ == "__main__":
    print("here")
    # extract_reference_program_data_filtered()

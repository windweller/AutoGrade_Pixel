import sklearn
from sklearn.neighbors import KernelDensity

from load_utils import extract_reference_program_data_filtered

class FeatureExtractor(object):
    def process(self, *argv):
        # pass in a list of features from a datapoint
        # the feature extractor will map it to a feature
        # since each program we have, have 8 observations, we need something to aggregate
        raise NotImplementedError

class TotalReward(FeatureExtractor):
    def process(self, traj_rewards):
        return sum(traj_rewards)

class AnomalyClassifier(object):
    pass


if __name__ == "__main__":
    print("here")
    # extract_reference_program_data_filtered()
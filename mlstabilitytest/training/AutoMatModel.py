import re
import numpy as np
import pandas as pd
from mlstabilitytest.training.MLModel import MLModel
from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import FeatureReducer, DataCleaner
from automatminer.automl import TPOTAdaptor, SinglePipelineAdaptor
from automatminer.pipeline import MatPipe
import matminer.featurizers.composition as cf
from pymatgen import Composition


class AutoMat(MLModel):
    """
    A wrapper for automatminer
    """

    def __init__(self, target):
        """
        Args:
            target (str) - The key of the target in the input dictionary

        """

        self.target = target

        self.config = {
            "learner": TPOTAdaptor(max_time_mins=600,
                                   max_eval_time_mins=60, n_jobs=2),
            "reducer": FeatureReducer(reducers=('corr', 'tree'),
                                      tree_importance_percentile=0.99),
            "autofeaturizer": AutoFeaturizer(preset="express", n_jobs=1),
            "cleaner": DataCleaner()
        }

    def preprocess(self, X):
        """
        Featurize an input data set.
        Args:
            X (dict): Dictionary of data points
        Returns:
            (np.array): An array of input features for training
            (np.array): Corresponding target property values
            (list of str): Corresponding plain text chemical formulae labels
        """

        feature_df = pd.DataFrame(X).transpose()
        feature_df['composition'] = feature_df.index.map(Composition)

        features = feature_df['composition'].to_numpy(copy=True)
        targets = feature_df[self.target].to_numpy(copy=True)
        labels = feature_df.index.to_numpy(copy=True)

        return features, targets, labels

    def fit(self, X, Y):
        """
        Fit the model to data
        Args:
            X (numpy array) - Input features
            Y (numpy array) - Target values
        """

        # Automatminer expects a dataframe as input to fit, containing compositions and target values
        input_dataframe = pd.DataFrame(list(zip(list(X), list(Y))), columns=[
                                       'composition', self.target])
        self.pipe = MatPipe(**self.config)

        self.pipe.fit(input_dataframe, self.target)

        return self

    def predict(self, X):
        """
        Make a prediction using a trained model
        Args:
            X (numpy array) - Input features
        Returns:
            (list of floats) - predictions
        """

        input_dataframe = pd.DataFrame(list(X), columns=['composition'])
        predictions = self.pipe.predict(input_dataframe)

        # Predict returns float32, which is not json seriallizable, so cast to float
        return [float(x) for x in predictions]

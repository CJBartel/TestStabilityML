import re
import numpy as np
import pandas as pd
from MLModel import MLModel
from matminer.featurizers.composition import ElementProperty, ElementFraction, Meredig
from xgboost import XGBRegressor
from pymatgen import Composition


class MatminerModel(MLModel):
    """
    A wrapper for models implemented through matminer featurizers
    After featurization, these models use an XBG model for regression
    """

    def __init__(self, model_name, target, max_depth=5, n_estimators=200):
        """
        Args:
            model_name (str) - The name of the model to use. One of ElFrac, Deml, Magpie, Meredig
            target (str) - The key of the target in the input dictionary
            max_depth (int) - XGBoost parameter max tree depth parameter
            n_estimators (int) - Number of XGBoost estimators
        """
        featurizer_dictionary = {"ElFrac": ElementFraction(), "Deml": ElementProperty.from_preset(
            "deml"), "Magpie": ElementProperty.from_preset("magpie"), "Meredig": Meredig()}
        try:
            self.featurizer = featurizer_dictionary[model_name]
        except KeyError:
            print("Invalid model selection. Valid choices are {}".format(
                ", ".join(model_dictionary.keys())))
            exit(1)

        self.target = target
        self.model = XGBRegressor(
            max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1)

    def preprocess(self, X):
        """
        Featurize an input data set.
        Args:
            X (dict): Dictionary of data points
        Returns:
            (np.array): An array of input features for training
            (np.array): Corresponding target property values
            (np.array): Corresponding plain text chemical formulae labels
        """

        feature_df = pd.DataFrame(X).transpose()
        feature_df['composition'] = feature_df.index.map(Composition)
        feature_df = self.featurizer.fit_featurize_dataframe(
            feature_df, 'composition', inplace=False, ignore_errors=True, pbar=False)
        feature_names = self.featurizer.feature_labels()
        feature_df = feature_df.fillna(value=0, axis="columns")

        features = np.array(feature_df[[i for i in feature_names]])
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
        self.model.fit(X, Y)

        return self

    def predict(self, X):
        """
        Make a prediction using a trained model
        Args:
            X (numpy array) - Input features
        Returns:
            (list of floats) - predictions
        """
        predictions = self.model.predict(X)

        #Predict returns float32, which is not json seriallizable, so cast to float
        return [float(x) for x in predictions]

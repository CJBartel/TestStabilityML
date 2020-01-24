from abc import ABC, abstractmethod
import re
import numpy as np
import pandas as pd


class MLModel(ABC):
    """
    An abstract base class for ML Models
    ML Models should implement fit and predict, SKLearn style
    Models should also implement preprocess, which takes input as a list of dicts
    and returns a suitable format for the model to learn on.
    """

    @abstractmethod
    def preprocess(self, X):
        """
        Featurize an input data set to a form suitable for use as input to fit and predict
        Args:
            X (dict): Dictionary of data points
        Returns:
            (np.array): An array of input features for training
            (np.array): Corresponding target property values
            (np.array): Corresponding plain text chemical formulae labels
        """
        pass

    @abstractmethod
    def fit(self, X, Y):
        """
        Fit the model to data
        Args:
            X (numpy array) - Input features
            Y (numpy array) - Target values
        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make a prediction using a trained model
        Args:
            X (numpy array) - Input features
        Returns:
            (list of floats) - predictions
        """
        pass



    def fit_and_predict(self, Xtrain, Ytrain, Xtest):
        """
        Convenience method to combine fit and predict
        Args:
            Xtrain (numpy array) - Input features to fit the model on
            Ytrain (numpy array) - Corresponding training target values
            Xtest (numpy array) - Input features to test on
        Returns:
            (list of floats) - predictions
        """

        return self.fit(Xtrain, Ytrain).predict(Xtest)

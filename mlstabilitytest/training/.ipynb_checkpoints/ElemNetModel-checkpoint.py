import re
import numpy as np
import pandas as pd
from collections import defaultdict
from mlstabilitytest.training.MLModel import MLModel
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import optimizers
# from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping


class ElemNet(MLModel):
    """
    ElemNet model implemented using Keras
    Input parsing from official implementation at
    https://github.com/NU-CUCIS/ElemNet

    Citations;
    Dipendra Jha, Logan Ward, Arindam Paul, Wei-keng Liao, Alok Choudhary, Chris Wolverton, and Ankit Agrawal, “ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition,” Scientific Reports, 8, Article number: 17593 (2018) [DOI:10.1038/s41598-018-35934-y]

    Dipendra Jha, Kamal Choudhary, Francesca Tavazza, Wei-keng Liao, Alok Choudhary, Carelyn Campbell, Ankit Agrawal, "Enhancing materials property prediction by leveraging computational and experimental data using deep transfer learning," Nature Communications, 10, Article number: 5316 (2019) [DOI: https:10.1038/s41467-019-13297-w]
    """

    def __init__(self, target):
        """
        Args:
            target (str) - The key of the target in the input dictionary

        """

        self.target = target

        self.elements = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V',
                         'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                         'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
                         'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
                         'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

        # Regular expression for parsing formula strings
        self.formulare = re.compile(r'([A-Z][a-z]*)(\d*)')

    def preprocess(self, X):
        """
        Featurize an input data set into ordered element fractions.
        Args:
            X (dict): Dictionary of data points
        Returns:
            (np.array): An array of input features for training
            (np.array): Corresponding target property values
            (np.array): Corresponding plain text chemical formulae labels
        """

        labels = []
        targets = []
        # Each row in features array is the elemental fraction of the corresponding element
        features = np.zeros((len(X.items()), len(self.elements)))

        for i, item in enumerate(X.items()):
            k, v = item
            # Formulae are keys of input dictionary, values are dictionary of properties
            labels.append(k.strip())
            targets.append(v[self.target])

            # Convert a string formula to an {element_name: number} dictionary
            parsed_formula = self.__parse_formula(k.strip())

            # Convert the dictionary to a numpy array, and normalize to element fractions
            total_n_eles = float(sum(parsed_formula.values()))

            for ele, number in parsed_formula.items():
                features[i][self.elements.index(ele)] = number / total_n_eles

        return features, np.array(targets), np.array(labels)

    def fit(self, X, Y):
        """
        Fit the model to data
        Args:
            X (numpy array) - Input features
            Y (numpy array) - Target values
        """

        # Create the Keras model
        self.__create_model()

        # Train for 200 epochs, stopping early if there's no improvement in validation loss within 20 epochs
        # Then restore the weights from the best epoch
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                                   verbose=1, mode='auto', baseline=None, restore_best_weights=True)]

        self.model.fit(X, Y, epochs=200, validation_split=0.1,
                       batch_size=64, callbacks=callbacks)

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

        # Predict returns float32, which is not json seriallizable, so cast to float
        return [float(x) for x in predictions]

    def __create_model(self):

        self.model = Sequential()
        for i in range(4):
            self.model.add(Dense(1024, activation='linear'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        for i in range(3):
            self.model.add(Dense(512, activation='linear'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        for i in range(3):
            self.model.add(Dense(256, activation='linear'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('linear'))

        self.model.add(Dropout(0.3))
        for i in range(2):
            self.model.add(Dense(64, activation='linear'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

        self.model.add(Dense(32, activation='linear'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(1, activation='linear'))
        adam = optimizers.Adam(learning_rate=0.0001)

        self.model.compile(loss='mean_absolute_error',
                           optimizer=adam, metrics=['mae'])

    def __parse_formula(self, formula):
        pairs = self.formulare.findall(formula)
        length = sum((len(p[0]) + len(p[1]) for p in pairs))
        assert length == len(formula)
        formula_dict = defaultdict(int)
        for el, sub in pairs:
            formula_dict[el] += float(sub) if sub else 1
        return formula_dict

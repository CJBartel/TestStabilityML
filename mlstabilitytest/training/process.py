from os.path import dirname, abspath, join
import json
from sklearn.model_selection import KFold
from MatminerModel import MatminerModel
from ElemNetModel import ElemNet
from AutoMatModel import AutoMat

base_path = dirname(dirname(abspath(__file__)))
data_path = join(base_path, "mp_data", "data")

#Dictionary of available models
model_dictionary = {"Deml": lambda target: MatminerModel('Deml', target),
                    "ElFrac": lambda target: MatminerModel('ElFrac', target),
                    "Magpie": lambda target: MatminerModel('Magpie', target),
                    "Meredig": lambda target: MatminerModel('Meredig', target),
                    "ElemNet": lambda target: ElemNet(target),
                    "AutoMat": lambda target: AutoMat(target),
                    }

#List of available target properties
target_list = ["Ed", "Ef"]


#Functions to perform training and prediction for a problem, given an input model and target property
def allMP(model, target):

    input_file = join(data_path, "hullout.json")

    print("Reading input data from {}".format(input_file))

    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("Preprocessing data")
    features, targets, labels = model.preprocess(input_data)

    predictions = dict()

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    iFold = 0
    for train_indices, test_indices in kf.split(features):
        print("Training on fold {}".format(iFold))
        iFold += 1

        features_train = features[train_indices]
        targets_train = targets[train_indices]

        features_test = features[test_indices]
        targets_test = targets[test_indices]
        labels_test = labels[test_indices]

        predictions_this_fold = model.fit_and_predict(
            Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

        predictions = {**predictions, **{labels_test[i]: predictions_this_fold[i] for i, x in enumerate(test_indices)}}

    return predictions


def LiMnTMO(model, target):
    input_file_all_mp = join(data_path, "hullout.json")

    input_file_LiMnTMO = join(data_path, "mp_LiMnTMO.json")

    print("Reading input data from {},{}".format(
        input_file_all_mp, input_file_LiMnTMO))

    # Load the training set
    with open(input_file_all_mp, 'r') as f:
        input_data_all_mp = json.load(f)

    # Load the test set
    with open(input_file_LiMnTMO, 'r') as f:
        input_data_LiMnTMO = json.load(f)

    # Make sure to exclude the test set from the training set
    input_data_all_mp = {k: v for k, v in input_data_all_mp.items(
    ) if k not in input_data_LiMnTMO.keys()}

    print("Preprocessing data")
    features_train, targets_train, _ = model.preprocess(input_data_all_mp)

    features_test, _, labels_test = model.preprocess(input_data_LiMnTMO)

    print("Training")
    predictions = model.fit_and_predict(
        Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

    predictions = {labels_test[i]: predictions[i]
                   for i, x in enumerate(labels_test)}

    return predictions


def smact(model, target):
    input_file_all_mp = join(data_path, "hullout.json")

    input_file_LiMnTMO = join(data_path, "mp_LiMnTMO.json")

    input_file_smact_LiMnTMO = join(data_path, "smact_LiMnTMO.json")

    print("Reading input data from {},{}, {}".format(
        input_file_all_mp, input_file_LiMnTMO, input_file_smact_LiMnTMO))

    # Load the training set
    with open(input_file_all_mp, 'r') as f:
        input_data_all_mp = json.load(f)

    # Load the test set
    with open(input_file_LiMnTMO, 'r') as f:
        input_data_LiMnTMO = json.load(f)

    # Load the smact formulae
    with open(input_file_smact_LiMnTMO, 'r') as f:
        input_data_smact_LiMnTMO = json.load(f)['smact']

    # The smact formulae are stored as a list of formulae
    # Convert to a dictionary of formulae: dummy value for consistency of data format
    # The dummy value is irrelevant since we only use these for testing
    dummy_dict = {target: 0 for target in target_list}

    input_data_smact_LiMnTMO = {
        formula: dummy_dict for formula in input_data_LiMnTMO}

    # Combine the smact and MP test sets
    input_data_LiMnTMO = {**input_data_LiMnTMO, **input_data_smact_LiMnTMO}

    # Make sure to exclude the test set from the training set
    input_data_all_mp = {k: v for k, v in input_data_all_mp.items(
    ) if k not in input_data_LiMnTMO.keys()}

    print("Preprocessing data")
    features_train, targets_train, _ = model.preprocess(input_data_all_mp)

    features_test, _, labels_test = model.preprocess(input_data_LiMnTMO)

    print("Training")
    predictions = model.fit_and_predict(
        Xtrain=features_train, Ytrain=targets_train, Xtest=features_test)

    predictions = {labels_test[i]: predictions[i]
                   for i, x in enumerate(labels_test)}

    return predictions

#Dictionary of available problem functions
problem_dictionary = {"allMP": allMP,
                      "LiMnTMO": LiMnTMO,
                      "smact": smact}

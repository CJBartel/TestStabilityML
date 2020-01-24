from os.path import dirname, abspath, join
from sys import argv
import json
from mlstabilitytest.training.process import problem_dictionary, target_list, model_dictionary

base_path = dirname(dirname(abspath(__file__)))
output_base_path = join(base_path, "mlstabilitytest", "ml_data")


def main(argv):
    """
    Train models on the MP data, and make predictions in a format suitable for hull analysis

    """
    try:
        problem = argv[1]
        target = argv[2]
        model_name = argv[3]
    except IndexError:
        print("Arguments should be Problem Target Model")
        exit(1)

    try:
        train_func = problem_dictionary[problem]
    except KeyError:
        print("Invalid problem selection {}. Valid choices are {}".format(problem,
                                                                          ", ".join(problem_dictionary.keys())))
        exit(1)
    try:
        if not target in target_list:
            raise ValueError
    except ValueError:
        print("Invalid target selection. Valid choices are {}".format(
            ", ".join(target_list)))
        exit(1)

    try:
        model = model_dictionary[model_name](target)
    except KeyError:
        print("Invalid model selection. Valid choices are {}".format(
            ", ".join(model_dictionary.keys())))
        exit(1)

    output_file = join(output_base_path, target, problem,
                       model_name, 'ml_input.json')

    print("Training {} to predict {} using the {} dataset".format(
        model_name, target, problem))

    predictions = train_func(model, target)

    print("Training complete, saving predictions to {}".format(output_file))
    with open(output_file, 'w') as f:
        json.dump(predictions, f)


if __name__ == '__main__':
    main(argv)

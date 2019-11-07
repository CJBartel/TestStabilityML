import os
from mlstabilitytest.StabilityAnalysis import StabilityAnalysis

models = ['elfrac', 'cgcnn', 'arXiv19', 'auto', 'npj16', 'prb16', 'prb14']
experiments = ['LiMnTMO', 'allMP']
path_to_examples = '/global/home/users/cbartel/bin/TestStabilityML/mlstabilitytest/examples/'

def process(model, experiment):
    data_dir = os.path.join(path_to_examples, experiment, model)
    data_file = 'ml_input.json'
    nprocs = 'all'
    obj = StabilityAnalysis(data_dir,
                            data_file,
                            experiment,
                            nprocs)
    obj.results_summary

def concatenate_results():
    data = {model : {experiment : read_json(os.path.join(path_to_examples, experiment, model, 'ml_results.json')) for experiment in experiments for model in models}}
    fjson = os.path.join('/global/home/users/cbartel/bin/TestStabilityML/mlstabilitytest/data/data/combined_model_results.json')
    return write_json(data, fjson)

def main():
    for experiment in experiments:
        print('\n ~~~ %s ~~~\n' % experiment)
        for model in models:
            print('\n %s ' % model)
            process(model, experiment)

if __name__ == '__main__':
    main()

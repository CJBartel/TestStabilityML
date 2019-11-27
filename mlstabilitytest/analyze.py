import os
from mlstabilitytest.utils import StabilityAnalysis, EdAnalysis

def main():
    models = ['ElFrac', 'Meredig', 'Magpie', 'AutoMat', 'ElemNet', 'Roost', 'Deml', 'CGCNN']
    experiments = ['LiMnTMO', 'allMP', 'smact']
    training_props = ['Ef', 'Ed']
    path_to_ml_data = '/Users/chrisbartel/Dropbox/postdoc/projects/ML_H/code/TestStabilityML/mlstabilitytest/ml_data'
    for training_prop in training_props:
        print('\n____ models trained on %s ____\n' % training_prop)
        for experiment in experiments:
            print('\n ~~~ %s ~~~\n' % experiment)
            for model in models:
                print('\n %s ' % model)
                process(training_prop, model, experiment, path_to_ml_data)
                
def process(training_prop, model, experiment, path_to_ml_data):
    """
    Args:
        training_prop (str) - 'Ef' if models trained on formation energies; 'Ed' if decomposition energies
        model (str) - ML model
        experiment (str) - 'allMP', 'LiMnTMO', or 'smact'
        path_to_ml_data (os.PathLike) - path to ml_data directory in .../TestStabilityML/mlstabilitytest/ml_data
    
    Returns:
        Runs all relevant analyses
        Prints a summary
    """
    if (model == 'CGCNN') and (experiment == 'smact'):
        print('CGCNN cannot be applied directly to the SMACT problem because the structures are not known')
        return
    data_dir = os.path.join(path_to_ml_data, training_prop, experiment, model)
    data_file = 'ml_input.json'
    if not os.path.exists(os.path.join(data_dir, data_file)):
        print('missing data for %s-%s' % (model, experiment))
        return
    if training_prop == 'Ef':
        nprocs = 'all'
        obj = StabilityAnalysis(data_dir,
                                data_file,
                                experiment,
                                nprocs)
    elif training_prop == 'Ed':
        obj = EdAnalysis(data_dir,
                         data_file,
                         experiment)
    else:
        raise NotImplementedError
    obj.results_summary
    print('got results')
    return

if __name__ == '__main__':
    main()

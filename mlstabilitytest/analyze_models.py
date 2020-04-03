import os
from mlstabilitytest.stability.StabilityAnalysis import StabilityAnalysis, EdAnalysis
from shutil import copyfile

here = os.path.abspath(os.path.dirname(__file__))
def main():
    models = ['ElFrac', 'Meredig', 'Magpie', 'AutoMat', 'ElemNet', 'Roost', 
              'CGCNN']
    """
    experiments = ['LiMnTMO', 'allMP', 'smact']
    training_props = ['Ef', 'Ed']
    """
    experiments = ['random1', 'random2', 'random3']
 #   experiments = experiments[::-1]
    training_props = ['Ef']
    path_to_ml_data = os.path.join(here, 'ml_data') 
    for training_prop in training_props:
        print('\n____ models trained on %s ____\n' % training_prop)
        for experiment in experiments:
            print('\n ~~~ %s ~~~\n' % experiment)
            experiment_dir = os.path.join(path_to_ml_data, training_prop, experiment)
            if not os.path.exists(experiment_dir):
                os.mkdir(experiment_dir)
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
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    data_file = 'ml_input.json'
    finput = os.path.join(data_dir, data_file)
    if 'random' in experiment:
        src = finput.replace(experiment, 'allMP')
        copyfile(src, finput)
    if not os.path.exists(finput):
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

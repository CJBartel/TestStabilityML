import os
from subprocess import call

def _names():
    return {'elfrac' : 'ElFrac',
            'prb14' : 'Meredig',
            'prb16' : 'Deml',
            'npj16' : 'Magpie',
            'auto' : 'AutoMat',
            'elemnet' : 'ElemNet',
            'arXiv19' : 'Roost',
            'cgcnn' : 'CGCNN'}

def convert_name(training_prop, exp, name, names):
    PARENT_DIR = '/global/home/users/cbartel/bin/TestStabilityML/mlstabilitytest/ml_data'
    old_dir = os.path.join(PARENT_DIR, training_prop, exp, name)
    new_dir = os.path.join(PARENT_DIR, training_prop, exp, names[name])
    print(old_dir)
    print(new_dir)
    call(['mv', old_dir, new_dir])

def main():
    names = _names()
    for training_prop in ['Ef', 'Ed']:
        for exp in ['allMP', 'LiMnTMO', 'smact']:
            for name in names:
                print('\n')
                convert_name(training_prop, exp, name, names)
#                return
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:23:57 2019

@author: chrisbartel
"""

from mlstabilitytest.StabilityAnalysis import StabilityAnalysis
import os

example_dir = os.path.join(os.path.dirname(__file__))

def npj16_LiMnTMO():
    """
    Takes ~60 s on 7 cores
    """
    
    data_dir = os.path.join(example_dir, 'examples', 'LiMnTMO', 'npj16')
    data_file = 'ml_input.json'
    experiment = 'LiMnTMO'
    nprocs = 'all'
    obj = StabilityAnalysis(data_dir, 
                            data_file, 
                            experiment,
                            nprocs)
    return obj.results_summary()

def npj16_allMP():
    """
    Takes ~15 min on 27 cores
    """
    data_dir = os.path.join(example_dir, 'examples', 'allMP', 'npj16')
    data_file = 'ml_input.json'
    experiment = 'all'
    nprocs = 'all'
    obj = StabilityAnalysis(data_dir, 
                            data_file, 
                            experiment,
                            nprocs)
    return obj.results()    

def npj_smact():
    """
    Takes ~ s on 7 cores
    """
    data_dir = os.path.join(example_dir, 'examples', 'smact', 'npj16')
    data_file = 'ml_input.json'
    experiment = 'smact'
    nprocs = 'all'
    obj = StabilityAnalysis(data_dir,
                            data_file,
                            experiment,
                            nprocs)
    
    return obj.smact_results
    
def main():
    return npj_smact()

if __name__ == '__main__':
    d = main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:35:53 2020

@author: chrisbartel
"""

import os
from mlstabilitytest.stability.utils import read_json, write_json
from mlstabilitytest.mp_data.data import Ef
from mlstabilitytest.stability.StabilityAnalysis import StabilityStats
import numpy as np

data_dir = '/Users/chrisbartel/Downloads/mp_fraction'

models = ['ElFrac', 'Meredig', 'Magpie', 'ElemNet', 'AutoMat', 'Roost']

training_amts = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8]

splits = range(5)

this_dir, this_filename = os.path.split(__file__)


def get_model_summary(model, training_amt, remake=False):
    print(model)
    print(training_amt)
    learning_dir = os.path.join(this_dir, 'ml_data', 'Ef', 'learning')
    if not os.path.exists(learning_dir):
        os.mkdir(learning_dir)
    model_dir = os.path.join(learning_dir, model)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    fjson = os.path.join(model_dir, '_'.join([str(training_amt), 'ml', 'results.json']))
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
    
    if training_amt == 0.8:
        fjson_in = os.path.join(this_dir, 'ml_data', 'Ef', 'allMP', model, 'ml_results.json')
        d = read_json(fjson_in)
        summary = {'mae' : {'mean' : d['stats']['Ef']['abs']['mean'],
                            'std' : 0}}
        return write_json(summary, fjson)
    
    fjson_in = os.path.join(data_dir, 
                         ''.join(['mp_fraction', str(training_amt)]), 
                         model, 
                         'ml_input.json')
    d = read_json(fjson_in)
    mp = Ef()
    out = {}
    for split in splits:
        data = d[str(split)]
        formulas = sorted(list(data.keys()))
        print('%i formulas' % len(formulas))
        
        actual = [mp[k]['Ef'] for k in formulas]
        pred = [data[k] for k in formulas]
        
        if (split == 0) and (model == 'ElemNet'):
            return formulas, actual, pred
        
        ss = StabilityStats(actual, pred)
        
        stats = ss.regression_stats
        out[split] = stats
        
    for i in out:
        print(i, out[i]['abs']['mean'])
        
    summary = {'mae' : {'mean' : np.mean([out[i]['abs']['mean'] for i in out]),
                        'std' : np.std([out[i]['abs']['mean'] for i in out])}}
    print(summary)
    return write_json(summary, fjson)

"""
def get_compoundwise_model_summary(model, training_amt):
    
    print(model)
    
    mp = Ef()
    
    fjson = os.path.join(data_dir, 
                         ''.join(['mp_fraction', str(training_amt)]), 
                         model, 
                         'ml_input.json')
    d = read_json(fjson)
    
    unique_formulas = [list(d[str(split)].keys()) for split in splits]
    unique_formulas = [j for i in unique_formulas for j in i]
    unique_formulas = list(set(unique_formulas))
    
    out = {}
    for formula in unique_formulas:
        
        curr_splits = [s for s in d if formula in d[s]]
        preds = [d[s][formula] for s in curr_splits]
        actual = mp[formula]['Ef']
        errors = [actual - pred for pred in preds]
        aerrors = [abs(e) for e in errors]
        out[formula] = np.mean(aerrors)
        
        
    return np.mean(list(out.values()))
"""
 
def main():
    """
    out = get_model_summary('ElemNet', 0.01, True)
    return out
    """
    remake = True
    out1 = {model : {training_amt : get_model_summary(model, training_amt, True) 
                    for training_amt in training_amts}
                    for model in models}  
    """
    out2 = {model : {training_amt : get_compoundwise_model_summary(model, training_amt) 
                    for training_amt in training_amts}
                    for model in models}  
            
                
    """     
    return out1

if __name__ == '__main__':
    out1 = main()

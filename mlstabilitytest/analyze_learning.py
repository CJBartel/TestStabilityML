#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:35:53 2020

@author: chrisbartel
"""

import os
from mlstabilitytest.stability.utils import read_json
from mlstabilitytest.mp_data.data import Ef
from mlstabilitytest.stability.StabilityAnalysis import StabilityStats
import numpy as np

data_dir = '/Users/chrisbartel/Downloads/mp_fraction'

models = ['ElFrac', 'Meredig', 'Magpie', 'ElemNet']

training_amts = [0.001, 0.01, 0.1, 0.2, 0.5]

splits = range(5)

def get_model_summary(model, training_amt):
    
    print(model)
    
    mp = Ef()
    
    fjson = os.path.join(data_dir, 
                         ''.join(['mp_fraction', str(training_amt)]), 
                         model, 
                         'ml_input.json')
    d = read_json(fjson)
    
    
    out = {}
    for split in splits:
        data = d[str(split)]
        formulas = sorted(list(data.keys()))
        print('%i formulas' % len(formulas))
        
        actual = [mp[k]['Ef'] for k in formulas]
        pred = [data[k] for k in formulas]
        
        ss = StabilityStats(actual, pred)
        
        stats = ss.regression_stats
        out[split] = stats
        
    summary = {'mae' : {'mean' : np.mean([out[i]['abs']['mean'] for i in out]),
                        'std' : np.std([out[i]['abs']['mean'] for i in out])}}
    return summary

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
    
def main():
    
    out1 = {model : {training_amt : get_model_summary(model, training_amt) 
                    for training_amt in training_amts}
                    for model in models}  

    out2 = {model : {training_amt : get_compoundwise_model_summary(model, training_amt) 
                    for training_amt in training_amts}
                    for model in models}  
            
                
        
    return out1, out2

if __name__ == '__main__':
    out1, out2 = main()
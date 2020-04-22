#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:35:53 2020

@author: chrisbartel
"""

import os
from mlstabilitytest.stability.utils import read_json
from mlstabilitytest.mp_data.data import hullout
from mlstabilitytest.stability.StabilityAnalysis import StabilityStats
import numpy as np
from prettytable import PrettyTable

data_dir = '/Users/chrisbartel/Downloads/allMP_class'

models = ['ElFrac', 'Meredig', 'Magpie', 'ElemNet']

here = os.path.abspath(os.path.dirname(__file__))

def get_model_summary(model):
    
    print(model)
        
    fjson = os.path.join(data_dir, 
                         model, 
                         'ml_input.json')
    d = read_json(fjson)
    
    d = {k : {'prob' : d[k],
              'stability' : True if d[k] >= 0.5 else False,
              'Ed' : -1 if d[k] >= 0.5 else 1} for k in d}
    
    mp = hullout()
    
    actual = [mp[k]['Ed'] for k in mp]
    pred = [d[k]['Ed'] for k in mp]
    
    
    summary = StabilityStats(actual, pred).classification_scores(0)
        
    return summary
    
def main():
    
    out = {model : get_model_summary(model) for model in models}     
    
    x = PrettyTable()
    
    header = ['', 'Acc', 'F1', 'FPR']
    x.field_names = header
    for m in models:
        row = [m] + ['%.3f' % np.round(out[m][prop], 3) for prop in ['accuracy',
                                                            'f1',
                                                            'fpr']]        
        x.add_row(row)
        
    print(x)

    
    return out

if __name__ == '__main__':
    out = main()
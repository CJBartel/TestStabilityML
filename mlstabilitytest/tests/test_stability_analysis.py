#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:10:49 2020

@author: chrisbartel
"""

import unittest
import os
from mlstabilitytest.stability.StabilityAnalysis import StabilityAnalysis
from mlstabilitytest.stability.utils import read_json
from time import time

here = os.path.abspath(os.path.dirname(__file__))

class UnitTestStabilityAnalysis(unittest.TestCase):
    
    def test_LiMnTMO(self):
        data_dir = os.path.join(here, 'data')
        data_file = os.path.join(data_dir, 'ml_input.json')
        out_files = [os.path.join(data_dir, f) for 
                     f in ['ml_results.json',
                           'ml_hullin.json',
                           'ml_hullout.json']]
        for f in out_files:
            if os.path.exists(f):
                os.remove(f)
        experiment = 'LiMnTMO'
        nprocs = 4
        start = time()
        obj = StabilityAnalysis(data_dir,
                                data_file,
                                experiment,
                                nprocs)
        check_output = read_json(os.path.join(here, '..', 'ml_data', 'Ef', 'LiMnTMO', 'ElFrac', 'ml_results.json'))
        mae_check = check_output['stats']['Ef']['abs']['mean']   
        new_output = obj.results(True)
        mae_new = new_output['stats']['Ef']['abs']['mean']
        end = time()
        print('\n%.2f processor-hours used' % ((end-start)*nprocs/3600))
        self.assertEqual(mae_new, mae_check)
        
if __name__ == '__main__':
    unittest.main()
        
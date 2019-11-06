#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:31:14 2019

@author: chrisbartel
"""

from compmatscipy.handy_functions import read_json, write_json
from compmatscipy.CompAnalyzer import CompAnalyzer
import os

from mlstabilitytest.data import Ef, hullin, hullout, spaces

def make_MP_LiMnTMO(d):
    TMs = ['Ti', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu']
    combos = [sorted(['Li', 'Mn', 'O', TM]) for TM in TMs]
    
    formulas = list(d.keys())
    formulas = [f for f in formulas if CompAnalyzer(f).els in combos]
    
    data = {f : d[f] for f in formulas}
    
    return write_json(data, os.path.join('data', 'data', 'mp_LiMnTMO.json'))

def make_smact_LiMnTMO():
    mp_LiMnTMO = read_json(os.path.join('data', 'data', 'mp_LiMnTMO.json')).keys()
    smact_LiMnTMO = read_json(os.path.join('data', 'data', 'smact_LiMnTMO.json')).keys()
    
    
def main():
    d = Ef()
    data = make_MP_LiMnTMO(d)
    return d, data

if __name__ == '__main__':
    d, data = main()
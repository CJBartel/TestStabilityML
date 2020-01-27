#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:00:21 2019

@author: chrisbartel
"""

import os
from mlstabilitytest.stability.utils import read_json
from shutil import unpack_archive

this_dir, this_filename = os.path.split(__file__)

def spaces():
    """
    {compound (str) : chemical space containing compound that is easy to eval (str)
        for compound in MP}
    """
    fjson = os.path.join(this_dir, "data", "spaces.json")
    return read_json(fjson)

def Ef():
    """
    {compound (str) : {'ID' : MP ID for ground-state structure (str),
                       'Ef' : formation energy (float, eV/atom)}
        for compound in MP}
    """
    fjson = os.path.join(this_dir, "data", "Ef.json")
    return read_json(fjson)

def hullin():
    """
    {chemical space (str) : {compound (str) : {'E' : formation energy (float, eV/atom),
                                               'amts' : {element (str) : fractional amt of element in compound (float)
                                                   for element in chemical space}}
                                for compound in chemical space}
        for chemical space in Materials Project}
    """
    ftar = os.path.join(this_dir, 'data', 'hullin.json.tar.gz')

    fjson = os.path.join(this_dir, 'data', "hullin.json")
    data_dir = os.path.join(this_dir, 'data')
    unpack_archive(ftar, data_dir)
    d = read_json(fjson)
    os.remove(fjson)
    return d

def hullout():
    """
    {compound (str) : {'Ef' : formation energy (float, eV/atom),
                       'Ed' : decomposition energy (float, eV/atom),
                       'rxn' : decomposition reaction (str),
                       'stability' : True if Ed <= 0 else False}
        for compound in Materials Project}
    """
    fjson = os.path.join(this_dir, "data", "hullout.json")
    return read_json(fjson)

def mp_LiMnTMO():
    """
    {compound (str) : {'ID' : MP ID for ground-state structure (str),
                       'Ef' : formation energy (float, eV/atom)}
        for compound in MP in the Li-Mn-TM-O quaternary space}
    """ 
    fjson = os.path.join(this_dir, "data", "mp_LiMnTMO.json")
    return read_json(fjson)

def smact_LiMnTMO():
    """
    {'smact' : [compounds (str) generated in Li-Mn-TM-O space with SMACT]}
    """ 
    fjson = os.path.join(this_dir, "data", "smact_LiMnTMO.json")
    return read_json(fjson)

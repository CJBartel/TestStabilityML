#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:00:21 2019

@author: chrisbartel
"""

import os
from compmatscipy.handy_functions import read_json

this_dir, this_filename = os.path.split(__file__)

def spaces():
    fjson = os.path.join(this_dir, "data", "spaces.json")
    return read_json(fjson)

def Ef():
    fjson = os.path.join(this_dir, "data", "Ef.json")
    return read_json(fjson)

def hullin():
    fjson = os.path.join(this_dir, "data", "hullin.json")
    return read_json(fjson)

def hullout():
    fjson = os.path.join(this_dir, "data", "hullout.json")
    return read_json(fjson)

def mp_LiMnTMO():
    fjson = os.path.join(this_dir, "data", "mp_LiMnTMO.json")
    return read_json(fjson)
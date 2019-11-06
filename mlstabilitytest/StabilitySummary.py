#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:06:34 2019

@author: chrisbartel
"""

from mlstabilitytest.StabilityStats import StabilityStats
from compmatscipy.CompAnalyzer import CompAnalyzer

class StabilitySummary(object):
    
    def __init__(self, 
                 mp, 
                 ml):
        
        self.mp = mp
        self.ml = ml
    
    @property
    def Ef(self):
        mp = self.mp
        ml = self.ml
        formulas = sorted(list(mp.keys()))
        return {'actual' : [mp[formula]['Ef'] for formula in formulas],
                'pred' : [ml[formula]['Ef'] for formula in formulas]}
        
    @property
    def stats_Ef(self):
        Ef = self.Ef
        actual, pred = Ef['actual'], Ef['pred']
        return StabilityStats(actual, pred).regression_stats
    
    @property
    def Ed(self):
        mp = self.mp
        ml = self.ml
        formulas = sorted(list(mp.keys()))
        return {'actual' : [mp[formula]['Ed'] for formula in formulas],
                'pred' : [ml[formula]['Ed'] for formula in formulas]}
        
    @property
    def stats_Ed(self):
        Ed = self.Ed
        actual, pred = Ed['actual'], Ed['pred']
        reg = StabilityStats(actual, pred).regression_stats
        cl = StabilityStats(actual, pred).classification_stats
        return {'reg' : reg,
                'cl' : cl}

    @property
    def rxns(self):
        mp = self.mp
        ml = self.ml
        formulas = sorted(list(mp.keys()))
        return {'actual' : [mp[formula]['rxn'] for formula in formulas],
                'pred' : [ml[formula]['rxn'] for formula in formulas]}

    @property
    def formulas(self):
        mp = self.mp
        return sorted(list(mp.keys()))

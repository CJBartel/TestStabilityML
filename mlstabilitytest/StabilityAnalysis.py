#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:49:07 2019

@author: chrisbartel
"""

import os
from compmatscipy.handy_functions import read_json, write_json
from compmatscipy.CompAnalyzer import CompAnalyzer
from compmatscipy.HullAnalysis import AnalyzeHull
from mlstabilitytest.data.data import Ef, mp_LiMnTMO, smact_LiMnTMO, hullin, hullout, spaces
from mlstabilitytest.StabilitySummary import StabilitySummary
import multiprocessing as multip
from time import time

def _update_hullin_space(ml, mp_hullin, space):
    """
    replace MP data for chemical space with ML data
    
    Args:
        ml (dict) - {formula (str) : {'Ef' : ML-predicted formation energy per atom (float)}}
        mp_hullin (dict) - mlstabilitytest.data.hullin
        space (str) - chemical space to update (format is '_'.join(sorted([element (str) for element in chemical space])))
        
    Returns:
        input data for analysis of one chemical space updated with ML-predicted data
    """
    
    ml_space = mp_hullin[space]
    
    for compound in ml_space:
        if (CompAnalyzer(compound).num_els_in_formula == 1) or (compound not in ml):
            continue
        else:
            ml_space[compound]['E'] = ml[compound]['Ef']
    
    return ml_space

def _assess_stability(hullin, spaces, compound):
    """
    determine the stability of a given compound by hull analysis
    
    Args:
        hullin (dict) - {space (str) : {formula (str) : {'E' : formation energy per atom (float),
                                                         'amts' : 
                                                             {element (str) : 
                                                                 fractional amount of element in formula (float)}}}}
        spaces (dict) - {formula (str) : smallest convex hull space that includes formula (str)}
        compound (str) - formula to determine stability for
        
    Returns:
        {'stability' : True if on the hull, else False,
         'Ef' : formation energy per atom (float),
         'Ed' : decomposition energy per atom (float),
         'rxn' : decomposition reaction (str)}
    """
    return AnalyzeHull(hullin, spaces[compound]).cmpd_hull_output_data(compound)

def _get_smact_hull_space(compound, mp_spaces):
    """
    TO-DO
    """
    els = set(CompAnalyzer(compound).els)
    for s in mp_spaces:
        space_els = set(s.split('_'))
        if (len(els.intersection(space_els)) == 4) and (len(space_els) < 7):
            return s
        
        
def _update_smact_space(ml,  mp_hullin, space):
    """
    TO-DO
    """
    ml_space = mp_hullin[space]
    for compound in ml_space:
        if (CompAnalyzer(compound).num_els_in_formula == 1) or (compound not in ml):
            continue
        else:
            ml_space[compound]['E'] = ml[compound]['Ef']

    for compound in ml:
        if set(CompAnalyzer(compound).els).issubset(set(space.split('_'))):
            ml_space[compound] = {'E' : ml[compound]['Ef'],
                                  'amts' : {el : CompAnalyzer(compound).amt_of_el(el)
                                   for el in space.split('_')}}

    return ml_space
        
def _get_stable_compounds(hullin, space):
    """
    TO-DO
    """
    return AnalyzeHull(hullin, space).stable_compounds



class StabilityAnalysis(object):
    """
    Perform stability analysis over all of Materials Project using ML-predicted formation energies
    
    Timing:
    The 'all' experiment takes *** s on 7 cores
    The 'LiMnTMO' experiment takes ~45 s on 7 cores
    
    """
    
    def __init__(self, 
                 data_dir, 
                 data_file, 
                 experiment='allMP',
                 nprocs='all'):
        """
        converts input data to convenient format
        
        Args:
            data_dir (os.PathLike) - place where input ML data lives and to generate output data
            data_file (str) - .json file with input ML data of form {formula (str) : formation energy per atom (float)}
            experiment (str) - 'all' for all MP compounds or 'LiMnTMO' for Li-Mn-TM-O (TM in ['V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu'])
            nprocs (str or int) - number of processors to parallelize analysis over ('all' --> all available processors)
        """
        
        start = time()
        
        print('\nChecking input data...')
        
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            
        input_data = read_json(os.path.join(data_dir, data_file))
        input_data = {CompAnalyzer(k).std_formula() : {'Ef' : float(input_data[k])}
                        for k in input_data}
        
        if experiment == 'allMP':
            mp = Ef()
            compounds = list(mp.keys())
        elif experiment == 'LiMnTMO':
            mp = mp_LiMnTMO()
            compounds = list(mp.keys())
        elif experiment == 'smact':
            mp = mp_LiMnTMO()
            smact = smact_LiMnTMO()
            compounds = list(set(list(mp.keys()) + smact['smact']))
        else:
            raise NotImplementedError
                    
        if set(compounds).intersection(set(list(input_data.keys()))) != set(compounds):
            print('ML dataset does not include all MP formulas!')
            print('Cannot perform analysis.')
            raise AssertionError
            
        input_data = {c : input_data[c] for c in compounds}
            
        self.compounds = compounds
        
        self.input_data = input_data
        
        self.data_dir = data_dir
        
        if nprocs == 'all':
            self.nprocs = multip.cpu_count() - 1
        else:
            self.nprocs = nprocs
    
        end = time()
        print('Data looks good.')
        print('Time elapsed = %.0f s.' % (end-start))
        
            
    def ml_hullin(self, remake=False):
        """
        generates input file for stability analysis using ML data
        
        Note on computational expense:
        -for experiment='allMP', requires ~5 min on 27 processors
        -for experiment='LiMnTMO', requires ~30 s on 7 processors

        Args:
            remake (bool) - repeat generation of file if True; else read file
            
        Returns:
            dictionary with ML data that can be processed by hull analysis program
            
            {space (str) : {formula (str) : {'E' : formation energy per atom (float),
                                                         'amts' : 
                                                             {element (str) : 
                                                                 fractional amount of element in formula (float)}}}}
                
            saves dictionary to file
        """
            
        
        fjson = os.path.join(self.data_dir, 'ml_hullin.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing stability input file: %s.' % fjson)
            return read_json(fjson)

        nprocs = self.nprocs
        print('\nGenerating stability input file on %i processors...' % nprocs)
        start = time()

        compounds = self.compounds
                
        mp_hullin = hullin()
        
        compound_to_space = spaces()
        
        relevant_spaces = list(set([compound_to_space[compound] for compound in compounds]))
    
        ml = self.input_data
        
        pool = multip.Pool(processes=nprocs)
        
        ml_spaces = pool.starmap(_update_hullin_space, 
                              [(ml, mp_hullin, space) 
                              for space in relevant_spaces])
    
        ml_hullin = dict(zip(relevant_spaces, ml_spaces))
        
        end = time()
        print('Writing to %s' % fjson)
        print('Time elapsed = %.0f s' % (end-start))
        
        return write_json(ml_hullin, fjson)
    
    def ml_hullout(self, remake=False):
        """
        generates output file with stability analysis using ML data

        Note on computational expense:
        -for experiment='allMP', requires ~10 min on 27 processors
        -for experiment='LiMnTMO', requires ~30 s on 7 processors
        
        Args:
            remake (bool) - repeat generation of file if True; else read file
            
        Returns:
            dictionary with stability analysis using ML-predicted formation energies

            {formula (str) : {'stability' : True if on the hull, else False,
                              'Ef' : formation energy per atom (float),
                              'Ed' : decomposition energy per atom (float),
                              'rxn' : decomposition reaction (str)}}
                
            saves dictionary to file
        """
        
        fjson = os.path.join(self.data_dir, 'ml_hullout.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing stability output file: %s.' % fjson)
            return read_json(fjson)
        
        ml_hullin = self.ml_hullin(False)
        nprocs = self.nprocs
        print('\nGenerating stability output file on %i processors...' % nprocs)
        start = time()

        compounds = self.compounds  
        
        hull_spaces = spaces()
        
        pool = multip.Pool(processes=nprocs)

        
        stabilities = pool.starmap(_assess_stability,
                                   [(ml_hullin, hull_spaces, compound)
                                   for compound in compounds])
        
        ml_hullout = dict(zip(compounds, stabilities))
        
        end = time()
        print('Writing to %s.' % fjson)
        print('Time elapsed = %.0f s.' % (end-start))
        
        return write_json(ml_hullout, fjson)
    
    def results(self, remake=False):
        """
        generates output file with summary data for rapid analysis
        
        Args:
            remake (bool) - repeat generation of file if True; else read file
            
        Returns:
            dictionary with results in convenient format for analysis
            
            TO-DO: finish comment
            
        """
        
        fjson = os.path.join(self.data_dir, 'ml_results.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing results file: %s' % fjson)
            return read_json(fjson)        
        
        

        ml_hullout = self.ml_hullout(False)
        
        print('\nCompiling results...')
        start = time()
        
        compounds = self.compounds
        mp_hullout = hullout()
        mp_hullout = {compound : mp_hullout[compound] for compound in compounds}
        
        obj = StabilitySummary(mp_hullout, ml_hullout)
        
        results = {'stats' : {'Ed' : obj.stats_Ed,
                              'Ef' : obj.stats_Ef},
                   'data' : {'Ed' : obj.Ed['pred'],
                             'Ef' : obj.Ef['pred'],
                             'rxns' : obj.rxns['pred'],
                             'formulas' : obj.formulas}}
    
        end = time()
        print('Writing to %s.' % fjson)
        print('Time elapsed = %.0f s.' % (end-start))
    
        return write_json(results, fjson)  

    @property
    def results_summary(self):
        """
        TO-DO
        """
        
        results = self.results(False)
        
        start = time()
        print('\nSummarizing performance...')
        
        Ef_MAE = results['stats']['Ef']['abs']['mean']
        Ed_MAE = results['stats']['Ed']['reg']['abs']['mean']
        
        tp, fp, tn, fn = [results['stats']['Ed']['cl']['0']['raw'][num] for num in ['tp', 'fp', 'tn', 'fn']]
        prec, recall, acc, f1 = [results['stats']['Ed']['cl']['0']['scores'][score] for score in ['precision', 'recall', 'accuracy', 'f1']]
        fpr = fp/(fp+tn)
        
        print('\nMAE on formation enthalpy = %.3f eV/atom' % Ef_MAE)
        print('MAE on decomposition enthalpy = %.3f eV/atom' % Ed_MAE)
        
        print('\nClassifying stable or unstable:')
        print('Precision = %.3f' % prec)
        print('Recall = %.3f' % recall)
        print('Accuracy = %.3f' % acc)
        print('F1 = %.3f' % f1)
        print('FPR = %.3f' % fpr)
        
        print('\nConfusion matrix:')
        print('TP | FP\nFN | TN = \n%i | %i\n%i | %i' % (tp, fp, fn, tn))
        
        end = time()
        print('\nTime elapsed = %i s' % (end-start))
        
    def smact_results(self, remake=False):
        """
        TO-DO
        """
        fjson = os.path.join(self.data_dir, 'ml_results.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing smact results from: %s.' % fjson)
            return read_json(fjson)
        
        nprocs = self.nprocs
        print('Analyzing SMACT compounds on %i processors...' % nprocs)
        
        start = time()
        mp_spaces = spaces()
        mp_spaces = list(set(list(mp_spaces.values())))
        mp_hullin = hullin()
        compounds = self.compounds
        ml = self.input_data
        
        pool = multip.Pool(processes=nprocs)
        
        relevant_spaces = pool.starmap(_get_smact_hull_space,
                                   [(compound, mp_spaces) for compound in compounds])
        smallest_spaces = dict(zip(compounds, relevant_spaces))
        relevant_spaces = list(set(relevant_spaces))
        
        end1 = time()
        print('Got %i relevant chemical spaces in %i s' % (len(relevant_spaces), end1-start))
        
        ml_spaces = pool.starmap(_update_smact_space,
                                 [(ml, mp_hullin, space)
                                 for space in relevant_spaces])
        ml_hullin = dict(zip(relevant_spaces, ml_spaces))
        
        end2 = time()
        print('Updated hullin in %i s' % (end2-end1))
        
        stable_compounds = pool.starmap(_get_stable_compounds,
                                    [(ml_hullin, space) for space in relevant_spaces])
        
        stable_compounds = list(set([j for i in stable_compounds for j in i]))
        end3 = time()
        print('Assessed stability in %i s' % (end3-end2))

        mp_hullout = hullout()
        mp_LiMnTMO_stable = [c for c in mp_hullout 
                             if c in compounds if mp_hullout[c]['stability']]
        
        pred_LiMnTMO_stable = [c for c in compounds if c in stable_compounds]

        results = {'compounds' : compounds,
                   'MP_stable' : mp_LiMnTMO_stable,
                   'pred_stable' : pred_LiMnTMO_stable}
        end = time()
        print('Time elapsed = %i s' % (end-start))
        return write_json(results, fjson)

    @property
    def smact_summary(self):
        results = self.smact_results(False)
        compounds, mp_LiMnTMO_stable, pred_LiMnTMO_stable = [results[k] for k in ['compounds', 'MP_stable', 'pred_stable']]
        
        print('%i compounds investigated' % len(compounds))
        print('%i are stable in MP' % len(mp_LiMnTMO_stable))
        print('%i (%.2f) are predicted to be stable' % (len(pred_LiMnTMO_stable), len(pred_LiMnTMO_stable)/len(compounds)))
        print('%i of those are stable in MP' % (len([c for c in pred_LiMnTMO_stable if c in mp_LiMnTMO_stable])))


    @property
    def Ed_results(self):

        ml = self.input_data
        mp_hullout = hullout()

        ml = {compound : {'Ed' : ml[compound]['Ef']} for compound in ml}

        obj = StabilitySummary(mp_hullout, ml)

        stats = obj.stats_Ed

        print(stats)

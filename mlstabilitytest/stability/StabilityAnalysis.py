#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:49:07 2019

@author: chrisbartel
"""

import os
from mlstabilitytest.stability.CompositionAnalysis import CompositionAnalysis
from mlstabilitytest.stability.HullAnalysis import AnalyzeHull
from mlstabilitytest.mp_data.data import Ef, mp_LiMnTMO, smact_LiMnTMO, hullin, hullout, spaces
from mlstabilitytest.stability.utils import read_json, write_json
import multiprocessing as multip
from time import time
from sklearn.metrics import confusion_matrix, r2_score
import numpy as np
import random
        
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
        if (CompositionAnalysis(compound).num_els_in_formula == 1) or (compound not in ml):
            continue
        else:
            ml_space[compound]['E'] = ml[compound]
    
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
    Args:
        compound (str) - compound to retrieve phase space for
        mp_spaces (list) - list of phase spaces in MP (str, '_'.join(elements))
    
    Returns:
        relevant chemical space (str) to determine stability compound
    """
    els = set(CompositionAnalysis(compound).els)
    for s in mp_spaces:
        space_els = set(s.split('_'))
        if (len(els.intersection(space_els)) == 4) and (len(space_els) < 7):
            return s
        
def _update_smact_space(ml,  mp_hullin, space):
    """
    Args:
        ml (dict) - {compound (str) - formation energy per atom (float)}
        mp_hullin (dict) - hull input file for all of MP
        space (str) - chemical space to update
        
    Returns:
        replaces MP formation energy with ML formation energies in chemical space
    """
    ml_space = mp_hullin[space]
    for compound in ml_space:
        if (CompositionAnalysis(compound).num_els_in_formula == 1) or (compound not in ml):
            continue
        else:
            ml_space[compound]['E'] = ml[compound]

    for compound in ml:
        if set(CompositionAnalysis(compound).els).issubset(set(space.split('_'))):
            ml_space[compound] = {'E' : ml[compound],
                                  'amts' : {el : CompositionAnalysis(compound).amt_of_el(el)
                                   for el in space.split('_')}}

    return ml_space
        
def _get_stable_compounds(hullin, space):
    """
    Args:
        hullin (dict) - hull input data
        space (str) - chemical space
        
    Returns:
        list of all stable compounds (str) in chemical space
    """
    return AnalyzeHull(hullin, space).stable_compounds

class StabilityAnalysis(object):
    """
    Perform stability analysis over all of Materials Project using ML-predicted formation energies
    
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

        finput = os.path.join(data_dir, data_file)
        input_data = read_json(finput)
        input_data = {CompositionAnalysis(k).std_formula() : float(input_data[k])
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
        elif 'random' in experiment:
            mp = Ef()
            compounds = list(mp.keys())
        else:
            raise NotImplementedError
                    
        if set(compounds).intersection(set(list(input_data.keys()))) != set(compounds):
            print('ML dataset does not include all MP formulas!')
            print('Cannot perform analysis.')
            raise AssertionError
            
        input_data = {c : input_data[c] for c in compounds}
        
        if 'random' in experiment:
            random.seed(int(experiment.split('random')[1]))
            errors = [mp[c]['Ef'] - input_data[c] for c in compounds]
            random.shuffle(errors)
            input_data = {compounds[i] : float(mp[compounds[i]]['Ef']+errors[i]) for i in range(len(errors))}
            input_data = write_json(input_data, finput)
            
        self.compounds = compounds
        
        self.input_data = input_data
        
        self.data_dir = data_dir
        
        self.experiment = experiment
        
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
            
            {'stats' : {'Ed' : decomposition energy prediction statistics (dict),
                        'Ef' : formation energy prediction statistics (dict)},
             'data' : {'Ed' : decomposition energy data (list of floats),
                       'Ef' : formation energy data (list of floats),
                       'rxns' : decomposition reactions (list of str),
                       'formulas' :  compounds (list of str)}}
            
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
        Args:
            
        Returns:
            prints a summary of key results
        """
        
        if self.experiment == 'smact':
            self.smact_summary
            return
        
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
        Args:
            remake (bool) - repeat generation of file if True; else read file
            
        Returns:
            performs stability analysis on smact compounds
            writes output data to file
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
        """
        Args:
            
        Returns:
            prints summary of key results for smact analysis
        """
        results = self.smact_results(False)
        compounds, mp_LiMnTMO_stable, pred_LiMnTMO_stable = [results[k] for k in ['compounds', 'MP_stable', 'pred_stable']]
        
        print('%i compounds investigated' % len(compounds))
        print('%i are stable in MP' % len(mp_LiMnTMO_stable))
        print('%i (%.2f) are predicted to be stable' % (len(pred_LiMnTMO_stable), len(pred_LiMnTMO_stable)/len(compounds)))
        print('%i of those are stable in MP' % (len([c for c in pred_LiMnTMO_stable if c in mp_LiMnTMO_stable])))
        
class EdAnalysis(object):
    """
    Assess performance of direct ML predictions on decomposition energy
    
    """    
    
    def __init__(self, 
                 data_dir, 
                 data_file, 
                 experiment='allMP'):

        """
        converts input data to convenient format
        
        Args:
            data_dir (os.PathLike) - place where input ML data lives and to generate output data
            data_file (str) - .json file with input ML data of form {formula (str) : formation energy per atom (float)}
            experiment (str) - 'all' for all MP compounds or 'LiMnTMO' for Li-Mn-TM-O (TM in ['V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu'])
        """
        
        start = time()
        
        print('\nChecking input data...')
        
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
            
        input_data = read_json(os.path.join(data_dir, data_file))
        
        input_data = {CompositionAnalysis(k).std_formula() : float(input_data[k])
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
        elif experiment == 'classifier':
            mp = Ef()
            compounds = list(mp.keys())
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
        
        self.experiment = experiment
            
        end = time()
        print('Data looks good.')
        print('Time elapsed = %.0f s.' % (end-start)) 
        
    @property
    def ml_hullout(self):
        """
        Args:
            
        Returns:
            converts input data into standard hull output data using ML-predicted Ed (dict)
            {compound (str) : {'Ef' : None (not inputted),
                               'Ed' : decomposition energy per atom (float),
                               'stability' : True if Ed <= 0 else False,
                               'rxn' : None (not determined)}}
        """
        ml_in = self.input_data
        compounds = self.compounds
        if self.experiment == 'classifier':
            return {c : {'Ef' : None,
                         'Ed' : -1 if ml_in[c] >= 0.5 else 1,
                         'stability' : True if ml_in[c] >= 0.5 else False,
                         'rxn' : None} for c in compounds}
        else:
            return {c : {'Ef' : None,
                         'Ed' : ml_in[c],
                         'stasbility' : True if ml_in[c] <= 0 else False,
                         'rxn' : None} for c in compounds}
        
    def results(self, remake=False):
        """
        generates output file with summary data for rapid analysis
        
        Args:
            remake (bool) - repeat generation of file if True; else read file
            
        Returns:
            dictionary with results in convenient format for analysis
        """
        
        fjson = os.path.join(self.data_dir, 'ml_results.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing results file: %s' % fjson)
            return read_json(fjson)        
        
        ml_hullout = self.ml_hullout
        
        print('\nCompiling results...')
        start = time()
        
        compounds = self.compounds
        mp_hullout = hullout()
        mp_hullout = {compound : mp_hullout[compound] for compound in compounds}
        
        obj = StabilitySummary(mp_hullout, ml_hullout)
        
        results = {'stats' : {'Ed' : obj.stats_Ed},
                   'data' : {'Ed' : obj.Ed['pred'],
                             'formulas' : obj.formulas}}
        if self.experiment == 'classifier':
            del results['stats']['Ed']['reg']
        end = time()
        print('Writing to %s.' % fjson)
        print('Time elapsed = %.0f s.' % (end-start))
    
        return write_json(results, fjson)
    
    @property
    def results_summary(self):
        """
        Prints summary of performance
        """
        
        if self.experiment == 'smact':
            self.smact_summary
            return
        
        results = self.results(False)
        
        start = time()
        print('\nSummarizing performance...')
        
        if self.experiment != 'classifier':
            Ed_MAE = results['stats']['Ed']['reg']['abs']['mean']
        else:
            Ed_MAE = np.nan
        tp, fp, tn, fn = [results['stats']['Ed']['cl']['0']['raw'][num] for num in ['tp', 'fp', 'tn', 'fn']]
        prec, recall, acc, f1 = [results['stats']['Ed']['cl']['0']['scores'][score] for score in ['precision', 'recall', 'accuracy', 'f1']]
        fpr = fp/(fp+tn)
        
        print('\nMAE on decomposition enthalpy = %.3f eV/atom\n' % Ed_MAE)
        
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
        Args:
            remake (bool) - repeat generation of file if True; else read file
        
        Returns:
            Analyzes smact results
        """
        fjson = os.path.join(self.data_dir, 'ml_results.json')
        if not remake and os.path.exists(fjson):
            print('\nReading existing smact results from: %s.' % fjson)
            return read_json(fjson)
                
        start = time()
        mp_spaces = spaces()
        mp_spaces = list(set(list(mp_spaces.values())))
        compounds = self.compounds
        ml = self.input_data
        
        stable_compounds = [c for c in compounds if ml[c] <= 0]
        

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
        """
        Args:
            
        Returns:
            prints summary of SMACT results
        """
        results = self.smact_results(False)
        compounds, mp_LiMnTMO_stable, pred_LiMnTMO_stable = [results[k] for k in ['compounds', 'MP_stable', 'pred_stable']]
        
        print('%i compounds investigated' % len(compounds))
        print('%i are stable in MP' % len(mp_LiMnTMO_stable))
        print('%i (%.2f) are predicted to be stable' % (len(pred_LiMnTMO_stable), len(pred_LiMnTMO_stable)/len(compounds)))
        print('%i of those are stable in MP' % (len([c for c in pred_LiMnTMO_stable if c in mp_LiMnTMO_stable])))

def _make_binary_labels(data, thresh):
    """
    Args:
        data (list) - list of floats
        thresh (float) - value to partition on 
    
    Returns:
        1 if value <= thresh else 0 for value in list
    """
    return [1 if v <= thresh else 0 for v in data]

class StabilityStats(object):
    """
    Perform statistical analysis on stability results
    """
    
    def __init__(self, actual, pred, 
                 percentiles=[1, 10, 25, 50, 75, 90, 99],
                 stability_thresholds=[0]):
        """
        Args:
            actual (list) - list of actual values (float) for some property
            pred (list) - list of predicted values (float) for some property
            percentiles (list) - list of percentiles (int) to obtain
            stability_thresholds (list) - list of thresholds (float) on which to classify as stable (below threshold) or unstable (above threshold)
        
        Returns:
            checks that actual and predicted lists are same length
        """
        if len(actual) != len(pred):
            raise ValueError
        self.actual = actual
        self.pred = pred
        self.percentiles = percentiles
        self.stability_thresholds = stability_thresholds

    @property
    def errors(self):
        """
        list of actual minus predicted
        """
        a, p = self.actual, self.pred
        return [a[i] - p[i] for i in range(len(a))]
    
    @property
    def abs_errors(self):
        """
        list of absolute value of actual minus predicted
        """
        errors = self.errors
        return [abs(e) for e in errors]

    @property
    def sq_errors(self):
        """
        list of (actual minus predicted) squared
        """
        errors = self.errors
        return [e**2 for e in errors]
    
    @property
    def mean_error(self):
        """
        mean error
        """
        return np.mean(self.errors)
    
    @property
    def mean_abs_error(self):
        """
        mean absolute error
        """
        return np.mean(self.abs_errors)
    
    @property
    def root_mean_sq_error(self):
        """
        root mean squared error
        """
        return np.sqrt(np.mean(self.sq_errors))
    
    @property
    def median_error(self):
        """
        median error
        """
        return np.median(self.errors)
    
    @property
    def median_abs_error(self):
        """
        median absolute error
        """
        return np.median(self.abs_errors)
    
    @property
    def r2(self):
        """
        correlation coefficient squared
        """
        return r2_score(self.actual, self.pred)
    
    @property
    def per_errors(self):
        """
        percentile errors (dict) {percentile : e such that percentile % of errors are < e}
        """
        percentiles = self.percentiles
        errors = self.errors
        return {int(percentiles[i]) : np.percentile(errors, percentiles)[i] for i in range(len(percentiles))}

    @property
    def per_abs_errors(self):
        """
        percentile absolute errors (dict) {percentile : |e| such that percentile % of |errors| are < |e|}
        """
        percentiles = self.percentiles
        errors = self.abs_errors
        return {int(percentiles[i]) : np.percentile(errors, percentiles)[i] for i in range(len(percentiles))}
        
    @property
    def regression_stats(self):
        """
        summary of stats
        """
        return {'abs' : {'mean' : self.mean_abs_error,
                         'median' : self.median_abs_error,
                         'per' : self.per_abs_errors},
                'raw' : {'mean' : self.mean_error,
                         'median' : self.median_error,
                         'per' : self.per_errors},
                'rmse' : self.root_mean_sq_error,
                'r2' : self.r2}
                
    def confusion(self, thresh):
        """
        Args:
            thresh (float) - threshold for stability (eV/atom)
        
        Returns:
            confusion matrix as dictionary
        """
        actual = _make_binary_labels(self.actual, thresh)
        pred = _make_binary_labels(self.pred, thresh)
        cm = confusion_matrix(actual, pred).ravel()
        labels = ['tn', 'fp', 'fn', 'tp']
        return dict(zip(labels, [int(v) for v in cm])) 
    
    def classification_scores(self, thresh):
        """
        Args:
            thresh (float) - threshold for stability (eV/atom)
        
        Returns:
            classification stats as dict
        """        
        confusion = self.confusion(thresh)
        tn, fp, fn, tp = [confusion[stat] for stat in ['tn', 'fp', 'fn', 'tp']]
        if tp+fp == 0:
            prec = 0
        else:
            prec = tp/(tp+fp)
        if tp+fn == 0:
            rec = 0
        else:
            rec = tp/(tp+fn)
        if prec+rec == 0:
            f1 = 0
        else:
            f1 = 2*(prec*rec)/(prec+rec)
        acc = (tp+tn)/(tp+tn+fp+fn)
        if fp+tn == 0:
            fpr = 0
        else:
            fpr = fp/(fp+tn)
        return {'precision' : prec,
                'recall' : rec,
                'f1' : f1,
                'accuracy' : acc,
                'fpr' : fpr}
    
    @property
    def classification_stats(self): 
        """
        summary of classification stats
        """
        threshs = self.stability_thresholds
        return {str(thresh) : {'raw' : self.confusion(thresh),
                               'scores' : self.classification_scores(thresh)} for thresh in threshs}

class StabilitySummary(object):
    """
    Summarize stability performance stats
    """
    
    def __init__(self, 
                 mp, 
                 ml):
        """
        Args:
            mp (dict) - dictionary of MP hull output data
            ml (dict) - dictionary of ML hull output data
        
        Returns:
            mp, ml
        """
        
        self.mp = mp
        self.ml = ml
    
    @property
    def Ef(self):
        """
        put actual and predicted formation energies in dict
        """
        mp = self.mp
        ml = self.ml
        formulas = sorted(list(mp.keys()))
        return {'actual' : [mp[formula]['Ef'] for formula in formulas],
                'pred' : [ml[formula]['Ef'] for formula in formulas]}
        
    @property
    def stats_Ef(self):
        """
        get stats on predicting formation energy
        """
        Ef = self.Ef
        actual, pred = Ef['actual'], Ef['pred']
        return StabilityStats(actual, pred).regression_stats
    
    @property
    def Ed(self):
        """
        put actual and predicted decomposition energies in dict
        """
        mp = self.mp
        ml = self.ml
        formulas = sorted(list(mp.keys()))
        return {'actual' : [mp[formula]['Ed'] for formula in formulas],
                'pred' : [ml[formula]['Ed'] for formula in formulas]}
        
    @property
    def stats_Ed(self):
        """
        get stats on predicting decomposition energy
        """
        Ed = self.Ed
        actual, pred = Ed['actual'], Ed['pred']
        reg = StabilityStats(actual, pred).regression_stats
        cl = StabilityStats(actual, pred).classification_stats
        return {'reg' : reg,
                'cl' : cl}

    @property
    def rxns(self):
        """
        get decomposition reactions
        """
        mp = self.mp
        ml = self.ml
        formulas = sorted(list(mp.keys()))
        return {'actual' : [mp[formula]['rxn'] for formula in formulas],
                'pred' : [ml[formula]['rxn'] for formula in formulas]}

    @property
    def formulas(self):
        """
        get compounds considered
        """
        mp = self.mp
        return sorted(list(mp.keys()))

import re
from itertools import combinations
import numpy as np

def gcd(a,b):
    """
    Args:
        a (float, int) - some number
        b (float, int) - another number
    
    Returns:
        greatest common denominator (int) of a and b
    """
    while b:
        a, b = b, a%b
    return a    

class CompositionAnalysis(object):
    """
    standardizes and analyzes chemical formulas
    """
    
    def __init__(self, formula):
        """
        Args:
            formula (str) - chemical composition
        
        Returns:
            
        """
        self.formula = formula
     
    @property
    def has_parentheses(self):
        """
        Args:
            
        Returns:
            True if formula has parentheses or False if not
        """
        formula = self.formula
        if ('(' in formula) or (')' in formula):
            return True
        else:
            return False
        
    @staticmethod
    def el_to_amt(formula_piece):
        """
        Args:
            formula_piece (str) - subset or all of chemical composition
        
        Returns:
            {element (str) : number of that element in formula_piece (int)}
        """
        el_amt_pairs = re.findall('([A-Z][a-z]\d*)|([A-Z]\d*)', formula_piece)
        el_amt_pairs = [[j[i] for i in range(len(j)) if j[i] != ''][0] for j in el_amt_pairs]
        amts = [re.findall('\d+', el_amt_pair) for el_amt_pair in el_amt_pairs]
        amts = [amt[0] if len(amt) > 0 else '' for amt in amts]
        els = [el_amt_pairs[i].strip(amts[i]) for i in range(len(amts))]
        amts = [int(amt) if amt != '' else 1 for amt in amts]
        d = {el : 0 for el in els}
        for i in range(len(els)):
            d[els[i]] += amts[i]
        return d

    def reduce_formula(self, formula):
        """
        Args:
            formula (str) - chemical composition that may or may not be one formula unit
        
        Returns:
            chemical composition (str) that is exactly one formula unit
        """
        amts = [int(i) for i in re.findall('\d+', formula)]
        if 1 in amts:
            return formula
        elif amts == [2]:
            return formula
        elif len(amts) == 1:
            els = re.findall('[A-Z][a-z]?', formula)
            return els[0]+'1'
        else:
            els = re.findall('[A-Z][a-z]?', formula)
            amt_combinations = list(combinations(amts, 2))
            individual_gcds = [gcd(combination[0], combination[1]) for combination in amt_combinations]
            overall_gcd = np.min(individual_gcds)
            new_amts = [int(amt/overall_gcd) for amt in amts]
            return ''.join([els[i]+str(new_amts[i]) for i in range(len(els))])

    def std_formula(self, reduce=True):
        """
        Args:
            reduce (bool) - the final formula should be exactly one formula unit (True) or may be as inputted (False)
        
        Returns:
            standardized chemical formula (str)
        """
        formula = self.formula
        if not self.has_parentheses:
            el_to_amt = self.el_to_amt(formula)
        else:
            parentheses_interiors = re.findall('\((.*?)\)', formula)
            parentheses_multipliers = re.findall('\(.*?\)(\d*)', formula)
            interior_to_multiplier = dict(zip(parentheses_interiors, parentheses_multipliers))
            interior_to_multiplier = {k : int(interior_to_multiplier[k]) if interior_to_multiplier[k] != '' else 1 for k in interior_to_multiplier}
            overall_el_to_amt = {}
            el_to_amts = []
            for interior in interior_to_multiplier:
                multiplier = interior_to_multiplier[interior]
                el_to_amt = self.el_to_amt(interior)
                el_to_amts.append({el : multiplier*el_to_amt[el] for el in el_to_amt})
            parentheses_components = ['('+parentheses_interiors[i]+')'+parentheses_multipliers[i] for i in range(len(parentheses_interiors))]
            for c in parentheses_components:
                formula = formula.replace(c, '')                
            el_to_amts.append(self.el_to_amt(formula))   
            for el_to_amt in el_to_amts:
                for el in el_to_amt:
                    if el not in overall_el_to_amt:
                        overall_el_to_amt[el] = el_to_amt[el]
                    else:
                        overall_el_to_amt[el] += el_to_amt[el]
            el_to_amt = overall_el_to_amt
        
        unreduced_formula = ''.join([''.join([el, str(el_to_amt[el])]) for el in sorted(list(el_to_amt.keys()))])
        if reduce:
            return self.reduce_formula(unreduced_formula)
        else:
            return unreduced_formula
        
    @property
    def els(self):
        """
        Args:
            
        Returns:
            sorted list of elements (str)
        """
        return re.findall('[A-Z][a-z]?', self.std_formula())
    
    def amts(self, reduce=True):
        """
        Args:
            reduce (bool) - the final formula should be exactly one formula unit (True) or may be as inputted (False)
        
        Returns:
            number of each element in the standardized formula (int) in the order of self.els, which is alphabetical
        """
        return [int(v) for v in re.findall('\d+', self.std_formula(reduce))]

    def els_to_amts(self, reduce=True):
        """
        Args:
            reduce (bool) - the final formula should be exactly one formula unit (True) or may be as inputted (False)
        
        Returns:
            {element (str) : stoichiometric amount of that element (int)}
        """
        return dict(zip(self.els, self.amts(reduce)))
                
    @property
    def num_els_in_formula(self):
        """
        Args:
            
        Returns:
            number (int) of unique elements in the formula
        """
        return len(self.els)
    
    def num_atoms_in_formula(self, reduce=True):
        """
        Args:
            reduce (bool) - the final formula should be exactly one formula unit (True) or may be as inputted (False)
        
        Returns:
            total number of atoms in the formula
        """        
        return np.sum(self.amts(reduce))
    
    @property
    def fractional_amts(self):
        """
        Args:

        Returns:
            fraction of each element in the standardized formula (int) in the order of self.els, which is alphabetical
        """        
        amts = self.amts()
        atoms_in_formula = self.num_atoms_in_formula()
        return [amt/atoms_in_formula for amt in amts]
    
    def amt_of_el(self, el, reduce=True):
        """
        Args:
            el (str) - element to check for amount of
            reduce (bool) - the final formula should be exactly one formula unit (True) or may be as inputted (False)
        
        Returns:
            the amount (int) of el in the formula
        """
        el_amts = dict(zip(self.els, self.amts(reduce)))
        if el in el_amts:
            return el_amts[el]
        else:
            return 0
        
    def fractional_amt_of_el(self, el):
        """
        Args:
            el (str) - element to check for fractional amount of
        
        Returns:
            the fractional amount (float) of el in the formula
        """        
        el_amts = dict(zip(self.els, self.fractional_amts))
        if el in el_amts:
            return el_amts[el]
        else:
            return 0
        
def main():
    return

if __name__ == '__main__':
    main()

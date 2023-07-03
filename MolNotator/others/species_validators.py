import numpy as np

class Species_validator:
    def __init__(self, mass_error):
        self.mass_error = mass_error
        self.selector = {
            'M1|p1H|pCH3OH' : self.get_arr_M1p1HpCH3OH,
            'M1|p1H|pHCOOH' :  self.get_arr_M1p1HpHCOOH,
            'M2|p1Na|' :  self.get_arr_M2p1Na,
            'M1|p1NH4|' :  self.get_arr_M1p1NH4,
            'M1|p1NH4|pCH3CN' :  self.get_arr_M1p1NH4pCH3CN,
            'M1|p1Na|' :  self.get_arr_M1p1Na,
            'M1|p1H|' :  self.get_arr_M1p1H,
            'M1|p1Na|pCH3CN' :  self.get_arr_M1p1NapCH3CN,
            'M1|p1H|pCH3CN' :  self.get_arr_M1p1HpCH3CN,
            'M2|p1H|' :  self.get_arr_M2p1H,
            'M2|p1H|pCH3CN' :  self.get_arr_M2p1HpCH3CN,
            'M2|p1NH4|' :  self.get_arr_M2p1NH4,
            'M2|p1K|' :  self.get_arr_M2p1K,
            'M1|p1K|' :  self.get_arr_M1p1K,
            'M1|p1K|pCH3OH' :  self.get_arr_M1p1KpCH3OH,
            'M2|p1Na|pCH3OH' :  self.get_arr_M2p1NapCH3OH,
            'M2|p1H|pHCOOH' :  self.get_arr_M2p1HpHCOOH,
            'M2|p1Na|pCH3CN' :  self.get_arr_M2p1NapCH3CN,
            'M2|p1H|pCH3OH' :  self.get_arr_M2p1HpCH3OH,
            'M3|p1H|' :  self.get_arr_M3p1H,
            'M3|p1Na|' :  self.get_arr_M3p1Na,
            'M3|p1NH4|' :  self.get_arr_M3p1NH4,
            'M4|p1K|' :  self.get_arr_M4p1K,
            'M4|p1H|' :  self.get_arr_M4p1H,
            'M4|p1NH4|' :  self.get_arr_M4p1NH4,
            'M4|p1Na|' : self.get_arr_M4p1Na,
            'M3|p1K|' :  self.get_arr_M3p1K,
            'M1|m1H|' :  self.get_arr_M1m1H,
            'M1|p1Cl|' :  self.get_arr_M1p1Cl,
            'M1|m1H|pC4H11N' :  self.get_arr_M1m1HpC4H11N,
            'M1|m1H|pHCOOH' :  self.get_arr_M1m1HpHCOOH,
            'M1|m2Hp1Na|pHCOOH' :  self.get_arr_M1m2Hp1NapHCOOH,
            'M1|m2Hp1Na|' :  self.get_arr_M1m2Hp1Na,
            'M1|m2Hp1K|' :  self.get_arr_M1m2Hp1K,
            'M2|m1H|pC4H11N' :  self.get_arr_M2m1HpC4H11N,
            'M2|m1H|pHCOOH' :  self.get_arr_M2m1HpHCOOH,
            'M2|m1H|' :  self.get_arr_M2m1H,
            'M2|p1Cl|' :  self.get_arr_M2p1Cl,
            'M2|m2Hp1Na|pHCOOH' :  self.get_arr_M2m2Hp1NapHCOOH,
            'M2|m2Hp1Na|' :  self.get_arr_M2m2Hp1Na,
            'M2|m2Hp1K|' :  self.get_arr_M2m2Hp1K,
            'M3|m1H|' :  self.get_arr_M3m1H,
            'M3|p1Cl|' :  self.get_arr_M3p1Cl,
            'M3|m2Hp1Na|pHCOOH' :  self.get_arr_M3m2Hp1NapHCOOH,
            'M3|m2Hp1Na|' :  self.get_arr_M3m2Hp1Na,
            'M4|m1H|' :  self.get_arr_M4m1H,
            'M4|p1Cl|' :  self.get_arr_M4p1Cl,
            'M4|m2Hp1Na|pHCOOH' :  self.get_arr_M4m2Hp1NapHCOOH,
            'M4|m2Hp1Na|' :  self.get_arr_M4m2Hp1Na
            }
    
    def validate(self, adduct_code, neutral, mass_array):
        arr_function = self.selector[adduct_code]
        values = arr_function(neutral)
        diff = np.abs(mass_array.reshape(-1, 1) - values)
        counter = np.sum(diff <= self.mass_error)
        return counter
        
    
    #-------------------------------------------------------- Negative m/z ----

    def get_mz_M1m1H(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = -1.007825,
                                  mol_count = 1,
                                  ion_charge = -1)
        return mz
    
    def get_mz_M1p1Cl(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 34.968853,
                                  mol_count = 1,
                                  ion_charge = -1)
        return mz
    
    def get_mz_M1m2Hp1Na(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 20.97412,
                                  mol_count = 1,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M1m1HpHCOOH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 44.997655,
                                  mol_count = 1,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M1m2Hp1NapHCOOH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 66.9796,
                                  mol_count = 1,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M1m2Hp1K(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 36.948057999999996,
                                  mol_count = 1,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M2m1H(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = -1.007825,
                                  mol_count = 2,
                                  ion_charge = -1)
        return mz
    
    def get_mz_M2p1Cl(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 34.968853,
                                  mol_count = 2,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M2m2Hp1Na(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 20.97412,
                                  mol_count = 2,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M2m2Hp1NapHCOOH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 66.9796,
                                  mol_count = 2,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M2m2Hp1K(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 36.948057999999996,
                                  mol_count = 2,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M3m1H(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = -1.007825,
                                  mol_count = 3,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M3p1Cl(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 34.968853,
                                  mol_count = 3,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M3m2Hp1Na(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 20.97412,
                                  mol_count = 3,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M3m2Hp1NapHCOOH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 66.9796,
                                  mol_count = 3,
                                  ion_charge = -1)
        return mz
        
    def get_mz_M3m2Hp1K(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 36.948057999999996,
                                  mol_count = 3,
                                  ion_charge = -1)
        return mz
    
    
    
    #-------------------------------------------------------- Positive m/z ----
    
    def get_mz_M1p1H(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 1.007825,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M1p1HpCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 42.034374,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M1p1HpCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 33.034040,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1K(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 38.963708,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1KpCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 79.990257,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1KpCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 70.989923,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1Na(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 22.989770,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1NapCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 64.016319,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1NapCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 55.015985,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1NH4(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 18.034374,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1NH4pCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 59.060923,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M1p1NH4pCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 50.060589,
                                  mol_count = 1,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M2p1H(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 1.007825,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M2p1HpCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 42.034374,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M2p1HpCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 33.034040,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M2p1K(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 38.963708,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M2p1KpCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 79.990257,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M2p1KpCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 70.989923,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M2p1Na(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 22.989770,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M2p1NapCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 64.016319,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M2p1NapCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 55.015985,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M2p1NH4(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 18.034374,
                                  mol_count = 2,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M3p1H(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 1.007825,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
        
    def get_mz_M3p1HpCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 42.034374,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    def get_mz_M3p1HpCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 33.034040,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M3p1K(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 38.963708,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M3p1KpCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 79.990257,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M3p1KpCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 70.989923,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M3p1Na(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                          adduct_mass = 22.989770,
                          mol_count = 3,
                          ion_charge = 1)
        return mz
    
    def get_mz_M3p1NapCH3CN(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 64.016319,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M3p1NapCH3OH(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 55.015985,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    def get_mz_M3p1NH4(self, neutral):
        mz = ion_mass_calculator(mol_mass = neutral,
                                  adduct_mass = 18.034374,
                                  mol_count = 3,
                                  ion_charge = 1)
        return mz
    
    
    #----------------------------------------------------- Positive arrays ----
    
    def get_arr_M1p1HpCH3CN(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1HpCH3OH(neutral)
            ])
        return values
    
    def get_arr_M1p1HpCH3OH(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1Na(neutral)
            ])
        return values
    
    def get_arr_M1p1HpHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral)
            ])
        return values
    
    def get_arr_M1p1H(self, neutral):
        values = np.array([])
        return values
    
    def get_arr_M1p1K(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral)
            ])
        return values
        
    def get_arr_M1p1KpCH3OH(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral)
            ])
        return values
        
    def get_arr_M1p1Na(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral)
            ])
        return values
    
    def get_arr_M1p1NapCH3CN(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1NapCH3OH(neutral)
            ])
        return values
        
    def get_arr_M1p1NapCH3OH(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral)
            ])
        return values
    
    def get_arr_M1p1NH4(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral)
            ])
        return values
        
    def get_arr_M1p1NH4pCH3CN(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral)
            ])
        return values
    
    def get_arr_M2p1H(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral)
            ])
        return values
        
    def get_arr_M2p1HpCH3CN(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral)
            ])
        return values
    
    def get_arr_M2p1HpCH3OH(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral)
            ])
        return values
    
    def get_arr_M2p1HpHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral)
            ])
        return values
        
    def get_arr_M2p1K(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral)
            ])
        return values
    
    def get_arr_M2p1Na(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1Na(neutral),
    
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral)
            ])
        return values
    
    def get_arr_M2p1NapCH3CN(self, neutral) : 
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1Na(neutral),
    
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral)
            ])
        return values
    
    def get_arr_M2p1NapCH3OH(self, neutral) : 
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1Na(neutral),
    
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral)
            ])
        return values
    
    def get_arr_M2p1NH4(self, neutral): 
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral)
            ])
        return values
        
    def get_arr_M3p1H(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral)
            
            ])
        return values
    
    def get_arr_M3p1K(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral)
            
            ])
        return values
        
    def get_arr_M3p1Na(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral)
            
            ])
        return values
    
    def get_arr_M3p1NH4(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral)
            
            ])
        return values
        
    def get_arr_M4p1H(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral),
            
            self.get_mz_M3p1H(neutral),
            self.get_mz_M3p1NH4(neutral),
            self.get_mz_M3p1Na(neutral),
            self.get_mz_M3p1K(neutral),
            
            self.get_mz_M3p1HpCH3OH(neutral),
            self.get_mz_M3p1HpCH3CN(neutral),
            self.get_mz_M3p1NapCH3OH(neutral),
            self.get_mz_M3p1NapCH3CN(neutral),
            self.get_mz_M3p1KpCH3OH(neutral),
            self.get_mz_M3p1KpCH3CN(neutral)
            
            ])
        return values
        
    def get_arr_M4p1K(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral),
            
            self.get_mz_M3p1H(neutral),
            self.get_mz_M3p1NH4(neutral),
            self.get_mz_M3p1Na(neutral),
            self.get_mz_M3p1K(neutral),
            
            self.get_mz_M3p1HpCH3OH(neutral),
            self.get_mz_M3p1HpCH3CN(neutral),
            self.get_mz_M3p1NapCH3OH(neutral),
            self.get_mz_M3p1NapCH3CN(neutral),
            self.get_mz_M3p1KpCH3OH(neutral),
            self.get_mz_M3p1KpCH3CN(neutral)
            
            ])
        return values
    
    def get_arr_M4p1Na(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral),
            
            self.get_mz_M3p1H(neutral),
            self.get_mz_M3p1NH4(neutral),
            self.get_mz_M3p1Na(neutral),
            self.get_mz_M3p1K(neutral),
            
            self.get_mz_M3p1HpCH3OH(neutral),
            self.get_mz_M3p1HpCH3CN(neutral),
            self.get_mz_M3p1NapCH3OH(neutral),
            self.get_mz_M3p1NapCH3CN(neutral),
            self.get_mz_M3p1KpCH3OH(neutral),
            self.get_mz_M3p1KpCH3CN(neutral)
            
            ])
        return values
    
    def get_arr_M4p1NH4(self, neutral):
        values = np.array([
            self.get_mz_M1p1H(neutral),
            self.get_mz_M1p1NH4(neutral),
            self.get_mz_M1p1Na(neutral),
            self.get_mz_M1p1K(neutral),
            
            self.get_mz_M1p1HpCH3OH(neutral),
            self.get_mz_M1p1HpCH3CN(neutral),
            self.get_mz_M1p1NapCH3OH(neutral),
            self.get_mz_M1p1NapCH3CN(neutral),
            self.get_mz_M1p1KpCH3OH(neutral),
            self.get_mz_M1p1KpCH3CN(neutral),
            
            self.get_mz_M2p1H(neutral),
            self.get_mz_M2p1NH4(neutral),
            self.get_mz_M2p1Na(neutral),
            self.get_mz_M2p1K(neutral),
            
            self.get_mz_M2p1HpCH3OH(neutral),
            self.get_mz_M2p1HpCH3CN(neutral),
            self.get_mz_M2p1NapCH3OH(neutral),
            self.get_mz_M2p1NapCH3CN(neutral),
            self.get_mz_M2p1KpCH3OH(neutral),
            self.get_mz_M2p1KpCH3CN(neutral),
            
            self.get_mz_M3p1H(neutral),
            self.get_mz_M3p1NH4(neutral),
            self.get_mz_M3p1Na(neutral),
            self.get_mz_M3p1K(neutral),
            
            self.get_mz_M3p1HpCH3OH(neutral),
            self.get_mz_M3p1HpCH3CN(neutral),
            self.get_mz_M3p1NapCH3OH(neutral),
            self.get_mz_M3p1NapCH3CN(neutral),
            self.get_mz_M3p1KpCH3OH(neutral),
            self.get_mz_M3p1KpCH3CN(neutral)
            
            ])
        return values

    
    #----------------------------------------------------- Negative arrays ----
    
    def get_arr_M1m1HpC4H11N(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral)
            ])
        
        return values
        
    def get_arr_M1m1HpHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral)
            ])
        return values

    def get_arr_M1m1H(self, neutral):
        values = np.array([])
        return values

    def get_arr_M1p1Cl(self, neutral):
        values = np.array([])
        
        return values

    def get_arr_M1m2Hp1NapHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral)
            ])

        return values

    def get_arr_M1m2Hp1Na(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral)
            ])
        
        return values

    def get_arr_M1m2Hp1K(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral)
            ])
        return values

    def get_arr_M2m1HpC4H11N(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral)
            ])
        
        return values
    
    def get_arr_M2m1HpHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m1HpHCOOH(neutral)
            ])
        
        return values

    def get_arr_M2m1H(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral)
            ])
        
        return values

    def get_arr_M2p1Cl(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral)
            ])
        
        return values

    def get_arr_M2m2Hp1NapHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral)
            ])
        
        return values

    def get_arr_M2m2Hp1Na(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral)
            ])
        
        return values

    def get_arr_M2m2Hp1K(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M3m1H(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M3p1Cl(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M3m2Hp1NapHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M3m2Hp1Na(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M4m1H(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral),
            
            self.get_mz_M3m1H(neutral),
            self.get_mz_M3p1Cl(neutral),
            self.get_mz_M3m2Hp1Na(neutral),
            self.get_mz_M3m2Hp1NapHCOOH(neutral),
            self.get_mz_M3m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M4p1Cl(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral),
            
            self.get_mz_M3m1H(neutral),
            self.get_mz_M3p1Cl(neutral),
            self.get_mz_M3m2Hp1Na(neutral),
            self.get_mz_M3m2Hp1NapHCOOH(neutral),
            self.get_mz_M3m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M4m2Hp1NapHCOOH(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral),
            
            self.get_mz_M3m1H(neutral),
            self.get_mz_M3p1Cl(neutral),
            self.get_mz_M3m2Hp1Na(neutral),
            self.get_mz_M3m2Hp1NapHCOOH(neutral),
            self.get_mz_M3m2Hp1K(neutral)
            ])
        
        return values

    def get_arr_M4m2Hp1Na(self, neutral):
        values = np.array([
            self.get_mz_M1m1H(neutral),
            self.get_mz_M1p1Cl(neutral),
            self.get_mz_M1m2Hp1Na(neutral),
            self.get_mz_M1m2Hp1NapHCOOH(neutral),
            self.get_mz_M1m2Hp1K(neutral),
            
            self.get_mz_M2m1H(neutral),
            self.get_mz_M2p1Cl(neutral),
            self.get_mz_M2m2Hp1Na(neutral),
            self.get_mz_M2m2Hp1NapHCOOH(neutral),
            self.get_mz_M2m2Hp1K(neutral),
            
            self.get_mz_M3m1H(neutral),
            self.get_mz_M3p1Cl(neutral),
            self.get_mz_M3m2Hp1Na(neutral),
            self.get_mz_M3m2Hp1NapHCOOH(neutral),
            self.get_mz_M3m2Hp1K(neutral)
            ])
        
        return values
import pandas as pd
import numpy as np
from tqdm import tqdm
import operator
import re

def sample_slicer_export(sample : str, csv_table, spectra, out_path : str):
    """
    Writes sub-mgf file specific to one sample.

    Parameters
    ----------
    sample : str
        Sample name in the table.
    csv_table : pandas.DataFrame
        Main table containing signal per sample.
    spectra : Spectra
        MolNotator Spectra class.
    out_path : str
        Output directory.

    Returns
    -------
    None.

    """
    spec_id = csv_table["spec_id"][csv_table[sample] > 0].tolist()
    new_sectra = slice_spectra(spectra, spec_id)
    new_sectra.write_mgf(f'{out_path}{sample}')




#---------------------------------------------- Spectrum & Spectra classes ----

class Spectrum:
    def __init__(self):
        self.mz = None
        self.intensity = None
        self.metadata = {}
        self.prec_mz = None
        self.rt = None
        self.charge = None
        self.tic = None
    
    def add_mz(self, mz_list : list):
        self.mz = np.array(mz_list)
        
    def add_intensities(self, int_list : list):
        self.tic = sum(int_list)
        self.intensity = np.array(int_list)
    
    def to_tuple(self):
        return tuple([self.prec_mz, self.rt, self.charge, self.tic] + list(self.metadata.values()))
    
    def to_mgf(self):
        mgf_string = "BEGIN IONS\n"
        mgf_string += "PEPMASS={}\n".format(self.prec_mz)
        mgf_string += "RTINSECONDS={}\n".format(self.rt)
        mgf_string += "CHARGE={}\n".format(self.charge)
        for key, value in self.metadata.items():
            mgf_string += "{}={}\n".format(key, value)
        for mz, intensity in zip(self.mz, self.intensity):
            mgf_string += "{} {}\n".format(mz, intensity)
        mgf_string += "END IONS\n\n"
        return mgf_string
    def correct_charge(self, op):
        self.charge = op(int(re.sub("[^0-9]", "", self.charge))) 

        

class Spectra:
    def __init__(self):
        self.spectrum = []
        self.data_frame = None
    
    def to_data_frame(self):
        data_frame = list()
        meta_keys = list(self.spectrum[0].metadata.keys())
        meta_keys = [s.lower() for s in meta_keys]
        columns = ["prec_mz", "rt", "charge", "TIC"] + meta_keys
        for s in self.spectrum:
            data_frame.append(s.to_tuple())
        return pd.DataFrame(data_frame, columns = columns)
    
    def to_mgf(self):
        mgf_string = ""
        for s in self.spectrum:
            mgf_string += s.to_mgf()
        return mgf_string
    
    def write_mgf(self, output_file_path):
        mgf_string = self.to_mgf()
        with open(output_file_path, 'w') as file:
            file.write(mgf_string)
    
    def correct_charge(self, ion_mode):
        if ion_mode == "NEG":
            op = operator.neg
        elif ion_mode == "POS":
            op = operator.pos
        for s in self.spectrum:
            s.correct_charge(op)
            

#--------------------------------------------------------------- MGF files ----
def read_mgf_file(file_path : str, mz_field : str = "pepmass", rt_field : str = "rtinseconds", charge_field : str = "charge", ion_mode : str = None):
    """
    Reads an MGF file and exports the spectra as a list

    Parameters
    ----------
    file_path : str
        MGF file name.

    Returns
    -------
    spectra : list
        List of spectra from the MGF file.

    """
    spectra = Spectra()
    current_spectrum = None
    mz_list = []
    int_list = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith('BEGIN IONS'):
                current_spectrum = Spectrum()
            
            elif line.startswith('END IONS'):
                if current_spectrum:
                    current_spectrum.add_mz(mz_list)
                    current_spectrum.add_intensities(int_list)
                    spectra.spectrum.append(current_spectrum)
                    current_spectrum = None
                    mz_list = []
                    int_list = []
            
            elif current_spectrum: 
                if mz_field in line.lower():
                    current_spectrum.prec_mz = float(line.split('=')[1])
                elif rt_field in line.lower():
                    current_spectrum.rt = float(line.split('=')[1])
                elif charge_field in line.lower():
                    current_spectrum.charge = line.split('=')[1]
                elif '=' in line:
                    key, value = line.split('=')
                    current_spectrum.metadata[key.strip()] = value.strip()
                elif line != "":
                    values = line.split()
                    mz_list.append(float(values[0]))
                    int_list.append(float(values[1]))
    
    if (ion_mode):
        spectra.correct_charge(ion_mode)
    
    return spectra

#------------------------------------------------------------ Cosine score ----

def collect_peak_pairs(spec1, spec2, tolerance):
    matches = []
    for i, mz1 in enumerate(spec1[0]):
        for j, mz2 in enumerate(spec2[0]):
            if abs(mz1 - mz2) <= tolerance:
                matches.append((i, j))
    if len(matches) == 0:
        return None
    matching_pairs = []
    for i, j in matches:
        int_spec1 = spec1[1][i]
        int_spec2 = spec2[1][j]
        matching_pairs.append((i, j, int_spec1 * int_spec2))
    return matching_pairs

def score_best_matches(matching_pairs, spec1, spec2, mz_power=0.0, intensity_power=1.0):
    score = 0.0
    used_matches = 0
    used1 = set()
    used2 = set()
    for i in range(len(matching_pairs)):
        if matching_pairs[i][0] not in used1 and matching_pairs[i][1] not in used2:
            score += matching_pairs[i][2]
            used1.add(matching_pairs[i][0])
            used2.add(matching_pairs[i][1])
            used_matches += 1

    score /= (np.sum(spec1[1] ** 2) ** 0.5 * np.sum(spec2[1] ** 2) ** 0.5)
    return score, used_matches

def calculate_cosine_score(spectrum1, spectrum2, tolerance):
    matching_pairs = collect_peak_pairs(spectrum1, spectrum2, tolerance)
    if matching_pairs is None or len(matching_pairs) == 0:
        return 0.0, 0
    score = score_best_matches(matching_pairs, spectrum1, spectrum2)
    return score

def spectrum_cosine_score(spectrum1, spectrum2, tolerance):
    spectrum1 = (spectrum1.mz, spectrum1.intensity)
    spectrum2 = (spectrum2.mz, spectrum2.intensity)
    return calculate_cosine_score(spectrum1, spectrum2, tolerance)

#-------------------------------------------------------------------- Misc ----

def slice_spectra(old_spectra, spec_id_list):
    new_spectra = Spectra()
    for i in spec_id_list:
        new_spectra.spectrum.append(old_spectra.spectrum[i])
    return new_spectra


#---------------------------------------------------- Data frame functions ----

"""reindexer.py - reindexer function for MolNotator"""
def reindexer(node_table, params):
    """
    Reindexes tables using the supplied "index_col" argument in the params dict
    and adds a "spec_id" column for the relative position of ions within the
    spectrum file.

    Parameters
    ----------
    node_table : pandas.DataFrame
        Dataframe containing the metadata for each ions.
    params : dict
        Dictionary containing the global parameters for the process.

    Returns
    -------
    node_table : pandas.DataFrame
        Same dataframe but reindexed.

    """
    node_table[params['index_col']] = node_table[params['index_col']].astype(int)
    node_table.set_index(params['index_col'], inplace = True)
    node_table.insert(0, 'spec_id', range(len(node_table)))
    return node_table

# remap node table columns using parameters
def remapper(data_table, params):
    data_table.rename(mapper = {"prec_mz" : params['mz_field'],
                                "rt" : params['rt_field']},
                      axis = 1,
                      inplace = True)
    return data_table


"""rt_slicer.py - rt_slicer function used by MolNotator"""
def rt_slicer(rt : float, rt_error : float, ion_id, input_table, rt_field : str) :
    """
    Returns a slice of a node table, given an RT, an RT error and a selected
    ion around which coeluted ions are to be search

    Parameters
    ----------
    rt : float
        Retention type of the selected ion.
    rt_error : float
        Retention time window to be used around RT.
    ion_id
        ID of the selected ion.
    input_table : TYPE
        Node table containing the retention time values.
    rt_field : str
        field containing the RT values.

    Returns
    -------
    slice_table : pandas.DataFrame
        Dataframe of ions coeluted with the selected ion.

    """
    rt_low = float(rt) - rt_error
    rt_high = float(rt) + rt_error
    sliced_table = input_table[input_table[rt_field].astype(float).between(rt_low,
                              rt_high, inclusive = "both")].copy()
    return sliced_table.drop(ion_id)

"""singleton_edges.py - singleton_edges function for the fragnotator module"""
def singleton_edges(node_table, edge_table):
    """
    Adds singleton edges to an edge table already containing precursor-fragment
    ion pairs.

    Parameters
    ----------
    node_table : pandas.DataFrame
        Dataframe containing all ions, provided by the fragnotator module
    edge_table : pandas.DataFrame
        Dataframe containing all already paired ions.

    Returns
    -------
    edge_table : pandas.DataFrame
        Dataframe/Edge table update with the singleton edges.

    """

    paired_nodes = set(list(edge_table['node_1']) + list(edge_table['node_2']))
    singleton_nodes = list(set(node_table.index) - paired_nodes)
    singleton_edge_table = pd.DataFrame(0.0, index = range(len(singleton_nodes)), 
                                        columns = edge_table.columns)
    singleton_edge_table['node_1'] = singleton_nodes
    singleton_edge_table['node_2'] = singleton_nodes
    edge_table['status'] = ["frag_edge"]*len(edge_table)
    singleton_edge_table['status'] = ["singleton"]*len(singleton_edge_table)
    edge_table = pd.concat([edge_table, singleton_edge_table], ignore_index=True)
    edge_table.reset_index(drop = True, inplace = True)
    return edge_table

"""fragnotator_edge_table.py - fragnotator_edge_table module for fragnotator"""
def fragnotator_edge_table(node_table, spectra, params):
    """
    Finds precursor-fragment ion pairs for in-source fragmentation, using the
    metadata in a node table and the spectra in the spectrum file.
    Parameters
    ----------
    node_table : pandas.DataFrame
        Dataframe containing metadata from the fragnotator module.
    spectra : list
        List of matchms.Spectrum objects from the fragnotator module.
    params : dict
        Dictionary containing the global parameters for the process.
    Returns
    -------
    edge_table : pandas.DataFrame
        Dataframe containing the precursor-fragment ion pairs.
    """

    # Get parameters
    score_threshold = params['fn_score_threshold']
    min_shared_peaks = params['fn_matched_peaks']
    rt_error = params['fn_rt_error']
    mass_error = params['fn_mass_error']
    rt_field = params['rt_field']
    mz_field = params['mz_field']
    

    # If the retention time is in minutes
    if params['rt_unit'] == "m":
        rt_error = rt_error/60

    # For each ion, search fragment candidates
    edge_table = list()
    for i in tqdm(node_table.index) :
        
        # Get ion 1 data (precursor)
        ion1_spec_id = node_table.loc[i, "spec_id"]
        ion1_rt = node_table.loc[i, rt_field]
        ion1_mz = node_table.loc[i, mz_field]
        ion1_msms = pd.Series(spectra.spectrum[ion1_spec_id].mz)
        
        # Find fragment candidate ions (below mz, similar RT)
        candidate_table = rt_slicer(ion1_rt, rt_error, i, node_table, rt_field)
        candidate_table = candidate_table[candidate_table[mz_field] < ion1_mz]
        
        # If no candidates are found, skip
        if len(candidate_table.index) == 0 : continue
        
        # Candidates must share their precursor ion with the precursor in MSMS
        for j in candidate_table.index:
            
            # Get ion 2 data (fragment)
            ion2_spec_id = node_table.loc[j, "spec_id"]
            ion2_mz = node_table.loc[j, mz_field]
            ion2_mz_low = ion2_mz - mass_error
            ion2_mz_high = ion2_mz + mass_error
            match = ion1_msms.between(ion2_mz_low, ion2_mz_high, inclusive = "both")
            if match.sum() > 0 : # if the frag candidate m/z is found in MSMS:
                ion2_msms = pd.Series(spectra.spectrum[ion2_spec_id].mz)
                matched_peaks = 0
                total_peaks = list(ion1_msms)
                for frag in ion2_msms : # find the number of matched peaks
                    frag_low = frag - mass_error
                    frag_high = frag + mass_error
                    frag_found = ion1_msms.between(frag_low, frag_high, inclusive = "both").sum()
                    if frag_found > 0 :
                        matched_peaks += 1
                    else :
                        total_peaks.append(frag)
                
                # Check number of matched peaks & matching score to validate frag
                if matched_peaks >= min_shared_peaks : # if number of matches above threshold
                    total_peaks = pd.Series(total_peaks)[total_peaks <= ion2_mz_high]
                    matching_score = round(matched_peaks / len(total_peaks),2)
                    if matching_score >= score_threshold:
                        edge_table.append((i, j, matched_peaks, len(total_peaks),
                                           matching_score,
                                           ion1_rt - node_table[rt_field][j],
                                           ion1_mz - node_table[mz_field][j]))
        
    # Results are stored in the edge table
    edge_table = pd.DataFrame(edge_table, columns = ['node_1', 'node_2',
                                                     'matched_peaks', 'total_peaks',
                                                     'matching_score', 'rt_gap',
                                                     'mz_gap'])

    return edge_table
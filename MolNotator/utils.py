import pandas as pd
import numpy as np
from tqdm import tqdm
import operator
import re
from datetime import datetime
from multiprocessing import Pool

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



def parallel_export_sampslicer(workers, samples, csv_table, spectra, out_path):
    with Pool(workers) as pool:
        pool.starmap(sample_slicer_export, tqdm([(sample, csv_table, spectra, out_path) for sample in samples]))






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
    
    def to_tuple(self, keys):
        tuple_vals = [self.prec_mz,
                      self.rt,
                      self.charge,
                      self.tic]
        for key in keys:
            tuple_vals.append(self.metadata.get(key))
        tuple_vals = tuple(tuple_vals)
        return tuple_vals
    
    def to_dict(self):
        values = {"prec_mz" : self.prec_mz,
                  "rt" : self.rt,
                  "charge" : self.charge,
                  "TIC" : self.tic}
        values.update(self.metadata)
        return values
    
    def to_mgf(self):
        mgf_string = "BEGIN IONS\n"
        mgf_string += "PEPMASS={}\n".format(self.prec_mz)
        mgf_string += "RTINSECONDS={}\n".format(self.rt)
        mgf_string += "CHARGE={}\n".format(self.charge)
        for key, value in self.metadata.items():
            mgf_string += "{}={}\n".format(key.upper(), value)
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
        self.fields = []
    
    def to_data_frame(self):
        data_frame = list()
        keys = self.fields.copy()
        keys.remove('prec_mz')
        keys.remove('rt')
        keys.remove('charge')
        keys.remove('TIC')
        for i in range(len(self.spectrum)):
            data_frame.append(self.spectrum[i].to_tuple(keys = keys))
        data_frame = pd.DataFrame(data_frame, columns = self.fields)
        
        return data_frame
    
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
    field_list = []

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
                    field_list = list(set(field_list))
            
            elif current_spectrum: 
                if mz_field in line.lower():
                    current_spectrum.prec_mz = float(line.split('=')[1])
                elif rt_field in line.lower():
                    current_spectrum.rt = float(line.split('=')[1])
                elif charge_field in line.lower():
                    current_spectrum.charge = line.split('=')[1]
                elif '=' in line:
                    key, value = line.split('=', 1)
                    current_spectrum.metadata[key.strip().lower()] = value.strip()
                    field_list.append(key.strip())
                elif line != "":
                    values = line.split()
                    mz_list.append(float(values[0]))
                    int_list.append(float(values[1]))
                    
    field_list.sort()
    field_list = [s.lower() for s in field_list]
    spectra.fields = ["prec_mz", "rt", "charge", "TIC"] + field_list
    
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



def print_time(txt):
    print(f"""{datetime.now().strftime("%H:%M:%S")} - {txt}""")
    


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

    def precursor_finder(mass_array, values, mass_error):
        kept = list()
        for i, mz in values:
            indices = np.where((mass_array >= mz-mass_error) & (mass_array <= mz+mass_error))
            if indices[0].size > 0 : kept.append(i)
        return kept
    
    def match_arrays(array1, array2, delta):
        # Calculate the absolute difference matrix
        diff = np.abs(array1[:, None] - array2)
        # Find pairs where difference <= delta
        matches = np.where(diff <= delta)
        return len(matches[0])

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
        ion1_spec_id = node_table.at[i, "spec_id"]
        ion1_rt = node_table.at[i, rt_field]
        ion1_mz = node_table.at[i, mz_field]
        ion1_msms = spectra.spectrum[ion1_spec_id].mz
        ion1_prec_mz = spectra.spectrum[ion1_spec_id].prec_mz
        ion1_msms = ion1_msms[np.where(ion1_msms <= ion1_prec_mz+0.5)]
        
        # Find fragment candidate ions (below mz, similar RT)
        candidate_table = rt_slicer(ion1_rt, rt_error, i, node_table, rt_field)
        candidate_table = candidate_table[candidate_table[mz_field] < ion1_mz]
    
        idx = precursor_finder(mass_array = ion1_msms,
                               values = list(zip(candidate_table.index.tolist(), node_table.loc[candidate_table.index, mz_field].tolist())),
                               mass_error = mass_error)
        
        candidate_table = candidate_table.loc[idx]
        
        # If no candidates are found, skip
        if len(candidate_table.index) == 0 : continue
        
        # Candidates must share their precursor ion with the precursor in MSMS
        
        for j in candidate_table.index:
        
            # Get ion 2 data (fragment)
            ion2_spec_id = node_table.at[j, "spec_id"]
            ion2_rt = node_table.at[j, rt_field]
            ion2_mz = node_table.at[j, mz_field]
            ion2_msms = spectra.spectrum[ion2_spec_id].mz
            ion2_prec_mz = spectra.spectrum[ion2_spec_id].prec_mz
            ion2_msms = ion2_msms[np.where(ion2_msms <= ion2_prec_mz+0.5)]
            
            matched_peaks = match_arrays(array1 = ion1_msms,
                                         array2 = ion2_msms,
                                         delta = mass_error)
            
            total_peaks = len(ion1_msms) + len(ion2_msms) - matched_peaks
            matching_score = round(matched_peaks / total_peaks,2)
            
            edge_table.append((i, j, matched_peaks, total_peaks, matching_score,
                               ion1_rt - ion2_rt,
                               ion1_mz - ion2_mz))
            

        
    # Results are stored in the edge table
    edge_table = pd.DataFrame(edge_table, columns = ['node_1', 'node_2',
                                                     'matched_peaks', 'total_peaks',
                                                     'matching_score', 'rt_gap',
                                                     'mz_gap'])
    edge_table = edge_table[edge_table['matched_peaks'] >= min_shared_peaks]
    edge_table = edge_table[edge_table['matching_score'] >= score_threshold]

    return edge_table


#----------------------------------------------------- Spectrum operations ----

def cosine_validation(ion_1_idx : int, node_table, spectrum_list : list,
                      ion_hypotheses_table, adduct_table_primary, params : dict):
    """
    For each ionisation hypothesis (ion 1 and 2 pairs), reports the
    cosine score from the cosine_table. In case of multiple hits in the
    "hit_indexes" columns, the one with the best cosine score is kept (chosen
    at random if both have the same score : either both completely unrelated
    or the same spectrum). The duplicate hits are transferred to a duplicate_df
    for annotation in the final network, they will take no part in calculation.
    Points are then awared according to the cosine score:
    1+cosine score if the score is above threshold, 0 if it is below.

    Parameters
    ----------
    ion_1_idx : int
        Current ion ID / index.
    node_table : pandas.DataFrame
        Node table with the ion metadata.
    spectrum_list : list
        List of matchms.Spectrum objects.
    ion_hypotheses_table : pandas.DataFrame
        Ion hypotheses table obtained from other functions.
    adduct_table_primary : pandas.DataFrame
        Primary adduct table from the parameter folder.
    params : dict
        Dictionary containing the global parameters for the process.

    Returns
    -------
    ion_hypotheses_table : pandas.DataFrame
        Ion hypotheses table with points awarded.
    duplicate_df : pandas.DataFrame
        Duplicates found in the ion hypotheses table.

    """
    
    # Load parameters
    cosine_threshold = params['an_cos_threshold']
    spec_id_1 = node_table.loc[ion_1_idx, 'spec_id']
    
    # Initialise variables
    score_table = list()

    # duplicate_df will contain adducts with roles considered redundant
    duplicate_df = pd.DataFrame(columns = ["ion_1_idx", "ion_1_adduct",
                                           "ion_2_idx", "ion_2_adduct",
                                           "selected_ion", "cosine_score",
                                           "matched_peaks", "different_group"])
    for i in ion_hypotheses_table.index:
        
        # Check adduct groups for ions 1 and 2
        group_1 = adduct_table_primary.loc[ion_hypotheses_table.loc[i, "Ion1_adduct"], "Group"]
        group_2 = adduct_table_primary.loc[ion_hypotheses_table.loc[i, "Ion2_adduct"], "Group"]
        if group_1 == group_2: different_group = False
        else: different_group = True
        
        # If only one ion 2, calculate scores
        if ion_hypotheses_table.loc[i, "hit_count"] == 1 :
            ion_2_idx = ion_hypotheses_table.loc[i,"hit_indexes"][0]
            spec_id_2 = node_table.loc[ion_2_idx, "spec_id"]
            
            score, n_matches = spectrum_cosine_score(spectrum1 = spectrum_list.spectrum[spec_id_1],
                                                     spectrum2 = spectrum_list.spectrum[spec_id_2],
                                                     tolerance = params['an_mass_error'])
            
            prod = score * n_matches
        # If several ion 2, select the best one based on scores
        else:
            selected_hit = pd.DataFrame(columns = ['cos', 'peaks', 'prod'])
            for hit in ion_hypotheses_table.loc[i,"hit_indexes"]:
                spec_id_2 = node_table.loc[hit, "spec_id"]
                
                score, n_matches = spectrum_cosine_score(spectrum1 = spectrum_list.spectrum[spec_id_1],
                                                         spectrum2 = spectrum_list.spectrum[spec_id_2],
                                                         tolerance = params['an_mass_error'])
                selected_hit.loc[hit] = [score, n_matches, score * n_matches]

            selected_hit.sort_values('prod', ascending = False, inplace =True)
            new_hit = selected_hit.index[0]
            score = selected_hit['cos'].iloc[0]
            n_matches = int(selected_hit['peaks'].iloc[0])
            prod = selected_hit['prod'].iloc[0]
            selected_hit.drop(new_hit, inplace = True)
            ion_hypotheses_table.loc[i, "hit_indexes"] = [[new_hit]]
            for j in selected_hit.index:
                tmp_idx = len(duplicate_df)
                duplicate_df.loc[tmp_idx] = [ion_1_idx,
                                    ion_hypotheses_table.loc[i, "Ion1_adduct"],
                                 j, ion_hypotheses_table.loc[i, "Ion2_adduct"],
                                 new_hit, selected_hit.loc[j, 'cos'], selected_hit.loc[j, 'peaks'],
                                 different_group]
        # If score is below threshold, nullify all.
        if score < cosine_threshold: 
            score = 0.0
            prod = 0.0
        score_table.append((different_group, score, n_matches, prod))
    
    # Convert scores to dataframe and merge with ion_hypothesis thable
    score_table = pd.DataFrame(score_table, columns = ['different_group',
                                                       'cosine_score',
                                                       'matched_peaks',
                                                       'product'])
    ion_hypotheses_table = pd.concat([ion_hypotheses_table, score_table], axis = 1)
    
    # Convert the hit_indexes col (which are now guaranteed to be only one), to int
    ion_hypotheses_table['hit_indexes'] = ion_hypotheses_table['hit_indexes'].str[0]
    
    # Filter by score and same group criterium
    tmp_bool = ion_hypotheses_table["different_group"] + ion_hypotheses_table["cosine_score"] >= cosine_threshold
    duplicate_bool = duplicate_df["different_group"] + duplicate_df["cosine_score"] >= cosine_threshold
    return ion_hypotheses_table[tmp_bool], duplicate_df[duplicate_bool]


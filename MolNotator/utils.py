import pandas as pd
import numpy as np
import sys
from pandas.core.common import flatten
from matchms.exporting import save_as_mgf




def sample_slicer_export(sample : str, csv_table, mgf_file : list, out_path : str):
    """
    Writes sub-mgf file specific to one sample.

    Parameters
    ----------
    sample : str
        Sample name in the table.
    csv_table : pandas.DataFrame
        Main table containing signal per sample.
    mgf_file : list
        List of spectra..
    out_path : str
        Output directory.

    Returns
    -------
    None.

    """
    mgf_idx = csv_table["spec_id"][csv_table[sample] > 0].tolist()
    new_mgf = [mgf_file[i] for i in mgf_idx]
    save_as_mgf(new_mgf, f'{out_path}{sample}')




#---------------------------------------------- Spectrum & Spectra classes ----

class Spectrum:
    def __init__(self):
        self.mz = None
        self.intensity = None
        self.metadata = {}
        self.prec_mz = None
        self.rt = None
        self.tic = None
    
    def add_mz(self, mz_list : list):
        self.mz = np.array(mz_list)
        
    def add_intensities(self, int_list : list):
        self.tic = sum(int_list)
        self.intensity = np.array(int_list)
    
    def to_tuple(self):
        return tuple([self.prec_mz, self.rt, self.tic] + list(self.metadata.values()))
    
    def to_mgf(self):
        mgf_string = "BEGIN IONS\n"
        mgf_string += "PEPMASS={}\n".format(self.prec_mz)
        mgf_string += "RTINSECONDS={}\n".format(self.rt)
        for key, value in self.metadata.items():
            mgf_string += "{}={}\n".format(key, value)
        for mz, intensity in zip(self.mz, self.intensity):
            mgf_string += "{} {}\n".format(mz, intensity)
        mgf_string += "END IONS\n\n"
        return mgf_string

class Spectra:
    def __init__(self):
        self.spectrum = []
        self.data_frame = None
    
    def to_data_frame(self):
        data_frame = list()
        meta_keys = list(self.spectrum[0].metadata.keys())
        meta_keys = [s.lower() for s in meta_keys]
        columns = ["prec_mz", "rt", "TIC"] + meta_keys
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
            



#--------------------------------------------------------------- MGF files ----
def read_mgf_file(file_path : str, mz_field : str = "pepmass", rt_field : str = "rtinseconds"):
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
                elif '=' in line:
                    key, value = line.split('=')
                    current_spectrum.metadata[key.strip()] = value.strip()
                elif line != "":
                    values = line.split()
                    mz_list.append(float(values[0]))
                    int_list.append(float(values[1]))
    
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

# Example usage
# spectrum1 = (np.array([90.0010, 100, 150, 200]), np.array([0.7, 0.2, 0.1, 0.5]))
# spectrum2 = (np.array([104.9, 100.0001, 140, 190, 90.0025]), np.array([0.4, 0.8, 0.2, 0.1, 0.5]))
# score = calculate_cosine_score(spectrum1, spectrum2, tolerance=0.2)
# print("Cosine score is {:.3f} with {} matched peaks".format(score[0], score[1]))




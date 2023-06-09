"""sample_slicer.py - sample_slicer module for MolNotator"""
import os
from tqdm import tqdm
import pandas as pd
from MolNotator.utils import sample_slicer_export
from MolNotator.utils import read_mgf_file

def sample_slicer(params : dict, ion_mode : str):
    """Splits the original spectrum file into several files, one for each sample.
    No spectrum from the spectrum file must be empty
    The CSV file is expected to have only one column per sample, for example,
    either peak area or peak height, but not both.
    When processing positive and negative mode data, it is important to
    differenciate both modes for each sample ("POS" and "NEG"), and to have
    otherwise the same exact same for each sample, i.e. "POS_sample_1" and 
    "NEG_sample_1".
    """

    # Load parameters
    if ion_mode == "NEG":
        csv_file= params['neg_csv']
        mgf_file= params['neg_mgf']
        in_path= params['neg_out_0']
        out_path= params['neg_out_1']
    elif ion_mode == "POS" :
        csv_file= params['pos_csv']
        mgf_file= params['pos_mgf']
        in_path= params['pos_out_0']
        out_path= params['pos_out_1']
    else:
        print('Ion mode must be either "NEG" or "POS"')
        return

    

    # Create out folder
    if not os.path.exists(f'{out_path}') :
        os.mkdir(f'{out_path}')
    
    # read the csv metadata file from mzmine
    csv_table = pd.read_csv(f'{in_path}{csv_file}', index_col = params['index_col'])
    
    # Get the sample list
    samples = pd.Series(csv_table.columns)
    samples = samples[samples.str.contains(params['col_suffix'])]
    samples = list(samples.str.replace(params['col_suffix'], '.mgf', regex = False))
    csv_table.columns = csv_table.columns.str.replace(params['col_suffix'], '.mgf', regex = False)
    
    # MZmine mgf file
    spectra = read_mgf_file(f'{in_path}{mgf_file}')

    # Export data
    for i in tqdm(range(len(samples))):
        sample_slicer_export(samples[i], csv_table, spectra, out_path)



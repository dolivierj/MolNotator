"""sample_slicer.py - sample_slicer module for MolNotator"""
import os
from tqdm import tqdm
import pandas as pd
from MolNotator.utils import read_mgf_file, print_time, sample_slicer_export, parallel_export_sampslicer

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
    print(f"------- SAMPLE SLICER : {ion_mode} -------")

    # Load parameters
    workers = params["workers"]
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
        print_time('Error: ion mode must be either "NEG" or "POS"')
        return

    

    # Create out folder
    if not os.path.exists(f'{out_path}') :
        os.mkdir(f'{out_path}')
    
    # read the csv metadata file from mzmine
    csv_table = pd.read_csv(f'{in_path}{csv_file}', index_col = params['index_col'])
    
    # Get the sample list
    samples = pd.Series(csv_table.columns)
    samples = samples[samples.str.contains(params['sample_pattern'])]
    samples = list(samples.str.replace(params['sample_pattern'], '.mgf', regex = False))
    csv_table.columns = csv_table.columns.str.replace(params['sample_pattern'], '.mgf', regex = False)
    
    # MZmine mgf file
    spectra = read_mgf_file(file_path = f'{in_path}{mgf_file}',
                            ion_mode = ion_mode)

    # Export data
    if workers == 1 :
        
        print_time("Exporting MGF files...")
        for i in tqdm(range(len(samples))):
            sample_slicer_export(samples[i], csv_table, spectra, out_path)
    else :
        print_time("Exporting MGF files...")
        parallel_export_sampslicer(workers = workers,
                                   samples = samples,
                                   csv_table = csv_table,
                                   spectra = spectra,
                                   out_path = out_path)
        print_time("Export complete.")
    
    
    



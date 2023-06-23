"""fragnotator.py - fragnotator module for MolNotator"""
import os
import pandas as pd
from MolNotator.utils import fragnotator_multiprocess, print_time

def fragnotator(params : dict, ion_mode : str):
    """
    Takes as input a spectrum file from a single sample to connect in-source 
    fragment ions to their in-source precursor.

    Parameters
    ----------
    params : dict
        Dictionary containing the global parameters for the process..
    ion_mode : str
        Either "POS" or "NEG", ion mode for the data..

    Returns
    -------
    Writes node and edge tables for each sample 

    """

    print(f"------- FRAGNOTATOR : {ion_mode} -------")

    # Load parameters
    frag_table = pd.read_csv("./params/" + params['fn_fragtable'], sep = '\t')

    workers = params["workers"]
    if ion_mode == "NEG" :
        in_path= params['neg_out_1']
        out_path= params['neg_out_2']
    elif ion_mode == "POS":
        in_path= params['pos_out_1']
        out_path= params['pos_out_2']
    else:
        print('Ion mode must be either "NEG" or "POS"')
        return

    # Create out dir
    if not os.path.exists(f'{out_path}') :
        os.mkdir(f'{out_path}')

    # Get file list and start processing for each sample
    files = pd.Series(os.listdir(f'{in_path}'))
    
    print_time("Processing files MGF files...")
    fragnotator_multiprocess(workers = workers,
                             files = files,
                             frag_table = frag_table,
                             in_path = in_path,
                             out_path = out_path,
                             ion_mode = ion_mode,
                             params = params)
    print_time("Processing complete.")
    
    return

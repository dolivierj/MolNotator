"""duplicate_filter.py - duplicate_filter function for MolNotator"""
import os
import pandas as pd
from MolNotator.others.duplicate_finder import duplicate_finder
from MolNotator.utils import read_mgf_file, slice_spectra, remapper, print_time

def duplicate_filter(params : dict, ion_mode : str):
    """
    Finds duplicate ions in a metabolomics experiment by loading the CSV and 
    spectrum files and by comparing the RT, m/z and cosine similarity values
    between ions. Ions with close values are deemed duplicates. Along with the 
    other duplicates, they will be deleted, leaving only a representative ion 
    (often the most intense). This deletion is operated on the CSV and the 
    spectrum file, which will both be exported in the duplicate filter folder.
    Parameters
    ----------
    params : dict
        Dictionary containing the global parameters for the process.
    ion_mode : str
        Either "POS" or "NEG", ion mode for the data.
    Returns
    -------
    CSV and MGF files, filtered, in the duplicate filter folder.
    """    
    
    print(f"------- DUPLICATE FILTER : {ion_mode} -------")
    
    # Load parameters
    index_col = params['index_col']
    rt_field = params['rt_field']
    mz_field = params['mz_field']
    
    if ion_mode == "NEG":
        csv_file = params['neg_csv']
        spectrum_file= params['neg_mgf']
        out_path= params['neg_out_0']
    elif ion_mode == "POS" :
        csv_file = params['pos_csv']
        spectrum_file= params['pos_mgf']
        out_path= params['pos_out_0']
    
    # create output dir:
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    # Load MZmine mgf and csv files
    print_time("Loading MGF and CSV files...")
    spectrum_list = read_mgf_file(file_path = f'{params["input_dir"]}{spectrum_file}',
                                  mz_field = params['mz_field'],
                                  rt_field = params['rt_field'],
                                  charge_field = params['charge_field'],
                                  ion_mode = ion_mode)
    csv_table = pd.read_csv(f'{params["input_dir"]}{csv_file}', index_col = index_col)
    
    # Format columns with ion modes:
    new_cols = csv_table.columns.tolist()
    new_cols = [ion_mode + "_" + col if (params['sample_pattern'] in col and col[:4] != f"{ion_mode}_") else col for col in new_cols]
    csv_table.columns = new_cols
    
    # Extract data from the MGF file
    print_time('Extracting MGF metadata...')
    node_table = spectrum_list.to_data_frame()
    
    # Set index to what the user selected
    node_table[index_col.lower()] = node_table[index_col.lower()].astype(int)
    node_table.set_index(index_col.lower(), inplace = True, drop = True)
    
    # Rename the columns according to user input
    node_table = remapper(node_table, params)
    
    # Remove duplicate columns
    drop_cols = csv_table.columns.intersection(set(node_table.columns))
    node_table.drop(drop_cols, axis = 1, inplace = True)
    node_table = node_table.merge(csv_table, left_index = True, right_index = True)
    
    # Format RT and prec mz fields to float
    node_table[f'{rt_field}'] = node_table[f'{rt_field}'].astype(float)
    node_table[f'{mz_field}'] = node_table[f'{mz_field}'].astype(float)
    
    # Correct rt unit from m to s if relevant
    if params['rt_unit'] == 'm':
        node_table[f'{rt_field}'] = node_table[f'{rt_field}']*60

    # Add spec_id, for the relative position of ions in the spectrum file
    node_table.insert(0, 'spec_id', range(len(node_table)))

    # If this step must be skipped :
    if params['df_skip'] :
        node_table.to_csv(f'{out_path}{csv_file}')
        spectrum_list.write_mgf(output_file_path = f'{out_path}{spectrum_file}')
        return
    
    # Get duplicates & delete them
    print_time('Removing duplicates...')
    dropped_ions = duplicate_finder(node_table = node_table,
                                       spectrum_list = spectrum_list,
                                       params = params,
                                       ion_mode = ion_mode)

    kept_ions = list(set(node_table.index) - set(dropped_ions))
    kept_ions.sort()
    kept_ions = [node_table.loc[i, "spec_id"] for i in kept_ions]
    
    mgf_file_new = slice_spectra(spectrum_list, kept_ions)

    node_table_new = node_table.drop(dropped_ions)
    
    # Reset the spec_id to account for dropped ions
    node_table_new['spec_id'] = range(len(node_table_new))
    
    # Export the data
    print_time('Exporting MGF and CSV files...')
    mgf_file_new.write_mgf(output_file_path = f'{out_path}{spectrum_file}')
    
    node_table_new.to_csv(f'{out_path}{csv_file}', index_label = index_col)
    perc = round(100*(len(dropped_ions)/len(spectrum_list.spectrum)),1)
    print_time('Export finished.')
    print_time(f'{len(dropped_ions)} ions removed out of {len(spectrum_list.spectrum)} ({perc}%)')
    return

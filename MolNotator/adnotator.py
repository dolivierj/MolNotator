import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from MolNotator.others.global_functions import *
from MolNotator.utils import read_mgf_file, cosine_validation, spectrum_cosine_score


def adnotator(params : dict, ion_mode : str):
    """
    Finds neutral-adduct pairs.

    Parameters
    ----------
    params : dict
        Dictionary containing the global parameters for the process..
    ion_mode : str
        Either "POS" or "NEG", ion mode for the data.

    Returns
    -------
    None.

    """
    
    # Load parameters:
    rt_error = params['an_rt_error']
    mass_error = params['an_mass_error']
    cosine_threshold = params['an_cos_threshold']
    idx_column = params['index_col']
    
    # If the retention time is in minutes
    if params['rt_unit'] == "m":
        rt_error = rt_error/60
    
    if ion_mode == "NEG":
        in_path_full= params['neg_out_0']
        csv_file= params['neg_csv']
        spectrum_file= params['neg_mgf']
        in_path_spec= params['neg_out_1']
        in_path_csv= params['neg_out_2']
        out_path_full= params['neg_out_3_1']
        out_path_samples= params['neg_out_3_2']
        adduct_table_primary_file= params['an_addtable_primary_neg']
        adduct_table_secondary_file= params['an_addtable_secondary_neg']
    elif ion_mode == "POS":
        in_path_full= params['pos_out_0']
        csv_file= params['pos_csv']
        spectrum_file= params['pos_mgf']
        in_path_spec= params['pos_out_1']
        in_path_csv= params['pos_out_2']
        out_path_full= params['pos_out_3_1']
        out_path_samples= params['pos_out_3_2']
        adduct_table_primary_file = params['an_addtable_primary_pos']
        adduct_table_secondary_file = params['an_addtable_secondary_pos']
    else:
        print('Ion mode must be either "NEG" or "POS"')
        return
    
    # Load files
    csv_file = in_path_full + csv_file
    spectrum_file = in_path_full + spectrum_file
    
    input_files = pd.DataFrame()
    input_files['spectrum_file'] = os.listdir(in_path_spec)
    input_files['base_name'] = input_files['spectrum_file'].copy().str.replace(f'{ion_mode}_', "")
    input_files['base_name'] = input_files['base_name'].str.replace('.mgf', '', regex = False)
    
    adduct_table_primary = pd.read_csv("./params/" + adduct_table_primary_file, sep = "\t")
    adduct_table_secondary = pd.read_csv("./params/" + adduct_table_secondary_file, sep = "\t")
    
    # Add additional metadata to the adduct table
    adduct_table_primary.loc[:,'Group_numeric'] = adduct_table_primary.loc[:,'Group'].replace({'H' : 0, 'Cl': 1, 'Na' : 2, 'K' : 3})
    adduct_table_primary.loc[:,"complexed"] = adduct_table_primary.loc[:,"Adduct_code"].str.split('|').str[-1] != ""
    adduct_table_primary.loc[:,"decomplexed"] = list(pd.Series(adduct_table_primary.index, index = adduct_table_primary.loc[:,"Adduct_code"])[adduct_table_primary.loc[:,"Adduct_code"].str.split('|').str[0:2].str.join('|') + "|"])
    adduct_table_primary.loc[:,'adduct_id'] = adduct_table_primary.index
    
    adduct_table_secondary.loc[:,'Group_numeric'] = adduct_table_secondary.loc[:,'Group'].replace({'H' : 0, 'Cl': 1, 'Na' : 2, 'K' : 3})
    adduct_table_secondary.loc[:,"complexed"] = adduct_table_secondary.loc[:,"Adduct_code"].str.split('|').str[-1] != ""
    adduct_table_secondary.loc[:,"decomplexed"] = list(pd.Series(adduct_table_secondary.index, index = adduct_table_secondary.loc[:,"Adduct_code"])[adduct_table_secondary.loc[:,"Adduct_code"].str.split('|').str[0:2].str.join('|') + "|"])
    adduct_table_secondary.loc[:,'adduct_id'] = adduct_table_secondary.index
    
    adduct_table_merged = pd.concat([adduct_table_primary, adduct_table_secondary], ignore_index=True)
    adduct_table_merged.loc[:,'adduct_id'] = adduct_table_merged.index
    
    
    # Create output folder
    if not os.path.isdir(out_path_full) :
        os.mkdir(out_path_full)
    
    
    # Start processing sample by sample
    for x in tqdm(input_files.index) :
        
        sample_base_name = input_files.at[x, "base_name"]
        
        # Load files for the processed sample
        edge_table = pd.read_csv(f"{in_path_csv}{ion_mode}_{sample_base_name}_edges.csv", 
                                 index_col = 'Index')
        node_table = pd.read_csv(f"{in_path_csv}{ion_mode}_{sample_base_name}_nodes.csv", 
                                index_col = params['index_col'])
        subspectrum_file = f'{in_path_spec}{input_files.at[x, "spectrum_file"]}'
        
        
        spectrum_list = read_mgf_file(file_path = subspectrum_file)
        
        # Filter out multi-charge ions
        idx = node_table.index[abs(node_table[params['charge_field']]) <= 1]
        node_table = node_table.loc[idx]
        

        # Get all possible pairs
        pairs_table = get_pairs(mass_error = mass_error,
                                rt_error = rt_error,
                                cosine_threshold = cosine_threshold,
                                adduct_df = adduct_table_primary,
                                node_table = node_table,
                                edge_table = edge_table,
                                spectrum_list = spectrum_list)
        
        # Sort two first columns and remove duplicates
        pairs_table = matched_masses_to_pd(pairs_table)
        pairs_table[['feature_id_1', 'feature_id_2']] = pairs_table[['feature_id_1', 'feature_id_2']].apply(lambda row: sorted(row), axis=1, result_type='expand')
        pairs_table = pairs_table.drop_duplicates(subset = ['feature_id_1', 'feature_id_2'], keep = 'last', ignore_index = True)
        pairs_table = np.array(pairs_table)
        
        # Get cohorts table
        cohort_table = pairs_to_cohorts(matched_masses = pairs_table,
                                        node_table = node_table)
        
        # To Dataframe
        cohort_table = cohort_table_to_pd(cohort_table = cohort_table,
                                          node_table = node_table,
                                          adduct_df = adduct_table_primary,
                                          adduct_col = "adduct_id")

        # Filter cohorts (merge equivalent cohorts, remove ions without annotations)
        cohort_table = filter_cohorts(df = cohort_table)
        
        # cohort_table, transition_table = supercohorts_tabler(cohort_table)

        # Produce the court table
        court_table = get_court_table(cohort_table)

        # Produce house table
        court_table = house_selection(court_table = court_table,
                                      supercohorts_table = cohort_table,
                                      node_table = node_table,
                                      transition_table = transition_table,
                                      merged_table = merged_table,
                                      spectrum_list = spectrum_list,
                                      params = params)
        

        # Update cross sample tables with the results
        cross_annotations, cross_points, cross_courts, cross_houses, cross_rules, cross_neutrals = cross_sample_report(court_table, cross_annotations, cross_points, cross_courts,
                            cross_houses, cross_rules, cross_neutrals, 
                            sample_base_name, cohort_table, 
                            adduct_table_primary, adduct_table_merged, node_table,
                            ion_mode, duplicate_table, spectrum_list, params)
    
        # Add secondary adducts
        cross_annotations, cross_points, cross_courts, cross_houses, cross_rules, cross_neutrals = get_secondary_adducts(cross_annotations, cross_points, cross_courts,
                                  cross_houses, cross_rules, cross_neutrals, sample_base_name,
                                  node_table, spectrum_list, adduct_table_primary, 
                                  adduct_table_secondary, ion_mode, params)
    
        # Export data
        cross_annotations.to_csv(out_path_full+"cross_sample_annotations.csv", index_label = idx_column)
        cross_courts.to_csv(out_path_full+"cross_sample_courts.csv", index_label = idx_column)
        cross_houses.to_csv(out_path_full+"cross_sample_houses.csv", index_label = idx_column)
        cross_points.to_csv(out_path_full+"cross_sample_points.csv", index_label = idx_column)
        cross_rules.to_csv(out_path_full+"cross_sample_rules.csv", index_label = idx_column)
        cross_neutrals.to_csv(out_path_full+"cross_sample_neutrals.csv", index_label = idx_column)
    
    # Load and merge fragnoted 
    merged_node_table, merged_edge_table = get_merged_tables(input_files, ion_mode, params)
    merged_edge_table['Adnotation'] = [None]*len(merged_edge_table)
    merged_node_table['Adnotation'] = [None]*len(merged_node_table)
    
    # Resolve the sample-specific unresolved ions by cross examining all samples:
    cross_annotations, cross_points, cross_courts, cross_houses, cross_rules, cross_neutrals = annotation_resolver(cross_annotations, cross_points, cross_courts, cross_houses, cross_rules, cross_neutrals)
    
    # Rebuild houses at a cross-sample level to check compatibilities between annotations
    cross_court_table, singletons = get_cross_courts(cross_annotations, cross_courts, cross_houses)    
    
    # Select the most likeley neutrals from the cross_court_table and report
    # the data on the merged node and edge tables.
    
    spectrum_list = read_mgf_file(file_path=spectrum_file)
    
    
    merged_node_table, merged_edge_table = cross_neutral_selection(spectrum_list = spectrum_list,
                                                                   cross_court_table = cross_court_table,
                                                                   cross_annotations = cross_annotations,
                                                                   cross_neutrals = cross_neutrals,
                                                                   merged_node_table = merged_node_table,
                                                                   merged_edge_table = merged_edge_table,
                                                                   adduct_table_merged = adduct_table_merged,
                                                                   ion_mode = ion_mode,
                                                                   params = params)
    
    
    
    # Update node table status:
    merged_node_table = update_node_table(merged_node_table, merged_edge_table,
                        cross_annotations, cross_rules, adduct_table_merged, params)
    
    # Update edge table with singletons:
    merged_edge_table = update_edge_table(merged_node_table, merged_edge_table,
                                          adduct_table_merged)
    
    # Export merged tables
    merged_node_table.to_csv(f"{out_path_full}node_table.csv", index_label = idx_column)
    merged_edge_table.to_csv(f"{out_path_full}edge_table.csv", index_label = "Index")
    
    
    # Also export sample wise slices of the tables for viewing at this stage
    if params['an_export_samples'] : 
        samplewise_export(merged_edge_table, merged_node_table, "adnotator", ion_mode, params)
    return

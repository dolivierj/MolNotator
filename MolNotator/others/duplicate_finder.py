"""duplicate_finder.py - duplicate_finder module for the duplicate_filter"""
import pandas as pd
from MolNotator.utils import spectrum_cosine_score
from sklearn.neighbors import KDTree


# Define a function that performs operations on each ion
def compare_ions(ion_pool, spectrum_list, mass_error):
    
    
    # Start doing a cosine comparison between duplicates
    cos_table = list()
    while len(ion_pool) > 1 :
        ion_1 = ion_pool[0]
        spectrum_1 = spectrum_list.spectrum[ion_1]
        rt_1 = spectrum_1.rt
        mz_1 = spectrum_1.prec_mz
        ion_pool.remove(ion_1)
        for ion_2 in ion_pool:
            spectrum_2 = spectrum_list.spectrum[ion_2]
            rt_2 = spectrum_2.rt
            mz_2 = spectrum_2.prec_mz
            score, matches = spectrum_cosine_score(spectrum_1,
                                                   spectrum_2,
                                                   tolerance = mass_error)

            cos_table.append((ion_1,
                              ion_2,
                              score,
                              matches,
                              abs(mz_1 - mz_2),
                              abs(rt_1 - rt_2)))
    cos_table = pd.DataFrame(cos_table, columns = ['ion_1', 'ion_2', 'cos',
                                                   'matched_peaks', 'd_mz', 'd_rt'])
    return cos_table

def find_duplicates_pairs(duplicates_table, node_table, spectrum_list, mass_error, cos_threshold):
    duplicates_table = duplicates_table[duplicates_table['count'] == 2].copy()
    pairs = duplicates_table['pool'].tolist()
    cos_list = list()
    droppable_ion = list()
    for i, j in pairs:
        score, matches = spectrum_cosine_score(spectrum_list.spectrum[node_table.at[i, "spec_id"]],
                                               spectrum_list.spectrum[node_table.at[j, "spec_id"]],
                                               tolerance = mass_error)
        cos_list.append(score)
        droppable_ion.append(node_table.loc[[i,j], "TIC"].idxmin())
        
    duplicates_table["score"] = cos_list
    duplicates_table["dropped"] = droppable_ion
    duplicates_table = duplicates_table[duplicates_table["score"] >= cos_threshold]
    
    dropped_idx = duplicates_table['dropped'].unique().tolist()
    
    return dropped_idx

def find_complex_duplicates(duplicates_table, node_table, spectrum_list, mass_error, cos_threshold):
    duplicates_table = duplicates_table[duplicates_table['count'] > 2].copy()
    dropped_ions = list()
    for i in duplicates_table.index:
        cos_table = compare_ions(ion_pool = list(duplicates_table.at[i, "pool"]),
                                 spectrum_list = spectrum_list,
                                 mass_error = mass_error)
        ion_pool = list(set(cos_table['ion_1'].tolist() + cos_table['ion_2'].tolist()))
        while len(ion_pool) > 0 :
            ion_seed = node_table.loc[ion_pool, 'TIC'].idxmax()
            ion_pool.remove(ion_seed)
            
            duplicates = cos_table.index[cos_table['ion_1'] == ion_seed].tolist() + cos_table.index[cos_table['ion_2'] == ion_seed].tolist()
            duplicates = cos_table.loc[duplicates].index[cos_table.loc[duplicates, 'cos'] >= cos_threshold] 
            duplicates = cos_table.loc[duplicates, 'ion_1'].tolist() + cos_table.loc[duplicates, 'ion_2'].tolist()
            duplicates = list(set(duplicates))
            
            if len(duplicates) == 0 : 
                continue
            
            duplicates.remove(ion_seed)
            duplicates.sort()
            ion_pool = list(set(ion_pool) - set(duplicates))
            dropped_ions += duplicates
    
    dropped_ions.sort()
    return dropped_ions

def duplicate_finder(node_table, spectrum_list, params, ion_mode):
    """
    Takes a node_table and an spectrum_list from the duplicate_filter module to 
    find duplicate ions (close RT, m/z and cosine values). Returns a list of 
    duplicate ions to be deleted along with the representative ion that is to
    be kept.
    Parameters
    ----------
    node_table : pandas.DataFrame
        Dataframe containing each ion's metadata, including RT, m/z and the 
        relative position in the spectrum file.
    spectrum_list : list
        list of matchms.Spectrum objects, to be used to compare each ions with
        cosine similarity.
    params : dict
        Dictionary containing the global parameters for the process.
    Returns
    -------
    duplicate_table : pandas.DataFrame
        Dataframe containing the representative ions to be kept and the duplicate
        ions to be deleted.
    """

    # Get parameters & load cosine function
    mass_error = params['df_mass_error']
    rt_error = params['df_rt_error']
    cos_threshold = params['df_cos_threshold']
    rt_field = params['rt_field']
    mz_field = params['mz_field']
    
    # If the retention time is in minutes
    if params['rt_unit'] == "m":
        rt_error = rt_error/60

        
    # Build a KDTree for each field
    rt_tree = KDTree(node_table[[rt_field]].values)
    mz_tree = KDTree(node_table[[mz_field]].values)
    
    # Find neighbors within rt_error and mass_error
    rt_inds = rt_tree.query_radius(node_table[[rt_field]].values, r=rt_error)
    mz_inds = mz_tree.query_radius(node_table[[mz_field]].values, r=mass_error)
    
    duplicates_table = list()
    duplicates_idx = list()
    
    for rt_neighbors, mz_neighbors in zip(rt_inds, mz_inds):
        # Find ions that are neighbors in both dimensions
        duplicate_ions = tuple(set(rt_neighbors).intersection(set(mz_neighbors)))
        duplicates_table.append((duplicate_ions, len(duplicate_ions)))
        
    duplicates_table = pd.DataFrame(duplicates_table, columns = ["pool", "count"])
    
    duplicates_table = duplicates_table[duplicates_table['count'] > 1]
    
    # Deal with non conflict duplicates (1 ion and its duplicate)
    duplicates_idx += find_duplicates_pairs(duplicates_table = duplicates_table,
                                            node_table = node_table,
                                            spectrum_list = spectrum_list,
                                            mass_error = mass_error,
                                            cos_threshold = cos_threshold)
    
    # Deal with conflict duplicates (1 ion with 2+ duplicates)
    duplicates_idx += find_complex_duplicates(duplicates_table = duplicates_table,
                                              node_table = node_table,
                                              spectrum_list = spectrum_list,
                                              mass_error = mass_error,
                                              cos_threshold = cos_threshold)
    
    # Return the list of dropped indices
    duplicates_idx.sort()
    return duplicates_idx
    

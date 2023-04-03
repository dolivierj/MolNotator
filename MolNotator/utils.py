import pandas as pd
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


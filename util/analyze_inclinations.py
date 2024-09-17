import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from analyze_period import prepare_df, compare_kois, string_to_list2
from plots import *
import re

def string_to_float_list(s):
    # Remove brackets and split by comma or space
    stripped = s.strip('[]')
    elements = re.split(r'[,\s]+', stripped)
    return [float(x) for x in elements if x]


def merge_and_aggregate_dataframes(dataframes, merge_column='KID', agg_dict=None):
    """
    Merges multiple DataFrames based on a specified column and aggregates the results.

    Args:
    dataframes (list): List of pandas DataFrames to merge
    merge_column (str): Column name to use for merging (default: 'KID')
    agg_dict (dict): Dictionary specifying aggregation functions for each column
                     (default: None, which will use 'first' for all columns)

    Returns:
    pandas.DataFrame: Merged and aggregated DataFrame
    """
    # Merge all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)

    # If no aggregation dictionary is provided, use 'first' for all columns except the merge column
    if agg_dict is None:
        agg_dict = {col: 'first' for col in merged_df.columns if col != merge_column}

    # Perform groupby and aggregation
    result = merged_df.groupby(merge_column).agg(agg_dict).reset_index()

    return result
def get_merged_quantiled_df(dfs_dir):
    dfs = []
    for i, file in enumerate(os.listdir(dfs_dir)):
        print("analyzing file ", file)
        if file.endswith('csv'):
            lag = file.strip('.csv').split('_')[-1]
            df = prepare_df(pd.read_csv(f'{dfs_dir}/{file}'), filter_giants=True, filter_eb=False,
                            teff_thresh=True, filter_non_ps=True)
            dfs.append(df)
    res = merge_and_aggregate_dataframes(dfs)
    return res

def plot_hists(df):
    hist(df, att='predicted inclination', theoretical='cos', dir='../inc_imgs', save_name='inc')
    threshold_hist(df, thresh_att='Teff', thresh=[3500, 5000, 6000, 6500], save_name='inc_t',
                   att='predicted inclination', dir='../inc_imgs')
    threshold_hist(df, thresh_att='Teff', thresh=[3500, 5000, 6000, 6500], save_name='inc_t',
                   att='predicted inclination', dir='../inc_imgs', sign='small')
    threshold_hist(df, thresh_att='Teff', thresh=[3500, 5000, 6000, 6500], save_name='inc_t',
                   att='predicted inclination', dir='../inc_imgs', sign='between', theoretical='cos')

    kois = pd.read_csv('tables/kois.csv')
    kois['kepler_name'] = kois['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])
    duplicates_mask = kois.duplicated(subset='KID', keep='first')
    kois = kois[~duplicates_mask]
    kois.reset_index(drop=True, inplace=True)
    kois_pred = df[df['KID'].isin(kois['KID'])]
    hist(df, df_other=kois_pred, att='predicted inclination', other_att='predicted inclination',
         save_name='inc_kois', )
    df_low_t = df[(df['Teff'] > 3500) & (df['Teff'] < 5000)]
    threshold_hist(df_low_t, thresh_att='R', thresh=[0,0.4,0.8, 1], save_name='inc_low_t_r',
                   att='predicted inclination', dir='../inc_imgs', sign='between')
    threshold_hist(df_low_t, thresh_att='logg', thresh=[0, 3.5], save_name='inc_low_t_r',
                   att='predicted inclination', dir='../inc_imgs', sign='between')


def compare_kois(df, refs):
    # kois = pd.read_csv('tables/kois.csv')
    # kois['kepler_name'] = kois['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])
    # duplicates_mask = kois.duplicated(subset='KID', keep='first')
    # kois = kois[~duplicates_mask]
    # kois.reset_index(drop=True, inplace=True)
    # kois_pred = df[df['KID'].isin(kois['KID'])]
    pred_with_refs = df.merge(refs, on='KID', suffixes=['', '_ref'])
    pred_errors = [pred_with_refs['predicted inclination'] - pred_with_refs['predicted inclination q_0.05'].values,
                   pred_with_refs['predicted inclination q_0.95'].values - pred_with_refs['predicted inclination']]
    refs['err_i'] = refs['err_i'].apply(string_to_list2)
    refs_error = np.array(pred_with_refs['err_i'].values)
    refs_error = np.array([string_to_float_list(s) for s in refs_error])
    pred_with_refs['marker'] = '*'

    plot_kois_comparison2(pred_with_refs, att1='i', att2='predicted inclination', err1 =refs_error,
                          err2=np.array(pred_errors).T, name='inc_kois', save_dir='../inc_imgs')
    # plt.errorbar(pred_with_refs['i'], pred_with_refs['predicted inclination'], yerr=pred_errors,
    #              fmt='*')
    # plt.scatter(pred_with_refs['i'], pred_with_refs['predicted inclination'])
    plt.show()


if __name__ == "__main__":
    print(os.getcwd())
    res = get_merged_quantiled_df('../inference/astroconf_play_exp13_kl_1')
    res.to_csv('tables/astroconf_play_exp13.csv')
    # res = pd.read_csv('tables/astroconf_play_exp12.csv')
    # res = pd.read_csv('tables/kepler_model_pred_exp45.csv')

    plot_hists(res)
    refs = pd.read_csv('tables/all_refs.csv')
    compare_kois(res, refs)

    print(len(res))
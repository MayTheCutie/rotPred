# __author:IlayK
# data:17/03/2024
import json
import time

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import ast
from PIL import Image
from scipy import stats
from matplotlib.legend_handler import HandlerTuple
from scipy.special import comb
import lightkurve as lk
from scipy import stats
# import stardate as sd


import re
import matplotlib as mpl
import os
from scipy.stats import ks_2samp
import warnings
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from astropy.io import fits
from astropy.table import Table, join, vstack
from scipy.signal import savgol_filter as savgol
# from utils import extract_qs, consecutive_qs, plot_fit
from plots import *
# from gyro.gyrointerp import gyro_age_posterior, gyro_age_posterior_list
# from gyro.gyrointerp import get_summary_statistics
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from stardate.lhf import age_model
from astropy.table import Table
from collections import defaultdict



warnings.filterwarnings("ignore")
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D


from utils import convert_to_list
import seaborn as sns

from scipy.signal import convolve

from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import pearsonr

mpl.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 32, 'figure.figsize': (16,10), 'lines.linewidth': 4})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["gray", "r", "c", 'm', 'brown', 'yellow'])
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r",  "black", "brown", "blue", "green",  "y",  "purple", "pink"])
plt.rcParams.update({'xtick.labelsize': 30, 'ytick.labelsize': 30})
plt.rcParams.update({'legend.fontsize': 26})



T_sun = 5770
R_sun = 6.96 * 1e9
L_sun = 3.85 * 1e26
sigma = 5.67 * 1e-8
J_radius_factor = 7
teff_hj_cool = 6200
prot_hj = 10


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def func(x, a, b):
    return a * x + b



kaimaka_paths = 'kepler/Kaimaka_planet_{}{}.txt'

p_columns = ['#', 'obj tags', 'obj count','koi id', 'raw id','identifier' ,'typ', 'coord1 (ICRS,J2000/2000)',
             'Mag U','Mag B','Mag V', 'Mag R','Mag I','spec. type' ,'#bib','#not']


def string_to_list2(string_array):
    '''
    convert string to list
    :param string_array:
    :return:
    '''
    try:
        # Use ast.literal_eval to safely evaluate the string as a literal
        return np.array(ast.literal_eval(string_array))
    except (ValueError, SyntaxError):
        # Handle cases where the string cannot be safely evaluated
        return None  # or any other appropriate actio


def string_to_list(string_array):
    try:
        # Use ast.literal_eval to safely evaluate the string as a literal
        array = ast.literal_eval(string_array)
        if len(array) >= 3:  # Ensure the array has at least 3 elements
            return array[2]
        else:
            return None  # Handle cases where the array doesn't have a position 2
    except (ValueError, SyntaxError):
        # Handle cases where the string cannot be safely evaluated
        return None  # or any other appropriate actio
# Load the .dat file into a pandas DataFrame

def string_to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan
def convert_probs_float_list(string):
    # Remove '[' and ']' characters from the string
    string = string.replace('[', '').replace(']', '')
    # Split the string by whitespace
    values = string.split()
    # Convert each value to float
    float_values = [float(value) for value in values]
    return float_values
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def median_agg(series):
    return np.median(series.tolist(), axis=0)




def read_raw_table(t_path, columns, start_idx=0, sep='\t',
                   col_type='label', clean_lines=True):
    if isinstance(columns, str):
        columns_df = pd.read_csv(columns, sep=sep)
        columns_df = columns_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        if col_type == 'label':
            columns = columns_df['Label'].values
        else:
            columns = columns_df.columns
    with open(t_path, 'r') as file:
        lines = file.readlines()
    # Parse each row to extract values and errors
    data = []
    for i, line in enumerate(lines):
        if i < start_idx:
            continue
        if clean_lines:
            line = re.sub(r'\s+', ',', line)
            line = re.sub(r',\*', '*', line)
            line = re.sub(r',+$', '', line)  # Remove trailing commas
        elements = line.rstrip('\n ').split(',')
        row = []
        print(elements)
        for j, e in enumerate(elements):
            if columns[j] == 'KID' or ('idx' in columns[j]):
                row.append(int(e))
            else:
                try:
                    row.append(float(e))
                except ValueError:
                    row.append(e)
        data.append(row)
        if i % 1000 == 0:
            print(i)
    df = pd.DataFrame(data, columns=columns)
    return df

def read_ascii_table_with_errors(t_path, columns, start_idx=1, sep='\t'):
    # Read the ASCII table into a list of strings
    with open(t_path, 'r') as file:
        lines = file.readlines()

    # Parse each row to extract values and errors
    data = []
    for line in lines:
        elements = line.rstrip('\n\t').split(sep)
        values = elements[:start_idx]
        for e in elements[start_idx:]:
            found = 0
            # Find all instances of the format "<number> +or- <error>"
            match = re.findall(r'([-+]?\d*\.?\d*)\s*\+or-\s*([-+]?\d*\.?\d*)', e)
            if match:
                found += 1
                values.extend([float(match[0][0]), [float(match[0][1]), float(match[0][1])]])

            # Find all instances of the format "${<number>}_{-<error>}^{+<error>}$"
            match = re.findall(r'\${([-+]?\d*\.?\d*)}_{-([-+]?\d*\.\d+)}\^{\+([-+]?\d*\.?\d*)}\$', e)
            if match:
                found += 1
                values.extend([float(match[0][0]), [float(match[0][1]), float(match[0][2])]])
            if not found:
                values.extend([e, [None, None]])
        print(len(values))
        data.append(values)

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Display the DataFrame
    return df

def adaptive_thresold(period, max_period=15, start_threshold=0.9, end_threshold=0.8):
    if period < 10:
        return start_threshold
    else:
        slope = (end_threshold - start_threshold) / (max_period - 10)
        return start_threshold + slope * (period - 10)

def fill_nan_np(x, interpolate=True):

    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(x))[0]

    # Find indices of NaN values
    nan_indices = np.where(np.isnan(x))[0]
    if interpolate:
        # Interpolate NaN values using linear interpolation
        interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])

        # Replace NaNs with interpolated values
        x[nan_indices] = interpolated_values
    else:
        x[nan_indices] = 0
    return x

def calc_stardate_age_per_row(row, p_att='predicted period'):
    bprp = row['bp_rp']  # Gaia BP - RP color.
    prot = row[p_att]
    log10_period = np.log10(prot)
    log10_age_yrs = age_model(log10_period, bprp)
    return 10 ** log10_age_yrs * 1e-6


def calc_gyro_age_myears_per_row(row, p_att='predicted period'):
    """
       from S. Barnes 2003 https://arxiv.org/pdf/0704.3068
       :param row: a row of the dataframe
       :return:
       """
    p = row[p_att]
    Teff = row['Teff']

    a = 0.77
    b = 0.553
    c = 0.472
    n = 0.519

    B_V = B_V_from_T(Teff)

    log_t = (1 / n) * (np.log10(p) - np.log10(a) - b * np.log10(B_V - c))
    return 10 ** (log_t)
def calc_gyro_age_myears(p, Teff):
    """
    from S. Barnes 2003 https://arxiv.org/pdf/0704.3068
    :param p: rotation period
    :return:
    """
    print(p, Teff)
    a = 0.77
    b = 0.553
    c = 0.472
    n = 0.519
    B_V = B_V_from_T(Teff)
    log_t = (1/n) * (np.log10(p) - np.log10(a) - b*np.log10(B_V - c))
    return 10**(log_t)


def B_V_from_T(T):
    """
    from F. J. Ballesteros 2012 https://arxiv.org/pdf/1201.1809
    :param T: Teff
    :return: B-V color index
    """
    a = 0.8464 * T
    b = 2.1344 * T - 4600 * 1.84
    c = 1.054 * T - 4600 * 2.32

    discriminant = b ** 2 - 4 * a * c

    x_positive = (-b + np.sqrt(discriminant)) / (2 * a)
    return x_positive

def T_from_B_V(B_V):
    """
    from F. J. Ballesteros 2012 https://arxiv.org/pdf/1201.1809
    :param B_V: color index
    :return: Teff
    """
    return 4600*(1/(0.92*B_V+1.7)+1/(0.92*B_V + 0.62))

def kinematic_gyro_age_comparison(df_kinematic, df_gyro, period_df):
    df_kinematic = df_kinematic.merge(period_df, on='KID')
    df_gyro = df_gyro.merge(period_df[['KID', 'predicted period', 'Teff',
                                           'mean_period_confidence', 'total error', 'eb_group']],
                                on='KID')
    # merged_df = df_gyro.merge(df_kinematic, on='KID', how='left')
    # for c in merged_df['Comp'].unique():
    #     reduced_df = merged_df[merged_df['Comp']==c]
    #     plt.scatter(reduced_df['predicted period'], reduced_df['age'], s=4, label=c)
    # plt.legend()
    # plt.colorbar(label='galactic component')
    # # plt.xlim((0,10))
    # plt.show()

    # ebs = merged_df[~merged_df['eb_group'].isna()]
    # ebs_gyro = df_gyro[~df_gyro['eb_group'].isna()]
    # plt.errorbar(ebs_gyro['predicted period'], ebs_gyro['age'], yerr=ebs_gyro[['e_age_up','e_age_low']].values)
    # plt.show()
    # print(len(df_kinematic), len(df_gyro))

def compare_ages(df_inference, dfs_compare, names, colors, p_att='Prot', method='barnes', save_dir='../imgs'):
    gaia_kepler = Table.read('tables/kepler_dr3_4arcsec.fits', format='fits').to_pandas()
    df_inference = df_inference.merge(gaia_kepler[['kepid', 'bp_rp', 'parallax', 'parallax_error']]
                                      .drop_duplicates('kepid'),
                              left_on='KID', right_on='kepid')
    if method == 'barnes':
        df_inference['age_myears'] = df_inference.apply(calc_gyro_age_myears_per_row, axis=1)
    else:
        df_inference['age_myears'] = df_inference.apply(calc_stardate_age_per_row, axis=1)
    for color, name, df in zip(colors, names, dfs_compare):
        print("calculating ages of ", name)
        df = df.merge(gaia_kepler[['kepid', 'bp_rp', 'parallax', 'parallax_error']]
                           .drop_duplicates('kepid'),
                           left_on='KID', right_on='kepid')
        if 'Teff' not in df.columns:
            df['Teff'] = df_inference[df_inference['KID'].isin(df['KID'])]['Teff']
        if method == 'barnes':
            df['age_myears'] = df.apply(lambda row: calc_gyro_age_myears_per_row(row, p_att), axis=1)
        else:
            df['age_myears'] = df.apply(lambda row: calc_stardate_age_per_row(row, p_att), axis=1)

        plt.hist(df['age_myears'], histtype='step',
                 density=True,
                 bins=np.linspace(0, 10000, 40),
                 label=name,
                 color=color,
                 linewidth=3
             )
        # plt.hist(df['age_myears'],
        #          histtype='step', density=True, bins=np.linspace(0, 10000, 40), label=name)
    plt.hist(df_inference['age_myears'], histtype='step',
             density=True,
             bins=np.linspace(0, 10000, 40),
             label='LightPred',
             color='r',
             linewidth=3
             )

    plt.xlabel(r"Age ($10^6$ year)")
    plt.ylabel("Density")
    # plt.ylim((0, 0.0007))
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'age_dist_ref_{method}.png'))
    plt.close()




# def mass_binning(kepler_inference, catalog, m_bins=[0, 0.8, 1.4, 3]):
#     df = kepler_inference.merge(catalog, on='KID')
#
#     fig, ax = plt.subplots()
#
#     for i, b in enumerate(m_bins[:-1]):
#         sub_df = df[(df['Mstar'] >= b) & (df['Mstar'] < m_bins[i + 1])]
#         sns.kdeplot(sub_df['predicted period'],
#                     label=fr'${b:.2f}*M_\odot < M < {m_bins[i + 1]:.2f}*M_\odot$ ({len(sub_df)} samples)',
#                     ax=ax,
#                     linewidth=2)
#         print("avg sigma error: ", sub_df['sigma error'].mean())
#
#     plt.legend(fontsize='small')
#     plt.show()

def mass_binning(df, m_bins=[0,1,1.4], save_dir='../imgs'):
    # m_bins = np.linspace(df['Mstar'].min(), df['Mstar'].max(), n_bins)
    for i,b in enumerate(m_bins[:-1]):
        sub_df = df[(df['Mstar'] >= b) & (df['Mstar'] < m_bins[i+1])]
        label = fr'${b:.2f}*M_\odot < M < {m_bins[i+1]:.2f}*M_\odot$, avg error (Days)'\
            f"-{sub_df['sigma error'].mean():.2f}" if i > 0 else fr'$M < {m_bins[i+1]:.2f}*M_\odot$, '\
             f"avg error (Days)-{sub_df['sigma error'].mean():.2f}"
        plt.hist(sub_df['predicted period'], histtype='step',
                 density=True,
                 bins=np.linspace(0,40,40),
                 label=label,
                 linewidth=3)
        print("avg sigma error: ", sub_df['sigma error'].mean())
    sub_df = df[df['Mstar'] >= m_bins[-1]]
    plt.hist(sub_df['predicted period'], histtype='step',
             density=True,
             bins=np.linspace(0, 40, 40),
             label=fr"${m_bins[-1]:.2f}*M_\odot < M$, avg error (Days) - {sub_df['sigma error'].mean():.2f} ",
             linewidth=3)
    print("avg sigma error: ", sub_df['sigma error'].mean())
    plt.legend(fontsize='small')
    plt.xlabel('Predicted Period (Days)', fontsize=30)
    plt.ylabel('Density', fontsize=30)
    plt.savefig(f'{save_dir}/mass_bins.png')
    plt.show()

def period_mass_bin(df, m=1, save_dir='../imgs'):
    sub_df = df[(df['Mstar'] >= m*0.9) & (df['Mstar'] < m*1.1)]
    plt.hexbin(sub_df['predicted period'], sub_df['sigma error'], mincnt=1)
    plt.title(fr'{m*0.9:.2f}$*M_\odot$ < M < {m*1.1:.2f}*$M_\odot$ ({len(sub_df)} samples)')
    print("avg sigma error in bin: ", sub_df['sigma error'].mean())
    plt.xlabel('Predicted Period (Days)', fontsize=30)
    plt.ylabel('Observational Error (Days)', fontsize=30)
    plt.savefig(f'{save_dir}/p_mass_bin.png')
    plt.show()

def period_metalicity(df, m_bins=[-1,-0.5,0,0.5], save_dir='../imgs'):
    df = kepler_inference.merge(catalog, on='KID')
    plt.hexbin(df['FeH'], df['predicted period'], mincnt=1)
    plt.xlabel(r'Surface Metalicity $([Fe/H])$')
    plt.ylabel('predicted period')
    plt.savefig(f'{save_dir}/metal_p.png')
    plt.show()
    for i,b in enumerate(m_bins[:-1]):
        sub_df = df[(df['FeH'] >= b) & (df['FeH'] < m_bins[i+1])]
        plt.hist(sub_df['predicted period'], histtype='step',
                 density=True,
                 bins=np.linspace(0,40,40),
                 label=fr'${b:.2f} < [Fe/H] < {m_bins[i+1]:.2f}$, avg error (Days)'
                       f"-{sub_df['sigma error'].mean():.2f}")
        print("avg sigma error: ", sub_df['sigma error'].mean())
    sub_df = df[df['FeH'] >= m_bins[-1]]
    plt.hist(sub_df['predicted period'], histtype='step',
             density=True,
             bins=np.linspace(0, 40, 40),
             label=fr"${m_bins[-1]:.2f} <= [Fe/H]$, avg error (Days) - {sub_df['sigma error'].mean():.2f} ")
    print("avg sigma error: ", sub_df['sigma error'].mean())
    plt.legend(fontsize='small')
    plt.xlabel('Predicted Period (Days)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/metal_bins.png')
    plt.show()



def calculate_error_bars(true, predicted, max_val=90):
    """
    calculate the error bars based on predictions std
    :param true: true predictions
    :param predicted: predictions
    :param max_val: maximum value
    :return: errs - the error (std from the mean) for each true integer value, mean - average for each integer value,
    mean_std_df - dataframe with all the predictions per each true integer value
    """
    df = pd.DataFrame({'true':true, 'predicted':predicted, 'diff':np.abs(true - predicted)}).sort_values('true')
    df['value'] = df['true'].round().astype(int)
    mean_std_df = df.groupby('value')['diff'].agg(['mean', 'std']).reset_index()
    df = df.merge(mean_std_df, on='value', how='left')

    df['lower_bound'] = df['value'] - (df['mean'] - 1 * df['std'])
    df['upper_bound'] = df['value'] + df['mean'] + 1 * df['std']
    df.ffill(inplace=True)
    mean_std_df.ffill(inplace=True)

    # df['std'].ffill(inplace=True)
    lower_bound = df.groupby('true')['lower_bound'].mean().reset_index()['lower_bound'].values
    upper_bound = df.groupby('true')['upper_bound'].mean().reset_index()['upper_bound'].values
    mean = df.groupby('true')['predicted'].mean().reset_index()['predicted'].values
    errs = np.clip(np.concatenate([lower_bound[None], upper_bound[None]]),0,None)

    return errs, mean, mean_std_df




def filter_samples(df1, df2):
    """
    find the intersection of kepler samples on two dataframes based on 'KID'
    :param df1: first dataframe
    :param df2: second dataframe
    :return: merged dataframe
    """
    merged_df = pd.merge(df1, df2, on='KID', how='left', indicator=True)
    merged_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge']).dropna(axis=1, how='all')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
    return merged_df


def giant_cond(x):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :param: x row in dataframe with columns - Teff, logg
    :return: boolean
    """
    logg, teff = x['logg'], x['Teff']
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh


def create_kois_mazeh(kepler_inference, mazeh_path='tables/Table_1_Periodic.txt', kois_path='tables/kois_no_fp.csv'):
    """
    get sub samples dataframes of kois and Mazeh objects
    :param kepler_inference: all kepler objects inference results
    :param mazeh_path: path to Mazeh csv
    :param kois_path: path to kois csv
    :return: 3 Dataframes - Mazeh sub sample, KOIs sub sample and non-KOIs sub sample
    """
    mazeh = pd.read_csv(mazeh_path)
    kois = pd.read_csv(kois_path)
    kois.sort_values(by='kepler_name', inplace=True)

    merged_df_mazeh = kepler_inference.merge(mazeh, on='KID', how='right')
    merged_df_mazeh.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    merged_df_kois = kepler_inference.merge(kois, on='KID')
    merged_df_kois.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    target_cols = ['Teff', 'KID', 'R','logg',
                   'kepler_name','planet_Prot','eb',
                   'confidence', 'koi_prad', 'sigma error']

    columns = [col for col in merged_df_kois.columns if 'period' in col or 'inclination' in col] + target_cols

    return merged_df_mazeh, merged_df_kois[columns]



def prepare_df(df, scale=False, filter_giants=True,
               filter_contaminants=True,
               filter_eb=True, filter_non_ps=False,
               teff_thresh=True, calc_errors=True):
    """
    prepare Dataframe for inference
    :param df: raw dataframe
    :param scale: if True, results are scaled
    :param filter_giants: filter out red gianet
    :param filter_eb: filter out eclipsing binaries
    :return: prepared dataframe
    """
    # teff = pd.read_csv('kepler/teff.csv')
    columns_to_lower = [col for col in df.columns if col.startswith('predicted') or col.endswith('confidence')]

    # Create a mapping dictionary
    column_mapping = {col: col.lower() for col in columns_to_lower}

    print("initial size: ", len(df))


    # Rename the specified columns
    df.rename(columns=column_mapping, inplace=True)
    if 'predicted inclination' not in df.columns and 'predicted inclination q_0.5' in df.columns:
        df.rename(columns={'predicted inclination q_0.5': 'predicted inclination',
                           'predicted period q_0.5': 'predicted period'},
                            inplace=True)
    if 'Teff' not in df.columns and 'KID' in df.columns:
        teff_df = pd.read_csv('tables/berger_catalog.csv')
        df['Teff'] = teff_df[df['KID'].isin(teff_df['KID']).astype(bool)]['Teff']
    try:
        err_model_p = pd.read_csv('tables/err_df_p.csv')
        err_model_i = pd.read_csv('tables/err_df_i.csv')
    except FileNotFoundError:
        err_model_p = None
        err_model_i = None
    if 'predicted inclination' in df.columns:
        if df['predicted inclination'].max() <= 2:
            inc_cols = [c for c in df.columns if ('inclination' in c) and ('confidence' not in c) ]
            df[inc_cols] = df[inc_cols] * 180 / np.pi
        if 'Inclination' in df.columns:
            if df['Inclination'].max() <= 2:
                df['Inclination'] = df['Inclination'] * 180 / np.pi
    # plt.xlim(0,1000)
    if 'KID' in df.columns:
        df['KID'] = df['KID'].astype(np.int64)
        eb = pd.read_csv('tables/kepler_eb.txt')
        df['eb'] = df['KID'].isin(eb['KID']).astype(bool)

        non_ps = pd.read_csv('tables/Table_2_Non_Periodic.txt')
        non_ps = non_ps.dropna()
        df = pd.merge(df, non_ps, how='left', on='KID')
        # df['w'] = df['w'].fillna(10)
        df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)
        df = df[df.columns.drop(list(df.filter(regex='_y$')))]
    # print(df['predicted inclination'].max())

        # print("after inclination scaling - max : ", df['predicted inclination'].max())
    if scale:
        df['predicted period'] *= 60
        df['predicted decay time'] *= 10
        # df['predicted inclination'] *= 90
    if 'inclination confidence' in df.columns:
        df['inclination confidence'] = 1 - np.abs(df['inclination confidence'])
        df['period confidence'] = 1 - np.abs(df['period confidence'])
        df['confidence'] = df['period confidence']
    else:
        df['inclination confidence'] = df['period confidence'] = df['confidence'] = None
    if teff_thresh:
        df = df[(df['Teff'] < 7000) & (df['Teff'] > 0)]

    df.fillna(value=0, inplace=True)
    if err_model_p is not None and calc_errors:
        rounded_inc = np.clip(np.round(df['predicted inclination']).astype(int), a_min=None, a_max=89)
        rounded_inc = np.clip(rounded_inc, a_min=0, a_max=len(err_model_i) - 1)
        inc_errors = err_model_i.iloc[rounded_inc]
        inc_errors_lower, inc_errors_upper = create_errorbars(inc_errors)
        df.loc[:, 'inclination model error lower'] = inc_errors_lower.values
        df.loc[:, 'inclination model error upper'] = inc_errors_upper.values

        rounded_p = np.round(df['predicted period']).astype(int)
        rounded_p = np.clip(rounded_p, a_min=0, a_max=len(err_model_p) - 1)
        p_errors = err_model_p.iloc[rounded_p]
        p_errors_lower, p_errors_upper = create_errorbars(p_errors)
        df.loc[:, 'period model error lower'] = p_errors_lower.values
        df.loc[:, 'period model error upper'] = p_errors_upper.values

    if filter_contaminants:
        all_cont, cont_dict = get_contaminants()
        df['contaminant'] = np.nan
        for cont_name, cont_df in cont_dict.items():
            cont_df['KID'] = cont_df['KIC'].apply(string_to_int)
            df.loc[df['KID'].astype(np.float64).isin(cont_df['KID']), 'contaminant'] = cont_name
        df = df[df['contaminant'].isna()]

    if filter_giants:
        df['main_seq'] = df.apply(giant_cond, axis=1)
        df = df[df['main_seq']==True]
    if filter_eb:
        df = df[df['eb'] == False]
    if filter_non_ps:
        df = df[df['w'] > 0.04]
    print("final size: ", len(df))
    return df

def create_errorbars(err_df):
    lower_bound = np.clip(err_df['mean'], a_min=0, a_max=None)
    upper_bound = err_df['mean']
    return lower_bound, upper_bound

def create_simulation_errors(df):
    err_model_p = pd.read_csv('tables/err_df_p.csv')
    rounded_p = np.round(df['predicted period']).astype(int)
    rounded_p = np.clip(rounded_p, a_min=0, a_max=len(err_model_p) - 1)
    p_errors = err_model_p.iloc[rounded_p]
    p_errors_lower, p_errors_upper = create_errorbars(p_errors)
    df.loc[:, 'simulation error'] = p_errors_lower.values
    return df
def compare_inferences(inferences_list, qs, dir='../imgs'):
    root = inferences_list[0]
    plt.hist(root['sin predicted inclination'], density=True, histtype='step', bins=20, label=f'qs {qs[0]}')

    for i, df in enumerate(inferences_list[1:]):
        root = root.merge(df, on='KID', suffixes=[' 0', f' {i+1}'])
        root[f'inclination_diff_{i+1}'] = np.abs(root['predicted inclination 0'] - root[f'predicted inclination {i+1}'])
        root[f'period_diff_{i+1}'] = np.abs(root['predicted period 0'] - root[f'predicted period {i+1}'])
        plt.hist(root[f'sin predicted inclination {i+1}'], density=True, histtype='step', bins=20, label=f'qs {qs[i+1]}')
        root = root.rename(columns=lambda x: x.rstrip(' 0'))

    plt.legend()
    plt.title("Kepler quarters comparison")
    plt.savefig(f"{dir}/q_compare.png")
    plt.show()
    plt.hexbin(np.arange(len(root)), root[f'inclination_diff_{i+1}'].values, gridsize=100, cmap='viridis', mincnt=1)
    plt.ylabel('Inclination difference')
    plt.savefig(f"{dir}/q_diff_i.png")
    plt.colorbar(label='Density')
    plt.show()
    plt.hexbin(np.arange(len(root)), root[f'period_diff_{i + 1}'], gridsize=100, cmap='viridis', mincnt=1)
    plt.ylabel("Period difference")
    plt.savefig(f"{dir}/q_diff_p.png")
    plt.colorbar(label='Density')
    plt.show()


def inference_diff(df1, df2, values_arr, att='period'):
    """
    plot 2 dataframes
    :param df1:
    :param df2:
    :param values_arr:
    :param att:
    :return:
    """
    df1 = df1.merge(df2, on='KID', suffixes=[' 0', ' 1'])
    matches = []
    for v in values_arr:
        df1[f'{att}_{v}_diff'] = np.abs(df1[f'predicted {att} 0'] - df1[f'predicted {att} 1'])
        matches.append(len(df1[df1[f'{att}_{v}_diff'] < v]))
        print(f"{att} {v} - {len(df1[df1[f'{att}_{v}_diff'] < v])} points")
    plt.plot(values_arr, matches)
    plt.xlabel(f"{att} differnece")
    plt.ylabel("number of points")
    plt.title(f"{att} difference between quarters")
    plt.show()


def filter_df_by_threshold(df1, df2, val, att='period'):
    """
    merge 2 Dataframes based on difference in attribute att
    """
    df1 = df1.merge(df2, on='KID', suffixes=[' 0', ' 1'])
    df1 = df1[np.abs(df1[f'predicted {att} 0'] - df1[f'predicted {att} 1']) < val]
    df1.rename(columns=lambda x: x.rstrip(' 0'), inplace=True)
    df1.drop(columns=df1.filter(like=' 1').columns, inplace=True)

    return df1

def compare_kois(all_kois, sample, merge_on='kepler_name', save_dir='../imgs'):
    """
    compare model inference on kepler object of interest (koi) with results from literature
    :param all_kois: model inference on all kois. Dataframe
    :param sample: sub sample of results to compare. Dataframe
    :param merge_on: column name that the comparison would be based on
    """
    all_kois[merge_on] = all_kois[merge_on].astype(str).apply(lambda x: x.lower().split(" ")[0])
    sample[merge_on] = sample[merge_on].astype(str).apply(lambda x: x.lower().split(" ")[0])
    all_kois = all_kois.sort_values(merge_on)
    sample = sample.sort_values(merge_on)
    all_kois['med predicted period'] = all_kois.groupby(merge_on)['predicted period'].transform('median')
    all_kois['med predicted inclination'] = all_kois.groupby(merge_on)['predicted inclination'].transform('median')

    duplicates_mask = all_kois.duplicated(subset=merge_on, keep='first')
    # Filter the DataFrame to keep only the first occurrence of each group
    all_kois = all_kois[~duplicates_mask]
    all_kois = all_kois[~all_kois['kepler_name'].isnull()]
    # bad = all_kois[all_kois['KID'] == 11709124]
    # print(bad['kepler_name'], type(bad['kepler_name']))
    # print("plotting lightcurves comparison of ", len(sample), " kois")
    # plot_refrences_lc(all_kois, sample, save_dir=save_dir)


    merged_df = all_kois.merge(sample, on=merge_on, suffixes=[' 0', ' 1'])
    merged_df.rename(columns=lambda x: x.rstrip(' 0'), inplace=True)
    merged_df['marker'] = merged_df['eb'].apply(lambda x: '*' if x else 'o')
    # merged_df = merged_df[~merged_df['prot'].isna()]
    prot_df = merged_df[~merged_df['med predicted period'].isnull()].reset_index()
    # prot_df = merged_df[~merged_df['kepler_name'].isnull()].reset_index()
    if 'prot' in prot_df.keys():
        # prot_df = prot_df[~prot_df['prot'].isnull()].reset_index()
        prot_df.to_csv('kois_rotation_ref.csv')
        p_err_sample = np.vstack(prot_df['err_prot'].to_numpy()).T
        # p_err_model = np.vstack([prot_df['period model error lower'].values[None],
        #                          prot_df['period model error upper'].values[None]]).T
        p_err_model = np.vstack([prot_df['sigma error'].values[None] / 2,
                                 prot_df['sigma error'].values[None] / 2]).T
        plot_kois_comparison(prot_df, 'med predicted period', 'prot',
                             err1=p_err_model, err2=p_err_sample, name='period', save_dir=save_dir)
        plot_kois_comparison2(prot_df, 'med predicted period', 'prot',
                              err1=p_err_model, err2=p_err_sample.T, name='period', save_dir=save_dir)

    merged_df = merged_df[merged_df['i'] <= 90].reset_index()
    merged_df = merged_df[~merged_df['med predicted inclination'].isnull()].reset_index()
    inc_err_sample = (np.vstack(merged_df['err_i'].to_numpy()).T).astype(np.float64)
    inc_err_model = np.vstack([prot_df['inclination model error lower'].values[None],
                             prot_df['inclination model error lower'].values[None]]).T
    # merged_df['mean predicted inclination'] = 90 - merged_df['mean predicted inclination']
    plot_kois_comparison(merged_df, 'med predicted inclination', 'i',
                         err1=inc_err_model, err2=inc_err_sample, name='inclination', save_dir=save_dir)
    plot_kois_comparison2(merged_df, 'med predicted inclination', 'i',
                          err1=inc_err_model, err2=inc_err_sample.T, name='inclination', save_dir=save_dir)


def compare_non_consistent_samples(dir_path, kepler_inference, ref_df, ref_name):
    if not os.path.exists(os.path.join(dir_path, 'imgs')):
        os.mkdir(os.path.join(dir_path, 'imgs'))
    save_dir = os.path.join(dir_path, 'imgs')
    for p in os.listdir(dir_path):
        if p.endswith('.npy'):
            kid = int(p.removesuffix('.npy'))
            model_p = kepler_inference[kepler_inference['KID']==kid]['predicted period'].values[0]
            ref_p = ref_df[ref_df['KID']==kid]['Prot'].values[0]
            title = fr'{kid}, $P_{{LightPred}}$: {model_p:.2f}, $P_{{{ref_name}}}$: {ref_p:.2f}'
            print(kid)
            file_path = os.path.join(dir_path, p)
            save_path = f'{save_dir}/{kid}.png'
            show_kepler_sample(file_path, title=title, save_path=save_path, numpy=True, zoom_length=180)


def prepare_kois_sample(paths, indicator='kepler_name'):
    def fill_missing(row):
        if row == '-':
            return None
        return float(row)

    kois = pd.read_csv('tables/kois.csv').reset_index(drop=True)
    kois['kepler_name'] = kois['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])
    duplicates_mask = kois.duplicated(subset='KID', keep='first')
    kois = kois[~duplicates_mask]
    kois.reset_index(drop=True, inplace=True)
    dfs = [pd.read_csv(p) for p in paths]
    ref_names = [p.split('.')[0] for p in paths]
    for i in range(len(ref_names)):
        dfs[i]['reference'] = ref_names[i]
        if 'kepler_name' not in dfs[i].keys():
            # Perform the comparison and retrieve 'kepler_name'
            merged_df = dfs[i].merge(kois, on='KID', how='left')
            # Assign 'kepler_name' from the merged DataFrame to a new column in dfs
            dfs[i]['kepler_name'] = merged_df['kepler_name']
        elif 'KID' not in dfs[i].columns or dfs[i]['KID'].isnull().any():
            dfs[i]['kepler_name'] = dfs[i]['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])
            # Create mapping dictionary from 'kepler_name' to 'KID'
            mapping = kois.set_index('kepler_name')['KID'].to_dict()
            # Map 'KID' based on 'kepler_name' correspondence
            dfs[i]['KID'] = dfs[i]['kepler_name'].map(mapping)
        dfs[i]['kepler_name'] = dfs[i]['kepler_name'].astype(str).apply(lambda x: x.lower().split(" ")[0])

    # sample_kois = pd.concat(dfs).drop_duplicates(subset=indicator, keep='first')
    sample_kois = pd.concat(dfs)
    sample_kois[indicator] = sample_kois[indicator].str.lower()  # Use str.lower() for better performance

    kids = sample_kois.loc[~sample_kois['KID'].isnull(), 'KID'].astype(np.int64)
    kepler_names_kids = kois[kois['KID'].astype(int).isin(kids.tolist())]

    # Remove duplicates based on 'KID' while keeping the first occurrence
    kepler_names_kids = kepler_names_kids.drop_duplicates(subset='KID', keep='first')

    # Merge to replace 'kepler_name' values in sample_kois
    sample_kois = sample_kois.merge(kepler_names_kids[['KID', 'kepler_name']], on='KID', how='left')

    # Update 'kepler_name' values only where 'KID' matches
    sample_kois['kepler_name'] = np.where(sample_kois['KID'].isin(kids), sample_kois['kepler_name_y'],
                                          sample_kois['kepler_name_x'])
    # Drop the redundant columns
    sample_kois.drop(columns=['kepler_name_x', 'kepler_name_y'], inplace=True)

    for column in sample_kois.columns:
        if column.startswith("err"):
            sample_kois[column] = sample_kois[column].apply(string_to_list2)
    if 'err_prot' not in sample_kois.keys():
        sample_kois['err_prot'] = None
    sample_kois['err_prot'] = sample_kois['err_prot'].apply(lambda x: [2, 2] if x is None else x)
    sample_kois['err_i'] = sample_kois['err_i'].apply(lambda x: [10, 10] if x[0] is None else x)
    sample_kois['i'] = sample_kois['i'].apply(fill_missing)
    # sample_kois['i'] = sample_kois['i'].apply(lambda x: x-90 if x > 90 else x)

    return sample_kois[['i', 'prot', 'err_i', 'err_prot', 'kepler_name', 'KID', 'reference']]


def filtered_inference(dfs, val, att='period'):
    filtered_df = dfs[0]
    for i in range(1, len(dfs)):
        filtered_df = filter_df_by_threshold(dfs[0], dfs[i], val, att=att)
    return filtered_df

def compare_references(ref1, ref2, name1, name2, p_att='Prot', save_dir='../imgs'):
    merged_df = ref1.merge(ref2, on='KID')
    p1 = merged_df[f'{p_att}_x']
    p2 = merged_df[f'{p_att}_y']
    acc10 = np.sum(np.abs(p1 - p2) <= p1 * 0.1) / len(merged_df)
    print(f"{name1} to {name2} accuracy 10%: {acc10}")
    plt.scatter(p1, p2, label=f"acc10p = {acc10:.2f}", s=3)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.savefig(f"{save_dir}/{name1}_{name2}.png")
    plt.clf()


def compare_period_on_mock(model_df, ref_df, ref_name='acf'):
    model_df.columns = model_df.columns.str.lower()

    merged_df = pd.merge(model_df, ref_df, left_index=True, right_index=True, suffixes=(' model', ' ref'))
    # print(np.sum(merged_df['period model'] - merged_df['period ref']))
    model_acc = np.sum(np.abs(merged_df['predicted period model'] - merged_df['period model']) <
                       merged_df['period model']/10) / len(merged_df)
    model_acc20 = np.sum(np.abs(merged_df['predicted period model'] - merged_df['period model']) <
                       merged_df['period model'] / 5) / len(merged_df)
    model_avg_error = np.mean(np.abs(merged_df['predicted period model'] - merged_df['period model']))
    ref_acc = np.sum(np.abs(merged_df['predicted period ref'] - merged_df['period ref']) <
                       merged_df['period ref']/10) / len(merged_df)
    ref_acc20 = np.sum(np.abs(merged_df['predicted period ref'] - merged_df['period ref']) <
                       merged_df['period ref']/5) / len(merged_df)
    ref_avg_error = np.mean(np.abs(merged_df['predicted period ref'] - merged_df['period ref']))
    # plt.scatter(merged_df['period model'], merged_df['predicted period model'], label='model')
    plt.scatter(merged_df['period ref'], merged_df['predicted period ref'], label=ref_name)
    plt.xlabel("True (Days)")
    plt.ylabel("Predicted (Days)")
    # plt.legend()
    # plt.title(f'acc10p: {ref_acc:.2f}')
    plt.savefig(f"../mock_imgs/{ref_name}.png")

    plt.show()

    fig, ax = plt.subplots()
    im = ax.scatter(merged_df['predicted period ref'], merged_df['predicted period model'],
                c=merged_df['period confidence'])
    cbar = fig.colorbar(im)
    cbar.ax.set_xlabel('confidence', fontdict={'fontsize': 14})
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(f"../mock_imgs/{ref_name}_period_comparison.png")
    plt.show()

    return model_acc, model_acc20, model_avg_error, ref_acc, ref_acc20, ref_avg_error

def find_non_ps(kepler_inference):
    non_ps = pd.read_csv('Table_2_Non_Periodic.txt')
    non_ps = non_ps.dropna()
    # non_ps = non_ps[non_ps['w'] < 0.06]
    non_ps.to_csv('non_ps.csv')
    ps = pd.read_csv('Table_1_Periodic.txt')
    all = pd.concat([non_ps, ps])
    all.to_csv('all_ps.csv')
    merged_df_mazeh, merged_df_kois, merged_df_no_kois = create_kois_mazeh(kepler_inference, mazeh_path='non_ps.csv',
                                                                           kois_path='kois.csv')
    print(len(merged_df_mazeh))
    plt.hist(merged_df_mazeh['predicted period'])
    plt.show()
    plt.scatter(merged_df_mazeh['Prot'], merged_df_mazeh['predicted period'], c=merged_df_mazeh['confidence'])
    plt.show()
    plt.scatter(merged_df_mazeh['w'], merged_df_mazeh['period confidence'])
    plt.show()


def read_csv_folder(dir_name, filter_thresh=5, att='period',
                    scale=False, calc_errors=True):
    print(f"*** reading files from kepler/{dir_name}")
    dfs = []
    atts = []
    for file in os.listdir(f"{dir_name}"):
        if file.endswith('csv'):
            print(file)
            df = prepare_df(
                pd.read_csv(f"{dir_name}/{file}", on_bad_lines='warn'),
                filter_eb=False, filter_giants=True, filter_non_ps=True, teff_thresh=True,
                scale=scale, calc_errors=calc_errors)
            print("current df len: ", len(df))
            if not len(dfs):
                dfs.append(df)
            else:
                if filter_thresh is not None:
                    filter_df = filter_df_by_threshold(dfs[0], df, filter_thresh, att=att)
                    print('filtered df len: ', len(filter_df))
                else:
                    filter_df = df
                    atts.append(df[['KID', f'predicted {att}']])
                dfs.insert(0,filter_df)
    # Concatenate the remaining rows from all dataframes
    # Get unique 'KID' values from the first dataframe
    unique_kids = dfs[0]['KID'].unique()

    # Filter rows in each dataframe based on the 'KID' values in the first dataframe
    if filter_thresh is not None:
        filtered_dfs = [df[df['KID'].isin(unique_kids)] for df in dfs]
    else:
        filtered_dfs = dfs

    # Merge dataframes based on the 'KID' column
    merged_df = pd.concat(filtered_dfs, ignore_index=True)

    # Take the median among all dataframes for the remaining rows
    if 'qs' in merged_df.columns:
        merged_df['qs'] = merged_df['qs'].apply(lambda x: ast.literal_eval(x))
        merged_df.drop(labels=['qs'], axis=1, inplace=True)
    if 'predicted inclination probability' in merged_df.columns:
        merged_df['predicted inclination probability'] = merged_df['predicted inclination probability'].apply(
            convert_probs_float_list)
        result_df = merged_df.groupby('KID').agg(median_agg)
    else:
        result_df = merged_df.groupby('KID').agg('median')
        std_df = merged_df.groupby('KID').agg('std')
        # plt.hexbin(result_df[f'{att} confidence'], std_df[f'{att}'],
        #            cmap='viridis', mincnt=1, label='Data')
        plt.scatter(np.arange(len(std_df)), std_df[f'predicted {att}'])
        # plt.xlabel('model confidence')
        plt.ylabel(f'Quarter Standard Deviation (Days)')
        plt.colorbar(label='Density')
        plt.savefig(f"../imgs/{att}_std_vs_conf.png")
        plt.close()
    if filter_thresh is None:
        max_diff_df = merged_df.groupby('KID')[f'predicted {att}'].agg(lambda x: x.max() - x.min())
        # max_diff_df.rename({f'predicted {att}': 'max_diff'}, inplace=True)
        result_df = pd.merge(result_df, max_diff_df, on='KID', how='inner', suffixes=['', ' max diff'])
        plt.hexbin(result_df[f'{att} confidence'], result_df[f'predicted {att} max diff'],
                   cmap='viridis', mincnt=1, label='Data')
        plt.xlabel('model confidence')
        plt.ylabel(f'Quarter Max Diff (Days)')
        plt.colorbar(label='Density')
        plt.savefig(f"../imgs/{att}_max_diff_vs_conf.png")
        plt.close()
    print(f"number of samples after filtering with {filter_thresh} days/degrees threshold : {len(result_df)}")
    return result_df


def median_list(lst):
    # Transpose the list of tuples to convert it into a tuple of lists
    lst_transposed = list(zip(*lst))

    # Calculate the median for each list separately
    median_values = [np.median(sublst) for sublst in lst_transposed]

    return median_values
def median_inference(dir_name):
    dfs = []
    for file in os.listdir(f"kepler/{dir_name}"):
        if file.endswith('csv'):
            print(file)
            df = prepare_df(
                pd.read_csv(f"kepler/{dir_name}/{file}", on_bad_lines='warn'),
                filter_eb=False, filter_giants=True, filter_non_ps=True, teff_thresh=False)
            dfs.append(df)
        print(f"number of samples after filtering: {len(dfs[-1])}")
    merged_df = pd.concat(dfs)
    std_df = merged_df.groupby('KID').std().reset_index()
    mean_df = merged_df.groupby('KID').mean().reset_index()
    # plt.hist(std_df['predicted period'])
    # plt.title('std of period predictions on different quarters')
    # plt.show()
    result_df = merged_df.groupby('KID').median().reset_index()
    # results_idx = std_df['predicted period'] < (mean_df['predicted period'] / 3)
    # plt.scatter(mean_df['predicted period'], std_df['predicted period'])
    # plt.show()
    results_idx = std_df['predicted period'] < 3

    results_idx_conf = mean_df['inclination confidence'] > 0.9

    plt.hist(result_df['inclination confidence'], histtype='step', label='inclination')
    plt.hist(result_df['confidence'], histtype='step', label='period')

    plt.title('median confidence histogram - kepler')
    plt.show()
    result_df = result_df[results_idx]
    print("final dataframe length: ", len(result_df))
    return result_df


def create_hist_factor(true, predicted, bins=90):
    # Calculate histograms
    hist_true, bins_true = np.histogram(true, bins=bins, density=False)
    hist_predicted, bins_predicted = np.histogram(predicted, bins=bins,
                                                  density=False)
    factors = np.ones(bins)
    # Initialize an array to store weights
    weights = np.zeros_like(predicted, dtype=float)
    # Calculate weights: true_val / predicted_val for each bin
    for i in range(len(predicted)):
        val = int(predicted[i])
        f = hist_true[val] / hist_predicted[val]
        factors[val] = f
        weights[i] = f

    np.save('hist_factors.npy', factors)


    # Plot the weighted histogram for the "predicted" array
    plt.figure(figsize=(10, 6))
    plt.hist(np.cos(predicted*np.pi/180), bins=40, weights=weights, alpha=0.7, label='Weighted Histogram', histtype='step')

    # plt.hist(predicted, histtype='step', bins=bins, label='predicted', )
    plt.hist(np.cos(true*np.pi/180), histtype='step',bins=40, label='true')
    # plt.hist(predicted, histtype='step', bins=bins, label='predicted', )
    # plt.bar(edges[:-1], new_hist,  align='center', alpha=0.3, label='Modified Histogram')
    plt.legend()
    plt.show()
    return
    # print('image saved at :', save_path)

def calc_consistency_curve(model_df, threshold, x_name, name, save_dir='../imgs'):
    model_df['p_diff'] = np.abs(model_df['predicted period'] - model_df[x_name])
    model_df['consistent'] = model_df.apply(lambda x: (x['p_diff'] < x[x_name] * threshold)
                                                      | (x['p_diff'] <= 1),
                                                        axis=1)
    bins = pd.cut(model_df[x_name],
                  bins=range(int(model_df[x_name].min()),
                             int(model_df[x_name].max()) + 2), right=False)
    grouped = model_df.groupby(bins)['consistent'].agg(['sum', 'count'])
    grouped['fraction'] = grouped['sum'] / grouped['count']
    bin_centers = [(bin.left + bin.right) / 2 for bin in grouped.index]
    plt.scatter(bin_centers, grouped['fraction'], color='blue')
    plt.xlabel('Period (Days)')
    plt.ylabel('Fraction of consistent values')
    plt.grid(True)
    plt.savefig(f'{save_dir}/consistency_{name}.png')
    plt.show()

def find_non_consistent_samples(df, p_att, ref_name, thresh_val, save_dir):
    df['p_diff'] = np.abs(df['predicted period'] - df[p_att])
    calc_consistency_curve(df, 0.4, p_att, ref_name)
    non_consistent_samples = df[(df['p_diff'] > df[p_att] * 0.4) & (df['p_diff'] > 1)]
    group2 = non_consistent_samples[non_consistent_samples[p_att] < thresh_val]
    group1 = non_consistent_samples[non_consistent_samples[p_att] > thresh_val]
    consistent_samples = df[df['p_diff'] < df[p_att] * 0.4]
    print("number of non consistent samples high periods: ", len(group1), "low periods: ", len(group2))

    plt.scatter(consistent_samples[p_att], consistent_samples['predicted period'], label='consistent')
    plt.scatter(group1[p_att], group1['predicted period'], label='non consistent group1')
    plt.scatter(group2[p_att], group2['predicted period'], label='non consistent group2')
    plt.legend()
    plt.xlabel(f'Period {ref_name}')
    plt.ylabel("Period LightPred")
    plt.savefig(f'{save_dir}/non_consistent_groups.png')
    plt.show()

    label1 = f"avg {consistent_samples['total error'].mean():.2f} Days"
    label2 = f"avg {group1['total error'].mean():.2f} Days"
    label3 = f"avg {group2['total error'].mean():.2f} Days"
    plt.hist(consistent_samples['total error'], density=True, histtype='step', label=label1)
    plt.hist(group1['total error'], density=True, histtype='step', label=label2)
    plt.hist(group2['total error'], density=True, histtype='step', label=label3)
    plt.legend()
    plt.xlabel('Total Error (Days)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/non_consistent_errors.png')
    plt.show()

    group1.to_csv('tables/non_consistent_group1.csv')
    group2.to_csv('tables/non_consistent_group2.csv')


def clusters_inference(kepler_inference, cluster_df, refs,
                       refs_names, ref_markers=['*', '+'], save_dir='../imgs'):
    # Merge dataframes and rename columns
    merged_df = cluster_df.merge(kepler_inference, on='KID')
    merged_df.rename(columns=lambda x: x.rstrip('_x'), inplace=True)

    # Define colormap and normalize 'period confidence' values
    cmap = plt.cm.viridis
    norm = plt.Normalize(merged_df['period confidence'].min(), merged_df['period confidence'].max())

    # Calculate period model error
    p_err_model = np.vstack([merged_df['period model error lower'].values[None],
                             merged_df['period model error lower'].values[None]])

    # Create a new figure and axis object
    fig, ax = plt.subplots()

    # Plot reference points
    ax.plot(merged_df['Prot'], merged_df['Prot'], color='r')  # Diagonal line for reference
    std = merged_df['predicted period'].std()
    std_high = merged_df[merged_df['period confidence'] > 0.9]['predicted period'].std()
    sc = ax.scatter(merged_df['Prot'], merged_df['predicted period'], c=merged_df['period confidence'],
                    label=f'model std: {std:.2f} ({std_high:.2f})', cmap=cmap, norm=norm)

    # Plot reference data
    for name, ref, mark in zip(refs_names, refs, ref_markers):
        suffix = '_' + name
        merged_df = merged_df.merge(ref, on='KID', suffixes=(None, suffix))
        std = merged_df[f'Prot_{name}'].std()
        ax.scatter(merged_df['Prot'], merged_df[f'Prot_{name}'], label=f'{name} std: {std:.2f}',
                   marker=mark, cmap=cmap, norm=norm)

    # Add colorbar with correct range
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Period Confidence')

    plt.legend()
    plt.xlabel("reference period")
    plt.ylabel("predicted period")
    plt.savefig(f'{save_dir}/clusters_meibom.png')
    plt.show()


def compare_distributions_to_mock(kepler_inference, mock_eval, save_dir='../imgs'):
    fig, ax = plt.subplots(1,2, figsize=(12,8))
    ax[0].hist(kepler_inference['predicted period'], histtype='step', density=True, bins=40)
    ax[0].hist(mock_eval['Period'], histtype='step', density=True, bins=40)
    ax[0].set_xlabel('Days')
    ax[1].hist(kepler_inference['predicted inclination'], histtype='step', density=True, bins=40)
    ax[1].hist(mock_eval['Inclination'], histtype='step', density=True, bins=40)
    ax[1].set_xlabel('Degrees')
    plt.savefig('../imgs/dist_comparison.png')
    plt.close()

def compare_period_distributions(kepler_inference, refs, refs_names, save_name):
    plt.hist(kepler_inference['predicted period'], histtype='step', bins=40, density=True,
             label=f'({len(kepler_inference)} samples)')
    for name, ref in zip(refs_names, refs):
        plt.hist(ref['Prot'], histtype='step', bins=40, density=True,
                 label=f'{name} - ({len(ref)} samples)')
    plt.xlabel('Days')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../imgs/{save_name}.png')
    plt.show()

def McQ_constrains(df):
    df = df[df['Teff'] < 6500]
    kois = pd.read_csv('tables/kois.csv')
    kois_confirmed = kois[kois['koi_disposition'] != 'FALSE POSITIVE']
    df['koi'] = df['KID'].isin(kois_confirmed['KID']).astype(bool)
    eb = pd.read_csv('tables/kepler_eb.txt')
    df['eb'] = df['KID'].isin(eb['KID']).astype(bool)
    df = df[df['koi']==False]
    df = df[df['eb']==False]
    return df


def get_sigma_error(model_df, regex='^diff_\d+_\d+$'):
    diff_columns = model_df.filter(regex=regex)

    # Step 2: Define a function to fit a Gaussian and calculate the standard deviation for each row
    def fit_gaussian_and_get_sigma(row):
        # Drop NaN values in the row
        values = row.dropna().values
        if len(values) > 1:  # Need at least two data points to fit a Gaussian
            mu, sigma = stats.norm.fit(values)
            return sigma
        else:
            return np.nan  # Return NaN if there are not enough data points

    model_df['sigma error'] = diff_columns.apply(fit_gaussian_and_get_sigma, axis=1)
    return model_df

def compare_consistency(model_dir, acf_dir, gps_dir, model_path=None, acf_path=None, gps_path=None):
    if gps_path is None:
        if gps_dir is not None:
            gps_df = get_consistency_df(gps_dir, target_att='predicted period', valid_thresh=0.7)
            gps_df.to_csv('tables/kepler_gps_pred.csv', index=False)
        else:
            gps_df = None
    else:
        gps_df = pd.read_csv(gps_path)
    if acf_path is None:
        match = re.search(r'exp(\d+)', acf_dir)
        if match:
            exp_num = match.group(1)
        else:
            exp_num = '?'
        print("acf exp num: ", exp_num)
        try:
            acf_df = get_consistency_df(acf_dir, target_att='acf_p', valid_thresh=0.1)
        except KeyError:
            acf_df = get_consistency_df(acf_dir, target_att='predicted acf_p', valid_thresh=0.1)
        acf_df.to_csv(f'tables/kepler_acf_pred_exp{exp_num}.csv', index=False)
    else:
        acf_df = pd.read_csv(acf_path)
    if model_path is None:
        match = re.search(r'exp(\d+)', model_dir)
        if match:
            exp_num = match.group(1)
        else:
            exp_num = '?'
        print("model exp num: ", exp_num)
        model_df = get_consistency_df(model_dir, target_att='predicted period', prepare=True, add_conf=True)
        print(f"*****final size model df: {len(model_df)}*******")
        exit()
        model_df.to_csv(f'tables/kepler_model_pred_exp{exp_num}.csv', index=False)
    else:
        model_df = pd.read_csv(model_path)
    # model_df = model_df[model_df['mean_period_confidence']>0.86]
    model_df['relative error'] = model_df['sigma error'] / (model_df['predicted period']+1e-3)
    try:
        acf_df['relative error']= acf_df['sigma error'] / (acf_df['acf_p'].abs()+1e-3)
    except KeyError:
        acf_df['relative error'] = acf_df['sigma error'] / (acf_df['predicted acf_p'].abs() + 1e-3)
    acf_df = McQ_constrains(acf_df)
    model_constrained_df = McQ_constrains(model_df)

    acf_valid_df = acf_df[acf_df['valid']]
    # gps_valid_df = gps_df[gps_df['predicted period'] > 0.8]
    model_df['valid'] = model_df['KID'].isin(acf_valid_df['KID']).astype(bool)
    model_df_valid = model_df[model_df['valid']]
    # model_df_gps = model_df[model_df['KID'].isin(gps_df['KID']).astype(bool)]
    # model_df_gps_valid = model_df[model_df['KID'].isin(gps_valid_df['KID']).astype(bool)]
    acf_valid_high_acc = acf_valid_df[acf_valid_df['total_acc'] == 21]
    acf_valid_high_acc = McQ_constrains(acf_valid_high_acc)
    print("number of samples acf valid ", len(acf_valid_df), "subsample with high consistency ", len(acf_valid_high_acc))
    mean_std_model = model_df['relative error'].mean()
    mean_std_model_constrained = model_constrained_df['relative error'].mean()
    mean_std_model_valid = model_df_valid['relative error'].mean()
    mean_std_acf = acf_df['relative error'].mean()
    mean_std_acf_valid = acf_valid_df['relative error'].mean()
    print("model/model_constrained/model_valid/acf/acf_valid average std error: ",
          mean_std_model, mean_std_model_constrained, mean_std_model_valid,
          mean_std_acf, mean_std_acf_valid)
    mean_std_df = pd.DataFrame({"LightPred": [mean_std_model],
                                "LightPred McQ14 constrained": [mean_std_model_constrained],
                                "LightPred acf valid subsample": [mean_std_model_valid],
                               "ACF McQ14 constrained": [mean_std_acf],
                                "ACF valid": [mean_std_acf_valid]})
    mean_std_df.to_csv("tables/mean_std_df.csv", index=False)
    model_high_acc = model_df[model_df['sigma error'] < 2.2]
    print("high acc number of samples: ", len(model_high_acc))
    # plot_consistency_vs_conf(model_df)
    plot_consistency_hist(model_df, acf_df, y_label='sigma error')
    # plot_consistency_hist(model_df, acf_df, y_label='relative error')
    plot_consistency_hist(model_df_valid, acf_valid_df,  y_label='sigma error',
                          suffix='valid', plot_rel=False)
    # plot_consistency_hist(model_df_valid, acf_valid_df, y_label='relative error', suffix='valid')
    # plot_consistency_hist(model_df_gps, gps_df, suffix='gps')
    # plot_consistency_hist(model_df_gps_valid, gps_valid_df, suffix='gps_valid')
    plot_difference_hist(model_df)

    # plot_confusion_matrix(model_df, model_name='LightPred', save_name='model_confusion')
    # plot_confusion_matrix(acf_valid_df, model_name='ACF', save_name='acf_confusion')
    # print(len(model_df[model_df['period confidence'] > 0.95]))
    # plot_confusion_matrix(model_df[model_df['period confidence'] > 0.95], model_name='LightPred', save_name='model_confusion_95')


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
def get_consistency_df(dfs_dir, target_att, prepare=False, thresh=6, add_conf=False, valid_thresh=0):
    for i, file in enumerate(os.listdir(dfs_dir)):
        print("analyzing file ", file)
        if file.endswith('csv'):
            lag = re.search(r'\d+', file).group(0)
            if prepare:
                df = prepare_df(pd.read_csv(f'{dfs_dir}/{file}'),
                                filter_giants=True, filter_eb=False,
                                teff_thresh=True, filter_non_ps=False)
            else:
                df = pd.read_csv(f'{dfs_dir}/{file}')
            if not add_conf and ('double_peaked' not in df.columns):
                df['double_peaked'] = False
            df['valid'] = df[target_att] >= valid_thresh
            if i == 0:
                tot_df = df
            else:
                selected_cols = ['KID', target_att, 'valid', 'double_peaked'] if not add_conf else \
                    ['KID', target_att,
                     'period confidence',
                     'inclination confidence',
                     'valid',
                     ]
                tot_df = tot_df.merge(df[selected_cols], on='KID', suffixes=['', f'_{lag}'])
                duplicates_kids = tot_df['KID'].duplicated()
                print("duplicates ", duplicates_kids.sum())

                tot_df['valid'] = tot_df['valid'] & tot_df[f'valid_{lag}']
                if not add_conf:
                    tot_df['double_peaked'] = tot_df['double_peaked'] + tot_df[f'double_peaked_{lag}']
                lag_acc10 = np.zeros(len(tot_df))
                lag_acc20 = np.zeros(len(tot_df))
                lag_acc30 = np.zeros(len(tot_df))
                lag_acc = np.zeros(len(tot_df))
                for j in range(i):
                    first_att = target_att if not j else f'{target_att}_{j}'
                    j_lag_diff = tot_df[first_att] - tot_df[f'{target_att}_{lag}']
                    j_lag_acc10 = np.abs(tot_df[f'{target_att}_{lag}'] * 0.1) > j_lag_diff
                    lag_acc10 += j_lag_acc10
                    j_lag_acc20 = np.abs(tot_df[f'{target_att}_{lag}'] * 0.2) > j_lag_diff
                    lag_acc20 += j_lag_acc20
                    j_lag_acc30 = np.abs(tot_df[f'{target_att}_{lag}'] * 0.3) > j_lag_diff
                    lag_acc30 += j_lag_acc30
                    j_lag_acc = j_lag_diff < thresh
                    lag_acc += j_lag_acc
                    tot_df[f'diff_{j}_{lag}'] = j_lag_diff
                tot_df[f'acc10_{lag}'] = lag_acc10
                tot_df[f'acc20_{lag}'] = lag_acc20
                tot_df[f'acc30_{lag}'] = lag_acc30
                tot_df[f'acc_{lag}'] = lag_acc
                # acf_df.rename(columns=lambda x: x.rstrip('_0'), inplace=True)
    print("aggregating acc predictions...")
    if not add_conf:
        tot_df['double_peaked'] = tot_df['double_peaked'] / (i+1)
        tot_df['double_peaked'] = tot_df['double_peaked'] > 0.5

    # tot_df['total_acc10'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc10' in k]), axis=1)
    # tot_df['total_acc20'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc20' in k]), axis=1)
    # tot_df['total_acc30'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc30' in k]), axis=1)
    tot_df['total_acc'] = tot_df.apply(lambda x: np.sum([x[k] for k in x.keys() if 'acc_' in k]), axis=1)
    # print("diff..")
    # tot_df['max_diff'] = tot_df.apply(lambda x: max([x[k] for k in x.keys() if 'diff' in k]), axis=1)
    tot_df['mean_diff'] = tot_df.apply(lambda x: np.mean([x[k] for k in x.keys() if 'diff' in k]), axis=1)
    # tot_df['std_p'] = tot_df.apply(lambda x: np.std([x[k] for k in x.keys() if target_att in k]), axis=1)
    if add_conf:
        tot_df['mean_period_confidence'] = (tot_df.apply
                                            (lambda x: np.mean([x[k] for k in x.keys() if
                                                            'confidence' in k]), axis=1))
    print("sigma error...")
    tot_df = get_sigma_error(tot_df)
    return tot_df

def multi_bin_error_correlation(model_df, ref_cat, coarse_name, coarse_bins, fine_name, points_thresh=30,
                                save_dir='../imgs'):
    if ref_cat is not None:
        merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                            'sigma error', 'mean_period_confidence',
                                            'total error', 'relative sigma error']], on='KID')
    else:
        merged_df = model_df
    ncols = int(np.ceil(len(coarse_bins) / 2))
    fig, axis = plt.subplots(nrows=2, ncols=ncols, figsize=(24, 12))

    # Flatten the axis array for easy indexing
    axis = axis.flatten()

    for i, c_bin in enumerate(coarse_bins):
        if i < len(coarse_bins) - 1:
            reduced_df = merged_df[(merged_df[coarse_name] > c_bin) & (merged_df[coarse_name] < coarse_bins[i + 1])]
            title = f'{coarse_name} Bin: {c_bin} - {coarse_bins[i + 1]}'
        else:
            reduced_df = merged_df[(merged_df[coarse_name] > c_bin)]
            title = f'{coarse_name} Bin: > {c_bin}'
        print(f'{coarse_name} - {c_bin} {len(reduced_df)} samples')
        x_vals = reduced_df[fine_name].values
        min_x, max_x = x_vals.min(), x_vals.max()
        bins = np.linspace(min_x, max_x, 100)
        bin_labels = bins[:-1]
        reduced_df[f'{fine_name}_bin'] = pd.cut(reduced_df[fine_name], bins=bins, labels=bin_labels)
        bin_counts = reduced_df[f'{fine_name}_bin'].value_counts().reset_index()
        bin_counts.columns = [f'{fine_name}_bin', 'count']
        valid_bins = bin_counts[bin_counts['count'] >= points_thresh][f'{fine_name}_bin']
        avg_total_error_per_bin = reduced_df[reduced_df[f'{fine_name}_bin'].isin(valid_bins)].groupby(f'{fine_name}_bin')[
            'relative sigma error'].mean().reset_index()
        ax = axis[i]
        ax.scatter(avg_total_error_per_bin[f'{fine_name}_bin'], avg_total_error_per_bin['relative sigma error'], s=100)
        ax.set_title(title)
        ax.set_xlabel(fine_name)
        ax.set_ylabel('Average relative Error')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{fine_name}_{coarse_name}_subplots.png")
    plt.show()

def bin_values(values: object, num_bins: object, threshold: object = 100) -> object:
    """
    Bins an array of values into a specified number of bins, aiming to keep the number of points in each bin similar.

    Parameters:
    - values (array-like): The array of values to bin.
    - num_bins (int): The desired number of bins.

    Returns:
    - bins (numpy.ndarray): The bin number for each value in the sorted array.
    - bin_edges (numpy.ndarray): The edges of the bins.
    """
    # Sort the values
    sorted_values = np.sort(values)

    # Calculate initial bin edges (equal-width bins)
    bin_edges = np.linspace(np.min(sorted_values), np.max(sorted_values), num_bins + 1)

    # Digitize the data to assign bins
    bins = np.digitize(sorted_values, bin_edges)

    # Calculate number of points in each bin
    counts_per_bin = np.bincount(bins)
    i = 0
    # Adjust bin edges or combine adjacent bins to achieve similar counts per bin
    while max(counts_per_bin) - min(counts_per_bin) > threshold:
        # Find bins with the most and least points
        max_bin = np.argmax(counts_per_bin)
        min_bin = np.argmin(counts_per_bin)


        # Adjust bin edge between these bins
        bin_edges[min_bin] = (bin_edges[min_bin] + bin_edges[min_bin + 1]) / 2

        # Recalculate bins and counts
        bins = np.digitize(sorted_values, bin_edges)
        counts_per_bin = np.bincount(bins)
        i += 1
        if i==100:
            break

    return bins, bin_edges


def equal_bin_samples(values, num_bins):
    # Sort the values
    sorted_values = np.sort(values)

    # Calculate the indices that will divide the sorted array into equal bins
    bin_indices = np.linspace(0, len(values), num_bins + 1, endpoint=True).astype(int)

    # Use these indices to get the bin edges from the sorted values
    bin_edges = sorted_values[bin_indices]

    # Assign each value to a bin based on these bin edges
    bin_assignments = np.digitize(values, bin_edges, right=True)

    # Return the bin assignments
    return bin_assignments
def error_correlation_kepler(model_df, ref_cat, ref_id, x_names, units,
                      y_name='total error', points_thresh=30, n_rows=2, n_cols=None, save_dir='../imgs'):
    if ref_cat is not None:
        merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence',
                                        'total error', 'relative sigma error']], on='KID')
    else:
        merged_df = model_df
    n_cols = n_cols or ((len(x_names) + 1) // n_rows)
    figsize = (16*(n_rows+1), 9*n_rows) if ((n_rows > 1) or (n_cols > 1)) else (16,9)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]))
    if (n_rows > 1) or (n_cols > 1):
        axes = axes.flatten()
    name = ''
    for i, (x_name, unit) in enumerate(zip(x_names, units)):
        print(x_name)
        ax = axes[i] if ((n_rows > 1) or (n_cols > 1)) else axes
        x_vals = merged_df[x_name].dropna()
        y_vals = merged_df[y_name].dropna()
        
        # Calculate Pearson correlation coefficient

        min_x, max_x = x_vals.min(), x_vals.max()
        bins = np.linspace(min_x, max_x, 200)
        bin_labels = bins[:-1]
        merged_df[f'{x_name}_bin'] = pd.cut(merged_df[x_name], bins=bins, labels=bin_labels)
        bin_counts = merged_df[f'{x_name}_bin'].value_counts().reset_index()
        bin_counts.columns = [f'{x_name}_bin', 'count']
        valid_bins = bin_counts[bin_counts['count'] >= points_thresh][f'{x_name}_bin']
        avg_total_error_per_bin = merged_df[merged_df[f'{x_name}_bin'].isin(valid_bins)].groupby(f'{x_name}_bin')[
           y_name].mean().reset_index()
        avg_total_error_per_bin = avg_total_error_per_bin.dropna()
        correlation, _ = pearsonr(avg_total_error_per_bin[f'{x_name}_bin'], avg_total_error_per_bin[y_name])
        ax.scatter(avg_total_error_per_bin[f'{x_name}_bin'], avg_total_error_per_bin[y_name], s=100,
                    label=f'Correlation: {correlation:.2f}')
        if unit is not None:
            label = rf"${x_name}$ (${unit}$)"
        else:
            label = rf"${x_name}$"
        ax.set_xlabel(label)
        ax.set_ylabel(rf"{y_name}")
        ax.legend()
        name = name + f'_{x_name}'
        # plt.savefig(f'{save_dir}/{x_name}_{y_name}.png')
        # plt.close()
        #
        # plt.hexbin(merged_df[x_name], merged_df[y_name], mincnt=1)
        # plt.xlabel(x_name)
        # plt.ylabel(r"$\frac{E_{rel}}{Conf}$")
        # plt.savefig(f'{save_dir}/{x_name}_{y_name}_hexbin.png')
        # plt.close()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_correlation_{name}.png')
    plt.close()

def pixel_binning(model_df, ref_cat, px_name, py_name, x_name, y_name, resolution, save_dir='../imgs'):
    # Merge the dataframes on 'KID'
    merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence',
                                        'total error', 'relative sigma error']], on='KID')

    # Extract px and py values
    px_vals = merged_df[px_name]
    py_vals = merged_df[py_name]

    # Define bins
    min_px, max_px = px_vals.min(), px_vals.max()
    min_py, max_py = py_vals.min(), py_vals.max()
    x_bins = np.arange(min_px, max_px + resolution, resolution)
    y_bins = np.arange(min_py, max_py + resolution, resolution)

    # Assign bins to each row
    merged_df[f'{px_name}_bin'] = pd.cut(merged_df[px_name], bins=x_bins, labels=x_bins[:-1])
    merged_df[f'{py_name}_bin'] = pd.cut(merged_df[py_name], bins=y_bins, labels=y_bins[:-1])

    # Drop rows with NaN bins (outside defined bin ranges)
    merged_df = merged_df.dropna(subset=[f'{px_name}_bin', f'{py_name}_bin'])

    # Calculate average x_name and y_name per bin
    avg_results = merged_df.groupby([f'{px_name}_bin', f'{py_name}_bin']).agg(
        avg_x=(x_name, 'mean'),
        avg_y=(y_name, 'mean')
    ).reset_index()

    # Plotting
    plt.scatter(avg_results['avg_x'], avg_results['avg_y'])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(f'{save_dir}/{x_name}_{y_name}_pixel_bin.png')
    plt.close()


def plot_avg_error_3d(model_df, ref_cat, x_name, y_name, points_thresh, ref_id, z_name='total error', save_dir='../imgs'):
    if ref_cat is not None:
        merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                            'sigma error', 'mean_period_confidence',
                                            'total error', 'relative sigma error']], on='KID')
    else:
        merged_df = model_df

    merged_df[x_name] = merged_df[x_name] /merged_df[x_name].max()
    merged_df[y_name] = merged_df[y_name] / merged_df[y_name].max()

    x_vals = merged_df[x_name]
    y_vals = merged_df[y_name]
    # Define bins
    min_x, max_x = x_vals.min(), x_vals.max()
    min_y, max_y = y_vals.min(), y_vals.max()
    x_bins = np.linspace(min_x, max_x, 200)
    y_bins = np.linspace(min_y, max_y, 200)

    # Create bin labels
    x_bin_labels = x_bins[:-1]
    y_bin_labels = y_bins[:-1]

    # Bin the data
    merged_df[f'{x_name}_bin'] = pd.cut(merged_df[x_name], bins=x_bins, labels=x_bin_labels)
    merged_df[f'{y_name}_bin'] = pd.cut(merged_df[y_name], bins=y_bins, labels=y_bin_labels)

    # Count points in each 2D bin
    bin_counts = merged_df.groupby([f'{x_name}_bin', f'{y_name}_bin']).size().reset_index(name='count')
    valid_bins = bin_counts[bin_counts['count'] >= points_thresh][[f'{x_name}_bin', f'{y_name}_bin']]

    # Calculate average total error per valid bin
    avg_total_error_per_bin = merged_df[
        merged_df[[f'{x_name}_bin', f'{y_name}_bin']].apply(tuple, axis=1).isin(valid_bins.apply(tuple, axis=1))
    ].groupby([f'{x_name}_bin', f'{y_name}_bin'])[z_name].mean().reset_index()

    # Convert bin labels to floats for plotting
    avg_total_error_per_bin[f'{x_name}_bin'] = avg_total_error_per_bin[f'{x_name}_bin'].astype(float)
    avg_total_error_per_bin[f'{y_name}_bin'] = avg_total_error_per_bin[f'{y_name}_bin'].astype(float)

    # Remove NaN values after binning
    avg_total_error_per_bin = avg_total_error_per_bin.dropna()

    # Create a grid for interpolation
    xi = np.linspace(avg_total_error_per_bin[f'{x_name}_bin'].min(), avg_total_error_per_bin[f'{x_name}_bin'].max(),
                     100)
    yi = np.linspace(avg_total_error_per_bin[f'{y_name}_bin'].min(), avg_total_error_per_bin[f'{y_name}_bin'].max(),
                     100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data
    zi = griddata((avg_total_error_per_bin[f'{x_name}_bin'], avg_total_error_per_bin[f'{y_name}_bin']),
                  avg_total_error_per_bin[z_name], (xi, yi), method='cubic')

    # Perform PCA
    scaler = StandardScaler()
    pca = PCA(n_components=3)
    pca_data = scaler.fit_transform(avg_total_error_per_bin[[f'{x_name}_bin', f'{y_name}_bin', z_name]])
    pca_result = pca.fit_transform(pca_data)

    # Plotting in 3D
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Surface plot
    # surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.8)
    #
    # # PCA vectors
    # pca_origin = np.mean(pca_data, axis=0)
    # print("pca 1: ", pca.components_[0])
    #
    # # for i, vector in enumerate(pca.components_):
    # #     ax.quiver(pca_origin[0], pca_origin[1], pca_origin[2],
    # #               vector[0], vector[1], vector[2],
    # #               color=['r', 'g', 'b'][i], label=f'PC{i + 1}', length=0.2)
    #
    # ax.set_xlabel(x_name)
    # ax.set_ylabel(y_name)
    # ax.set_zlabel(r"$\frac{E_{rel}}{Conf}$")
    #
    # # Color bar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    #
    # # Legend for PCA vectors
    # ax.legend()
    #
    # plt.savefig(f'{save_dir}/{x_name}_{y_name}_{ref_id}_surface_pca.png')
    # plt.close()

    avg_total_error_per_bin.dropna()
    pca_vals = avg_total_error_per_bin[f'{x_name}_bin'] * pca.components_[0][0] \
                + avg_total_error_per_bin[f'{y_name}_bin'] * pca.components_[0][1]
    err_vals = avg_total_error_per_bin[z_name]
    correlation, _ = pearsonr(pca_vals, err_vals)

    fig, axis = plt.subplots()
    axis.scatter(pca_vals, err_vals,
                label=rf"$\alpha={pca.components_[0][0]:.2f}$" + "\n" +
                      rf"$\beta={pca.components_[0][1]:.2f}$" + "\n" +
                      f"Correlation {correlation:.2f}")
    xlabel = rf"$\alpha*{x_name}$ + $\beta*{y_name}$"
    axis.set_xlabel(xlabel)
    axis.set_ylabel('total error')
    axis.legend()
    plt.savefig(f'{save_dir}/{x_name}_{y_name}_{ref_id}_2d_pca.png')
    plt.close()
    return avg_total_error_per_bin, pca, correlation

def multiplot_avg_error_3d(model_df,
                           ref_cat,
                           x_names,
                           y_name,
                           points_thresh,
                           ref_id,
                           z_name='total error',
                           save_dir='../imgs'):
    n_rows = 2
    n_cols = (len(x_names)+1) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 18))
    axes = axes.flatten()
    for i, x_name in enumerate(x_names):
        print(x_name)
        avg_total_error_per_bin, pca, cor = plot_avg_error_3d(model_df, ref_cat, x_name, y_name, points_thresh,
                                   ref_id, z_name=z_name)
        axes[i].scatter(avg_total_error_per_bin[f'{x_name}_bin'] * pca.components_[0][0]
                     + avg_total_error_per_bin[f'{y_name}_bin'] * pca.components_[0][1],
                     avg_total_error_per_bin[z_name],
                     label=rf"$\alpha={pca.components_[0][0]:.2f}$" + "\n" +
                           rf"$\beta={pca.components_[0][1]:.2f}$" + "\n" +
                        f"Correlation {cor:.2f}")
        xlabel = rf"$\alpha*{x_name}$ + $\beta*{y_name}$"
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(z_name)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3d_multiplot_error_{z_name}.png')
    plt.close()


def error_correlation_lkg(model_df, ref_cat, save_dir='../imgs'):
    merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence',
                                        'total error', 'relative sigma error']], on='KID')
    high_conf = merged_df[merged_df['mean_period_confidence'] > 0.94]
    plt.scatter(merged_df['RVel'], merged_df['predicted period'], c=merged_df['relative sigma error'])
    plt.xlabel('RV')
    plt.ylabel('Prot (Days)')
    plt.colorbar(label='relative obs error')
    plt.close()

    plt.scatter(np.abs(merged_df['RVel']), merged_df['predicted period'], c=merged_df['mean_period_confidence'])
    plt.xlabel('|RV|')
    plt.ylabel('Prot (Days)')
    plt.colorbar(label='model confidence')
    plt.close()

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter_plot = sns.scatterplot(data=merged_df,
                                   x=merged_df['RVel'].abs(),
                                   y='predicted period',
                                   hue='Comp', palette='viridis', s=20)


    # Customize plot
    plt.xlabel('|RVel|')
    plt.ylabel('predicted period')
    plt.legend()
    plt.show()

    plt.hexbin(merged_df['kmag'], merged_df['relative sigma error'], mincnt=1, gridsize=50)
    plt.xlabel('kepler magnitude')
    plt.ylabel('relative obs error (Days)')
    plt.savefig(f'{save_dir}/kmag_err_lkg.png')
    plt.close()

    plt.hexbin(merged_df['kmag'], merged_df['mean_period_confidence'], mincnt=1, gridsize=50)
    plt.xlabel('kepler magnitude')
    plt.ylabel('relative obs error (Days)')
    plt.savefig(f'{save_dir}/kmag_conf_lkg.png')
    plt.close()

    scatter_binned_df(merged_df, 'kmag', 'total error', [10,11,12,13,14,15,16],
                      bin_labels=['10-11', '11-12', '12-13', '13-14', '14-15', '15-16'],
                      bar=False, save_name='kmag_err_lkg')


def error_correlation_lamost(model_df, ref_cat, save_dir='../imgs'):
    merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence',
                                        'total error', 'relative sigma error']], on='KID')
    active = merged_df[merged_df['logR+HK'] > -4.69]
    inactive = merged_df[(merged_df['logR+HK'] > -5.167) & (merged_df['logR+HK'] < -4.69)]
    very_inactive = merged_df[(merged_df['logR+HK'] < -5.167)]
    for df, name in zip([active, inactive, very_inactive], ['active', 'inactive', 'very inactive']):
        print(f"******{name}********")
        print(len(df), 'points')
        med_error = df['sigma error'].mean()
        print(' error: ', med_error)
        rel_error = df['relative sigma error'].mean()
        print(' relative sigma error: ', rel_error)
        med_conf = df['mean_period_confidence'].mean()
        print(' conf: ', med_conf)
        tot = df['total error'].mean()
        print(' total error: ', tot)
    plt.hexbin(merged_df['logR+HK'], merged_df['sigma error'], mincnt=1)
    plt.xlabel(r"$log(R_{HK})$")
    plt.ylabel("Obs. Error (Days)")
    plt.savefig(f'{save_dir}/sigma_rhk_lamost.png')
    plt.close()
    plt.hexbin(merged_df['logR+HK'], merged_df['relative sigma error'], mincnt=1)
    plt.xlabel(r"$log(R_{HK})$")
    plt.ylabel("relative Obs. Error (Days)")
    plt.savefig(f'{save_dir}/sigma_rhk_lamost.png')
    plt.close()
    plt.hexbin(merged_df['logR+HK'], merged_df['mean_period_confidence'], mincnt=1)
    plt.xlabel(r"$log(R_{HK})$")
    plt.ylabel("Model Confidence")
    plt.savefig(f'{save_dir}/conf_rhk_lamost.png')
    plt.close()
    plt.hexbin(merged_df['logR+HK'], merged_df['total error'], mincnt=1)
    plt.xlabel(r"$log(R_{HK})$")
    plt.ylabel("total Error (Days)")
    plt.savefig(f'{save_dir}/err_rhk_lamost.png')
    plt.close()

    # bins = np.linspace(-6,-4, 30)
    # bins = np.round(bins, 2)
    # bins_labels = [f'{str(bins[i])}' for i in range(len(bins[:-1]))]
    # scatter_binned_df(merged_df, 'logR+HK', 'total error',
    #                   bin_edges=bins, bin_labels=bins_labels,
    #                   bar=True, save_name='log_rhk_err_lamost')
def error_correlation_cks(model_df, ref_cat_cks, ref_cat_lamost, save_dir='../imgs'):
    merged_df = ref_cat_cks.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence',
                                        'total error', 'relative sigma error']], on='KID')
    merged_df_lamost = ref_cat_lamost.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence',
                                        'total error', 'relative sigma error']], on='KID')

    high_conf_df = merged_df[(merged_df['Prot_flag'] == 3) & (merged_df['logg'] > 4)]
    active = merged_df[merged_df['log(R`HK)']>-4.69]
    inactive = merged_df[(merged_df['log(R`HK)']>-5.167) & (merged_df['log(R`HK)']<-4.69)]
    very_inactive = merged_df[(merged_df['log(R`HK)']<-5.167)]
    high_conf_df['diff_p'] = high_conf_df['Prot_phot'] - high_conf_df['predicted period']
    med_conf_df = merged_df[merged_df['Prot_flag'] > 1]
    diff_p_high = high_conf_df[high_conf_df['diff_p'] > 3]
    diff_p_low = high_conf_df[high_conf_df['diff_p'] < 3]
    print("total error diff high: ", diff_p_high['total error'].mean())
    print("total error diff low: ", diff_p_low['total error'].mean())
    for df, name in zip([active, inactive, very_inactive], ['active', 'inactive', 'very inactive']):
        print(f"******{name}********")
        print(len(df), ' points')
        med_error = df['sigma error'].mean()
        print(name, ' error: ', med_error)
        med_conf = df['mean_period_confidence'].mean()
        print(name, ' conf: ', med_conf)
        tot = df['total error'].mean()
        print(name, ' total error: ', tot)
    # plt.scatter(high_conf_df['Teff'], high_conf_df['P_rot'])
    # plt.show()
    # plt.scatter(high_conf_df['Teff'], high_conf_df['Prot_phot'])
    # plt.show()
    # plt.scatter(high_conf_df['Teff'], high_conf_df['predicted period'])
    # plt.show()

    bins = np.linspace(-6, -4, 30)
    # bins = np.round(bins, 2)
    bins_labels = [bins[i] for i in range(len(bins[:-1]))]
    cks_bin_result = scatter_binned_df(merged_df, 'log(R`HK)', 'total error',
                      bin_edges=bins, bin_labels=bins_labels,
                      bar=False, save_name='log_rhk_rel_err_cks')

    lamost_bin_result = scatter_binned_df(merged_df_lamost, 'logR+HK', 'total error',
                                       bin_edges=bins, bin_labels=bins_labels,
                                       bar=False, save_name='log_rhk_rel_err_cks')

    plt.scatter(cks_bin_result['log(R`HK)_bin'], cks_bin_result['total error'], s=100, label='CKS')
    plt.scatter(lamost_bin_result['logR+HK_bin'], lamost_bin_result['total error'],
                  s=100, label='LAMOST')
    plt.legend()
    plt.ylabel('Total Relative Error')
    plt.xlabel(r"$log(R_{HK})$")
    plt.savefig(f'{save_dir}/log_rhk_tot_err_cks_lamost.png')
    plt.show()
def error_correlations_berger(model_df, ref_cat):
    merged_df = ref_cat.merge(model_df[['KID', 'predicted period',
                                        'sigma error', 'mean_period_confidence', 'total error']], on='KID')

    for pred_att in ['total error']:
        for att, thresh in zip(['Dist', 'Teff', 'Lstar', 'Avmag', 'FeH'], [[0, 500, 1000, 2000],
                                                                    [2300, 3900, 5300, 6000, 7000],
                                                                    [-2,-1.5, 0,1.5, 2],
                                                                    [0, 0.1,0.2,0.4, 0,5],
                                                                    [-2, -1, 0, 1, 2]]):
            # threshold_hist(merged_df,
            #                att=pred_att,
            #                thresh_att=att,
            #                thresh=thresh,
            #                sign='between',
            #                save_name=f'{pred_att}_{att}',
            #                x_label=pred_att)
            bins = np.linspace(thresh[0], thresh[-1], 30)
            bins_labels = [bins[i] for i in range(len(bins[:-1]))]
            bin_result = scatter_binned_df(merged_df, att, 'total error',
                                               bin_edges=bins, bin_labels=bins_labels,
                                               bar=False, save_name=f'log_{att}_rel_err_berger')

            # plt.hexbin(merged_df[att], merged_df[pred_att], mincnt=1)
            # plt.xlabel(att)
            # plt.ylabel(pred_att)
            # plt.show()

def contamination_analysis(model_df, save_dir='../imgs'):
    for flag in range(12):
        for suffix in ['', '_p']:
            flag_name = f'flag1{suffix}'
            if flag_name in model_df.columns:
                cont = model_df[model_df[flag_name]==flag]
                if len(cont):
                    plt.hist(cont['predicted period'], density=True,
                             bins=15, histtype='step', linewidth=5, label=f"{len(cont)} samples")
                    plt.hist(model_df['predicted period'], density=True, bins=15, linewidth=3,
                                                    histtype='step', label=f"all samples")
                    plt.xlabel('Period (Days)')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.savefig(f'{save_dir}/santos_flags_{flag}{suffix}_period.png')
                    plt.close()
                    ks_test = ks_2samp(cont['predicted period'], model_df['predicted period'])
                    print(f"flag {flag}  period - ks : {ks_test}, avgs : {cont['predicted period'].mean():.2f},"
                          f" {model_df['predicted period'].mean():.2f}")
                    print("number of samples ", len(cont))

                    plt.hist(cont['total error'], density=True, bins=15, linewidth=5,
                             histtype='step', label=f"{len(cont)} samples")
                    plt.hist(model_df['total error'], density=True, bins=15,linewidth=3,
                             histtype='step', label=f"all samples")
                    ks_test = ks_2samp(cont['total error'], model_df['total error'])
                    print(f"flag {flag} error - ks : {ks_test}, avgs : {cont['total error'].mean():.2f},"
                          f" {model_df['total error'].mean():.2f}")
                    plt.xlabel('Total Error')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.savefig(f'{save_dir}/santos_flags_{flag}{suffix}_error.png')
                    plt.close()
                else:
                    print(f"flag {flag} no samples!")

    print(len(cont))

def cum_counts(df):

    # Step 2: Bin the 'predicted period' into integer bins
    df['predicted period bin'] = df['predicted period'].astype(int)

    # Step 3: Count occurrences of 'predicted period bin' for each 'KID'
    histogram = df['predicted period bin'].value_counts().sort_index()
    cumulative_histogram = histogram.cumsum()

    # Step 4: Plot the cumulative histogram
    plt.plot(cumulative_histogram.index, cumulative_histogram.values, marker='o', color='skyblue')
    plt.xlabel('Predicted Period (Binned)')
    plt.ylabel('Binaries Count')
    plt.grid(True)
    plt.show()
def kepler_binaries(model_df, ref=None, save_dir='../imgs'):
    all_gaia_binaries = Table.read('tables/gaia_binaries.fits', format='fits')
    kepler_gaia_table = Table.read('tables/kepler_dr3_4arcsec.fits', format='fits')
    all_gaia_binaries.rename_column('source_id1', 'source_id')
    kepler_gaia_binaries1 = join(all_gaia_binaries, kepler_gaia_table, keys='source_id')
    all_gaia_binaries.rename_column('source_id', 'source_id1')
    all_gaia_binaries.rename_column('source_id2', 'source_id')
    kepler_gaia_binaries2 = join(all_gaia_binaries, kepler_gaia_table, keys='source_id')
    kepler_gaia_binaries = vstack([kepler_gaia_binaries1, kepler_gaia_binaries2])
    kepler_gaia_binaries = kepler_gaia_binaries.to_pandas()
    kepler_gaia_binaries = kepler_gaia_binaries[~kepler_gaia_binaries['kepid'].duplicated()]
    kepler_gaia_df = model_df.merge(kepler_gaia_table.to_pandas(), left_on='KID', right_on='kepid')

    # gaia_binary = pd.read_csv('tables/kepler_gaia_binaries.csv')
    koi_binary = pd.read_csv('tables/kois_binaries.csv')
    # model_df = model_df[model_df['Dist'] < 500]
    print("number of samples: ", len(model_df))
    lightpred_gaia_binaries = kepler_gaia_binaries.merge(model_df[['KID',
                                              'Dist',
                                              'Bin',
                                              'predicted period',
                                              'eb',
                                               'eb_orbital_period',
                                              'sigma error', 'total error',
                                              'mean_period_confidence',
                                              'FeH']], left_on='kepid', right_on='KID')
    # cum_counts(gaia_binary)
    # kois = pd.read_csv('tables/kois.csv')

    lightpred_gaia_binaries['theta_arcsec'] = lightpred_gaia_binaries['sep_AU'] / lightpred_gaia_binaries['Dist']
    fast_binaries = lightpred_gaia_binaries[(lightpred_gaia_binaries['predicted period'] < 7) &
                                            (lightpred_gaia_binaries['predicted period'] > 3)]
    fast_and_wide = fast_binaries[fast_binaries['sep_AU'] > 50]
    fast_and_close = fast_binaries[~(fast_binaries['sep_AU'] > 50)]
    print("all fast rotators: ", len(model_df[(model_df['predicted period'] < 7)
                                              & (model_df['predicted period'] > 3)]))
    print("lightpred-gaia_binaries: ", len(lightpred_gaia_binaries), 'fast binaries: ', len(fast_binaries),
          'fast and wide: ', len(fast_and_wide), 'fast and close: ', len(fast_and_close))
    lightpred_gaia_binaries_single = lightpred_gaia_binaries[lightpred_gaia_binaries['theta_arcsec'] <= 4]
    lightpred_gaia_binaries_wide = lightpred_gaia_binaries[lightpred_gaia_binaries['theta_arcsec'] > 4]
    lightpred_gaia_binaries_single['phot_g_mag_diff'] =\
        lightpred_gaia_binaries_single['phot_g_mean_mag2'] - lightpred_gaia_binaries_single['phot_g_mean_mag1']
    threshold_hist(lightpred_gaia_binaries_single, 'phot_g_mag_diff', [1,3], 'gaia3_binaries',
                   att='predicted period', sign='between', bins=15)
    threshold_hist(lightpred_gaia_binaries_single, 'phot_g_mag_diff', [1, 3], 'gaia3_binaries',
                   att='total error', sign='between', bins=15)

    plt.hist(gaia_binary['predicted period'], density=True, linewidth=3,)
    plt.savefig(f'{save_dir}/gaia_binaries_period.png')


    berger_binaries = model_df[
        ['KID', 'predicted period', 'total error', 'Bin', 'Dist', 'Teff', 'kmag']]
    berger_binaries.dropna(inplace=True)
    threshold_hist(berger_binaries, 'Bin', [-1,0],
                   att='predicted period', sign='between',
                   bins=20, save_name='berger_binaries')
    santos_binaries_p = model_df[['KID',
                                  'predicted period',
                                  'total error',
                                  'Fl1_p',
                                  'Prot_santos2019',
                                  'flag1',
                                  'flag1_p'
                                  ]]
    santos_binaries_p.dropna(inplace=True)
    plt.hist(santos_binaries_p['predicted period'], histtype='step', density=True, label='lightPred period')
    # plt.hist(santos_binaries_p['Prot_santos2019'], histtype='step', density=True, label='santos periods')
    # plt.hist(model_df[model_df['eb']==True]["predicted period"], histtype='step', density=True, label='EB periods')
    plt.title('Santos 2019 binaries')
    plt.legend()
    plt.savefig('../imgs/santos_binaries.png')
    plt.close()

    simonian_binaries = model_df[['KID', 'predicted period', 'total error', 'dK']]
    simonian_binaries.dropna(inplace=True)
    plt.hist(simonian_binaries['predicted period'], histtype='step', density=True, label='lightPred period')
    plt.close()
    plt.scatter(simonian_binaries['dK'], simonian_binaries['predicted period'])
    plt.ylabel('period')
    plt.close()
    plt.scatter(simonian_binaries['dK'], simonian_binaries['total error'])
    plt.ylabel('error')
    plt.close()

    santos_binaries_no_p = model_df[['KID', 'predicted period', 'total error', 'Fl1_no_p']]
    santos_binaries_no_p.dropna(inplace=True)
    # plt.scatter(gaia_binary['predicted acf_p'], gaia_binary['predicted period'], )
    # diff = np.abs(gaia_binary['predicted period'] - gaia_binary['predicted acf_p'])
    # acc10 = (diff < gaia_binary['predicted acf_p'] * 0.1).sum() / len(diff)
    # plt.title(f"acc10p {acc10}")
    # plt.show()
    # koi_binary = koi_binary.merge(model_df, on='KID')
    # gaia_binary['koi_binary'] = gaia_binary['kepid'].isin(koi_binary['KID'])
    # koi_binary['gaia_binary'] = koi_binary['KID'].isin(gaia_binary['kepid'])
    # plt.hist(model_df['predicted period'], histtype='step',bins=20, density=True, label='all samples')
    # plt.hist(kois['predicted period'], histtype='step',bins=20, density=True, label='all kois')
    # plt.hist(gaia_binary['predicted period'], histtype='step',bins=20, density=True, label='gaia binaries')
    # plt.hist(koi_binary['predicted period'], histtype='step',bins=20, density=True, label='kois binaries')
    # plt.legend()
    # plt.xlabel("predicted period (Days)")
    # plt.show()
    #
    # plt.hist(model_df['total error'], histtype='step', bins=20, density=True, label='all samples')
    # plt.hist(kois['total error'], histtype='step', bins=20, density=True, label='all kois')
    # plt.hist(gaia_binary['total error'], histtype='step', bins=20, density=True, label='gaia binaries')
    # plt.hist(koi_binary['total error'], histtype='step', bins=20, density=True, label='kois binaries')
    # plt.legend()
    # plt.xlabel("total error")
    # plt.show()
    #
    # gaia_binary['Teff_diff'] = (gaia_binary['dr2_rv_template_teff1'] - gaia_binary['dr2_rv_template_teff2']).abs()
    # plt.hexbin(gaia_binary['sep_AU'], gaia_binary['total error'], mincnt=1)
    # plt.show()
    # plt.hexbin(gaia_binary['R_chance_align'], gaia_binary['total error'], mincnt=1)
    # plt.show()


def compare_kois_eb_hj(kepler_inference, save_dir='../imgs'):
    kois = pd.read_csv('tables/kois.csv')
    kois = kois[kois['koi_disposition'] != 'FALSE POSITIVE']
    print("number of total kois: ", len(kois))
    inference_kois = kepler_inference.merge(kois, on='KID')
    print("number of kois in catalog: ", len(inference_kois))
    inference_hj = inference_kois[(inference_kois['koi_prad'] > J_radius_factor)
                                  & (inference_kois['planet_Prot'] < prot_hj)]
    eb = pd.read_csv('tables/kepler_eb.txt')
    inference_eb = kepler_inference.merge(eb, on='KID')

    hist(kepler_inference, df_other=inference_kois, other_name='kois',
         save_name='p_kois', att='predicted period')
    hist(kepler_inference, df_other=inference_hj, other_name='HJ',
         save_name='p_hj', att='predicted period')
    hist(kepler_inference, df_other=inference_eb, other_name='Eclipsing Binaries',
         save_name='p_eb', att='predicted period')

    fig, ax = plt.subplots(2,1)
    ax[0].scatter(inference_eb['period'],
                inference_eb['predicted period'])
    ax[1].scatter(inference_eb['period'],
                inference_eb['predicted period'])
    ax[1].set_ylim(0,15)
    ax[1].set_xlim(0,15)
    ax[0].set_xlim(0, 100)
    ax[1].set_xlabel("Orbital Period (Days)")
    plt.tight_layout(rect=[0.1, 0, 1, 1])
    fig.text(0.04, 0.5, 'Predicted Stellar Period (Days)', va='center', rotation='vertical')

    plt.savefig(f'{save_dir}/p_eb_scatter_all.png')
    plt.close()

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(inference_eb['period'],
                  inference_eb['predicted acf_p'])
    ax[1].scatter(inference_eb['period'],
                  inference_eb['predicted acf_p'])
    ax[1].set_ylim(0, 15)
    ax[1].set_xlim(0, 15)
    ax[0].set_xlim(0, 100)
    ax[1].set_xlabel("Orbital Period (Days)")
    plt.tight_layout(rect=[0.1, 0, 1, 1])
    fig.text(0.04, 0.5, 'Predicted Stellar Period (Days)', va='center', rotation='vertical')

    plt.savefig(f'{save_dir}/p_eb_scatter_acf.png')
    plt.close()

    plt.scatter(inference_eb[inference_eb['Teff_x'] < 6200]['period'],
                inference_eb[inference_eb['Teff_x'] < 6200]['predicted period'],
                label=r'$Teff < 6200 K$')
    plt.scatter(inference_eb[inference_eb['Teff_x'] > 6200]['period'],
                inference_eb[inference_eb['Teff_x'] > 6200]['predicted period'],
                label=r'$Teff > 6200 K$', alpha=0.5)
    plt.xlim(0, 100)
    plt.xlabel("orbital period")
    plt.ylabel("stellar period")
    plt.legend()
    # plt.colorbar(label='period confidence')
    plt.savefig('../imgs/p_eb_scatter.png')
    plt.close()

    plt.scatter(inference_hj[inference_hj['Teff_x'] < 6200]['planet_Prot'],
                inference_hj[inference_hj['Teff_x'] < 6200]['predicted period'], label=r'$Teff>6200 K$')
    plt.scatter(inference_hj[inference_hj['Teff_x'] > 6200]['planet_Prot'],
                inference_hj[inference_hj['Teff_x'] > 6200]['predicted period'], label=r'$Teff<6200 K$')
    plt.xlabel("HJ period")
    plt.ylabel("stellar period")
    plt.legend()
    plt.savefig('../imgs/p_hj_scatter.png')
    plt.close()
    return inference_kois, inference_hj, inference_eb

def selective_merge_dfs(df_1, df_2, columns_to_update, key_column='KID'):
    """
    Update values in df_2 with values from df_1 where the key_column matches.

    Parameters:
    df_1 (pd.DataFrame): The dataframe containing the values to prioritize.
    df_2 (pd.DataFrame): The dataframe to be updated.
    key_column (str): The column name to join on (e.g., 'KID').
    columns_to_update (list): List of columns to update.

    Returns:
    pd.DataFrame: Updated dataframe.
    """
    # Merge the dataframes on the key_column with a left join
    merged_df = pd.merge(df_2, df_1, on=key_column, how='left', suffixes=('_df2', '_df1'))

    # Update the columns
    for col in columns_to_update:
        if col in df_2.columns:
            # If the column is in df_2, update it with values from df_1 where available
            merged_df[col] = merged_df[f'{col}_df1'].combine_first(merged_df[f'{col}_df2'])
            # Drop the temporary columns
            merged_df = merged_df.drop([f'{col}_df2', f'{col}_df1'], axis=1)
    return merged_df

def convert_cols_to_float(df, cols):
    for c in cols:
        if df[c].dtype == 'int64':
            continue
        df[c] = df[c].apply(lambda x: float(x.strip().split(',')[0])
        if (len(x) and len(x.strip())) else float('nan'))
        # df[c] = df[c].str.strip().replace('', float('nan'))
    return df

def merge_all_cats(kepler_inference):
    acf_df = pd.read_csv('tables/kepler_acf_pred_exp14.csv')
    acf_df_7 = pd.read_csv('tables/kepler_acf_pred_exp7.csv')
    acf_df_7.rename(columns={'predicted acf_p': 'predicted acf_p no doubles'}, inplace=True)
    eb_df = pd.read_csv('tables/kepler_eb.txt')
    eb_df.rename(columns={'period': 'eb_orbital_period'}, inplace=True)
    acf_df.rename(columns={'double_peaked': 'second_peak'}, inplace=True)
    mcq2014 = pd.read_csv('tables/Table_1_Periodic.txt')
    mcq2014.rename(columns={'Prot':'Prot_mcq14'}, inplace=True)
    berger_cat = pd.read_csv('tables/berger_catalog.csv')
    berger2018 = pd.read_csv('tables/berger2018.txt', sep='\t')
    santos2019_p = convert_cols_to_float(pd.read_csv(
                                    'tables/santos2019_period.txt', sep=';'),
                                    cols=['Fl1'])
    santos2019_p.rename(columns={'Fl1': 'Fl1_p',  'Prot': 'Prot_santos2019'}, inplace=True)
    santos2019_no_p = convert_cols_to_float(pd.read_csv
                                         ('tables/santos2019_no_period.txt', sep=';'),
                                          cols=['Fl1'])
    santos2019_no_p.rename(columns={'Fl1': 'Fl1_no_p', 'Prot': 'Prot_santos19'}, inplace=True)

    simonian2019 = pd.read_csv('tables/simonian2019.txt', sep='\t')

    santos2021 = pd.read_csv('tables/santos2021_full.txt', sep=';')
    flags_cols = [col for col in santos2021.columns if 'flag' in col]
    santos2021 = convert_cols_to_float(santos2021, cols=flags_cols)
    santos2021_p = convert_cols_to_float(pd.read_csv('tables/santos2021_p.txt', sep=';'),
                                         cols=['flag1'])
    santos2021_p.rename(columns={'flag1': 'flag1_p', 'Prot': 'Prot_santos21'}, inplace=True)
    reinhold23 = pd.read_csv('tables/reinhold2023.csv')
    reinhold23.rename(columns={'Prot':'Prot_reinhold23'}, inplace=True)
    r_var = pd.read_csv('tables/r_var.csv')
    s_ph = pd.read_csv('tables/s_ph.csv')


    kepler_inference = selective_merge_dfs(berger_cat, kepler_inference, columns_to_update=['Teff',
                                                                                            'logg',
                                                                                            'Dist',
                                                                                            'Lstar',
                                                                                            'FeH',
                                                                                            'Mstar'])
    kepler_inference = selective_merge_dfs(acf_df[['KID', 'predicted acf_p', 'second_peak']], kepler_inference,
                                           columns_to_update=['predicted acf_p', 'second_peak'])
    kepler_inference = selective_merge_dfs(acf_df_7[['KID', 'predicted acf_p no doubles']], kepler_inference,
                                           columns_to_update=['predicted acf_p no doulbes'])
    kepler_inference = selective_merge_dfs(berger2018[['KID', 'Bin']], kepler_inference,
                                           columns_to_update=['Bin'])
    kepler_inference = selective_merge_dfs(santos2019_p[['KID', 'Fl1_p','Prot_santos2019']], kepler_inference,
                                           columns_to_update=['Fl1', 'Prot_santos2019']
                                           )
    kepler_inference = selective_merge_dfs(santos2019_no_p[['KID', 'Fl1_no_p']], kepler_inference,
                                           columns_to_update=['Fl1_no_p']
                                           )

    kepler_inference = selective_merge_dfs(simonian2019[['KID', 'dK']], kepler_inference,
                                           columns_to_update=['dK'])

    kepler_inference = selective_merge_dfs(santos2021[flags_cols + ['KID']], kepler_inference,
                                           columns_to_update=flags_cols)

    kepler_inference = selective_merge_dfs(santos2021_p[['KID', 'flag1_p', 'Prot_santos21']], kepler_inference,
                                           columns_to_update=['flag1_p', 'Prot_santos21'])

    kepler_inference = selective_merge_dfs(r_var[['R_{var}', 'kmag', 'KID']], kepler_inference,
                                           columns_to_update=['R_{var}', 'kmag'])

    kepler_inference = selective_merge_dfs(s_ph[['KID', 's_{ph}']], kepler_inference,
                                           columns_to_update=['s_{ph}'])

    kepler_inference = selective_merge_dfs(mcq2014[['KID', 'w', 'Prot_mcq14']], kepler_inference,
                                           columns_to_update=['w', 'Prot_mcq14'])
    kepler_inference = selective_merge_dfs(reinhold23[['KID', 'Prot_reinhold23']], kepler_inference,
                                           columns_to_update=['Prot_reinhold23'])

    kepler_inference = selective_merge_dfs(eb_df[['KID', 'eb_orbital_period']], kepler_inference,
                                           columns_to_update=['eb_orbital_period'])

    kepler_inference = kepler_inference[~kepler_inference['KID'].duplicated()]

    return kepler_inference


def gaia_binaries(model_df):
    gaia = Table.read('tables/kepler_gaia_binaries.csv')


def eb_analysis(model_df, save_dir='../imgs'):

    # eb_df = pd.read_csv('tables/kepler_eb.txt')
    ebs = model_df[model_df['eb']==True]
    lurie = pd.read_csv('tables/lurie2017.txt', sep=';')
    lurie_async = pd.read_csv('tables/lurie_async.txt', sep=';')
    lurie_async['async']=True
    ebs = ebs.merge(lurie[['KIC', 'Class']], left_on='KID', right_on='KIC', how='left')
    ebs = ebs.merge(lurie_async, on='KID',  how='left')

    ebs['p_ratio'] = ebs['eb_orbital_period'] / ebs['predicted period']
    ebs['acf_ratio'] = ebs['eb_orbital_period'] / ebs['predicted acf_p']
    fast_orb = ebs[ebs['eb_orbital_period'] < 10]
    low_sync = fast_orb[(fast_orb['p_ratio'] > 0.5) & (fast_orb['p_ratio'] < 1.2)]
    sync = fast_orb[(fast_orb['p_ratio'] > 0.69) & (fast_orb['p_ratio'] < 1.2)]
    high_sync = fast_orb[(fast_orb['p_ratio'] > 0.92) & (fast_orb['p_ratio'] < 1.2)]
    avg_ratio = sync['p_ratio'].mean()

    print("fraction 0.5 - 1.2: ", len(sync)/len(fast_orb), "avg: ", avg_ratio)
    print("fraction 0.69 - 1.2: ", len(sync)/len(fast_orb), "avg: ", avg_ratio)
    print("fraction 0.92 - 1.2: ", len(high_sync)/len(fast_orb), "avg: ", avg_ratio)

    calc_consistency_curve(ebs[ebs['eb_orbital_period']<50], 0.4, x_name='eb_orbital_period', name='eb')
    model_df['eb_group'] = np.nan
    sample1 = ebs[ebs['eb_orbital_period'] > 10]
    sample2 = ebs[(ebs['eb_orbital_period'] < 10) & (ebs['eb_orbital_period'] > 3)]
    sample2['p_diff'] = np.abs(sample2['predicted period'] - sample2['eb_orbital_period'])
    sample2['p_acc'] = sample2['p_diff'] / sample2['eb_orbital_period']
    sample3_cond = (sample2['p_acc'] > 0.4) & (sample2['p_diff'] > 2)
    sample4_cond = (sample2['p_acc'] < 0.4) | (sample2['p_diff'] < 2)

    sample3 = sample2[sample3_cond]
    sample4 = sample2[sample4_cond]

    print("un-synchronized fraction ", len(sample3)/len(sample2))
    print("synchronized fraction ", len(sample4)/len(sample2))


    fig, ax = plt.subplots()
    ax.scatter(sample1['eb_orbital_period'], sample1['p_ratio'], label='sample1', color='gray')
    ax.scatter(sample3['eb_orbital_period'], sample3['p_ratio'], label='sample3', color='red')
    ax.scatter(sample4['eb_orbital_period'], sample4['p_ratio'], label='sample4', color='cyan')
    # plt.legend()
    ax.set_xlabel(r'$P_{orb}$ (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{rot}$')
    # ax.set_xlim((0,50))

    # Set the x and y scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.hlines(y=2, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=1, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=0.5, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')

    # plt.xlim((0,50))
    # plt.xlabel('Orbital Period (Days)')
    # plt.ylabel('Predicted Period (Days)')
    plt.savefig(f'{save_dir}/eb_classes.png')
    plt.close()

    plt.hist(sample1['total error'], density=True, linewidth=3, histtype='step',
             label=f"avg {sample1['total error'].mean():.3f} Days")
    plt.hist(sample3['total error'], density=True, linewidth=3, histtype='step',
             label=f"avg {sample3['total error'].mean():.3f} Days")
    plt.hist(sample4['total error'], density=True, linewidth=3, histtype='step',
             label=f"avg {sample4['total error'].mean():.3f} Days")
    plt.legend()
    plt.xlabel('Total Error')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/eb_errors.png')
    plt.close()

    plt.hist(sample1['mean_period_confidence'], density=True, linewidth=3, histtype='step',
             label=f"average {sample1['mean_period_confidence'].mean():.2f}", color='gray')
    plt.hist(sample3['mean_period_confidence'], density=True, linewidth=3, histtype='step',
             label=f"average {sample3['mean_period_confidence'].mean():.2f}", color='red')
    plt.hist(sample4['mean_period_confidence'], density=True, linewidth=3, histtype='step',
             label=f"average {sample4['mean_period_confidence'].mean():.2f}", color='cyan')
    plt.legend()
    plt.xlabel('Period (Days)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/eb_confs.png')
    plt.close()

    unique_classes = ebs['Class'].unique()
    color_map = {cls: f'C{i}' for i, cls in enumerate(unique_classes)}
    # color_map[np.nan] = 'black'
    fig, ax = plt.subplots()
    scatter_points = []
    for cls, color in color_map.items():
        class_data = ebs[ebs['Class'] == cls]
        if len(class_data):
            print(f"number of {cls} samples - ", len(class_data), "color - ", color)
            scatter_point = ax.scatter(class_data['eb_orbital_period'], class_data['p_ratio'], c=color, label=cls)
            scatter_points.append(scatter_point)

    ax.hlines(y=2, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=1, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=0.5, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    # Add the legend
    ax.legend(loc='best')

    # Set the axis labels
    ax.set_xlabel(r'$P_{orb}$ (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{rot}$')
    # ax.set_xlim((0,50))

    # Set the x and y scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(f'{save_dir}/ebs_scatter_lurie_classes.png')
    plt.show()


    fig, ax = plt.subplots()
    scatter_points = []
    for cls, color in color_map.items():
        class_data = ebs[ebs['Class'] == cls]
        if len(class_data):
            print(f"number of {cls} samples - ", len(class_data))
            scatter_point = ax.scatter(class_data['eb_orbital_period'], class_data['acf_ratio'], c=color, label=cls)
            scatter_points.append(scatter_point)

    ax.hlines(y=2, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=1, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=0.5, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    # Add the legend
    ax.legend(loc='best')

    # Set the axis labels
    ax.set_xlabel(r'$P_{orb}$ (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{ACF}$')
    # ax.set_xlim((0,50))

    # Set the x and y scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(f'{save_dir}/ebs_scatter_lurie_classes_acf.png')
    plt.show()

    fig, ax = plt.subplots()
    scatter = ax.scatter(ebs['eb_orbital_period'], ebs['p_ratio'], c=ebs['mean_period_confidence'])
    fig.colorbar(scatter, label='confidence')
    ax.set_xlabel(r'$P_{orb}$ (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{rot}$')
    # ax.set_xlim((0,50))

    ax.hlines(y=2, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=1, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=0.5, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')

    # Set the x and y scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(f'{save_dir}/ebs_scatter_conf.png')
    plt.show()

    fig, ax = plt.subplots()
    scatter = ax.scatter(ebs['eb_orbital_period'], ebs['p_ratio'], c=ebs['total error'])
    fig.colorbar(scatter, label='Total Error')
    ax.set_xlabel(r'$P_{orb}$ (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{rot}$')
    # ax.set_xlim((0,50))

    ax.hlines(y=2, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=1, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')
    ax.hlines(y=0.5, xmin=0, xmax=ebs['eb_orbital_period'].max(), linestyles='dashed', color='gray')

    # Set the x and y scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(f'{save_dir}/ebs_scatter_error.png')
    plt.show()


    fig, ax = plt.subplots()
    ax.scatter(ebs['eb_orbital_period'], ebs['p_ratio'])
    ax.scatter(ebs[ebs['async']==True]['eb_orbital_period'], ebs[ebs['async']==True]['p_ratio'], label='async')
    ax.set_xlabel('Orbital Period (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{rot}$ (Days)')

    # Set the x and y scales to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    fig, ax = plt.subplots()
    scatter = ax.scatter(ebs['eb_orbital_period'], ebs['p_ratio'], c=ebs['mean_period_confidence'])
    fig.colorbar(scatter, label='confidence')
    # plt.xlim([0,50])
    # plt.ylim((0,3))
    ax.set_xlabel('Orbital Period (Days)')
    ax.set_ylabel(r'$P_{orb}/P_{rot}$ (Days)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(f'{save_dir}/ebs_scatter_conf.png')
    plt.close()
    exit()

    plt.scatter(ebs['eb_orbital_period'], ebs['p_ratio'], c=ebs['total error'])
    plt.colorbar(label='Total Error')
    plt.xlim([0, 50])
    plt.ylim(0,3)
    plt.xlabel('Orbital Period (Days)')
    plt.ylabel(r'$P_{orb}/P_{rot}$ (Days)')
    plt.savefig(f'{save_dir}/ebs_scatter_err.png')
    plt.close()

    # plt.scatter(model_df['predicted period'], model_df['FeH'], label='all samples', alpha=0.5)
    plt.scatter(sample1['predicted period'], sample1['FeH'])
    plt.scatter(sample3['predicted period'], sample3['FeH'])
    plt.scatter(sample4['predicted period'], sample4['FeH'])
    plt.show()

    print("number of ebs: ", len(ebs))
    return model_df
def plot_kepler_inference(kepler_inference, save_dir):
    plot_gyro(kepler_inference, save_dir=save_dir)
    period_correlations(kepler_inference, ['FeH', 'Dist', 'Lstar', 'Teff', 'logg', 'w'])
    compare_period(kepler_inference, p_att='Prot_mcq14', ref_name='McQ14', save_dir=save_dir)
    find_non_consistent_samples(kepler_inference,p_att='Prot_mcq14', ref_name='McQ14', thresh_val=3, save_dir=save_dir)
    compare_period(kepler_inference, p_att='Prot_reinhold23', ref_name='Reinhold23', save_dir=save_dir)
    compare_period(kepler_inference, p_att='Prot_santos21', ref_name='Santos21', save_dir=save_dir)
    compare_non_consistent_samples(r'..\non_consistent_group1',
                                   kepler_inference, ref1, 'McQ14')
    compare_non_consistent_samples(r'..\non_consistent_group2',
                                   kepler_inference, ref1, 'McQ14')
    inference_kois, inference_hj, inference_eb = compare_kois_eb_hj(kepler_inference, save_dir=save_dir)

    kepler_inference = eb_analysis(kepler_inference, save_dir=save_dir)
    eb_analysis(kepler_inference, save_dir=save_dir)
    contamination_analysis(kepler_inference, save_dir=save_dir)
    kepler_binaries(kepler_inference, save_dir=save_dir)
    error_analysis(kepler_inference, save_dir=save_dir)
    hist_binned_by_att(kepler_inference, att='predicted period',
                       bins=[3500, 5200, 6000, 7000, 7600], bin_att='Teff',
                       save_name='p_binned_Teff', save_dir=save_dir)
    mass_binning(kepler_inference, save_dir=save_dir)
    period_mass_bin(kepler_inference, save_dir=save_dir)

def plot_gyro(model_df, save_dir='../imgs'):
    lightpred = pd.read_csv('tables/gyrointerp_lightPred4.csv')
    lightpred_merged = model_df.merge(lightpred, on='KID')
    mcq14 = pd.read_csv('tables/gyrointerp_mcQ14_3.csv')
    reinhold = pd.read_csv('tables/gyrointerp_reinhold.csv')
    print(f'number of samples gyro :\n lightPred: {len(lightpred[~lightpred.isna()])}\n'
          f'McQuillan: {len(mcq14[~mcq14.isna()])}\n Rienhold: {len(reinhold[~reinhold.isna()])} \n'
          f'lightPred merged {len(lightpred_merged)}')

    light_mcq = lightpred_merged.merge(mcq14, on='KID')

    plt.hist(mcq14['age'], density=True, histtype='step', bins=20, label='McQ14', linewidth=3)
    plt.hist(lightpred_merged['age'], density=True, histtype='step', bins=20, label='LightPred', linewidth=3)
    plt.hist(reinhold['age'], density=True, histtype='step', bins=20, label='Reinhold23', linewidth=3)
    plt.legend()
    plt.xlabel('Age (Myear)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/gyro_compare.png')
    plt.show()

    plt.hist((mcq14['e_age_up'] + mcq14['e_age_low'])/2 , density=True, histtype='step', bins=20, label='McQ14', linewidth=3)
    plt.hist((lightpred_merged['e_age_up'] + lightpred_merged['e_age_low']), density=True,
             histtype='step', bins=20, label='LightPred', linewidth=3)
    plt.hist((reinhold['e_age_up'] + reinhold['e_age_low']), density=True, histtype='step',
             bins=20, label='Reinhold23', linewidth=3)
    plt.legend()
    plt.xlabel('Age Error (Myear)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/gyro_errors.png')
    plt.show()

    plt.hist(light_mcq['age_y'], density=True, histtype='step', bins=20, label='McQ14', linewidth=3)
    plt.hist(light_mcq['age_x'], density=True, histtype='step', bins=20, label='LightPred', linewidth=3)
    plt.legend()
    plt.xlabel('Age (Myear)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/gyro_light_mcq.png')
    plt.show()

    plt.hist((light_mcq['e_age_up_x'] + light_mcq['e_age_low_x'])/2, density=True,
             histtype='step', bins=20, label='McQ14', linewidth=3)
    plt.hist((light_mcq['e_age_up_y'] + light_mcq['e_age_low_y'])/2, density=True,
             histtype='step', bins=20, label='LightPred', linewidth=3)
    plt.legend()
    plt.xlabel('Age Error (Myear)')
    plt.ylabel('Density')
    plt.savefig(f'{save_dir}/gyro_light_mcq_errors.png')
    plt.show()





def plot_logg_teff(kepler_inference):
    all_samples = pd.read_csv('tables/berger_catalog.csv')
    # santos = pd.read_csv('tables/santos2021_full.txt', sep='\t')
    # Scatter plots
    plt.scatter(all_samples['Teff'], all_samples['logg'], s=1, label='Berger')
    plt.scatter(kepler_inference['Teff'], kepler_inference['logg'], s=1, label='Our')
    # plt.scatter(santos['Teff'], santos['logg'], s=1, label='Santos2021', alpha=0.3)
    # plt.scatter(santos['Teff'], santos['logg'], s=1, label='santos')

    # Axis inversions
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    # Show legend
    plt.legend()

    # Show plot
    plt.close()

    binaries = kepler_inference[kepler_inference['Bin'] != 0][['Teff', 'logg', 'Bin']]
    binaries.dropna(inplace=True)
    print("number of binaries :", len(binaries))
    plt.scatter(all_samples['Teff'], all_samples['logg'], s=1, label='Berger')
    plt.scatter(binaries['Teff'], binaries['logg'], s=1, label='Binaries')

    # Axis inversions
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    # Show legend
    plt.legend()

    # Show plot
    plt.show()


def period_correlations(model_df, x_names, save_dir='../imgs'):
    for x_name in x_names:
        plt.scatter(model_df[x_name], model_df['predicted period'], s=1)
        plt.xlabel(x_name)
        plt.ylabel('Period (Days)')
        plt.savefig(f'{save_dir}/{x_name}_period.png')
        plt.close()

        bins_edges = stats.mstats.mquantiles(model_df[x_name], [1 / 4, 2 / 4, 3 / 4])
        bins_edges = bins_edges
        for i, bin in enumerate(bins_edges):
            if i == 0:
                reduced_df = model_df[model_df[x_name] < bin]
                label = f"{x_name} < {bin:.2f}"
            elif i == len(bins_edges) - 1:
                reduced_df = model_df[model_df[x_name] > bin]
                label = f"{x_name} > {bin:.2f}"
            else:
                reduced_df = model_df[(model_df[x_name] >= bin) & (model_df[x_name] < bins_edges[i+1])]
                label = f"{bin:.2f} <= {x_name} < {bins_edges[i+1]:.2f}"
            label = label + f' {len(reduced_df)} samples'
            plt.hist(reduced_df['predicted period'], bins=20, density=True, histtype='step', label=label)
        plt.legend()
        plt.savefig(f'{save_dir}/{x_name}_period_hist.png')
        plt.close()



def error_analysis(kepler_inference, save_dir='../imgs'):
    kic = pd.read_csv('tables/kepler_input_catalog_dr25.txt', sep='\t')
    cks = pd.read_csv('tables/CKS_activity.csv')
    kep_bonus_all = pd.read_csv('tables/kepler_bonus.csv')
    kep_bonus = kep_bonus_all.merge(kepler_inference['KID'], left_on='kic', right_on='KID')
    kep_bonus = kep_bonus.merge(kic[['KIC', '_RA', '_DE',]], left_on='KID', right_on='KIC')
    kep_bonus = kep_bonus.merge(kepler_inference[['KID', 'Lstar']], on='KID')

    # pixel_binning(kepler_inference, kep_bonus, '_RA', '_DE',
    #               'crowdsap', 'total error', 0.1)
    #
    kep_bonus = kep_bonus[kep_bonus['Lstar'] != 0.0]
    kep_bonus['norm-crowdsap'] = kep_bonus['crowdsap'] / (kep_bonus['Lstar'])


    error_correlation_kepler(kepler_inference, None,
                             x_names=['Teff', 'Dist', 'kmag',
                                      's_{ph}', 'Lstar', 'R_{var}'
                                      ],
                             units=['K', 'pc', None,
                                    None, 'L_\odot', None],
                             n_rows=2, ref_id='',
                             save_dir=save_dir
                             )


    multiplot_avg_error_3d(kepler_inference, None,
                           x_names= ['kmag', 'Dist', 'FeH', 'Teff', 's_{ph}', 'Lstar'],
                           y_name='R_{var}',
                           points_thresh=30,
                           ref_id='all')
def download_eb_dataset():
    eb_dataset = pd.read_csv('tables/eb_dataset.csv')
    for i, row in eb_dataset.iterrows():
        kid = row['KID']
        if f"{kid}.png" not in os.listdir(f"samples/eb/{int(row['target'])}"):
            res = lk.search_lightcurve(f'KIC {kid}', cadence='long', author='kepler')
            lc = res.download_all().stitch()
            lc.plot()
            plt.savefig(f"samples/eb/{int(row['target'])}/{kid}.png")
            plt.close()
        else:
            print(f"{kid} already exist")
def get_contaminants():
    base_root = 'tables/contaminants'
    res = {}
    for dir in os.listdir(base_root):
        df = None
        for i, filename in enumerate(os.listdir(os.path.join(base_root, dir))):
            if i == 0:
                df = pd.read_csv(os.path.join(base_root, dir, filename), sep='\t')
            else:
                df = pd.concat([df, pd.read_csv(os.path.join(base_root, dir, filename), sep='\t')])
        res[dir] = df
    all_dfs = [r for r in res.values()]
    total_df = pd.concat(all_dfs)
    return total_df, res

def aggregate_results(df, target_att='predicted period'):
    selected_columns = df.filter(regex=target_att).columns
    df[target_att] = df[selected_columns].mean(axis=1)
    return df


def set_low_p(kepler_inference, other_att='Prot', threshold=4, ref_name='McQ14'):
    filtered_other = other_df[(other_df[other_att] < threshold) & (other_df[other_att] > 0)
                              & (other_df['KID'].isin(kepler_inference['KID']))]
    print("number of low p acf points :", len(filtered_other))
    update_dict = filtered_other.set_index('KID')[other_att].to_dict()
    mask = kepler_inference['KID'].isin(update_dict.keys())
    kepler_inference.loc[mask, 'predicted period'] = kepler_inference['KID'].map(update_dict)
    if 'sigma error' in other_df.columns:
        sigma_dict = filtered_other.set_index('KID')['sigma error'].to_dict()
        kepler_inference.loc[mask, 'sigma error'] = kepler_inference['KID'].map(sigma_dict)
    kepler_inference.loc[mask, 'method'] = ref_name
    return kepler_inference


def get_doubles_from_acf(df, p_label):
    # df['period_diff'] = np.abs(df['predicted period']*2 - df['Prot'])
    df['doubles'] = np.abs(df['predicted period'] * 2 - df[p_label]) < df[p_label] * 0.2
    print("number of second peak acf: ", len(df[(df['doubles']==True) & (df['second_peak']==True)]))
    df.loc[(df['doubles'] == True) & (df['second_peak']==True), 'predicted period'] = \
        df.loc[(df['doubles'] == True) & (df['second_peak']==True), p_label]
    return df

def apply_constraints(df, contaminants=True, doubles=True, conf=None, low_p=True, error=None):
    print("****number of samples****")
    print("before constraints: ", len(df))
    if contaminants:
        df = df[df['flag1'] != 6]
        df = df[df['flag1'] != 5]
        df = df[df['flag1'] != 4]
        print("after removing contaminants: ", len(df))
    if doubles:
        df = get_doubles_from_acf(df, p_label='Prot_mcq14')
        print("after applying acf seconf peak: ", len(df))
    if low_p:
        print('number of low acf - ', len(df[(df['predicted acf_p no doubles'] < 3) &
                                            (df['predicted acf_p no doubles'] > 0)]))
        df = df[~((df['predicted acf_p no doubles'] > 0) & (df['predicted acf_p no doubles'] <= 3))]
        print("after removing fast rotators: ", len(df))
        df = df[~((df['eb_orbital_period'] < 3) & (df['eb_orbital_period'] > 0))]
        print("after removing eb fast rotators: ", len(df))
    if conf is not None:
        df = df[df['mean_period_confidence'] > conf]
        print("after applying confidence threshold: ", len(df ))
    if error is not None:
        df = df[df['total error'] < error]
        print("after applying error threshold: ", len(df))
    return df


def test_santos_flags(df):
    print("number of total samples: ", len(df))
    for i in range(12):
        res = df[df['flag1']==i]
        print(f'number of samples with flag1=={i}: ', len(res))

def create_final_predictions(df_path):
    kepler_inference = aggregate_results(pd.read_csv(df_path))

    kepler_inference['relative error'] = (kepler_inference['sigma error']
                                       / (kepler_inference['predicted period']))
    kepler_inference['total error'] = (kepler_inference['relative error']
                                       / (kepler_inference['mean_period_confidence']))
    kepler_inference = merge_all_cats(kepler_inference)

    kepler_inference = apply_constraints(kepler_inference, contaminants=True,
                                                            doubles=True,
                                                            conf=0.86,
                                                            low_p=True,
                                                            error=None)
    test_santos_flags(kepler_inference)
    plot_kepler_inference(kepler_inference,  save_dir='../imgs')
    kepler_inference = kepler_inference.round(decimals=3).sort_values(by='KID')
    kepler_inference.rename(columns={'sigma error':'observational error'}, inplace=True)
    kepler_inference_clean = kepler_inference[['KID', 'Teff', 'R', 'logg',
                                               'predicted period', 'observational error', 'relative error',
                                                'mean_period_confidence', 'total error',
                                               ]]
    kepler_inference_clean.to_csv("tables/kepler_predictions_clean.csv", index=False)
    print("number of all samples: ", len(kepler_inference_clean))

def aggregate_dfs_from_gpus(folder_name, num_qs=7, start_q=0, num_ranks=4,
                            file_name='kepler_inference_full'):
    if not os.path.exists(f'{folder_name}'):
        os.mkdir(f'{folder_name}')
    for q in range(start_q, start_q + num_qs):
        for rank in range(num_ranks):
            if not rank:
                df = pd.read_csv(f'{folder_name}_ranks/'
                                 f'{file_name}_{q}_rank_{rank}.csv')
            else:
                df = pd.concat([df, pd.read_csv
                (f'{folder_name}_ranks/'
                 f'{file_name}_{q}_rank_{rank}.csv')], ignore_index=True)
        df.to_csv(f'{folder_name}/{file_name}_{q}.csv', index=False)

def mock_inference(eval_dir, name, prepare=False, quantiles=False):
    if 'eval.csv' not in os.listdir(eval_dir):
        return
    mock_eval = pd.read_csv(f'{eval_dir}/eval.csv')
    # inc_cols = [c for c in mock_eval.columns if 'inclination' in c.lower()]
    # mock_eval[inc_cols] = mock_eval[inc_cols]*180/np.pi

    columns_to_lower = [col for col in mock_eval.columns
                        if col.startswith('predicted') or col.endswith('confidence')]
    column_mapping = {col: col.lower() for col in columns_to_lower}
    mock_eval.rename(columns=column_mapping, inplace=True)
    if prepare:
        mock_eval = prepare_df(mock_eval,
               filter_giants=False, filter_eb=False, teff_thresh=False, filter_contaminants=False)
    if quantiles:
        scatter_predictions_quantiled(mock_eval['Inclination'], mock_eval['predicted inclination 0.5'],
                                      mock_eval['predicted inclination 0.1'], mock_eval['predicted inclination 0.9'],
                                      80, name='inc_80', units='Deg')
        scatter_predictions_quantiled(mock_eval['Inclination'], mock_eval['predicted inclination 0.5'],
                                      mock_eval['predicted inclination 0.05'], mock_eval['predicted inclination 0.95'],
                                      90, name='inc_80', units='Deg')

        scatter_predictions_quantiled(mock_eval['Period'], mock_eval['predicted period 0.5'],
                                      mock_eval['predicted period 0.1'], mock_eval['predicted period 0.9'],
                                      80, name='inc_80', units='Deg')
        scatter_predictions_quantiled(mock_eval['Period'], mock_eval['predicted period 0.5'],
                                      mock_eval['predicted period 0.05'], mock_eval['predicted period 0.95'],
                                      90, name='inc_80', units='Deg')
    scatter_predictions(mock_eval['Period'], mock_eval['predicted period'], mock_eval['period confidence'],
                        name=f'period_{name}_clean', units='Days', show_acc=False, vmin=0.75, dir=eval_dir )
    scatter_predictions(mock_eval['Inclination'], mock_eval['predicted inclination'],
                        mock_eval['inclination confidence'],
                        name=f'inc_{name}_clean', units='Deg', show_acc=False, vmin=0.75, dir=eval_dir)

    scatter_predictions(mock_eval['Period'], mock_eval['predicted period'], mock_eval['period confidence'],
                        name=f'period_{name}', units='Days', dir=eval_dir)
    scatter_predictions(mock_eval['Inclination'], mock_eval['predicted inclination'],
                        mock_eval['inclination confidence'],
                        name=f'inc_{name}', units='Deg', dir=eval_dir)


    mock_eval['diff'] = np.abs(mock_eval['predicted period'] - mock_eval['Period'])
    mock_eval['diff_norm'] = mock_eval['diff'] / mock_eval['predicted period']
    plt.hexbin(mock_eval['period confidence'], mock_eval['diff'], gridsize=100, mincnt=1, cmap='viridis')
    plt.xlabel("Confidence")
    plt.ylabel(r"Absolute Error (Days)")
    plt.colorbar(label='Counts')
    plt.savefig(f'{eval_dir}/p_vs_conf_{name}.png')
    plt.close()

    mock_eval['inc_diff'] = np.abs(mock_eval['predicted inclination'] - mock_eval['Inclination'])
    plt.hexbin(mock_eval['inclination confidence'], mock_eval['diff'],gridsize=100, mincnt=1, cmap='viridis')
    plt.xlabel("confidence")
    plt.ylabel(r"$|i_{True} - i_{Predicted}|$ (Deg)")
    plt.colorbar(label='Counts')
    plt.savefig(f'{eval_dir}/i_vs_conf_{name}.png')
    plt.close()

    plt.hist(mock_eval['Period'], density=True, bins=40, histtype='step', label='Ground Truth')
    plt.hist(mock_eval['predicted period'], density=True, bins=40, histtype='step', label='Prediction')
    plt.xlabel('Period (Days)')
    plt.ylabel('PDF')
    plt.legend()
    plt.savefig(f'{eval_dir}/p_hist_{name}.png')
    plt.close()

    plt.hist(mock_eval['Inclination'], density=True, bins=40, histtype='step', label='Ground Truth')
    plt.hist(mock_eval['predicted inclination'], density=True, bins=40, histtype='step', label='Prediction')
    plt.xlabel('Inclination (Deg)')
    plt.ylabel('PDF')
    plt.legend()
    plt.savefig(f'{eval_dir}/inc_hist_{name}.png')
    plt.close()

    # error_correlation(mock_eval, None, '', x_names=['Period'], y_name=['diff_norm'])




def gps_test(df_path):
    import statsmodels.api as sm
    df = pd.read_csv(f'../inference/{df_path}')
    df_gps = df[df['method'] == 'gps']
    df_acf = df[df['method'] == 'acf']
    model_acf = sm.OLS(df_acf['period'], df_acf['predicted period']).fit()
    slope_acf = model_acf.params[0]
    plt.scatter(df_acf['period'], df_acf['predicted period'])
    plt.plot(df_acf['predicted period'], df_acf['predicted period'] * slope_acf)
    acc10_acf = np.array(np.abs(df_acf['period'] - df_acf['predicted period']) < df_acf['period'] * 0.1).astype(
        np.int8)
    acc10p_acf = acc10_acf.sum() / len(acc10_acf)
    plt.title(f"acc10p {acc10p_acf}, slope = {slope_acf}")
    plt.show()

    df_gps_valid = df_gps[(df_gps['predicted period'] > 1) & (df_gps['predicted period'] < 11)]
    model = sm.OLS(df_gps_valid['period'], df_gps_valid['predicted period']).fit()
    slope = model.params[0]
    plt.scatter(df_gps_valid['predicted period'], df_gps_valid['period'])
    plt.plot(df_gps_valid['predicted period'], df_gps_valid['predicted period'] * slope)
    plt.title(f"slope - {1/slope}")
    plt.show()

def show_gyro_results(cache_id='lightPred2'):
    skip=False
    if f'gyrointerp_{cache_id}.csv' in os.listdir('tables'):
        res_df = pd.read_csv(f'tables/gyrointerp_{cache_id}.csv' )
        skip=True

    t = time.time()
    if not skip:
        csvpaths = os.listdir(f'gyro/{cache_id}')
        csvpaths = [os.path.join(f'gyro/{cache_id}', f) for f in csvpaths if 'posterior' in f]

        ids = []
        ages = []
        e_ages_up = []
        e_ages_low = []
        print("total number of files: ", len(csvpaths))
        for i, csvpath in enumerate(sorted(csvpaths)):
            try:
                filename = os.path.basename(csvpath)
                id = filename.split('_')[0]
                df = pd.read_csv(csvpath)
                r = get_summary_statistics(df.age_grid.values, df.age_post.values)
                ages.append(r['median'])
                e_ages_up.append(r['+1sigma'])
                e_ages_low.append(r['-1sigma'])
                ids.append(id)
                if len(ids) != len(ages):
                    print('stop at: ', i, 'id is ', id)
            except Exception as e:
                continue
            if i % 1000 == 0:
                print(i)
                print(len(ids))
            # if i > 5000:
            #     print(len(ids))
            #     break
        print(len(ids), len(ages), len(e_ages_up), len(e_ages_low))
        res_df = pd.DataFrame({'KID': ids, 'age': ages, 'e_age_up': e_ages_up, 'e_age_low': e_ages_low})
        print("number of valid samples: ", len(res_df.dropna()))
        res_df.to_csv(f'tables/gyrointerp_{cache_id}.csv', index=False)
        plt.hist(ages)
        plt.show()
    else:
        plt.hist(res_df['age'], histtype='step', density=True)
        plt.show()
    print('that took: ', time.time() - t)


def gyrointerp_test(model_df, prot_label, err_prot_label, chache_id='lightPred'):
    teff_df = pd.read_csv('tables/berger_catalog.csv')
    model_df = model_df
    model_df = model_df.merge(teff_df[['KID', 'Teff', 'E_Teff','e_Teff']], on='KID', suffixes=['', '_berger'])
    model_df = model_df[(model_df['Teff'] > 3800) & (model_df['Teff'] < 6200)]
    N_stars = os.cpu_count()
    Teffs = model_df['Teff_berger'].values
    err_t_up = model_df['E_Teff'].values
    err_t_down = model_df['e_Teff'].values
    Teff_errs = (err_t_up - err_t_down)/2
    Prots = model_df[prot_label].values
    Prot_errs = model_df[err_prot_label].values
    cache_id = chache_id
    age_grid = np.linspace(0, 2600, 500)
    star_ids = model_df['KID'].values

    # Prot = Prots[10]
    # Prot_err = Prot_errs[10]
    # Teff = Teffs[10]
    # Teff_err = Teff_errs[10]
    # age_posterior = gyro_age_posterior(
    #     Prot, Teff,
    #     Prot_err=Prot_err, Teff_err=Teff_err,
    #     age_grid=age_grid
    # )
    # print(age_posterior)
    # csvpaths = gyro_age_posterior_list(
    #     cache_id, Prots, Teffs, Prot_errs=Prot_errs, Teff_errs=Teff_errs,
    #     star_ids=star_ids, age_grid=age_grid,
    #     interp_method="pchip_m67"
    # )

    csvpaths = os.listdir(f'gyro/{chache_id}')
    csvpaths = [os.path.join(f'gyro/{chache_id}', f) for f in csvpaths if 'posterior' in f]

    ids = []
    ages = []
    e_ages_up = []
    e_ages_low = []
    print("total number of files: ", len(csvpaths))
    for i, (csvpath, Prot, Teff) in enumerate(zip(sorted(csvpaths), Prots, Teffs)):
        filename = os.path.basename(csvpath)
        id = filename.split('_')[0]
        try:
            ids.append(id)
            df = pd.read_csv(csvpath)
            r = get_summary_statistics(df.age_grid.values, df.age_post.values)
            ages.append(r['median'])
            e_ages_up.append(r['+1sigma'])
            e_ages_low.append(r['-1sigma'])
        except Exception:
            continue
        msg = f"Age = {r['median']} +{r['+1sigma']} -{r['-1sigma']} Myr."
        # print(f"Teff {int(Teff)} Prot {Prot:.2f} {msg}")
        # print(i)
        if i > 500:
            break
    res_df = pd.DataFrame({'KID':ids, 'age': ages, 'e_age_up': e_ages_up, 'e_age_low': e_ages_low})
    print("number of valid samples: ", len(res_df.dropna()))
    res_df.to_csv(f'tables/gyrointerp_{cache_id}.csv', index=False)
    plt.hist(ages)
    plt.show()


def aigrian_test():

    model_on_aigrain = prepare_df(pd.read_csv('../inference/aigrain_data/astroconf_exp52.csv'),
                                  filter_giants=False,
                                  filter_eb=False,
                                  teff_thresh=False,
                                  filter_contaminants=False)
    acf_on_aigrain = pd.read_csv('../inference/aigrain_data/acf_results_data_aigrain2_clean.csv')
    gps_on_aigrain = pd.read_csv('../inference/aigrain_data/gps_results_data_aigrain2_dual.csv')
    gps_subdf = gps_on_aigrain[gps_on_aigrain['predicted period'] < gps_on_aigrain['predicted period'].max()]
    acf_subdf = acf_on_aigrain[acf_on_aigrain['predicted period'] > 0]
    model_sub_acf = model_on_aigrain[acf_on_aigrain['predicted period'] > 0]
    model_sub_gps = model_on_aigrain[gps_on_aigrain['predicted period'] < gps_on_aigrain['predicted period'].max()]

    scatter_predictions(model_on_aigrain['Period'], model_on_aigrain['predicted period'],
                        model_on_aigrain['period confidence'],
                        name='period', units='Days', title='LightPred', show_acc=False )
    scatter_predictions(acf_on_aigrain['period'], acf_on_aigrain['predicted period'],
                        conf=None, title='ACF', show_acc=False,
                        name='period_acf_aigrain', units='Days', )
    scatter_predictions(gps_on_aigrain['period'], gps_on_aigrain['predicted period'],
                        conf=None, title='GPS', show_acc=False,
                        name='period_gps_aigrain', units='Days', )
    scatter_predictions(acf_subdf['period'], acf_subdf['predicted period'],
                        conf=None, title='ACF', show_acc=False,
                        name='period_acf_aigrain_subset', units='Days', )
    scatter_predictions(gps_subdf['period'], gps_subdf['predicted period'],
                        conf=None, title='GPS', show_acc=False,
                        name='period_gps_aigrain_subset', units='Days', )
    scatter_predictions(model_sub_acf['Period'], model_sub_acf['predicted period'],
                        model_sub_acf['period confidence'],
                        name='period_sub_acf', units='Days', title='LightPred', show_acc=False )
    scatter_predictions(model_sub_gps['Period'], model_sub_gps['predicted period'],
                        model_sub_gps['period confidence'],
                        name='period_sub_acf', units='Days', title='LightPred', show_acc=False)
    model_acc, model_acc20, model_error, acf_acc, acf_acc20, acf_error = compare_period_on_mock(model_on_aigrain, acf_on_aigrain)
    print("resuls acf - ", acf_acc, acf_acc20, acf_error)
    model_acc, model_acc20, model_error, gps_acc, gps_acc20, gps_error = compare_period_on_mock(model_on_aigrain, gps_on_aigrain,
                                                                        ref_name='gps')
    (model_acf_acc, model_acf_acc20, model_acf_error, subset_acf_acc, subset_acf_acc20,
     subset_acf_error) = compare_period_on_mock(model_sub_acf, acf_subdf)
    (model_gps_acc, model_gps_acc20, model_gps_error, subset_gps_acc, subset_gps_acc20,
     subset_gps_error) = compare_period_on_mock(model_sub_gps, gps_subdf)
    print("resuls acf - ", acf_acc, acf_acc20, acf_error)
    print("results gps - ", gps_acc, gps_acc20, gps_error)
    print("resuls acf subset - ", subset_acf_acc, subset_acf_acc20, subset_acf_error)
    print("results gps subset - ", subset_gps_acc, subset_gps_acc20, subset_gps_error)
    print("results model - ", model_acc, model_acc20, model_error)
    print("results model acf subset - ", model_acf_acc, model_acf_acc20, model_acf_error)
    print("results model gps subset - ", model_gps_acc, model_gps_acc20, model_gps_error)
    print("fraction of points acf subset: ", len(acf_subdf)/ len(acf_on_aigrain))
    print("fraction of points gps subset: ", len(gps_subdf)/ len(gps_on_aigrain))
    aigrain_acc = .68
    aigrain_acc20 = .80
    acc10 = [model_acc, aigrain_acc, acf_acc, gps_acc]
    acc20 = [model_acc20, aigrain_acc20, acf_acc20, gps_acc20]
    errs = [model_error, acf_error, gps_error]
    errs_valid = [model_acf_error, subset_acf_error, subset_gps_error]
    names = ['LightPred', 'Aigrain et al.', 'ACF ours', 'GPS ours']
    fig, axis = plt.subplots(1,2, figsize=(26,14))
    axis[0].scatter(names, acc10, label='< 10% Error', color='r', s=400)
    axis[0].scatter(names, acc20, label='< 20% Error', color='orange', s=400)
    for i in range(len(names)):
        axis[0].plot([names[i], names[i]], [acc10[i], acc20[i]], color='gray', linestyle='--', linewidth=2)
    axis[0].legend()
    axis[0].set_ylabel('Accuracy (%)')
    axis[1].scatter(['LightPred', 'ACF ours', 'GPS ours'], errs, label='All Samples', color='r', s=400)
    axis[1].scatter(['LightPred', 'ACF ours', 'GPS ours'], errs_valid, label='ACF Valid points', color='orange', s=400)
    for i in range(len(errs)):
        axis[1].plot([['LightPred', 'ACF ours', 'GPS ours'][i],['LightPred', 'ACF ours', 'GPS ours'][i]], [errs[i], errs_valid[i]], color='gray', linestyle='--',
                     linewidth=2)
    axis[1].legend()
    axis[1].set_ylabel('Mean Absolute Error (Days)')
    plt.tight_layout()
    plt.savefig('../imgs/aigrain_test.png')
    plt.show()

    # res_df = pd.DataFrame({"acc10":[model_acc], "acc20":[model_acc20],
    #                        "acf_acc10":[acf_acc], "acf_acc20":[acf_acc20],
    #                        "gps_acc10":[gps_acc], "gps_acc20":[gps_acc20],})
    #

def get_crowdsap(kepler_df):
    results = []
    kids = []
    for i, row in kepler_df.iterrows():
        kid = row['KID']
        res = lk.search_lightcurve(f'KIC {kid}', cadence='long', author='kepler')
        res = res.download_all()
        crowds = []
        for r in res.data:
            meta = r.meta
            crowds.append(meta['CROWDSAP'])
        results.append(np.mean(crowds))
        kids.append(kid)
        print(i, len(results))
        if i > 10:
            break

def download_samples(dir, num_samples=100):
    df_path = [f for f in os.listdir(dir) if f.endswith('csv')][0]
    df = pd.read_csv(os.path.join(dir, df_path))
    for i, row in df.iterrows():
        kid = row['KID']
        print("downloading ", kid)
        res = lk.search_lightcurve(f'KIC {kid}', cadence='long', author='kepler')
        lc = res.download_all().stitch()
        lc.plot()
        plt.savefig(f'{dir}/{kid}.png')
        plt.close()
        if i > num_samples:
            break

def download_multiple_dirs(dirs=['cpcb', 'eb', 'pollution']):
    for dir in dirs:
        download_samples(f'samples/{dir}')

def add_teff(df_dir, teff_dir):
    regex = 'q_(\d)*'
    for p in os.listdir(df_dir):
        match = re.search(regex, p)
        if match:
            q = match.group(1)
        df = pd.read_csv(os.path.join(df_dir, p))
        for t in os.listdir(teff_dir):
            q_t = t.removesuffix('.csv').split('_')[-1]
            if q_t == q:
                df_t = pd.read_csv(os.path.join(teff_dir, t))
                df['Teff'] = df_t.merge(df, on='KID')['Teff']
                df.to_csv(os.path.join(df_dir, p), index=False)


def compare_periods_different_runs(df1, df2):
    df_merged = df1.merge(df2, on='KID', suffixes=[' 1', ' 2'])
    df_merged['diff'] = np.abs(df_merged['predicted period 1'] - df_merged['predicted period 2'])
    plt.hexbin(df_merged['predicted period 1'], df_merged['predicted period 2'], mincnt=1, gridsize=100)
    plt.show()
    plt.hexbin(np.arange(len(df_merged['diff'])), df_merged['diff'], mincnt=1, gridsize=100)
    plt.show()
    plt.hist(df_merged['predicted period 1'], bins=40, histtype='step', density=True, label='df1')
    plt.hist(df_merged['predicted period 2'], bins=40, histtype='step', density=True, label='df2')
    plt.legend()
    plt.show()

def ensemble_mock(paths, group_col=None, prepare=True):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        if prepare:
               df = prepare_df(filter_giants=False, filter_eb=False, teff_thresh=False, filter_contaminants=False)
        dfs.append(df)
    concatenated_df = pd.concat(dfs, axis=0)
    if group_col is None:
        final_df = concatenated_df.groupby(concatenated_df.index).mean()
    else:
        numeric_columns = concatenated_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_df = concatenated_df[numeric_columns]
        mean_df = numeric_df.groupby(group_col).mean()
        non_numeric_df = dfs[0].select_dtypes(exclude=['float64', 'int64'])
        non_numeric_df[group_col] = dfs[0][group_col]
        non_numeric_df.set_index(group_col, inplace=True)

        final_df = non_numeric_df.join(mean_df, on=group_col)
    # scatter_predictions(mean_df['Period'], mean_df['predicted period'], mean_df['period confidence'],
    #                     name='ensemble', units='Days')
    # scatter_predictions(mean_df['Inclination'], mean_df['predicted inclination'], mean_df['inclination confidence'],
    #                     name='ensemble', units='Deg')
    return final_df
def test_dataset():
    all_samples = pd.read_csv('tables/all_kepler_samples.csv')
    qs = list(range(3,17))
    errs = []
    for i in range(500):
        idx = np.random.randint(0, len(all_samples))
        kic = all_samples.loc[idx, 'KID']
        num_qs = all_samples.loc[idx, 'number_of_quarters']
        search_results = lk.search_lightcurve('KIC '+ str(kic), mission='kepler', cadence='long',
                                              quarter=qs)
        print(len(search_results) - num_qs)
        if (len(search_results) - num_qs):
            errs.append(idx)

    print(len(errs)/i)


def eb_classifier(df):
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.model_selection import GridSearchCV, KFold, train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd

    # Load and prepare the original dataset
    dataset = pd.read_csv('tables/eb_dataset.csv')
    dataset['target'] = dataset['target'].astype('int64')
    dataset = dataset[['target', 'total error', 'R_{var}', 's_{ph}', 'Teff', 'kmag', 'mean_period_confidence']]
    dataset.fillna(0, inplace=True)

    # Split into training and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.4, random_state=0)
    y_train = train_set['target']
    X_train = train_set.drop(columns='target')
    y_test = test_set['target']
    X_test = test_set.drop(columns='target')

    # Train the Random Forest model
    model = RF(min_samples_leaf=10, random_state=0)
    param_grid = {"n_estimators": [10, 20, 50, 100]}
    cv = KFold(n_splits=4, shuffle=True, random_state=0)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        return_train_score=True,
        cv=cv,
    ).fit(X_train, y_train)

    # Store the best Random Forest model
    best_rf_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_rf_model.predict(X_test)

    # Evaluate performance on the test set
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"***Random Forest***")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # # Load the new dataframe
    # new_dataset = pd.read_csv('tables/another_dataset.csv')
    df_dataset = df[['total error', 'R_{var}', 's_{ph}', 'Teff', 'kmag', 'mean_period_confidence']]
    df_dataset.fillna(0, inplace=True)

    # Apply the model to the new dataframe
    new_predictions = best_rf_model.predict(df_dataset)
    df['eb_classification'] = new_predictions

    # If the new dataframe contains ground truth labels, evaluate the performance
    # if 'target' in new_dataset.columns:
    #     y_new = new_dataset['target']
    #     accuracy_new = accuracy_score(y_new, new_predictions)
    #     report_new = classification_report(y_new, new_predictions)
    #     print(f"***Performance on New Dataset***")
    #     print(f"Test Accuracy: {accuracy_new}")
    #     print(f"Classification Report:\n{report_new}")

    return df

def compare_acf_mcq(exp_num):
    acf_df = pd.read_csv(f'tables/kepler_acf_pred_exp{exp_num}.csv')
    mcq_df = pd.read_csv('tables/Table_1_Periodic.txt')
    merged_df = acf_df.merge(mcq_df, on='KID')
    merged_df['diff'] = np.abs(merged_df['Prot'] - merged_df['predicted acf_p'])
    acc = (merged_df['diff'] < merged_df['Prot']*0.1).sum() / len(merged_df)
    plt.scatter(merged_df['Prot'], merged_df['predicted acf_p'])
    plt.title(f'experiment {exp_num}, accuracy - {acc:.2f}, {len(merged_df)} samples')
    plt.show()

def stardate(model_df, exp_name):

    gaia_kepler = Table.read('tables/kepler_dr3_4arcsec.fits', format='fits').to_pandas()
    model_df = model_df.merge(gaia_kepler[['kepid', 'bp_rp', 'parallax', 'parallax_error']].drop_duplicates('kepid'),
                              left_on='KID', right_on='kepid')
    ages = []
    ages_iso = []
    errs = []
    ages_mcq14 = []
    ages_iso_mcq14 = []
    errs_mcq14 = []
    for i, row in model_df.iterrows():
        # err_teff = (row['E_Teff'] + row['e_Teff'])/2
        # err_logg = (row['E_logg'] + row['e_logg'])/2
        # err_feh = (row['E_FeH'] + row['e_FeH'])/2
        # iso_params = {"teff": (row['Teff'], err_teff),  # Teff with uncertainty.
        #               "logg": (row['logg'], err_logg),  # logg with uncertainty.
        #               "feh": (row['FeH'], err_feh),  # Metallicity with uncertainty.
        #               "parallax": (row['parallax'], row['parallax_error']),  # Parallax in milliarcseconds.
        #               "maxAV": row['Avmag']}  # Maximum extinction
        bprp = row['bp_rp']  # Gaia BP - RP color.

        prot_mcq14, prot_err_mcq14 = row['Prot'], row['Prot_err']
        log10_period_mcq14 = np.log10(prot_mcq14)
        log10_age_yrs_mcq14 = age_model(log10_period_mcq14, bprp)
        ages_mcq14.append(10 ** log10_age_yrs_mcq14 * 1e-9)
        # star = sd.Star(iso_params, prot=prot_mcq14, prot_err=prot_err_mcq14)  # Here's where you add a rotation period
        # try:
        #     star.fit(max_n=1000)
        #     age, errm, errp, samples = star.age_results()
        # except Exception as e:
        #     print(e)
        age, errm, errp, samples = None, None, None, None
        ages_iso_mcq14.append(age)
        errs_mcq14.append((errm, errp))
        # print("stellar age = {0:.2f} + {1:.2f} - {2:.2f}".format(age, errp, errm))

        prot, prot_err = row['predicted period'], row['sigma error']
        log10_period = np.log10(prot)
        log10_age_yrs = age_model(log10_period, bprp)
        ages.append((10 ** log10_age_yrs) * 1e-9)
        # if i % 1000 == 0:
        #     print(i)
        # star = sd.Star(iso_params, prot=prot, prot_err=prot_err)  # Here's where you add a rotation period
        # try:
        #     star.fit(max_n=1000)
        #     age, errm, errp, samples = star.age_results()
        # except Exception as e:
        #     print(e)
        age, errm, errp, samples = None, None, None, None
        ages_iso.append(age)
        errs.append((errm, errp))
        # print("stellar age = {0:.2f} + {1:.2f} - {2:.2f}".format(age, errp, errm))
    len_mcq14 = len(model_df[~model_df['Prot'].isna()])
    bins = np.linspace(0,10, 50)
    plt.hist(ages, density=True, histtype='step', bins=bins, label=f'LightPred {len(ages)} samples')
    plt.hist(ages_mcq14, density=True, histtype='step', bins=bins, label=f'McQ14 {len_mcq14} samples')
    plt.legend()
    plt.show()
    model_df['age'] = ages
    model_df['age_iso'] = ages_iso
    model_df['age_iso_err'] = errs
    model_df['age_mcq14'] = ages_mcq14
    model_df['age_iso_mcq14'] = ages_iso_mcq14
    model_df['age_iso_mcq14_err'] = errs_mcq14
    model_df.to_csv(f'tables/stardate_iso_age_{exp_name}.csv')


def test_mock_data(num_tests=10):
    from sklearn.model_selection import train_test_split
    Nlc = 50000
    sim_labels = pd.read_csv('../butter/data_cos_old/simulation_properties.csv')
    res = []
    p = []
    for _ in range(num_tests):
        idxf_list = [f'{idx:d}'.zfill(int(np.log10(Nlc)) + 1) for idx in range(Nlc)]
        train_list, test_list = train_test_split(idx_list, test_size=0.1)
        train_list, val_list = train_test_split(train_list, test_size=0.1)
        train_df = sim_labels.iloc[train_list]
        test_df = sim_labels.iloc[test_list]
        test_res, test_p = ks_2samp(sim_labels['Period'], test_df['Period'])
        res.append(test_res)
        p.append(test_p)
    print(np.mean(res), np.mean(p))

if __name__ == "__main__":
    create_final_predictions('tables/kepler_model_pred_exp45.csv')



import os
from matplotlib import pyplot as plt
import glob
import json
import itertools
import numpy as np
import pandas as pd
import re
import torch
# import wandb
import yaml
from collections import OrderedDict
import time
import lightkurve as lk
# from butterpy_local import Spots
from sklearn.model_selection import train_test_split
from astropy.io import fits
from typing import Callable, Dict, Optional, Tuple, Type, Union, List
import collections
import contextlib
import re
import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)   

from util.period_analysis import analyze_lc


def read_fits(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a FITS file and returns the PDCSAP_FLUX and TIME columns as numpy arrays.

    Args:
        filename (str): The path to the FITS file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The PDCSAP_FLUX and TIME columns as numpy arrays.
    """
    
    with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          meta = hdulist[0].header
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    return x,time, meta

def fill_nan_np(x:np.ndarray, interpolate:bool=True):
    """
    fill nan values in a numpy array

    Args:
         x (np.ndarray): array to fill
         interpolate (bool): whether to interpolate or not

    Returns:
        np.ndarray: filled array
    """
    non_nan_indices = np.where(~np.isnan(x))[0]
    nan_indices = np.where(np.isnan(x))[0]
    if len(nan_indices) and len(non_nan_indices):
        if interpolate:
            # Interpolate NaN values using linear interpolation
            interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])
            # Replace NaNs with interpolated values
            x[nan_indices] = interpolated_values
        else:
            x[nan_indices] = 0
    return x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def replace_zeros_with_average(arr: np.ndarray) -> np.ndarray:
    """
    Replace zero values in an array with the average of neighboring values.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: The array with zero values replaced by the average of neighboring values.
    """
    # Find the indices of zero values
    zero_indices = np.where(arr == 0)[0]
    # print(len(arr))

    # Iterate over each zero index and replace with the average of neighboring values
    for idx in zero_indices:
        before_idx = idx - 1 if idx > 0 else idx
        after_idx = idx + 1 if idx < len(arr) - 1 else idx

        # Find the nearest non-zero neighbors
        while before_idx in zero_indices and before_idx >= 0:
            before_idx -= 1
        while after_idx in zero_indices and after_idx < (len(arr) - 1):
            after_idx += 1

        # Replace zero with the average of neighboring values
        arr[idx] = (arr[before_idx] + arr[after_idx]) / 2

    return arr

def dataset_weights(dl:torch.utils.data.DataLoader, Nlc:int):
    """
    Calculate weights for each sample in the dataset
    based on the number of light curves with the same inclination.

    Args:
        dl (torch.utils.data.Dataloader): The dataloader.
        Nlc (int): The number of light curves.
    Returns:
        torch.Tensor: The weights for each sample in the dataset.
    """
    incl = (np.arcsin(np.random.uniform(0, 1, Nlc))*180/np.pi).astype(np.int16)
    unique, counts = np.unique(incl, return_counts=True)
    weights = torch.zeros(0)
    for i, (x,y,_,_) in enumerate(dl):
        w = 1/counts[(y[:,0]*180/np.pi).squeeze().numpy().astype(np.int16)]
        weights = torch.cat((weights, torch.tensor(w)), dim=0)
    return weights



def load_model(data_dir:str, model:torch.nn.Module,
                distribute:bool, device:torch.device,
                  to_ddp:bool=False, load_params:bool=False):
    """
    Load a model from a directory.

    Args:
        data_dir (str): The directory containing the model files.
        model (torch.nn.Module): The model class.
        distribute (bool): Whether the model was distributed.
        device (torch.device): The device to load the model on.
        to_ddp (bool, optional): distribute the model
        load_params (bool, optional): Load the model parameters.

    Returns:
        Tuple[torch.nn.Module, Dict, str]: The model, network parameters, and model name.
    """
    print("data dir ", data_dir)
    if load_params:
        try:
            with open(f'{data_dir}/net_params.yml', 'r') as f:
                net_params = yaml.load(f, Loader=yaml.FullLoader)
            model = model(**net_params).to(device)
        except FileNotFoundError:
            net_params = None
            model = model.to(device)
    else:
        net_params = None
        model = model.to(device)
    model_name = model.__class__.__name__
    state_dict_files = glob.glob(data_dir + '/*.pth')
    print("loading model from ", state_dict_files[-1])
    
    state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=device)
    if distribute:
        print("loading distributed model")
        # Remove "module." from keys
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                while key.startswith('module.'):
                    key = key[7:]
            new_state_dict[key] = value
        state_dict = new_state_dict
    print(model)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model,net_params,model_name

def load_results(log_path:str, exp_num:int):
    """
    load results from a log path

    Args:
        log_path (str): The path to the log directory.
        exp_num (int): The experiment number.

    Returns:
        List[Dict]: The results from the log directory.
    """
    fit_res = []
    folder_path=f"{log_path}/{exp_num}" #folderpath
    print("folder path ", folder_path, "files: ", os.listdir(folder_path))
    json_files = glob.glob(folder_path + '/*.json')
    print("json files: ", json_files)
    for f in json_files:
        filename = os.path.basename(f)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            output = json.load(f)
        fit_res.append(output)
    return fit_res


def plot_fit(
    fit_res: dict,
    fig: plt.figure = None,
    log_loss: bool = False,
    legend: bool = None,
    train_test_overlay: bool = False,
):
    """
    Plot fit results.

    Args:
        fit_res (dict): The fit results.
        fig (plt.figure, optional): The figure to plot on. Defaults to None.
        log_loss (bool, optional): Whether to plot the loss on a log scale. Defaults to False.
        legend (bool, optional): The legend to use. Defaults to None.
        train_test_overlay (bool, optional): Whether to overlay the train and test results. Defaults to False.

    Returns:
        Tuple[plt.figure, plt.axes]: The figure and axes.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 1 if np.isnan(fit_res['train_acc']).any() else 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()
    if ncols > 1:
        p = itertools.product(enumerate(["train", "val"]), enumerate(["loss", "acc"]))
    else:
        p = itertools.product(enumerate(["train", "val"]), enumerate(["loss"]))
    for (i, traintest), (j, lossacc) in p:
        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data =fit_res[attr]
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes


def plot_all(root_dir:str):
    """
    Plot all fits in a directory.

    Args:
        root_dir (str): The root directory.
    """
    for d in os.listdir(root_dir):
        print(d)
        if os.path.isdir(os.path.join(root_dir, d)):
            fit_res = load_results(root_dir, d)
            if fit_res:
                print("plotting fit for ", d)
                fig, axes = plot_fit(fit_res[0], legend=d, train_test_overlay=True, only_loss=True)
                plt.savefig(f"{root_dir}/{d}/fit.png")

def remove_leading_zeros(s:str):
    """
    Remove leading zeros from a string of numbers and return as integer
    """
    # Remove leading zeros
    s = s.lstrip('0')
    # If the string is now empty, it means it was all zeros originally
    if not s:
        return 0
    # Convert the remaining string to an integer and return
    return int(s)


def create_train_test(data_dir:str):
    """
    Create a train and test set from a directory of data.

    Args:
        data_dir (str): The directory containing the data.

    Returns:
        Tuple[List[str], List[str]]: The train and test sets.
    """
    test = []
    train = []
    df_test = pd.read_csv("/kepler/lightPred/Table_1_Periodic.txt", sep=",")
    kids = set(df_test["KID"].values)
    for d in os.listdir(data_dir):
        d_s = d.split("/")[-1].split("-")[0]
        numbers = re.findall(r'\d+', d_s)[0]
        k = remove_leading_zeros(numbers)
        if k in kids:
            test.append(d)
        else:
            train.append(d)
    return train, test


def extract_object_id(file_name:str):
    """
    Extract the object ID from a file name.

    Args:
        file_name (str): The file name.

    Returns:
        str: The object ID.
    """
    match = re.search(r'kplr(\d{9})-\d{13}_llc.fits', file_name)
    return match.group(1) if match else None


def create_kepler_df(kepler_path:str, table_path:str=None):
    """
    Create a DataFrame of Kepler data files.

    Args:
        kepler_path (str): The path to the Kepler data files.
        table_path (str, optional): The path to the table of Kepler data. Defaults to None.
    Returns:
        pd.DataFrame: The DataFrame of Kepler data files.
    """

    data_files_info = []
    for file in os.listdir(kepler_path):
        obj_id = extract_object_id(file)
        if obj_id:
            data_files_info.append({'KID': obj_id, 'data_file_path':os.path.join(kepler_path, file) })
    if len(data_files_info) == 0:
        print("no files found in ", kepler_path)
        return pd.DataFrame({'KID':[], 'data_file_path':'[]'})
    kepler_df = pd.DataFrame(data_files_info)
    kepler_df['KID'] = kepler_df['KID'].astype('int64')
    kepler_df['data_file_path'] = kepler_df['data_file_path'].astype('string')

    if table_path is None:
        return kepler_df
    table_df = pd.read_csv(table_path)
    final_df = table_df.merge(kepler_df, on='KID', how='inner', sort=False)
    return final_df

def multi_quarter_kepler_df(root_kepler_path:str, Qs:List, table_path:str=None):
    """
    Create a DataFrame of multi-quarter Kepler data files.

    Args:
        root_kepler_path (str): The root path to the Kepler data files.
        Qs (List): The list of quarters to include.
        table_path (str, optional): The path to the table of Kepler data. Defaults to None.
    Returns:
        pd.DataFrame: The DataFrame of multi-quarter Kepler data files.
    """
    
    print("creating multi quarter kepler df with Qs ", Qs, "table path " , table_path)
    dfs = []
    for q in Qs:
        kepler_path = os.path.join(root_kepler_path, f"Q{q}")
        print("kepler path ", kepler_path)
        df = create_kepler_df(kepler_path, table_path)
        print("length of df ", len(df))
        dfs.append(df)
    if 'Prot' in dfs[0].columns:
        if 'Prot_err' in dfs[0].columns:
            merged_df = pd.concat(dfs).groupby('KID').agg({'Prot': 'first', 'Prot_err': 'first', 'Teff': 'first',
            'logg': 'first', 'data_file_path': list}).reset_index()
        else:
            merged_df = pd.concat(dfs).groupby('KID').agg({'Prot': 'first', 'data_file_path': list}).reset_index()
    elif 'i' in dfs[0].columns:
        merged_df = pd.concat(dfs).groupby('KID').agg({'i': 'first', 'data_file_path': list}).reset_index()
    else:
        merged_df = pd.concat(dfs).groupby('KID')['data_file_path'].apply(list).reset_index()
    merged_df['number_of_quarters'] = merged_df['data_file_path'].apply(lambda x: len(x))
    return merged_df


def convert_to_list(string_list:str):
    """
    Convert a string representation of a list to a list.

    Args:
        string_list (str): The string representation of the list.

    Returns:
        List: The list.
    """
    # Extract content within square brackets
    matches = re.findall(r'\[(.*?)\]', string_list)
    if matches:
        # Split by comma, remove extra characters except period, hyphen, underscore, and comma, and strip single quotes
        cleaned_list = [re.sub(r'[^A-Za-z0-9\-/_,.]', '', s) for s in matches[0].split(',')]
        return cleaned_list
    else:
        return []

def convert_to_tuple(string:str):
    """
    Convert a string representation of a tuple to a tuple.

    Args:
        string (str): The string representation of the tuple.

    Returns:
        Tuple: The tuple.
    """
    values = string.strip('()').split(',')
    return tuple(int(value) for value in values)

def convert_ints_to_list(string:str):
    """
    Convert a string representation of a list of integers to a list of integers.

    Args:
        string (str): The string representation of the list of integers.

    Returns:
        List: The list of integers.
    """
    values = string.strip('()').split(',')
    return [int(value) for value in values]

def convert_floats_to_list(string:str):
    """
    Convert a string representation of a list of floats to a list of floats.

    Args:
        string (str): The string representation of the list of floats.
    Returns:
        List: The list of floats.
    """
    string = string.replace(' ', ',')
    string = string.replace('[', '')
    string = string.replace(']', '')
    numbers = string.split(',')    
    return [float(num) for num in numbers if len(num)]

def extract_qs(path:str):
    """
    Extract the quarters numbers from a string.

    Args:
        path (str): The string containing the quarter numbers.

    Returns:
        List: The list of quarter numbers.
    """
    qs_numbers = []
    for p in path:
        match = re.search(r'[\\/]Q(\d+)[\\/]', p)
        if match:
            qs_numbers.append(int(match.group(1)))
    return qs_numbers

def consecutive_qs(qs_list:List[int]):
    """
    calculate the length of the longest consecutive sequence of 'qs'
    Args:
        qs_list (List[int]): The list of quarter numbers.
    Returns:
        int: The length of the longest consecutive sequence of 'qs'.
    """

    max_length = 0
    current_length = 1
    for i in range(1, len(qs_list)):
        if qs_list[i] == qs_list[i-1] + 1:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1
    return max(max_length, current_length)

def find_longest_consecutive_indices(nums:List[int]):
    """
    Find the indices of the longest consecutive sequence of numbers.
    Args:
        nums (List[int]): The list of numbers.
    Returns:
        Tuple[int, int]: The start and end indices of the longest consecutive sequence.
    """
    start, end = 0, 0
    longest_start, longest_end = 0, 0
    max_length = 0

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            end = i
        else:
            start = i

        if end - start > max_length:
            max_length = end - start
            longest_start = start
            longest_end = end

    return longest_start, longest_end

def get_all_samples_df(num_qs:int=8, read_from_csv:bool=True):
    """
    Get all samples DataFrame.
    Args:
        num_qs (int, optional): The minimum number of quarters. Defaults to 8.
    Returns:
        pd.DataFrame: The DataFrame of all samples.
    """
    if read_from_csv:
        kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')
    else:
        kepler_df = multi_quarter_kepler_df('data/', table_path=None, Qs=np.arange(3,17))
    try:
        kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
    except TypeError:
        pass
    kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
    kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
    if num_qs is not None:
        kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]
    kepler_df['longest_consecutive_qs_indices'] = kepler_df['longest_consecutive_qs_indices'].apply(convert_ints_to_list)
    return kepler_df

def break_samples_to_segments(num_qs:int):
    """
    Break samples to segments.

    Args:
        num_qs (int): The number of quarters.

    Returns:
        pd.DataFrame: The DataFrame of samples.
    """
    kois_df = pd.read_csv('tables/ref_merged.csv')
    kois_df.dropna(subset=['longest_consecutive_qs_indices'], inplace=True)
    kois_df['longest_consecutive_qs_indices'] = kois_df['longest_consecutive_qs_indices'].apply(
        lambda x: tuple(map(int, x.strip('[]').split(','))))
    kois_df['data_file_path'] = kois_df['data_file_path'].apply(convert_to_list)
    res_df = pd.DataFrame(columns=kois_df.columns)
    for idx, row in kois_df.iterrows():
        if len(row['data_file_path']) < num_qs:
            continue
        # print(len(row['data_file_path']), num_qs)
        for j in range(len(row['data_file_path']) - num_qs):
            sub_row = row.copy()
            sub_samples = row['data_file_path'][j:j + num_qs]
            sub_row['data_file_path'] = sub_samples
            sub_row['qs'] = sub_row['qs'][j:j + num_qs]
            sub_row['consecutive_qs'] = num_qs
            sub_row['longest_consecutive_qs_indices'] = (0, num_qs)
            sub_row['number_of_quarters'] = num_qs
            sub_row['KID'] = f"{row['KID']}_{j}"
            sub_row['kepler_name'] = f"{row['kepler_name']}_{j}"
            sub_df = pd.DataFrame(sub_row).transpose()
            res_df = pd.concat([res_df, sub_df], ignore_index=True)
    return res_df

def read_kepler_row(row:pd.Series, skip_idx:int=0, num_qs:int=17):
    """
    Read a row from the Kepler DataFrame.

    Args:
        row (pd.Series): The row from the DataFrame.
        skip_idx (int, optional): The index to skip. Defaults to 0.
        num_qs (int, optional): The number of quarters. Defaults to 17.
    Returns:
        Tuple[np.ndarray, Dict, List]: The light curve, information, and effective quarters.
    """
    try:
        q_sequence_idx = row['longest_consecutive_qs_indices']
        info = dict()
        if q_sequence_idx is np.nan:
            q_sequence_idx = (0, 0)
            # x_tot, meta = np.zeros((self.seq_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None}
            # effective_qs = []
        if isinstance(q_sequence_idx, str):
            q_sequence_idx = q_sequence_idx.strip('()').split(',')
            q_sequence_idx = [int(i) for i in q_sequence_idx]
        if q_sequence_idx[1] > q_sequence_idx[0] and skip_idx < (q_sequence_idx[1] - q_sequence_idx[0]):
            for i in range(q_sequence_idx[0] + skip_idx, q_sequence_idx[1]):
            # print(row['data_file_path'])
                x,time,meta = read_fits(row['data_file_path'][i])
                x /= x.max()
                x = fill_nan_np(np.array(x), interpolate=True)
                if i == q_sequence_idx[0] + skip_idx:
                    x_tot = x.copy()
                else:
                    border_val = np.mean(x) - np.mean(x_tot)
                    x -= border_val
                    x_tot = np.concatenate((x_tot, np.array(x)))
                if i == num_qs:
                    break
            effective_qs = row['qs'][q_sequence_idx[0]: q_sequence_idx[1]]
            info['Teff'] = meta['TEFF'] if meta['TEFF'] is not None else 0
            info['R'] = meta['RADIUS'] if meta['RADIUS'] is not None else 0
            info['logg'] = meta['LOGG'] if meta['LOGG'] is not None else 0
        else:
            effective_qs = []
            x_tot, info = None, {'TEFF': None, 'RADIUS': None, 'LOGG': None}
    except (TypeError, ValueError, FileNotFoundError)  as e:
        print("Error: ", e)
        effective_qs = []
        x_tot, info = None, {'TEFF': None, 'RADIUS': None, 'LOGG': None}
    return x_tot, info, effective_qs

def kepler_collate_fn(batch:List):
    """
    Collate function for the Kepler dataset.        
    """
    # Separate the elements of each sample tuple (x, y, mask, info) into separate lists
    xs, ys, masks, masks_y, infos, infos_y = zip(*batch)

    # Convert lists to tensors
    xs_tensor = torch.stack(xs, dim=0)
    ys_tensor = torch.stack(ys, dim=0)
    masks_tensor = torch.stack(masks, dim=0)
    masks_y_tensor = torch.stack(masks_y, dim=0)
    return xs_tensor, ys_tensor, masks_tensor, masks_y_tensor, infos, infos_y



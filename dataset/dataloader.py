import copy
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import acf as A
import random
from util.utils import *
from matplotlib import pyplot as plt
from scipy.signal import stft, correlate2d
from scipy import signal
import time
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter as savgol
import csv
from dataset.sampler import DynamicRangeSampler

# import torchaudio.transforms as T


cad = 30
DAY2MIN = 24*60
# min_p, max_p = 0,70
# min_lat, max_lat = 0, 80
# min_cycle, max_cycle = 1, 10
# min_i, max_i = 0, np.pi/2
# min_tau, max_tau = 1,10 
# min_n, max_n = 0, 5000
# min_shear, max_shear = 0, 1
T_SUN = 5777
# boundary_values_dict = {'Period': (min_p, max_p), 'Inclination': (min_i, max_i),
#  'Decay Time': (min_tau, max_tau), 'Cycle Length': (min_cycle, max_cycle), 'Spot Max': (min_lat, max_lat),
#  'n_spots': (min_n, max_n), 'Shear': (min_shear, max_shear)}

non_period_table_path = "/data/lightPred/Table_2_Non_Periodic.txt"
kepler_path = "/data/lightPred/data"


def create_boundary_values_dict(df):
  boundary_values_dict = {}
  for c in df.columns:
    if c not in boundary_values_dict.keys():
      if c == 'Butterfly':
        boundary_values_dict[c] = bool(df[c].values[0])
      else:
        min_val, max_val = df[c].min(), df[c].max()
        boundary_values_dict[f'min {c}'] = float(min_val)
        boundary_values_dict[f'max {c}'] = float(max_val)
  return boundary_values_dict

def mask_array(array, mask_percentage=0.15, mask_value=-1, vocab_size=1024):
  if len(array.shape) == 1:
    array = array.unsqueeze(0)
  len_s = array.shape[1]  
  inverse_token_mask = torch.ones(len_s, dtype=torch.bool)  

  mask_amount = round(len_s * mask_percentage)  
  for _ in range(mask_amount):  
      i = random.randint(0, len_s - 1)  

      if random.random() < 0.8:  
          array[:,i] = mask_value  
      else:
          array[:,i] = random.randint(0, vocab_size - 1)  
      inverse_token_mask[i] = False  
  return array, inverse_token_mask

class SubsetF(Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        if self.indices.shape == ():
            print('this happens: Subset')
            return 1
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx % len(self.indices)]]

class TimeSsl(Dataset):
    """
    A dataset for time series data with self-supervised learning tasks.

    """
    def __init__(self, root_dir:str, path_list:List,
                   df:pd.DataFrame=None,
                    t_samples:int=512,
                    skip_idx:int=0,
                    num_qs:int=8,
                    norm:str='std',
                    transforms:object=None,
                    acf:bool=False,
                    return_raw:bool=False):
      """
      A dataset for time series data with self-supervised learning tasks.
      Args:
          root_dir (str): root directory
          path_list (List): list with samples paths
          df (pd.DataFrame, optional): dataframe of samples. Defaults to None.
          t_samples (int, optional): length of samples. Defaults to 512.
          skip_idx (int, optional): skipping index. Defaults to 0.
          num_qs (int, optional): number of quarters. Defaults to 8.
          norm (str, optional): normalizing method. Defaults to 'std'.
          transforms (object, optional): data transformation. Defaults to None.
          acf (bool, optional): calculate ACF. Defaults to False.
          return_raw (bool, optional): return raw lightcurve (in case of ACF calculation). Defaults to False.
      """
      # self.idx_list = idx_list
      self.path_list = path_list
      self.cur_len = None
      self.df = df
      self.root_dir = root_dir
      self.seq_len = t_samples
      self.norm = norm
      self.skip_idx = skip_idx
      self.num_qs = num_qs
      self.transforms = transforms
      self.acf = acf
      self.return_raw = return_raw
      self.length = len(self.df) if self.df is not None else len(self.path_list)
      self.num_bad_samples = 0
      
    def __len__(self):
        return self.length
    
    def read_data(self, idx, interpolate=True):
        path = self.paths_list[idx]
        filename = os.path.join(self.root_dir, path)
        try:
          x, time, meta = read_fits(filename)
          x = fill_nan_np(x, interpolate=interpolate)
          self.cur_len = len(x)
        except TypeError as e:
            print("TypeError: ", e)
            return np.zeros((1,self.cur_len)), np.zeros((1,self.cur_len)) 
        if self.seq_len:
          f = interp1d(time, x)
          new_t = np.linspace(time[0], time[-1], self.seq_len)
          x = f(new_t)
        # x = torch.tensor(x)
        return x, meta

    def read_row(self, idx):
        row = self.df.iloc[idx]
        if 'prot' in row.keys():
          y_val = row['prot']
        elif 'Prot' in row.keys():
          y_val = row['Prot']
        else:
          y_val = row['predicted period']
        # print(row['KID'])
        try:
          q_sequence_idx = row['longest_consecutive_qs_indices']
          if q_sequence_idx is np.nan:
              q_sequence_idx = (0, 0)
              # x_tot, meta = np.zeros((self.seq_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None}
              # effective_qs = []
          if isinstance(q_sequence_idx, str):
              q_sequence_idx = q_sequence_idx.strip('()').split(',')
              q_sequence_idx = [int(i) for i in q_sequence_idx]
          if q_sequence_idx[1] > q_sequence_idx[0] and self.skip_idx < (q_sequence_idx[1] - q_sequence_idx[0]):
              for i in range(q_sequence_idx[0] + self.skip_idx, q_sequence_idx[1]):
                # print(row['data_file_path'])
                x,time,meta = read_fits(row['data_file_path'][i])

                x /= x.max()
                x = fill_nan_np(np.array(x), interpolate=True)
                if i == q_sequence_idx[0] + self.skip_idx:
                  x_tot = x.copy()
                else:
                  border_val = np.mean(x) - np.mean(x_tot)
                  x -= border_val
                  x_tot = np.concatenate((x_tot, np.array(x)))
                if i == self.num_qs:
                  break
              effective_qs = row['qs'][q_sequence_idx[0]: q_sequence_idx[1]]
              self.cur_len = len(x)
          else:
              self.num_bad_samples += 1
              effective_qs = []
              x_tot, meta = np.zeros((self.seq_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None, 'KMAG': None}
          # meta['qs'] = row['qs']
        except (TypeError, ValueError, FileNotFoundError, OSError)  as e:
            print("Error: ", e)
            effective_qs = []
            x_tot, meta = np.zeros((self.seq_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None, 'KMAG': None}
        return x_tot, meta, effective_qs, y_val
    
    def __getitem__(self, idx):
        if self.df is not None:
          x, meta, qs = self.read_row(idx)
        else:
          x, meta =  self.read_data(idx).float()
          x /= x.max()
        info = {'idx': idx, 'qs': qs}
        if self.transforms is not None:
          x, mask, info = self.transforms(x, mask=None, info=info)
          if self.seq_len > x.shape[0]:
            x = np.pad(x, ((0, self.seq_len - x.shape[-1]), (0,0)), "constant", constant_values=0)
            if mask is not None:
              mask = np.pad(mask, ((0, self.seq_len - mask.shape[-1]), (0,0)), "constant", constant_values=0)
        x = fill_nan_np(x, interpolate=True)
        x = torch.tensor(x.T[:, :self.seq_len]).unsqueeze(0)
        if mask is not None:
          mask = torch.tensor(mask.T[:, :self.seq_len]).unsqueeze(0)
        if self.ssl_tf is not None:
          x1 = self.ssl_tf(copy.deepcopy(x.transpose(1,2))).transpose(1,2).squeeze(0)
          x2 = self.ssl_tf(copy.deepcopy(x.transpose(1,2))).transpose(1,2).squeeze(0)
          x1 = (x1 - x1.mean())/(x1.std()+1e-8)
          x2 = (x2 - x2.mean())/(x2.std()+1e-8)
          return x1.float(), x2.float()
        else:
          return x, torch.zeros((1,self.seq_len))
        
class KeplerDataset(TimeSsl):
  """
  A dataset for Kepler data.
  """
  def __init__(self, root_dir:str,
                path_list:List,
                df:pd.DataFrame=None,
                mask_prob:float=0,
                mask_val:float=-1,
                np_array:bool=False,
                prot_df:pd.DataFrame=None,
                keep_ratio:float=0.8,
                random_ratio:float=0.2,
                uniform_bound:int=2,
                target_transforms:object=None,
                **kwargs):
    """
    dataset for Kepler data
    Args:
        root_dir (str): root directory of the data
        path_list (List): list of paths to the data
        df (pd.DataFrame, optional): dataframe with the data. Defaults to None.
        mask_prob (float, optional): masking probability
        mask_val (float, optional): masking value. Defaults to -1.
        np_array (bool, optional): flag to load data as numpy array. Defaults to False.
        prot_df (pd.DataFrame, optional): refernce Dataframe (like McQ14). Defaults to None.
        keep_ratio (float, optional): ratio of masked values to keep. Defaults to 0.8.
        random_ratio (float, optional): ratio of masked values to convert into random numbers. Defaults to 0.2.
        uniform_bound (int, optional): bound for random numbers range. Defaults to 2.
        target_transforms (object, optional): transformations to target. Defaults to None.
    """
    super().__init__(root_dir, path_list, df=df, **kwargs)
    # self.df = df
    # self.length = len(self.df) if self.df is not None else len(self.paths_list)
    self.mask_prob = mask_prob
    self.mask_val = mask_val
    self.np = np_array
    self.keep_ratio = keep_ratio
    self.random_ratio = random_ratio
    self.uniform_bound = uniform_bound
    self.target_transforms = target_transforms
    self.prot_df = prot_df
    if df is not None and 'predicted period' not in df.columns:
      if prot_df is not None:
        self.df = pd.merge(df, prot_df[['KID', 'predicted period']], on='KID')
      else:
        self.df['predicted period'] = np.nan
      


  def mask_array(self, array, mask_percentage=0.15, mask_value=-1):
      # if len(array.shape) == 1:
      #   array = array.unsqueeze(0)
      len_s = array.shape[-1]
      inverse_token_mask = torch.ones_like(array, dtype=torch.bool)

      mask_amount = round(len_s * mask_percentage)
      for _ in range(mask_amount):  
          i = random.randint(0, len_s - 1)  

          if random.random() < 0.95:  
              array[:, i] = mask_value
          else:
              array[:, i] = random.uniform(array.min(),array.max())
          inverse_token_mask[:, i] = False
      return array, inverse_token_mask
  
  def read_np(self, idx):
    x = np.load(os.path.join(self.root_dir, self.path_list[idx]))
    return torch.tensor(x), dict()


  def apply_mask_one_channel(self, x, mask):
      r = torch.rand_like(x)
      keep_mask = (~mask | (r <= self.keep_ratio)).to(x.dtype)
      random_mask = (mask & (self.keep_ratio < r)
                     & (r <= self.keep_ratio + self.random_ratio)).to(x.dtype)
      # token_mask = (mask & ((1 - self.token_ratio) < r)).to(x.dtype)
      xm, xM = -self.uniform_bound, self.uniform_bound
      out = x * keep_mask + (torch.rand_like(x) * (xM - xm) + xm) * random_mask
      out[torch.isnan(out)] = 0.
      return out

  def apply_mask(self, x, mask):
      if mask is None:
          out = x
          mask = torch.zeros_like(x).bool()
          return out, mask
      mask = torch.tensor(mask.T[:, :self.seq_len])
      out = x.clone()
      for c in range(x.shape[0]):
          out[c] = self.apply_mask_one_channel(x[c], mask)
      mask = mask.repeat(x.shape[0],1)
      return out, mask

  def __getitem__(self, idx):
    tic = time.time()
    if self.df is not None:
      x, meta, qs, p_val = self.read_row(idx)
    elif self.np:
      x, meta = self.read_np(idx)
      qs = [] # to be implemented
      p_val = np.nan
    else:
      x, meta = self.read_data(idx).float()
      x /= x.max()
      qs = [] # to be implemented
      p_val = np.nan
    info = {'idx': idx}
    info['qs'] = qs
    info['period'] = p_val
    info_y = copy.deepcopy(info)
    x /= x.max()
    target = x.copy()
    mask = None
    if self.transforms is not None:
          x, mask, info = self.transforms(x, mask=None, info=info)
          if self.seq_len > x.shape[0]:
            x = F.pad(x, ((0,0, 0, self.seq_len - x.shape[-1])), "constant", value=0)
            if mask is not None:
              mask = F.pad(mask, ((0,0,0, self.seq_len - mask.shape[-1])), "constant", value=0)
          x = x.T[:, :self.seq_len].nan_to_num(0)
    else:
       x = torch.tensor(x)
    if self.target_transforms is not None:
            target, mask_y, info_y = self.target_transforms(target, mask=None, info=info_y)
            if self.seq_len > target.shape[0]:
                target = F.pad(target, ((0,0,0, self.seq_len - target.shape[-1])), "constant", value=0)
                if mask_y is not None:
                  mask_y = F.pad(mask_y, ((0,0,0, self.seq_len - mask_y.shape[-1])), "constant", value=0)
            target = target.T[:, :self.seq_len].nan_to_num(0)
    else:
        target = x.clone()
        mask_y = None
    # print(x.shape, target.shape)
    x,mask = self.apply_mask(x, mask)
    target, mask_y = self.apply_mask(target, mask_y)

    if len(meta):
      info['Teff'] = meta['TEFF'] if meta['TEFF'] is not None else 0
      info['R'] = meta['RADIUS'] if meta['RADIUS'] is not None else 0
      info['logg'] = meta['LOGG'] if meta['LOGG'] is not None else 0
      info['kmag'] = meta['KMAG'] if meta['KMAG'] is not None else 0
    info['path'] = self.df.iloc[idx]['data_file_path'] if self.df is not None else self.path_list[idx]
    info['KID'] = self.df.iloc[idx]['KID'] if self.df is not None else self.path_list[idx].split("/")[-1].split("-")[0].split('kplr')[-1]
    toc = time.time()
    info['time'] = toc - tic
    return x.float(), target.float(), mask, mask_y, info, info_y

   
class KeplerNoiseDataset(KeplerDataset):
  """
  A dataset for Kepler data with noise
  """
  def __init__(self, root_dir:str,
            path_list:List,
            df:pd.DataFrame=None,
            **kwargs ):
    """
    A dataset for Kepler data as noise for simulations
    Args:
        root_dir (str): root directory
        path_list (List): list of paths to the data
        df (pd.DataFrame, optional): Dataframe of samples. Defaults to None.
    """
    super().__init__(root_dir, path_list, df=df, acf=False, norm='none', **kwargs)
    self.samples = []
    # print(f"preparing kepler data of {len(self.df)} samples...")
    for i in range(len(self.df)):
      # if i % 1000 == 0:
        # print(i, flush=True)
      x, masked_x, inv_mask, info = super().__getitem__(i)
      self.samples.append((x, masked_x, inv_mask, info))

  def __getitem__(self, idx):
    x, masked_x, inv_mask, info = self.samples[idx]
    return x.float(), masked_x.squeeze().float(), inv_mask, info 

class KeplerLabeledDataset(KeplerDataset):
  """
  A dataset for Kepler data with labels
  """
  def __init__(self, root_dir, path_list, cos_inc=False,
               classification=False, num_classes=10, boundaries_dict=dict(), **kwargs):
      super().__init__(root_dir, path_list, **kwargs)
      self.cos_inc = cos_inc
      self.cls = classification
      self.num_classes = num_classes
      self.boundary_values_dict = boundaries_dict

  def get_labels(self, row):
    min_p, max_p = self.boundary_values_dict['min Period'], self.boundary_values_dict['max Period']
    min_i, max_i = self.boundary_values_dict['min Inclination'], self.boundary_values_dict['max Inclination']
    p = (row['prot'] - min_p)/(max_p - min_p)
    if 'i' in row.keys():
      if self.cos_inc:
        i = np.cos(row['i']*np.pi/180)
      else:
        i = (row['i']*np.pi/180 - min_i)/(max_i - min_i)
    else:
      i = -1
    y = torch.tensor([i, p]).float()
    return y
  
  def get_cls_labels(self, row):
    y = row['i']
    sigma = np.mean([np.abs(y-row['err_i'][0]), np.abs(row['err_i'][1]-y)])
    if self.cos_inc:
      y = np.cos(y*np.pi/180)
      cls = np.linspace(0,1, self.num_classes)
      sigma = sigma/50
    else:
      cls = np.linspace(0, 90, self.num_classes)
    probabilities = np.exp(-0.5 * ((y - cls) / sigma) ** 2)
    return torch.tensor(probabilities), y

  def __getitem__(self, idx): 
    x, y, mask, mask_y, info, info_y = super().__getitem__(idx)
    if self.df is not None:
      row = self.df.iloc[idx]
    if self.cls:
      y, y_val = self.get_cls_labels(row)
      info_y['y_val'] = y_val
    else:
      y = self.get_labels(row)
    info['path'] = str(info['path'])
    info['qs'] = str(info['qs'])
    info_y['qs'] = str(info_y['qs'])
    return x.float(), y, mask, mask_y, info, info_y


class TimeSeriesDataset(Dataset):
  """
  A dataset for time series data.
  """
  def __init__(self, root_dir:str,
              idx_list:List,
              labels:List=['Inclination', 'Period'],
              t_samples:int=None,
              norm:str='std',
              transforms:object=None,
              acf=False,
              return_raw=False,
              cos_inc=False,
              freq_rate=1/48,
              init_frac=0.4,
              dur=360,
              spots=False,
              period_norm=False,
              classification=False,
              num_classes=None):
      """
      A dataset for supervised time series data.
      Args:
          root_dir (str): root directory
          idx_list (List): list of indices
          labels (List, optional): labels to be used. Defaults to ['Inclination', 'Period'].
          t_samples (int, optional): length of samples. Defaults to None.
          norm (str, optional): normalizing method. Defaults to 'std'.
          transforms (object, optional): data transformations. Defaults to None.
          acf (bool, optional): calculate ACF. Defaults to False.
          return_raw (bool, optional): return raw lightcurve. Defaults to False.
          cos_inc (bool, optional): cosine of inclination. Defaults to False.
          freq_rate (float, optional): frequency rate. Defaults to 1/48.
          init_frac (float, optional): fraction of the lightcurve to cut. Defaults to 0.4.
          dur (int, optional): duration in days. Defaults to 360.
          spots (bool, optional): include spots. Defaults to False.
          period_norm (bool, optional): normalize lightcurve by period. Defaults to False.
          classification (bool, optional): prepare labels for cls task. Defaults to False.
          num_classes ([type], optional): number of classes. Defaults to None.
      """
                 
      self.idx_list = idx_list
      self.labels = labels
      self.length = len(idx_list)
      self.p_norm = period_norm
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      lc_dir = 'simulations'
      self.lc_path = os.path.join(root_dir, lc_dir)
      self.spots_path = os.path.join(root_dir, "spots")
      self.loaded_labels = pd.read_csv(self.targets_path)
      self.norm=norm
      self.boundary_values_dict = create_boundary_values_dict(self.loaded_labels)
      if num_classes == None and self.labels is not None:
        self.num_classes = len(labels)
      else:
        self.num_classes = num_classes
      self.transforms = transforms
      self.spots = spots
      self.acf = acf
      self.return_raw = return_raw
      self.freq_rate = freq_rate
      self.init_frac = init_frac
      self.dur = dur 
      self.seq_len = t_samples
      if self.seq_len is None:
         self.seq_len = int(self.dur/self.freq_rate)
      self.cos_inc = cos_inc
      self.step = 0
      self.samples = []
      self.cls = classification

  def read_spots(self, idx):
    spots = pd(os.path.join(self.spots_path, f"spots_{idx}.pqt")).values
    return spots

  def crop_spots(self, spots, info):
    if 'left_crop' in info:
      left_crop, right_crop = info['left_crop'], info['right_crop']
      left_day, right_day = int(left_crop*self.freq_rate), int(right_crop*self.freq_rate)
      spots = spots[np.logical_and(spots[:,0] > left_day, spots[:,0] < right_day)]
      spots[:,0] -= left_day
    return spots

  def create_spots_arr(self, idx, info, x):
        spots_data = self.read_spots(self.idx_list[idx])
        init_day = int(1000 * self.init_frac)
        spots_data = spots_data[spots_data[:, 0] > init_day]
        spots_data[:, 0] -= init_day
        if not self.p_norm:
          spots_data = self.crop_spots(spots_data, info)
        # normalize to [0,1]
        spots_data[:, 1] /= np.pi/2
        spots_data[:, 2] /= 2*np.pi

        spots_arr = np.zeros((2, x.shape[-1]))
        spot_t = (spots_data[:, 0] / self.freq_rate).astype(np.int64)
        spots_arr[:, spot_t] = spots_data[:,1:3].T
        return spots_arr

  def interpolate(self, x):
      f = interp1d(x[:,0], x[:,1])
      new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
      x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
      return x

  
  def get_labels(self, sample_idx):
      y = self.loaded_labels.iloc[sample_idx]
      if self.labels is None:
        return y
      y = torch.tensor([y[label] for label in self.labels])
      for i,label in enumerate(self.labels):
        if label == 'Inclination' and self.cos_inc:
          y[i] = np.cos(y[i])
        elif label == 'Period' and self.p_norm:
          y[i] = 1
        else:
          min_val, max_val = self.boundary_values_dict[f'min {label}'], self.boundary_values_dict[f'max {label}']
          y[i] = (y[i] - min_val)/(max_val - min_val)
      if len(self.labels) == 1:
        return y.float()
      return y.squeeze(0).squeeze(-1).float()
  
  def get_cls_labels(self, sample_idx, sigma=5, att='Inclination'):
      y = self.loaded_labels.iloc[sample_idx][att]
      if att == 'Inclination':
          if self.cos_inc:
            y = np.cos(y)
            cls = np.linspace(0,1, self.num_classes)
            sigma = sigma/50
          else:
            y = y*180/np.pi
            cls = np.linspace(0, 90, self.num_classes)
      else:
          cls = np.linspace(0, self.boundary_values_dict[att], self.num_classes)
      probabilities = np.exp(-0.5 * ((y - cls) / sigma) ** 2)
      probabilities /= np.sum(probabilities)
      return probabilities, y
      
  def __len__(self):
      return self.length
  
  def __getitem__(self, idx):
      s = time.time()
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      p = self.loaded_labels.iloc[sample_idx]['Period']
      info = {'idx': idx, 'period': p}
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      x = x[int(self.init_frac*len(x)):,:]
      x[:,1] = fill_nan_np(x[:,1], interpolate=True)
      if self.transforms is not None:
        x, _, info = self.transforms(x[:,1], mask=None,  info=info, step=self.step)
        if self.seq_len > x.shape[0] and (not self.p_norm):
          x = F.pad(x, (0, self.seq_len - x.shape[-1], 0,0), mode="constant", value=0)
        info['idx'] = idx
      else:
        x = x[:,1]
      # x = x.T[:, :self.seq_len]
      x = x.nan_to_num(0)

      if not self.cls:
        y = self.get_labels(sample_idx)
      else:
        y, y_val = self.get_cls_labels(sample_idx)
        info['y_val'] = y_val
      if self.spots:
        if len(x.shape) == 1:
          x = x.unsqueeze(0)
        spots_arr = self.create_spots_arr(idx, info, x)
        x = torch.cat((x, torch.tensor(spots_arr).float()), dim=0)
      end = time.time()
      if torch.isnan(x).sum():
        print("nans! in idx: ", idx)
      info['time'] = end-s
      return x.float(), y, torch.ones_like(x), info
  
class HybridDataset(TimeSeriesDataset):
  """
  A dataset for hybrid data
  """
  def __init__(self, root_dir:str,
              idx_list:List,
              labels:List=['Inclination', 'Period'],
              t_samples:int=None,
              norm:str='std',
              transforms:object=None,
              acf=False,
              return_raw=False,
              cos_inc=False,
              freq_rate=1/48,
              init_frac=0.4,
              dur=360,
              spots=False,
              period_norm=False,
              classification=False,
              num_classes=None,
              **kwargs):
      """
      A dataset for hybrid data
      Args:
          root_dir (str): root directory
          idx_list (List): list of indices
          labels (List, optional): labels to be used. Defaults to ['Inclination', 'Period'].
          t_samples (int, optional): length of samples. Defaults to None.
          norm (str, optional): normalizing method. Defaults to 'std'.
          transforms (object, optional): data transformations. Defaults to None.
          acf (bool, optional): calculate ACF. Defaults to False.
          return_raw (bool, optional): return raw lightcurve. Defaults to False.
          cos_inc (bool, optional): cosine of inclination. Defaults to False.
          freq_rate (float, optional): frequency rate. Defaults to 1/48.
          init_frac (float, optional): fraction of the lightcurve to cut. Defaults to 0.4.
          dur (int, optional): duration in days. Defaults to 360.
          spots (bool, optional): include spots. Defaults to False.
          period_norm (bool, optional): normalize lightcurve by period. Defaults to False.
          classification (bool, optional): prepare labels for cls task. Defaults to False.
          num_classes ([type], optional): number of classes. Defaults to None.
      """
      super().__init__(root_dir, idx_list, labels=labels, t_samples=t_samples, norm=norm, transforms=transforms,
                       acf=acf, return_raw=return_raw, cos_inc=cos_inc, freq_rate=freq_rate, init_frac=init_frac,
                       dur=dur, spots=spots, period_norm=period_norm, classification=classification, num_classes=num_classes)
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.loaded_labels = pd.read_csv(self.targets_path)
  def __getitem__(self, idx):
      s  =time.time()
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      p = self.loaded_labels.iloc[sample_idx]['Period']
      info = {'idx': idx, 'period': p}
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      x = x[int(self.init_frac*len(x)):,:]
      x[:,1] = fill_nan_np(x[:,1], interpolate=True)
      x1, _, info1 = self.transforms[0](x[:,1], mask=None,  info=info, step=self.step)
      x2, _, info2 = self.transforms[1](x[:,1], mask=None,  info=info, step=self.step)
      if self.seq_len > x.shape[0] and (not self.p_norm):
        x1 = F.pad(x1, (0, self.seq_len - x1.shape[-1], 0,0), mode="constant", value=0)
        x2 = F.pad(x2, (0, self.seq_len - x2.shape[-1], 0,0), mode="constant", value=0)
      info1['idx'] = idx
      info2['idx'] = idx
      x1 = x1.T[:, :self.seq_len].nan_to_num(0)
      x2 = x2.T[:, :self.seq_len].nan_to_num(0)
      y = self.get_labels(sample_idx)
      info1['time'] = time.time() - s
      return x1.float(), x2.float(), y, torch.ones_like(x1), info1, info2

class DynamicRangeDataset(Dataset):
  """
  A dataset for dynamic range data
  """
  def __init__(self, root_dir:str,
               indices:List,
              labels:List=['Inclination', 'Period'],
              seq_len:int=None,
              transforms:object=None,
              initial_max_label=0.2,
              increment=0.05,
              threshold=10,
              num_samples = np.inf,
              **kwargs):
      """
      A dataset for dynamic range data
      Args:
          root_dir (str): root directory
          idx_list (List): list of indices
          labels (List, optional): labels to be used. Defaults to ['Inclination', 'Period'].
          t_samples (int, optional): length of samples. Defaults to None.
          norm (str, optional): normalizing method. Defaults to 'std'.
          transforms (object, optional): data transformations. Defaults to None.
          acf (bool, optional): calculate ACF. Defaults to False.
          return_raw (bool, optional): return raw lightcurve. Defaults to False.
          cos_inc (bool, optional): cosine of inclination. Defaults to False.
          freq_rate (float, optional): frequency rate. Defaults to 1/48.
          init_frac (float, optional): fraction of the lightcurve to cut. Defaults to 0.4.
          dur (int, optional): duration in days. Defaults to 360.
          spots (bool, optional): include spots. Defaults to False.
          period_norm (bool, optional): normalize lightcurve by period. Defaults to False.
          classification (bool, optional): prepare labels for cls task. Defaults to False.
          num_classes ([type], optional): number of classes. Defaults to None.
      """
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.indices_list = indices
      self.loaded_labels = pd.read_csv(self.targets_path)
      self.root_dir = root_dir
      self.boundary_values_dict = create_boundary_values_dict(self.loaded_labels)
      self.samples_paths = [os.path.join(f'{root_dir}/simulations', p) for p in
                             os.listdir(f'{root_dir}/simulations') if p.endswith('.pqt')]
      self.transform = transforms
      self.seq_len = seq_len
      self.num_samples = num_samples
      self.labels_vals = labels
      self.labels = []
      self.samples = []
      self.current_max_label = initial_max_label
      self.increment = increment
      self.threshold = threshold
      self.iteration_counter = 0
      print("loading data...")
      self.load_data()
      self.update_filtered_indices()

  def get_labels(self, row):
      if self.labels_vals is None:
        return row
      y = torch.tensor([row[label] for label in self.labels_vals])
      for i,label in enumerate(self.labels_vals):
          min_val, max_val = self.boundary_values_dict[f'min {label}'], self.boundary_values_dict[f'max {label}']
          y[i] = (y[i] - min_val)/(max_val - min_val)
      if len(self.labels_vals) == 1:
        return y.float()
      return y.squeeze(0).squeeze(-1).float()
  
  def load_data(self):
     with open(self.targets_path, 'r') as f:
          for i, row in self.loaded_labels.iloc[self.indices_list].iterrows():
              sim_num = row['Simulation Number']
              sim_num_str = f'{sim_num:d}'.zfill(int(np.log10(len(self.loaded_labels)))+1)
              labels = self.get_labels(row)
              self.labels.append(labels)
              self.samples.append(os.path.join(self.root_dir, f'simulations/lc_{sim_num_str}.pqt'))
              if i > self.num_samples:
                break

  def __len__(self):
        return len(self.indices_list)
  
  def __getitem__(self, idx):
        t = time.time()
        x = pd.read_parquet(self.samples[idx]).values
        x = x[:, 1]
        y = self.labels[idx]
        if self.transform is not None:
            x, _, info = self.transform(x, mask=None, info=dict())
            if self.seq_len > x.shape[0]:
                x = F.pad(x, (0, self.seq_len - x.shape[-1], 0, 0), mode="constant", value=0)
        else:
            x = torch.Tensor(x[None,:])
        x = x.nan_to_num(0)
        info['idx'] = idx
        info['time'] = time.time() - t
        return x, y, torch.ones_like(x), info
  
  def update_filtered_indices(self):
        self.filtered_indices = [i for i, label in enumerate(self.labels) if (1 - label[0] <= self.current_max_label)]
        print("filtered indices: ", len(self.filtered_indices))

  def expand_label_range(self):
        self.current_max_label += self.increment
        self.current_max_label = min(1, self.current_max_label)
        print("current max label: ", self.current_max_label)
        self.update_filtered_indices()


class DynamicRangeDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self.iteration_counter = 0
        self.threshold = dataset.threshold
        sampler = DynamicRangeSampler(dataset)
        super().__init__(dataset, sampler=sampler, *args, **kwargs)

    def __iter__(self):
        self.iteration_counter += 1
        if self.iteration_counter >= self.threshold:
            self.dataset.expand_label_range()
            self.iteration_counter = 0  # Reset the counter after expanding the range
        return super().__iter__()

class MultiCopyDataset(Dataset):
    def __init__(self, dataset,
                 num_copies=2,
                 ):
        self.dataset = dataset
        self.num_copies = num_copies
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        res = []
        for i in range(self.num_copies):
            data = self.dataset[idx]
            res.append(data[0])
        return torch.stack(res), *data[1:]
        

class TimeSeriesGraphDataset(TimeSeriesDataset):
    def __init__(self, time_series_data, periods, num_noisy_edges=5):
        self.time_series_data = time_series_data
        self.periods = periods
        self.num_noisy_edges = num_noisy_edges

    def __len__(self):
        return len(self.time_series_data)
    
    def __getitem__(self, idx):
        series = self.time_series_data[idx]
        period = self.periods[idx]
        
        # Create edges for temporal connections
        edges_temporal = [(i, i + 1) for i in range(len(series) - 1)]
        
        # Define periodic edges (connecting with nodes that are one period away)
        edges_periodic = [(i, (i + period) % len(series)) for i in range(len(series))]
        
        # Define noisy edges (example: random connections)
        edges_noisy = [(random.randint(0, len(series) - 1), random.randint(0, len(series) - 1)) for _ in range(self.num_noisy_edges)]
        
        # Combine all edges
        edges = edges_temporal + edges_periodic + edges_noisy
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(list(set(edges)).T, dtype=torch.long)
        
        # Create a PyTorch Geometric Data object
        data = Data(x=torch.tensor(series).view(-1, 1), edge_index=edge_index)
        
        return data, period
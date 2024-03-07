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
from astropy.io import fits
from lightPred.utils import create_kepler_df
from matplotlib import pyplot as plt
from lightPred.utils import fill_nan_np, replace_zeros_with_average
from pyts.image import GramianAngularField
from scipy.signal import stft, correlate2d
from scipy import signal
import time
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter as savgol
import torchaudio.transforms as T


cad = 30
DAY2MIN = 24*60
min_p, max_p = 0,60
min_lat, max_lat = 0, 80
min_cycle, max_cycle = 1, 10
min_i, max_i = 0, np.pi/2
min_tau, max_tau = 1,10 
min_n, max_n = 0, 5000
min_shear, max_shear = 0, 1
T_SUN = 5777
boundary_values_dict = {'Period': (min_p, max_p), 'Inclination': (min_i, max_i),
 'Decay Time': (min_tau, max_tau), 'Cycle Length': (min_cycle, max_cycle), 'Spot Max': (min_lat, max_lat),
 'n_spots': (min_n, max_n), 'Shear': (min_shear, max_shear)}

non_period_table_path = "/data/lightPred/Table_2_Non_Periodic.txt"
kepler_path = "/data/lightPred/data"



def fill_nans(x):
  # Find indices of NaN values
  nan_indices = torch.isnan(x)

  # Create an array of indices where NaN values are located
  nan_indices_array = torch.arange(len(x))[nan_indices]

  # Initialize a copy of the tensor to store the interpolated values
  interpolated_tensor = x.clone()

  # Fill NaN values with 0 to ignore them in interpolation
  x[nan_indices] = 0

  # Forward fill NaN values with the previous non-NaN value
  non_nan_indices = torch.arange(len(x))[~nan_indices]
  x[nan_indices] = torch.cat([x[non_nan_indices[0]], x[nan_indices]])

  # Backward fill NaN values with the next non-NaN value
  x[nan_indices] = torch.cat([x[nan_indices], x[non_nan_indices[-1]]])

  # Find indices of NaN values after forward and backward filling
  nan_indices_after_fill = torch.arange(len(x))[nan_indices]

  # Perform linear interpolation
  interpolated_tensor[nan_indices_after_fill] = \
   (x[nan_indices_after_fill + 1] + x[nan_indices_after_fill - 1]) / 2
  return interpolated_tensor

def read_fits(filename):
    # print("reading fits file: ", filename)
    with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          meta = hdulist[0].header
          # print(header)
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    # teff = meta['TEFF']
    return x,time, meta

def add_kepler_noise(x, non_p_df, factor=4):
    # plt.plot(x)
    # plt.savefig("/data/tests/x.png")
    # plt.clf()
    row = non_p_df.sample(n=1)
    noise,time = read_fits(row['data_file_path'].values[0])
    f = interp1d(time, noise)
    new_t = np.linspace(time[0], time[-1], x.shape[-1])
    noise = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
    noise = torch.tensor(noise.astype(np.float64))[:,1]
    valid_values = noise[~torch.isnan(noise)]
    mean_value = torch.mean(valid_values)
    noise = torch.where(torch.isnan(noise), mean_value, noise)
    x_range = x.max() - x.min()
    # print("x_range:", x_range, "noise_max: ", noise.max(), "noise_min: ", noise.min(), "num_nans: ", torch.isnan(noise).sum(), "num_points: ", noise.shape[0])
    noise_normalized = ((noise - noise.min()) / (noise.max() - noise.min()))*x_range/factor
    # print("noise shape: ", noise_normalized.shape)
    # print(noise_normalized)
    # plt.plot(noise_normalized)
    # plt.savefig("/data/tests/noise.png")
    # plt.clf()
    return x + noise_normalized



def remove_leading_zeros(s):
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

def quantize_tensor(tensor, num_classes):
    thresholds = torch.linspace(0,1,num_classes+1)
    # print("thersh: ", thresholds)
    # print("tensor: ", tensor)
    classification_matrix = torch.zeros((tensor.size(0), len(thresholds) - 1))  
    for i in range(len(thresholds) - 1):
        mask = torch.logical_and(thresholds[i + 1] >= tensor, thresholds[i] < tensor)  # Create a boolean mask for elements above or equal to the threshold
        classification_matrix[:, i] = mask  # Assign the binary tensor to the corresponding class column in the classification matrix
    return classification_matrix

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

class TimeSsl(Dataset):
    def __init__(self, root_dir, path_list,  df=None, t_samples=512, norm='std', ssl_tf=None, transforms=None, acf=False, return_raw=False):
        # self.idx_list = idx_list
        self.path_list = path_list
        self.cur_len = None
        self.df = df
        self.root_dir = root_dir
        self.seq_len = t_samples
        self.norm = norm
        self.ssl_tf = ssl_tf
        self.transforms = transforms
        self.acf = acf
        self.return_raw = return_raw
        self.length = len(self.df) if self.df is not None else len(self.path_list)


        
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
        # print(row['KID'])
        try:
          for i in range(len(row['data_file_path'])):
            x,time,meta = read_fits(row['data_file_path'][i])
            x /= x.max()
            x = fill_nan_np(np.array(x), interpolate=True)
            if i == 0:
              x_tot = x.copy()
            else:
              border_val = np.mean(x) - np.mean(x_tot)
              x -= border_val
              x_tot = np.concatenate((x_tot, np.array(x)))
          x = torch.tensor(x_tot)
          self.cur_len = len(x)
        except (TypeError,OSError, FileNotFoundError)  as e:
            print("Error: ", e)
            x, meta = torch.zeros((self.cur_len)), {'TEFF': None, 'RADIUS': None, 'LOGG': None}
        return x, meta
    
    def __getitem__(self, idx):
        # if idx % 1000 == 0:
        #   print(idx)
        if self.df is not None:
          x, meta = self.read_row(idx)
        else:
          x, meta =  self.read_data(idx).float()
          x /= x.max()
        if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=dict())
        if self.acf:
          x = A(x, nlags=len(x))
        x = torch.tensor(x)
        x = torch.nan_to_num(x, torch.nanmean(x))
        x = x.unsqueeze(0).unsqueeze(0)
        # print("before augmentation: ", x.shape)
        # x = self.x_samples[idx].unsqueeze(0).unsqueeze(0)
        if self.norm == 'std':
          x = (x - x.mean())/(x.std()+1e-8)
        elif self.norm == 'median':
          x /= x.median()
        elif self.norm == 'minmax':
          mini = x.min(dim=-1).values.view(-1,1).float()
          maxi = x.max(dim=-1).values.view(-1,1).float()
          x = (x-mini)/(maxi - mini)
        if self.ssl_tf is not None:
          x1 = self.ssl_tf(copy.deepcopy(x))
          x2 = self.ssl_tf(copy.deepcopy(x))
          x1 = (x1 - x1.mean())/(x1.std()+1e-8)
          x2 = (x2 - x2.mean())/(x2.std()+1e-8)
          return x1.squeeze(0).float(), x2.squeeze(0).float()
        else:
          return x, torch.zeros((1,self.seq_len))
        
        
class MaskedSSL(TimeSsl):
   def __init__(self, root_dir, path_list, t_samples=1024, norm='minmax', transforms=None, vocab_size=1024, mask_prob = 0.15, mask_val= -1,
                cls_val = 0):
    super().__init__(root_dir, path_list, t_samples, norm=norm, transforms=transforms)
    self.vocab_size = vocab_size
    self.mask_prob = mask_prob
    self.mask_val = mask_val
    self.cls_val = cls_val
    self.transforms = transforms

   def mask_array(self, array, mask_percentage=0.15, mask_value=-1):
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
              array[:,i] = random.randint(0, 1)  
          inverse_token_mask[i] = False  
      return array, inverse_token_mask

   def __getitem__(self, idx):
      x = self.read_data(idx)
      x = torch.tensor(fill_nan_np(x, interpolate=True))
      if self.transforms is not None:
        x, _, info = self.transforms(x, mask=None, info=dict())
      # xcf = A(x[:,1], nlags=len(x[:,0]))
      # x = torch.stack([torch.tensor(x[:,1]),torch.tensor(xcf)])
      x = torch.tensor(x[:,1]).unsqueeze(0)
      if self.norm == 'std':
          x = (x - x.mean(dim=-1)[:,None])/(x.std(dim=-1)[:,None] + 1e-8)
      elif self.norm == 'minmax':
        mini = x.min(dim=-1).values.view(-1,1).float()
        maxi = x.max(dim=-1).values.view(-1,1).float()
        x = torch.clamp((x-mini)/(maxi - mini)*self.vocab_size, min=0, max=self.vocab_size)
      masked_x, inv_mask = mask_array(x.clone(), mask_percentage=self.mask_prob, mask_value=self.mask_val)
      x[:,0] = self.cls_val
      if torch.isnan(x).any():
        print(f"idx {idx} - sample {idx} is nan")
      return masked_x.float(), inv_mask, x
   


class KeplerDataset(TimeSsl):
  def __init__(self, root_dir, path_list, df=None, mask_prob=0, mask_val=-1, np=False,  **kwargs):
    super().__init__(root_dir, path_list, df=df, **kwargs)
    # self.df = df
    # self.length = len(self.df) if self.df is not None else len(self.paths_list)
    self.mask_prob = mask_prob
    self.mask_val = mask_val
    self.np = np

  def mask_array(self, array, mask_percentage=0.15, mask_value=-1):
      # if len(array.shape) == 1:
      #   array = array.unsqueeze(0)
      len_s = array.shape[0]  
      inverse_token_mask = torch.ones(len_s, dtype=torch.bool)  

      mask_amount = round(len_s * mask_percentage)
      for _ in range(mask_amount):  
          i = random.randint(0, len_s - 1)  

          if random.random() < 0.95:  
              array[i] = mask_value  
          else:
              array[i] = random.uniform(array.min(),array.max())  
          inverse_token_mask[i] = False  
      return array, inverse_token_mask
  
  def read_np(self, idx):
    x = np.load(os.path.join(self.root_dir, self.path_list[idx]))
    return torch.tensor(x), dict()

  def __getitem__(self, idx):
    if self.df is not None:
      x, meta = self.read_row(idx)
    elif self.np:
      x, meta = self.read_np(idx)
    else:
      x, meta =  self.read_data(idx).float()
      x /= x.max()
    info = dict()
    x /= x.max()
    if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=info)
          x = x.squeeze()
    if self.acf:
      xcf = torch.tensor(A(x, nlags=len(x)))
      if self.return_raw:
        x = torch.stack([torch.tensor(xcf), torch.tensor(x)])
      else:
        x = torch.tensor(xcf).unsqueeze(0)
    else:
       x = x.unsqueeze(0)
    if self.norm == 'std':
      x = (x - x.mean())/(x.std()+1e-8)
    elif self.norm == 'median':
      x /= x.median()
    elif self.norm == 'minmax':
      mini = x.min(dim=-1).values.view(-1,1).float()
      maxi = x.max(dim=-1).values.view(-1,1).float()
      x = (x-mini)/(maxi - mini)
    if self.mask_prob > 0:
      masked_x, inv_mask = self.mask_array(x.clone(), mask_percentage=self.mask_prob, mask_value=self.mask_val)
    else:
      masked_x = x.clone()
      inv_mask = torch.ones_like(x).bool()
    torch.nan_to_num(masked_x, torch.nanmean(masked_x))
    torch.nan_to_num(x, torch.nanmean(x))
    # print(meta['TEFF'], meta['RADIUS'], meta['LOGG'])
    info['idx'] = idx
    if len(meta):
      info['Teff'] = meta['TEFF'] if meta['TEFF'] is not None else 0
      info['R'] = meta['RADIUS'] if meta['RADIUS'] is not None else 0
      info['logg'] = meta['LOGG'] if meta['LOGG'] is not None else 0
    info['path'] = self.df.iloc[idx]['data_file_path'] if self.df is not None else self.path_list[idx]
    info['KID']  = self.df.iloc[idx]['KID'] if self.df is not None else self.path_list[idx].split("/")[-1].split("-")[0].split('kplr')[-1]
    return x.float(), masked_x.squeeze().float(), inv_mask, info


class KeplerNoiseDataset(KeplerDataset):
  def __init__(self, root_dir, path_list, df=None, **kwargs ):
    super().__init__(root_dir, path_list, df=df, acf=False, norm='none', **kwargs)
    self.samples = []
    print(f"preparing kepler data of {len(self.df)} samples...")
    for i in range(len(self.df)):
      if i % 1000 == 0:
        print(i, flush=True)
      x, masked_x, inv_mask, info = super().__getitem__(i)
      self.samples.append((x, masked_x, inv_mask, info))

  def __getitem__(self, idx):
    x, masked_x, inv_mask, info = self.samples[idx]
    return x.float(), masked_x.squeeze().float(), inv_mask, info 

class KeplerLabeledDataset(KeplerDataset):
  def __init__(self, root_dir, path_list, **kwargs):
      super().__init__(root_dir, path_list, **kwargs)


  def __getitem__(self, idx): 
    x, masked_x, inv_mask, info = super().__getitem__(idx)
    if self.df is not None:
      row = self.df.iloc[idx]
    if 'Prot' in row:
      val = (row['Prot'] - boundary_values_dict['Period'][0])\
      /(boundary_values_dict['Period'][1]-boundary_values_dict['Period'][0])
    elif 'i' in row:
      val = (row['Inclination'] - boundary_values_dict['Inclination'][0])\
      /(boundary_values_dict['Inclination'][1]-boundary_values_dict['Inclination'][0])
    y = torch.tensor(val)
    return x.float(), y, inv_mask, info

class TFCKeplerDataset(KeplerDataset):
  def __init__(self, root_dir, path_list, **kwargs):
      super().__init__(root_dir, path_list, **kwargs)    
  def __len__(self):
      return self.length
  
  def __getitem__(self, idx):
      x_t, masked_x, inv_mask, info = super().__getitem__(idx)
      x_f = torch.fft.fft(x_t[0]).abs().unsqueeze(0) # add channel dimension
      # if self.norm == 'std':
      #   x = ((x-x.mean())/(x.std()+1e-8))
      # elif self.norm == 'median':
      #   x /= x.median()
      # elif self.norm == 'minmax':
      #   x = (x-x.min())/(x.max()-x.min())
      
      return x_t.float(), x_f.float(), info

  def get_labels(self, sample_idx):
      y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      y = torch

class DenoisingDataset(Dataset):

  def __init__(self, root_dir, idx_list, t_samples=512, norm='std', noise_factor=4):
      self.idx_list = idx_list
      self.length = len(idx_list)
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.lc_path = os.path.join(root_dir, "simulations")
      self.seq_len = t_samples
      self.norm=norm
      self.non_p_df = create_kepler_df(kepler_path, non_period_table_path)
      self.noise_factor = noise_factor


  def __len__(self):
      return self.length

  def __getitem__(self, idx):
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      # x = x[int(0.4*len(x)):,:]
      if self.seq_len:
        f = interp1d(x[:,0], x[:,1])
        new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
        x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
        x = torch.tensor(x.astype(np.float32))[:,1]
        x_noise = add_kepler_noise(x, self.non_p_df, factor=self.noise_factor)
        if torch.isnan(x).any() or torch.isnan(x_noise).any():
          print("nans in dataset at index ", idx)
      return x.unsqueeze(0).float(), x_noise.unsqueeze(0).float()



class TimeSeriesDataset(Dataset):
  def __init__(self, root_dir, idx_list, labels=['Inclination', 'Period'], t_samples=None, norm='std', transforms=None,
                noise=False, spectrogram=False, n_fft=1000, acf=False, return_raw=False,cos_inc=False,
                 wavelet=False, freq_rate=1/48, init_frac=0.4, dur=360, kep_noise=None, prepare=True):
      self.idx_list = idx_list
      self.labels = labels
      self.length = len(idx_list)
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.lc_path = os.path.join(root_dir, "simulations")
      self.loaded_labels = pd.read_csv(self.targets_path)
      self.seq_len = t_samples
      self.norm=norm
      self.num_classes = len(labels)
      self.transforms = transforms
      self.noise = noise
      self.n_fft = n_fft
      self.spec = spectrogram
      self.acf = acf
      self.return_raw = return_raw
      self.wavelet = wavelet
      self.freq_rate = freq_rate
      self.init_frac = init_frac
      self.dur = dur 
      self.cos_inc = cos_inc
      self.kep_noise = kep_noise
      self.maxstds = 0.159
      self.step = 0
      self.weights = torch.zeros(self.length)
      self.samples = []
      if prepare:
        self.prepare_data()
      self.prepare = prepare

  def add_kepler_noise(self, x, max_ratio=1, min_ratio=0.5):
        std = x.std()
        idx = np.random.randint(0, len(self.kep_noise))
        x_noise,_,_,info = self.kep_noise[idx]
        
        noise_std = np.random.uniform(std*min_ratio, std*max_ratio)
        x_noise = (x_noise - x_noise.mean()) / (x_noise.std() + 1e-8) *  noise_std + 1
        x = x*x_noise.squeeze().numpy()
        return x

  def wavelet_from_np(self, lc, wavelet=signal.morlet2,
                        w=6,
                        period=None,
                        minimum_period=None,
                        maximum_period=None,
                        sample_rate = 30/(24*60),
                        period_samples=512):
        time, flux = lc[:,0], lc[:,1]
        time -= time[0]
        flux -= flux.mean()
        if sample_rate is None:
            sample_rate = 0.5 * (1./(np.nanmedian(np.diff(time))))    
        nyquist = 0.5 * (1./sample_rate)

        if period is None:
            if minimum_period is None:
                minimum_period = 1/nyquist
            if maximum_period is None:
                maximum_period = time[-1]
            # period = np.geomspace(minimum_period, maximum_period, period_samples)
            period = np.linspace(minimum_period, maximum_period, period_samples)
        else:
            if any(b is not None for b in [minimum_period, maximum_period]):
                print(
                    "Both `period` and at least one of `minimum_period` or "
                    "`maximum_period` have been specified. Using constraints "
                    "from `period`.", RuntimeWarning)

        widths = w * nyquist * period / np.pi
        cwtm = signal.cwt(flux, wavelet, widths, w=w)
        power = np.abs(cwtm)**2 / widths[:, np.newaxis]
        phase = np.angle(cwtm)
        return power, phase, period
  
  def interpolate(self, x):
      f = interp1d(x[:,0], x[:,1])
      new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
      x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
      return x
  
  def create_data(self, x):
    if self.acf:
      xcf = A(x, nlags=int(self.dur/self.freq_rate) - 1)
      if self.wavelet:
        power, phase, period = self.wavelet_from_np(x, period_samples=int(self.dur/self.freq_rate))
        gwps = power.sum(axis=-1)
        grad = 1 + np.gradient(gwps)/(2/period)
        x = np.vstack((grad[None], xcf[None]))
      elif self.return_raw:
        x = np.vstack((xcf[None], x[None]))
      else:
        x = xcf 
    elif self.wavelet:
      power, phase, period = self.wavelet_from_np(x, period_samples=int(self.dur//self.freq_rate))
      gwps = power.sum(axis=-1)
      grad = 1 + np.gradient(gwps)/(2/period)
      x = grad
    else:
      x = x[:,1]
    return  torch.tensor(x.astype(np.float32))
  
  def normalize(self, x):
    if self.norm == 'std':
        m = x.mean(dim=-1).unsqueeze(-1)
        s = x.std(dim=-1).unsqueeze(-1)
        x = (x-m)/(s+1e-8)
    elif self.norm == 'median':
        # median = x.median(dim=-1).values.unsqueeze(-1)
        # x = x / (median + 1e-8)
        x /= x.median(dim=-1).values.unsqueeze(-1)
    elif self.norm == 'minmax':
        mn = x.min(dim=-1).values.unsqueeze(-1)
        mx = x.max(dim=-1).values.unsqueeze(-1)
        x = (x-mn)/(mx-mn)
    return x
  
  def get_labels(self, sample_idx):
      # y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      y = self.loaded_labels.iloc[sample_idx]
      if self.labels is None:
        return y
      y = torch.tensor([y[label] for label in self.labels])
      # if 'Inclination' in self.labels:
      #    y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
      for i,label in enumerate(self.labels):
        if label == 'Inclination' and self.cos_inc:
          y[i] = np.cos(np.pi/2 - y[i])
        elif label in boundary_values_dict.keys():
          y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
      if len(self.labels) == 1:
        return y.squeeze(-1).float()
      return y.squeeze(0).squeeze(-1).float()

  def get_weight(self, y, counts):
      inc_idx = self.labels.index('Inclination')
      inc = y[int(inc_idx)]*(boundary_values_dict['Inclination'][1] - boundary_values_dict['Inclination'][0]) + boundary_values_dict['Inclination'][0]
      inc = int(inc.item()*180/np.pi)
      w = 1/(counts[inc])
      # print('inc: ', inc, 'weight: ', w, 'raw inc: ', y[inc_idx])
      # weights = torch.cat((weights, torch.tensor(w)), dim=0)
      return torch.tensor(w)
  
  def prepare_data(self):
      print("loading dataset...")
      all_inclinations = np.arange(0, 91)
      counts = np.zeros_like(all_inclinations)
      incl = (np.arccos(np.random.uniform(0, 1, self.length))*180/np.pi).astype(np.int16)
      unique, unique_counts = np.unique(incl, return_counts=True)
      counts[unique] = unique_counts
      indices = np.argsort(counts)
      # plt.hist(incl)
      # plt.savefig("/data/tests/counts.png")
      # plt.clf()
      counts = replace_zeros_with_average(counts)
      weights = []
      stds = []  
      for i in range(self.length):
        info = {'idx': i}
        starttime= time.time()
        if i % 1000 == 0:
          print(i, flush=True)
        # try:
        sample_idx = remove_leading_zeros(self.idx_list[i])
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[i]}.pqt")).values
        x = x[int(self.init_frac*len(x)):,:]
        time1 = time.time()
        if self.seq_len:
          x = self.interpolate(x)
        time2 = time.time()
        if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=info)
        time3 = time.time()
        x[:,1] = fill_nan_np(x[:,1], interpolate=True)
        time4 = time.time()
        x = self.create_data(x)
        time5 = time.time()
        y = self.get_labels(sample_idx)
        time6 = time.time()
        weights.append(self.get_weight(y, counts))
        stds.append(x.std().item())
        self.samples.append((x,y, info))

        # print("times: ", "\nread ", time1-starttime, "\ninterpolate ", time2-time1,
        #        "\ntransforms ", time3-time2, "\nfill nans: ", time4-time3,
        #          "\ncreate data ", time5-time4, "\nget labels ",  time6-time5, "\ntot time: ", time6-starttime)
        # except Exception as e:
        #   print("Exception: ", e)
      # self.weights = torch.stack(weights)
      # print(self.weights.shape, torch.max(self.weights))
      # plt.hist(self.weights)
      # plt.savefig("/data/tests/weights.png")
      # plt.clf()
      self.maxstds = np.max(stds)
      return
          
      
      
  def __len__(self):
      return self.length
  
  def __getitem__(self, idx):
      s = time.time()
      if not self.prepare:
        info = {'idx': idx}
        sample_idx = remove_leading_zeros(self.idx_list[idx])
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        t1 = time.time()
        x = x[int(self.init_frac*len(x)):,:]
        if self.seq_len:
          x = self.interpolate(x)
        t2 = time.time()
        if self.transforms is not None:
          x, _, info = self.transforms(x[:,1], mask=None, info=dict(), step=self.step)
          # x = savgol(x, 49, 1, mode='mirror')
          info['idx'] = idx
        t3 = time.time()
        x = fill_nan_np(x, interpolate=True)
        t4 = time.time()
        x = self.create_data(x)
        # print("x shape: ", x.shape)
        t5 = time.time()
        x = self.normalize(x)
        t6 = time.time()
        x = x.nan_to_num(0)
        # t4 = time.time()
        y = self.get_labels(sample_idx)
        t7 = time.time()
        self.step += 1
      else:
        x, y, info = self.samples[idx]
        x = self.normalize(x)
        x = x.nan_to_num(0)
      if self.spec:
        spec = T.Spectrogram(n_fft=self.n_fft, win_length=4, hop_length=4)
        x_spec = spec(x)
        # t6 = time.time()
        return x_spec.unsqueeze(0), y, x.float(), info
      # print("times: ", t1-s, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5,t7-t6,  "tot time: ", t7-s)
      return x.float(), y, torch.ones_like(x), info

class TimeSeriesDataset2(Dataset):
  def __init__(self, root_dir, idx_list, labels=['Inclination', 'Period'], p_norm=True, t_samples=512, norm='std', num_classes=2, transforms=None,
                noise=False, spectrogram=False, n_fft=1000, prepare=True):
      self.idx_list = idx_list
      self.labels = labels
      self.length = len(idx_list)
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.lc_path = os.path.join(root_dir, "simulations")
      self.seq_len = t_samples
      self.norm=norm
      self.num_classes = num_classes
      self.transforms = transforms
      self.noise = noise
      self.n_fft = n_fft
      self.spec = spectrogram
      self.p_norm = p_norm
      self.prepare = prepare
      self.weights = torch.ones(self.length)
      self.samples = []
      if prepare:
        self.prepare_data()
      
  def prepare_data(self):
      print("loading dataset...")
      all_inclinations = np.arange(0, 91)
      counts = np.zeros_like(all_inclinations)
      incl = (np.arcsin(np.random.uniform(0, 1, self.length))*180/np.pi).astype(np.int16)
      unique, unique_counts = np.unique(incl, return_counts=True)
      counts[unique] = unique_counts
      weights = torch.zeros(0)   
      for i in range(self.length):
        if i % 1000 == 0:
          print(i)
        try:
          sample_idx = remove_leading_zeros(self.idx_list[i])
          x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[i]}.pqt")).values
          x = x[int(0.4*len(x)):,:]
          if self.seq_len:
            f = interp1d(x[:,0], x[:,1])
            new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
            x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
          x = fill_nan_np(x[:,1], interpolate=True)
          x = torch.tensor(x.astype(np.float32))
          
             
          if self.transforms is not None:
            x, _, info = self.transforms(x, mask=None, info=dict())
            info['idx'] = i
          if self.norm == 'std':
            x = ((x-x.mean())/(x.std()+1e-8))
          elif self.norm == 'median':
            x /= x.median()
          elif self.norm == 'minmax':
            x = (x-x.min())/(x.max()-x.min())
          x = x.nan_to_num(0)

          y = self.get_labels(sample_idx)
          y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
          w = 1/(counts[(y['Inclination']*180/np.pi).astype(np.int16)] + 1e-8)
          weights = torch.cat((weights, torch.tensor(w)), dim=0)
          if self.labels is None:
            self.samples.append((x,y))
            return 
          y = torch.tensor([y[label] for label in self.labels])
          if 'Inclination' in self.labels:
            y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
          for i,label in enumerate(self.labels):
            if label in boundary_values_dict.keys():
              y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
          if len(self.labels) == 1:
            y =  y.squeeze(-1).float()
          else:
            y = y.squeeze(0).squeeze(-1).float()
          self.samples.append((x,y))
        except Exception as e:
          print("Exception: ", e)
      # self.weights = weights
      return
  
  def __len__(self):
      return self.length

  def __getitem__(self, idx):
      # s = time.time()
      
      info = {'idx': idx}
      if self.prepare:
        x, y = self.samples[idx]
      else:
        sample_idx = remove_leading_zeros(self.idx_list[idx])
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        # t1 = time.time()
        x = x[int(0.5*len(x)):,:]
        row = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
        if self.p_norm:
          t, x = self.preiod_norm(x, row['Period'].values[0], 10)
        # x = fill_nan_np(x, interpolate=True)
        x = torch.tensor(x[:,1].astype(np.float32))

        # t2 = time.time()
        if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=dict())
          info['idx'] = idx
        # t3 = time.time()
        if self.noise:
          x = add_kepler_noise(x, self.non_p_df, factor=self.noise_factor)
        if self.norm == 'std':
          x = ((x-x.mean())/(x.std()+1e-8))
        elif self.norm == 'median':
          x /= x.median()
        elif self.norm == 'minmax':
          x = (x-x.min())/(x.max()-x.min())
        x = torch.tensor(x).nan_to_num(0)
        # t4 = time.time()
        y = self.get_labels(sample_idx)
      # t5 = time.time()
      if self.spec:
        spec = T.Spectrogram(n_fft=self.n_fft, win_length=4, hop_length=4)
        x_spec = spec(x)
        # t6 = time.time()
        # print("times: ", t1-s, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, "tot time: ", t6-s)
        return x_spec.unsqueeze(0), y, x.float(), info
      # print("times: ", t1-s, t2-t1, t3-t2, t4-t3, t5-t4,  "tot time: ", t5-s)
      return x.float(), y, torch.ones_like(x), info
  
  def get_labels(self, sample_idx):
      y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      if self.labels is None:
        return y
      y = torch.tensor([y[label] for label in self.labels])
      if 'Inclination' in self.labels:
         y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
      for i,label in enumerate(self.labels):
        if label in boundary_values_dict.keys():
          y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
      
      if len(self.labels) == 1:
        return y.squeeze(-1).float()
      return y.squeeze(0).squeeze(-1).float()
  
  def preiod_norm(self, lc, period, num_ps):
    # plt.plot(lc[:,0], lc[:,1])
    time, flux = lc[:, 0], lc[:, 1]
    time = time - time[0]
    new_sampling_rate = period / 1000  
    new_time = np.arange(0, period * num_ps, new_sampling_rate)
    new_flux = np.interp(new_time, time, flux)
    return new_time, new_flux

class TimeSeriesClassifierDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, num_classes=10, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.cls = num_classes
  def __getitem__(self, idx):
    x,y = super().__geti
    y1 = quantize_tensor(y[0].unsqueeze(0), self.cls)
    y2 = quantize_tensor(y[1].unsqueeze(0), self.cls)
    return x.float(), torch.cat([y1,y2], dim=-1).float().squeeze(0)
 
class CleanDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, noise_ds, transforms, max_noise=0.4, min_noise=0.02):
      self.idx_list = idx_list
      self.length = len(idx_list)
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.lc_path = os.path.join(root_dir, "simulations")
      self.transforms = transforms
      self.kep_noise = noise_ds
      self.max_noise = max_noise
      self.min_noise = min_noise
      self.step = 0
  
  def __getitem__(self, idx):
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      x = x[int(0.4*len(x)):,:]
      x = fill_nan_np(x[:,1], interpolate=True)
      if self.transforms is not None:
        x, _, info = self.transforms(x, mask=None, info=dict())
        info['idx'] = idx
      x = torch.tensor(x.astype(np.float32))
      x = x.nan_to_num(0)
      x_transf = self.add_kepler_noise(x)
      x_transf = x_transf.nan_to_num(0)
      return x_transf.unsqueeze(-1), x.unsqueeze(-1), x.unsqueeze(-1), info

  def __len__(self):
      return self.length
  

class FourierDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, labels=['Inclination', 'Period'], n_days=512, norm='none', transforms=None):
    super().__init__(root_dir, idx_list, labels, t_samples=None, norm=norm,  transforms=transforms)
    
    self.rate = 48 #samples per day
    self.dt = 1/self.rate
    self.time = np.arange(0, n_days, self.dt)
    self.n = len(self.time)
    self.n_days = n_days
    
  def __getitem__(self, idx):
      x,y,_,info = super().__getitem__(idx)
      signal_fft = np.fft.fftshift(np.fft.fft(x, self.n))
      PSD = np.abs(signal_fft)
      Phase = np.angle(signal_fft)
      freq = (1 / (self.dt * self.n)) * np.arange(-self.n//2, self.n//2)
      dc = self.n_days * self.rate // 2
      PSD[dc] = 0
      fft_results = PSD[dc - self.n_days:dc + self.n_days]
      fft_phase= Phase[dc - self.n_days:dc + self.n_days]
      x_fft = torch.tensor(np.concatenate((fft_results, fft_phase)))
      return x_fft.float(), y, torch.ones_like(x_fft), info
          

class ACFDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, labels=['Inclination', 'Period'], t_samples=512,
                norm='std', return_raw=True, transforms=None, kep_noise=None):
    super().__init__(root_dir, idx_list, labels, t_samples, norm=norm,
                      transforms=transforms, acf=False, wavelet=False, kep_noise=kep_noise, prepare=False)
    self.return_raw = return_raw

  def __getitem__(self, idx):
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      # print(idx, sample_idx)
      try:
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        # x = x[int(0.5*len(x)):,:]
        info = dict()
      except Exception as e:
         print("Exception: ", e)
         return torch.zeros((2,self.seq_len)), torch.zeros((2)), torch.zeros((2,self.seq_len)), dict()
      if self.seq_len:
          x = self.interpolate(x)
      else:
        x = x[:,1]
      if self.transforms is not None:
        x, _, info = self.transforms(x, mask=None, info=dict())
      if self.kep_noise:
        x = self.add_kepler_noise(x)
      x = fill_nan_np(x, interpolate=True)
      xcf = A(x, nlags=len(x))
      xcf = fill_nan_np(xcf, interpolate=True)
      if np.isnan(xcf).any():
        pass
      x = torch.tensor(x)
      if self.return_raw:
        x = torch.stack([x,torch.tensor(xcf)])
      else:
        x = torch.tensor(xcf).unsqueeze(0)
      # y[0] /= 90
      if self.norm == 'std':
        x = (x - x.mean(dim=-1)[:,None])/(x.std(dim=-1)[:,None] + 1e-8)
      elif self.norm == 'median':
        x /= x.median()
      elif self.norm == 'minmax':
        mini = x.min(dim=-1).values.view(-1,1).float()
        maxi = x.max(dim=-1).values.view(-1,1).float()
        x = (x-mini)/(maxi - mini)
        x[0,:] = 0
      x = x.nan_to_num(0)
      if torch.isnan(x).any():
        print(f"idx {idx} - sample {sample_idx} is nan")
      
      y = super().get_labels(sample_idx)

      return x.float(), y, torch.ones_like(x), info

  

class ACFClassifierDataset(ACFDataset):
  def __init__(self, root_dir, idx_list, num_classes=10, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.cls = num_classes
  def __getitem__(self, idx):
    x,y, mask, info = super().__getitem__(idx)
    y1 = quantize_tensor(y[0].unsqueeze(0), self.cls)
    y2 = quantize_tensor(y[1].unsqueeze(0), self.cls)
    return x.float(), torch.cat([y1,y2], dim=-1).float().squeeze(0), torch.ones_like(x), info
  

class TimeImageDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, cls=10, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.cls = cls
      
  def __len__(self):
      return self.length
  
  def __getitem__(self, idx):
      x,y, mask, info = super().__getitem__(idx)
      if self.cls:
        y = quantize_tensor(y[0].unsqueeze(0), self.cls)
        # y2 = quantize_tensor(y[1].unsqueeze(0), self.cls)
      gaf = GramianAngularField()
      x_gram = gaf.transform(x.reshape(1,-1)).squeeze()
      x_gram = torch.tensor(x_gram).unsqueeze(0)
      return x_gram.float(), y.squeeze(0).squeeze(-1).float(), mask, info

class ACFImageDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, lags=None, image_transform=None, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.lags = lags
    self.image_transform = image_transform
    print("image transform: ", self.image_transform)
      
  def __len__(self):
      return self.length

  def auto_correlation_2d(self, x):
    x0 = x - x.mean()
    indices = torch.arange(x0.size(0)).unsqueeze(0) - torch.arange(x0.size(0)).unsqueeze(1)    
    lagged_data = x0[indices.abs()]
    corr = lagged_data * lagged_data[0].unsqueeze(0)
    return corr
  
  def __getitem__(self, idx):
      start = time.time()
      info = {'idx': idx}
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      # x = x[int(0.4*len(x)):,:]
      if self.seq_len:
        f = interp1d(x[:,0], x[:,1])
        new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
        x = np.concatenate(f(new_t)[:,None])
      if self.transforms is not None:
        x, _, info = self.transforms(x, mask=None, info=dict())

      x_t = fill_nan_np(x.numpy(), interpolate=True)
      xcf = A(x_t, nlags=len(x_t))
      xcf = torch.tensor(xcf.astype(np.float32))
      xcf = (xcf - xcf.mean())/(xcf.std() + 1e-8)

      x_im = self.auto_correlation_2d(x)
      # x_im = (x_im - x_im.min()) / (x_im.max() - x_im.min()) 
      if self.image_transform is not None:
        x_im = self.image_transform(x_im.unsqueeze(0)) 
      x_im = x_im.nan_to_num(0)
      xcf = xcf.nan_to_num(0)
      y = super().get_labels(sample_idx)
      return x_im.float(), y.squeeze(0).squeeze(-1).float(), xcf,  info

class LatDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, t_samples=1024, norm='minmax', transforms=None, vocab_size=1024, mask_prob = 0.15, mask_val= -1,
                cls_val = 0):
    super().__init__(root_dir, idx_list, t_samples, norm=norm, transforms=transforms)
    self.vocab_size = vocab_size
    self.mask_prob = mask_prob
    self.mask_val = mask_val
    self.cls_val = cls_val
    self.transforms = transforms

  def __getitem__(self, idx):
    sample_idx = remove_leading_zeros(self.idx_list[idx])
    x,_, mask, info = super().__getitem__(idx)
    y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
    y = torch.tensor(y['Spot Max']) > 45
    return x.float().unsqueeze(0), y.long().squeeze(), mask, info
  
class IncDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, t_samples=1024, norm='minmax', transforms=None, vocab_size=1024, mask_prob = 0.15, mask_val= -1,
                cls_val = 0):
    super().__init__(root_dir, idx_list, t_samples, norm=norm, transforms=transforms)
    self.vocab_size = vocab_size
    self.mask_prob = mask_prob
    self.mask_val = mask_val
    self.cls_val = cls_val
    self.transforms = transforms

  def __getitem__(self, idx):
    sample_idx = remove_leading_zeros(self.idx_list[idx])
    x,_, mask, info = super().__getitem__(idx)
    ratio = self.get_depth_width(x)
    

    row = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
    y = torch.tensor(row['Inclination'])
    params = torch.tensor([ratio, row['Period'], row['Decay Time'], row['Cycle Length'], row['Spot Max'], row['Activity Rate'],
                           row["Butterfly"]])
    params = params.nan_to_num(0)
    return x.float(), y.float(), params.float(), info
  
  def get_depth_width(self, x):
    peaks = find_peaks(x, prominence=0.1)[0]
    prominences = peak_prominences(x, peaks)[0]
    highest_idx = np.argsort(prominences)[-10:]
    prominences = prominences[highest_idx]
    widths = peak_widths(x, peaks[highest_idx], rel_height=0.1)[0]
    ratio = prominences / widths
    return ratio.mean()

class PlagDataset(ACFDataset):
  def __init__(self, root_dir, idx_list, dur, lag_len, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.dur = dur
    self.lag_len = lag_len
    self.time = np.arange(0, self.dur, cad / DAY2MIN)

  def __getitem__(self, idx):
    sample_idx = remove_leading_zeros(self.idx_list[idx])
    x, y, mask, info = super().__getitem__(idx)
    # p = x[:,0]*(boundary_values_dict['Period'][1] - boundary_values_dict['Period'][0]) + boundary_values_dict['Period'][0]
    # p = y['Period'].values[0]
    y_df = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
    p = y_df['Period'].values[0]
    if self.return_raw:
      lag1_a, lag2_a = self.find_peaks_in_lag( x[1,:], self.time, p)
      if lag1_a is None:
        num_channels = (self.return_raw + 1)*2
        return torch.zeros((num_channels, self.lag_len)), y, mask, info
      pad1 = self.lag_len - len(lag1_a)
      pad2 = self.lag_len - len(lag2_a)
      lag_xcf = torch.tensor(np.vstack([np.pad(lag1_a, (0, pad1)), np.pad(lag2_a, (0, pad2))]))
      lag1, lag2 = self.find_peaks_in_lag( x[0,:], self.time, p)
      pad1 = self.lag_len - len(lag1)
      pad2 = self.lag_len - len(lag2)
      lag_x = torch.tensor(np.vstack([np.pad(lag1, (0, pad1))[None], np.pad(lag2, (0, pad2))[None]]))
      lag_x = torch.vstack([lag_x, lag_xcf])
    else:
      lag1, lag2 = self.find_peaks_in_lag( x[0,:], self.time, p)
      if lag1 is None:
        num_channels = (self.return_raw + 1)*2
        return torch.zeros((num_channels, self.lag_len)), y, mask, info
      pad1 = self.lag_len - len(lag1)
      pad2 = self.lag_len - len(lag2)
      lag_x = torch.tensor(np.vstack([np.pad(lag1, (0, pad1))[None], np.pad(lag2, (0, pad2))[None]]))
      lag_x = torch.vstack([lag_x, lag_xcf])
    
    return lag_x.float(), y, mask, info

  def find_peaks_in_lag(self, x, time, p):
        peaks= find_peaks(x, prominence=0.1)[0]
        valleys = find_peaks(-x, prominence=0.1)[0]
        peaks_valleys = np.concatenate([peaks, valleys])
        if len(peaks) == 0:
            return None, None
        first_peak = peaks[0]
        # print(first_peak, len(x), p, p//2*DAY2MIN//(cad))
        # p1_indices = np.arange(fi
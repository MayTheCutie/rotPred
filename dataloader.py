import copy
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import acf as A
import random
from astropy.io import fits
from lightPred.utils import create_kepler_df
from matplotlib import pyplot as plt
from lightPred.utils import fill_nan_np


min_p, max_p = 0,60
min_i, max_i = 0, np.pi/2

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
          header = hdulist[1].header
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    return x,time

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
    def __init__(self, root_dir, paths_list, t_samples=512, norm='std', ssl_tf=None, transforms=None, acf=False, return_raw=False):
        # self.idx_list = idx_list
        self.length = len(paths_list) if paths_list is not None else 0
        self.root_dir = root_dir
        self.paths_list = paths_list
        self.seq_len = t_samples
        self.norm=norm
        self.ssl_tf = ssl_tf
        self.transforms = transforms
        self.acf = acf
        self.return_raw = return_raw

        
    def __len__(self):
        return self.length
    
    def read_data(self, idx, interpolate=True):
        path = self.paths_list[idx]
        filename = os.path.join(self.root_dir, path)
        try:
          x, time = read_fits(filename)
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
        return x
    
    def __getitem__(self, idx):
        # if idx % 1000 == 0:
        #   print(idx)
        x = self.read_data(idx)
        x = torch.tensor(fill_nan_np(x, interpolate=True))
        # min, max = torch.nanmin(x), torch.nanmax(x)
        # x = ((x-x.min())/(x.max()-x.min())).unsqueeze(0).unsqueeze(0)
        if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=dict())
        if self.acf:
          x = A(x, nlags=len(x))
        x = torch.tensor(x)
        x = x.unsqueeze(0).unsqueeze(0)
        # print("before augmentation: ", x.shape)
        # x = self.x_samples[idx].unsqueeze(0).unsqueeze(0)
        if self.ssl_tf is not None:
          x1 = self.ssl_tf(copy.deepcopy(x))
          x2 = self.ssl_tf(copy.deepcopy(x))
          x1 = (x1 - x1.mean())/(x1.std()+1e-8)
          x2 = (x2 - x2.mean())/(x2.std()+1e-8)
          return x1.squeeze(0).float(), x2.squeeze(0).float()
        else:
          return x, torch.zeros((1,self.seq_len))
        
        
class MaskedSSL(TimeSsl):
   def __init__(self, root_dir, paths_list, t_samples=1024, norm='minmax', transforms=None, vocab_size=1024, mask_prob = 0.15, mask_val= -1,
                cls_val = 0):
    super().__init__(root_dir, paths_list, t_samples, norm=norm, transforms=transforms)
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
  def __init__(self, root_dir, path_list, df=None, mask_prob=0.15, mask_val=-1,  **kwargs):
    super().__init__(root_dir, path_list, **kwargs)
    self.df = df
    self.length = len(self.paths_list) if self.df is None else len(self.df)
    self.cur_len = None
    self.mask_prob = mask_prob
    self.mask_val = mask_val

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

  def __getitem__(self, idx):
    if self.df is not None:
      row = self.df.iloc[idx]
      # print(row['KID'])
      try:
        for i in range(len(row['data_file_path'])):
          x,time = read_fits(row['data_file_path'][i])
          x /= x.max()
          x = fill_nan_np(np.array(x), interpolate=True)
          if i == 0:
            x_tot = x.copy()
          else:
            border_val = np.mean(x[0:10]) - np.mean(x_tot[-10:-1])
            x -= border_val
            x_tot = np.concatenate((x_tot, np.array(x)))
        x = torch.tensor(x_tot/x_tot.max())
        self.cur_len = len(x)
      except TypeError as e:
          print("TypeError: ", e)
          x = torch.zeros((self.cur_len))
    else:
      x =  self.read_data(idx).float()
      x /= x.max()
    
    info = dict()
    if self.transforms is not None:
          x, _, info = self.transforms(x, mask=None, info=info)
    if self.acf:
      xcf = torch.tensor(A(x, nlags=len(x)))
      if self.return_raw:
        x = torch.stack([x,torch.tensor(xcf)])
      else:
        x = torch.tensor(xcf).unsqueeze(-1)
    masked_x, inv_mask = self.mask_array(x.clone(), mask_percentage=self.mask_prob, mask_value=self.mask_val)
    torch.nan_to_num(masked_x, torch.nanmean(masked_x))
    torch.nan_to_num(x, torch.nanmean(x))

    info['idx'] = idx
    info['path'] = row['data_file_path'] if self.df is not None else self.paths_list[idx]
    info['KID']  = row['KID'] if self.df is not None else self.paths_list[idx].split("/")[-1].split("-")[0].split('kplr')[-1]
    return x.float(), masked_x.squeeze().float(), inv_mask, info


class KeplerLabeledDataset(Dataset):
  def __init__(self, df,  t_samples=512, norm='std', transforms=None, acf=False, return_raw=False):
      self.df = df
      self.seq_len = t_samples
      self.norm=norm
      self.transforms = transforms
      self.acf = acf 
      self.return_raw = return_raw

  def __len__(self):
      return len(self.df)
  
  def __getitem__(self, idx): 
    row = self.df.iloc[idx]
    try:
      row = self.df.iloc[idx]
      for i in range(len(row['data_file_path'])):
        x,time = read_fits(row['data_file_path'][i])
        x /= x.max()
        x = fill_nan_np(np.array(x), interpolate=True)
        if i == 0:
          x_tot = x
        else:
          border_val = x[0] - x_tot[-1]
          x -= border_val
          x_tot = np.concatenate((x_tot, np.array(x))) 
      x = x_tot
    except TypeError as e:
        print("TypeError: ", e)
        return torch.zeros((1,self.seq_len)), torch.zeros((1,self.seq_len)) 
    if self.seq_len:
      f = interp1d(time, x)
      new_t = np.linspace(time[0], time[-1], self.seq_len)
      x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
      x = torch.tensor(x.astype(np.float32))[:,1]
    else:
      x = torch.tensor(x.astype(np.float32))
    # x = torch.nan_to_num(x, torch.nanmean(x))
    if self.transforms is not None:
      x, _, info = self.transforms(x, mask=None, info=dict())
    if self.acf:
      xcf = torch.tensor(A(x, nlags=len(x)))
      if self.return_raw:
        x = torch.stack([x,torch.tensor(xcf)])
      else:
        x = torch.tensor(xcf).unsqueeze(-1)
    x = (x - x.mean())/(x.std()+1e-8)
    y = {'Period': torch.tensor(row['Prot'])/max_p, 'period_err': torch.tensor(row['Prot_err']),
          'Teff': torch.tensor(row['Teff']),'logg': torch.tensor(row['logg'])} 
    return x.float(), y

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
  def __init__(self, root_dir, idx_list, t_samples=512, norm='std', num_classes=10, vocab_size=30521, transforms=None,
                noise=False, noise_factor=4):
      self.idx_list = idx_list
      self.length = len(idx_list)
      self.targets_path = os.path.join(root_dir, "simulation_properties.csv")
      self.lc_path = os.path.join(root_dir, "simulations")
      self.seq_len = t_samples
      self.norm=norm
      self.cls = num_classes
      self.vocab_size = vocab_size
      self.transforms = transforms
      self.noise = noise
      self.noise_factor = noise_factor
      if self.noise:
        self.non_p_df = create_kepler_df(kepler_path, non_period_table_path)
      
  def __len__(self):
      return self.length
  
  def __getitem__(self, idx):
      info = {'idx': idx}
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      # x = x[int(0.4*len(x)):,:]
      if self.seq_len:
        f = interp1d(x[:,0], x[:,1])
        new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
        x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
      y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      y = torch.tensor([y['Inclination'], y['Period']]).float()
      y[0] = (y[0] - min_i)/(max_i-min_i)
      y[1] = (y[1] - min_p)/(max_p-min_p)
      x = torch.tensor(x.astype(np.float32))[:,1]
      if self.transforms is not None:
        x, _, info = self.transforms(x, mask=None, info=dict())
        info['idx'] = idx
      if self.noise:
        x = add_kepler_noise(x, self.non_p_df, factor=self.noise_factor)
      if self.norm == 'std':
        x = ((x-x.mean())/(x.std()+1e-8))
      elif self.norm == 'minmax':
        x = ((x-x.min())/(x.max()-x.min())*self.vocab_size).to(torch.long)
      return x.float(), y.squeeze(0).squeeze(-1).float(), torch.ones_like(x), info
  
class TimeSeriesClassifierDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, num_classes=10, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.cls = num_classes
  def __getitem__(self, idx):
    x,y = super().__geti
    tem__(idx)
    y1 = quantize_tensor(y[0].unsqueeze(0), self.cls)
    y2 = quantize_tensor(y[1].unsqueeze(0), self.cls)
    return x.float(), torch.cat([y1,y2], dim=-1).float().squeeze(0)
        
        
class ACFDataset(TimeSeriesDataset):
  def __init__(self, root_dir, idx_list, t_samples=512, norm='std', vocab_size=30521, return_raw=True, transforms=None):
    super().__init__(root_dir, idx_list, t_samples, norm=norm, vocab_size=vocab_size)
    self.return_raw = return_raw
    self.transforms = transforms

  def __getitem__(self, idx):
      sample_idx = remove_leading_zeros(self.idx_list[idx])
      # print(idx, sample_idx)
      try:
        x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
        x = x[int(0.5*len(x)):,:]
        info = dict()
      except Exception as e:
         print("Exception: ", e)
         return torch.zeros((2,self.seq_len)), torch.zeros((2)), torch.zeros((2,self.seq_len)), dict()
      if self.seq_len:
          f = interp1d(x[:,0], x[:,1])
          new_t = np.linspace(x[:,0][0], x[:,0][-1], self.seq_len)
          x = np.concatenate((new_t[:,None], f(new_t)[:,None]), axis=1)
      y = pd.read_csv(self.targets_path, skiprows=range(1,sample_idx+1), nrows=1)
      if self.transforms is not None:
        x, _, info = self.transforms(x, mask=None, info=dict())
      x[:,1] = fill_nan_np(x[:,1], interpolate=True)
      xcf = A(x[:,1], nlags=len(x[:,1]))
      xcf = fill_nan_np(xcf, interpolate=True)
      if np.isnan(xcf).any():
        pass
        # print(f"xcf idx {idx} - sample {self.idx_list[idx]} is nan")
        # fig, ax = plt.subplots(1,2)
        # ax[0].plot(x[:,1])
        # ax[0].set_title("x")
        # ax[1].plot(xcf)
        # ax[1].set_title("xcf")
        # plt.savefig(f"/data/tests/xcf_with_nans_{self.idx_list[idx]}.png")
        # print(x[:,1])
      x = torch.tensor(x[:,1])
      if self.return_raw:
        x = torch.stack([x,torch.tensor(xcf)])
      else:
        x = torch.tensor(xcf).unsqueeze(0)
      y = torch.round(torch.tensor([y['Inclination']*180/np.pi, y['Period']]))
      y[1] = (y[1] - min_p)/(max_p-min_p)
      # y[0] = (y[0] - min_i)/(max_i-min_i)
      y[0] /= 90
      if self.norm == 'std':
          x = (x - x.mean(dim=-1)[:,None])/(x.std(dim=-1)[:,None] + 1e-8)
      elif self.norm == 'minmax':
        mini = x.min(dim=-1).values.view(-1,1).float()
        maxi = x.max(dim=-1).values.view(-1,1).float()
        x = (x-mini)/(maxi - mini)
        x[0,:] = 0
      x = x.nan_to_num(0)
      if torch.isnan(x).any():
        print(f"idx {idx} - sample {sample_idx} is nan")
      return x.float(), torch.squeeze(y,dim=-1).float(), torch.ones_like(x), info

class ACFClassifierDataset(ACFDataset):
  def __init__(self, root_dir, idx_list, num_classes=10, **kwargs):
    super().__init__(root_dir, idx_list, **kwargs)
    self.cls = num_classes
  def __getitem__(self, idx):
    x,y, mask, info = super().__getitem__(idx)
    y1 = quantize_tensor(y[0].unsqueeze(0), self.cls)
    y2 = quantize_tensor(y[1].unsqueeze(0), self.cls)
    return x.float(), torch.cat([y1,y2], dim=-1).float().squeeze(0), torch.ones_like(x), info
  


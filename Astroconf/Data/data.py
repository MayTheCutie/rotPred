from glob import glob
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler

import os

import torchaudio.transforms as T
from scipy.interpolate import interp1d

import pandas as pd
from tqdm import tqdm

from .utils import longcadence, get_boxsize, preprocess_norm_mp, table_read, tokenizer, tokenize_path, pad_and_mask, preprocess_single_mock

from .transforms import *

from .sampler import DistributedSamplerWrapper

cad = 30
DAY2MIN = 24*60
min_p, max_p = 0,60
min_lat, max_lat = 0, 80
min_cycle, max_cycle = 1, 10
min_i, max_i = 0, 1
min_tau, max_tau = 1,10 
T_SUN = 5777
boundary_values_dict = {'Period': (min_p, max_p), 'Inclination': (min_i, max_i),
 'Decay Time': (min_tau, max_tau), 'Cycle Length': (min_cycle, max_cycle), 'Spot Max': (min_lat, max_lat),}

class Seismic_set(Dataset):
  '''Dataset class for single-quarter preprocessed light curves'''
  def __init__(self, paths, segment_len=4000):
    self.segment_len = segment_len
    lc_path, label_path = paths
    self.target = 'numax' if 'numax' in label_path else 'logg'

    label_df = table_read(label_path) 
    label_kids, label = label_df['KIC'].values, label_df[self.target].values
    lc_paths = sorted(glob(lc_path+'*.npy'))
    data = []
    kids = []
    for path in lc_paths[:]:#如果该dataset类出错，减小读取长度以缩短调试时间
      kid = int(path.split('/')[-1].split('.')[0])
      if kid in label_kids:
        lc = np.load(path)
        kids.append(kid)
        data.append(torch.from_numpy(lc).float())
    kids = np.array(kids)
    label = label[np.isin(label_kids, kids)]
    self.data = data
    self.kids = kids
    
    if self.target == 'numax':
      label = torch.log10(torch.FloatTensor(label))
      mean = label.mean()
      std = label.std()
      label = (label - mean) / std
      self.mean = mean.numpy()
      self.std = std.numpy() # result analysis is usually done in numpy
    else:
      label = torch.FloatTensor(label)
    self.label = label[:, None]
    self.mean_label = self.label.mean()
    self.info = f"{len(data)} stars from {len(lc_paths)} candidates and {len(label_df)} labels.\n"
    print(self.info)

  def __getitem__(self, idx):
    lc = self.data[idx]
    segment_len = self.segment_len
    if self.transform:
      if len(lc) > segment_len:
        start = np.random.randint(0, len(lc) - segment_len)
        lc = lc[start:start+segment_len]
    else:
      lc = lc[:segment_len]
    return lc, self.label[idx], self.kids[idx]

  def __len__(self):
    return len(self.data)

class PDCSAP_MQ(Dataset):
  '''PDCSAP dataset and quarter-awareness in order for 
  a sample-level comparison with asteroseismology'''
  def __init__(self, paths, segment_len=4000, transform=True):
    self.segment_len = segment_len
    self.transform = transform
    lc_dir, label_path, radius_path = paths
    self.target = 'numax' if 'numax' in label_path else 'logg'
    
    label_df, radius_df = table_read(label_path), table_read(radius_path)
    label_df = label_df.merge(radius_df, on='KIC', how='inner')
    label_kids = label_df['KIC'].values
    label = label_df[self.target].values
    radii = label_df['R'].values
    
    lc_paths = sorted(glob(lc_dir+'*.npy'))
    num_candidates = len(lc_paths)
    data_kids = np.sort(np.array([int(path.split('/')[-1].split('.')[0]) for path in lc_paths]))
    
    label = label[np.isin(label_kids, data_kids)]
    radii = radii[np.isin(label_kids, data_kids)]
    lc_paths = np.array(lc_paths)[np.isin(data_kids, label_kids)]
    data_kids = data_kids[np.isin(data_kids, label_kids)]
    
    data = []
    quarters = []
    start_times = []
    end_times = []
    boxsizes = []
    idx = []
    valid_lengths = []
    for Q in range(18):
      start_times.append(longcadence.iloc[Q]['t_first_cadence']-54833)
      end_times.append(longcadence.iloc[Q]['t_last_cadence']-54833)
    for i in range(int(len(lc_paths))):
      star = []
      quarter_flag = []
      lc = np.load(lc_paths[i])
      lc = lc[:, lc[1]!=0]
      for Q, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        quarter = lc[:, (lc[0]>=start_time) & (lc[0]<=end_time)]
        valid_lengths.append(quarter.shape[-1])
        if quarter.shape[-1] >= 3000:
          star.append(quarter)
          quarter_flag.append(Q)
      if len(star) >= 1:
        data.append(star)
        quarters.append(np.array(quarter_flag))
        radius = radii[i]
        boxsize = get_boxsize(radius)
        boxsizes.append(boxsize)
        idx.append(i)
    with mp.Pool(mp.cpu_count()*6) as pool:
      results = pool.map(preprocess_norm_mp, zip(data, boxsizes, quarters))
    
    data, stds, quarters = map(list, zip(*results))
    self.maxstd = max(stds)
    self.valid_lengths = valid_lengths 
    
    data = [[torch.from_numpy((quarter-quarter.mean())/quarter.std()/(np.log(self.maxstd/quarter.std())+1)).float() for quarter in star] for star in data]
    # data = [(star-star.mean())/std/(np.log(self.maxstd/std)+1) for star, std in zip(data, stds)]
    # data = [torch.from_numpy(star).float() for star in data]
    
    self.data = data
    self.quarters = quarters
    label = label[idx]
    
    if self.target == 'numax':
      label = torch.log10(torch.FloatTensor(label))
      mean = label.mean()
      std = label.std()
      label = (label - mean) / std
      self.mean = mean.numpy()
      self.std = std.numpy() # result analysis is usually done in numpy
    else:
      label = torch.FloatTensor(label)
    self.label = label[:, None]
    self.mean_label = self.label.mean()
    self.kids = data_kids[idx]
    self.boxsizes = boxsizes
    self.info = f"{sum([len(star) for star in data])} quarters from {len(data)} stars from {num_candidates} candidates and {len(label_df)} labels.\n"
    print(self.info)

  def __getitem__(self, idx):
    star = self.data[idx]
    segment_len = self.segment_len

    if self.transform:
      lengths = [quarter.shape[-1] for quarter in star]
      starts = [np.random.randint(0, length - segment_len) for length in lengths]
      star = torch.stack([quarter[start:start+segment_len] for quarter, start in zip(star, starts)])
    else:
      star = torch.stack([quarter[:segment_len] for quarter in star])

    return star, self.label[idx], self.kids[idx]

  def __len__(self):
    return len(self.data)
  
class PDCSAP_MQ_semisupervision(Dataset):
  '''PDCSAP_MQ adjusted for semi-supervised learning.'''
  def __init__(self, paths, segment_len=4000, transform=True):
    self.segment_len = segment_len
    self.transform = transform
    lc_dir, label_path, radius_path = paths
    self.target = 'numax' if 'numax' in label_path else 'logg'
    
    label_df, radius_df = table_read(label_path), table_read(radius_path)
    label_df = label_df.merge(radius_df, on='KIC', how='inner')
    label_kids = label_df['KIC'].values
    label = label_df[self.target].values
    radius_kids = radius_df['KIC'].values
    self.kic_to_label = {kid: lbl for kid, lbl in zip(label_kids, label)}
    self.kic_to_radius = {kid: r for kid, r in zip(radius_kids, radius_df['R'].values)}
    
    lc_paths = sorted(glob(lc_dir+'*.npy'))
    num_candidates = len(lc_paths)
    data_kids = np.sort(np.array([int(path.split('/')[-1].split('.')[0]) for path in lc_paths]))
    lc_paths = np.array(lc_paths)[np.isin(data_kids, radius_kids)]
    data_kids = data_kids[np.isin(data_kids, radius_kids)]
    label_kids = label_kids[np.isin(label_kids, data_kids)]
    label = np.array([self.kic_to_label.get(kid, np.nan) for kid in data_kids])
    radii = np.array([self.kic_to_radius.get(kid) for kid in data_kids])
    
    data = []
    quarters = []
    start_times = []
    end_times = []
    boxsizes = []
    idx = []
    valid_lengths = []
    for Q in range(18):
      start_times.append(longcadence.iloc[Q]['t_first_cadence']-54833)
      end_times.append(longcadence.iloc[Q]['t_last_cadence']-54833)
    for i in range(int(len(lc_paths))):
      star = []
      quarter_flag = []
      lc = np.load(lc_paths[i])
      lc = lc[:, lc[1]!=0]
      for Q, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        quarter = lc[:, (lc[0]>=start_time) & (lc[0]<=end_time)]
        valid_lengths.append(quarter.shape[-1])
        if quarter.shape[-1] >= 3000:
          star.append(quarter)
          quarter_flag.append(Q)
      if len(star) >= 1:
        data.append(star)
        quarters.append(np.array(quarter_flag))
        radius = radii[i]
        boxsize = get_boxsize(radius)
        boxsizes.append(boxsize)
        idx.append(i)
    with mp.Pool(mp.cpu_count()*6) as pool:
      results = pool.map(preprocess_norm_mp, zip(data, boxsizes, quarters))
    
    data, stds, quarters = map(list, zip(*results))
    self.maxstd = np.load('PDCSAP_ast_maxstd.npy')
    self.valid_lengths = valid_lengths 
    
    data = [[torch.from_numpy((quarter-quarter.mean())/quarter.std()/(np.log(self.maxstd/quarter.std())+1)).float() for quarter in star] for star in data]
    # data = [(star-star.mean())/std/(np.log(self.maxstd/std)+1) for star, std in zip(data, stds)]
    # data = [torch.from_numpy(star).float() for star in data]
    
    self.data = data
    self.quarters = quarters
    label = label[idx]
    
    if self.target == 'numax':
      mean = np.nanmean(label)
      std = np.nanstd(label)
      label = (label - mean) / std
      label = torch.log10(torch.FloatTensor(label))
      self.mean = mean.numpy()
      self.std = std.numpy() # result analysis is usually done in numpy
    else:
      label = torch.FloatTensor(label)
    self.label = label[:, None]
    self.mean_label = self.label.mean()
    self.kids = data_kids[idx]
    self.boxsizes = boxsizes
    self.info = f"{sum([len(star) for star in data])} quarters from {len(data)} stars from {num_candidates} candidates and {len(label_df)} labels.\n"
    print(self.info)

  def __getitem__(self, idx):
    star = self.data[idx]
    segment_len = self.segment_len

    if self.transform:
      lengths = [quarter.shape[-1] for quarter in star]
      starts = [np.random.randint(0, length - segment_len) for length in lengths]
      star = torch.stack([quarter[start:start+segment_len] for quarter, start in zip(star, starts)])
    else:
      star = torch.stack([quarter[:segment_len] for quarter in star])

    return star, self.label[idx], self.kids[idx]

  def __len__(self):
    return len(self.data)
  
class Kepseismic(Dataset):
  '''Full Kepseismic dataset with all available logg label. '''
  def __init__(self, paths, segment_len=4000, transform=True):
    self.segment_len = segment_len
    self.transform = transform
    lc_dir, label_path = paths
    self.target = 'numax' if 'numax' in label_path else 'logg'

    label_df = table_read(label_path)
    label_kids = label_df['KIC'].values
    label = label_df[self.target].values
    lc_paths = np.sort(np.array(glob(lc_dir+'*.npy')))
    data_kids = np.sort(np.array([int(path.split('/')[-1].split('.')[0]) for path in lc_paths]))
    lc_paths = lc_paths[np.isin(data_kids, label_kids)]
    data_kids = data_kids[np.isin(data_kids, label_kids)]
    label = label[np.isin(label_kids, data_kids)]
    label_kids = label_kids[np.isin(label_kids, data_kids)]
    
    data = []
    quarters = []
    start_times = []
    end_times = []
    stds = []
    idx = []
    for Q in range(18):
      start_times.append(longcadence.iloc[Q]['t_first_cadence'])
      end_times.append(longcadence.iloc[Q]['t_last_cadence'])
    for i in range(int(len(lc_paths))):
      star = []
      quarter_flag = np.zeros(18)
      lc = np.load(lc_paths[i])
      lc[1] = lc[1]/1e6
      lc = lc[:, lc[1]!=0]
      for Q, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        quarter = lc[:, (lc[0]>=start_time) & (lc[0]<=end_time)]
        if quarter.shape[-1] >= 4100:
          quarter_start, quarter_end = quarter[0,0], quarter[0,-1]
          time_interp = np.arange(quarter_start, quarter_end, 30./(60.*24.))
          flux_interp = np.interp(time_interp, quarter[0], quarter[1])
          if len(flux_interp) >= 4001:
            star.append(flux_interp)
            quarter_flag[Q] = 1
      if len(star) >= 1:
        data.append(star)
        quarters.append(quarter_flag)
        stds.append(np.concatenate(star).std())
        idx.append(i)
    
    self.maxstd = max(stds)
    
    data = [[torch.from_numpy((quarter-quarter.mean())/quarter.std()/(np.log(self.maxstd/quarter.std())+1)).float() for quarter in star] for star in data]
    
    self.data = data
    self.quarters = quarters
    label = label[idx]
    if self.target == 'numax':
      label = torch.log10(torch.FloatTensor(label))
      mean = label.mean()
      std = label.std()
      label = (label - mean) / std
      self.mean = mean.numpy()
      self.std = std.numpy() # result analysis is usually done in numpy
    else:
      label = torch.FloatTensor(label)
    self.label = label[:, None]
    self.mean_label = self.label.mean()
    self.kids = data_kids[idx]
    self.info = f"{sum([len(star) for star in data])} quarters from {len(data)} stars from {len(lc_paths)} candidates and {len(label_df)} labels.\n"
    print(self.info)
        
  def __getitem__(self, idx):
    star = self.data[idx]
    segment_len = self.segment_len

    if self.transform:
      lengths = [quarter.shape[-1] for quarter in star]
      starts = [np.random.randint(0, length - segment_len) for length in lengths]
      star = torch.stack([quarter[start:start+segment_len] for quarter, start in zip(star, starts)])
    else:
      star = torch.stack([quarter[:segment_len] for quarter in star])

    return star, self.label[idx], self.kids[idx]
  
  def __len__(self):
    return len(self.data)
  
class Kepseismic_normed(Dataset):
  '''Full Kepseismic dataset with all available logg label, normalized. '''
  def __init__(self, paths):
    lc_dir, label_path, radius_path = paths
    label_df, radius_df = table_read(label_path), table_read(radius_path)
    label_df = label_df.merge(radius_df, on='KIC', how='inner')
    label_kids = label_df['KIC'].values
    label = label_df[self.target].values
    radii = label_df['R'].values
    
    lc_paths = sorted(glob(lc_dir+'*.npy'))
    data_kids = np.array([int(path[-13:-4]) for path in lc_paths])
    
    label = label[np.isin(label_kids, data_kids)]
    radii = radii[np.isin(label_kids, data_kids)]
    lc_paths = np.array(lc_paths)[np.isin(data_kids, label_kids)]
    data_kids = data_kids[np.isin(data_kids, label_kids)]
    
    data = []
    quarters = []
    start_times = []
    end_times = []
    boxsizes = []
    idx = []
    for Q in range(18):
      start_times.append(longcadence.iloc[Q]['t_first_cadence'])
      end_times.append(longcadence.iloc[Q]['t_last_cadence'])
    for i in range(int(len(lc_paths))):
      star = []
      quarter_flag = np.zeros(18)
      lc = np.load(lc_paths[i])
      lc[1] = lc[1]/1e6
      lc = lc[:, lc[1]!=0]
      for Q, (start_time, end_time) in enumerate(zip(start_times, end_times)):
        quarter = lc[:, (lc[0]>=start_time) & (lc[0]<=end_time)]
        if quarter.shape[-1] >= 4100:
          star.append(quarter)
          quarter_flag[Q] = 1
      if len(star) >= 1:
        data.append(star)
        quarters.append(quarter_flag)
        radius = radii[i]
        boxsize = get_boxsize(radius)
        boxsizes.append(boxsize)
        idx.append(i)
    
    with mp.Pool(mp.cpu_count()*6) as pool:
      results = pool.map(preprocess_norm_mp, zip(data, boxsizes))
    
    data, stds = map(list, zip(*results))
    self.maxstd = max(stds)
    
    data = [(star-star.mean())/std/(np.log(self.maxstd/std)+1) for star, std in zip(data, stds)]
    data = [torch.from_numpy(star).float() for star in data]
    
    self.data = data
    self.quarters = quarters
    label = label[idx]
    
    if 'numax' in label_path:
      self.target = 'numax'
      label = torch.log10(torch.FloatTensor(label))
      mean = label.mean()
      std = label.std()
      label = (label - mean) / std
      self.mean = mean.numpy()
      self.std = std.numpy() # result analysis is usually done in numpy
    else:
      self.target = 'logg'
      label = torch.FloatTensor(label)
    self.label = label
    self.mean_label = self.label.mean()
    self.kids = data_kids[idx]
    self.info = f"{sum([len(star) for star in data])} quarters from {len(data)} stars from {len(lc_paths)} candidates and {len(label_df)} labels.\n"
    print(self.info)

  def __getitem__(self, idx):
    return self.data[idx], self.label[idx], self.kids[idx]
  
  def __len__(self):
    return len(self.data)
  
class Kepseismic_token(Dataset):
  '''Full Kepseismic dataset with all available logg label. Also providing customized data format for Transformer'''
  def __init__(self, paths):
    lc_dir, label_path = paths
    label_df = pd.read_csv(label_path)
    label_kids = label_df['KIC'].values
    label = label_df['logg'].values
    lc_paths = sorted(glob(lc_dir+'*.npy'))
    data_kids = sorted(np.array([int(path[-13:-4]) for path in lc_paths]))
    label = label[np.isin(label_kids, data_kids)]
    label_kids = label_kids[np.isin(label_kids, data_kids)]
    self.start_time = longcadence['t_first_cadence'].min()
    self.end_time = longcadence['t_last_cadence'].max()
    
    data = []
    stds = []
    idx = []
    num_tokens = []
    for i in range(int(len(lc_paths))):
      lc = np.load(lc_paths[i])
      lc[1] = lc[1]/1e6
      try: 
        tokens = tokenizer(lc)  
      except:
        continue
      if np.isnan(tokens).any():
        continue
      data.append(tokens)
      stds.append(tokens[:, 1:].std())
      num_tokens.append(len(tokens))
      idx.append(i)
    
    self.maxstd = max(stds)
    self.max_len = max(num_tokens)
    self.time_scale = (self.end_time-self.start_time)
    
    for i, (tokens, std) in enumerate(zip(data, stds)):
      tokens[:, 0] = (tokens[:, 0]-self.start_time)/self.time_scale-0.5
      
      tokens[:, 1:] = tokens[:, 1:]/(np.log(self.maxstd/std)+1)
      
      data[i] = torch.from_numpy(tokens).float()
    
    self.data = data
    self.label = label[idx]
    self.mean_label = self.label.mean()
    self.kids = label_kids[idx]
    self.info = print(f'{sum(num_tokens)} tokens from {len(data)} light curves from {len(label_df)} candidate stars.\n')
    print(self.info)

  def __getitem__(self, idx):
    return self.data[idx], self.label[idx], self.kids[idx]
  
  def __len__(self):
    return len(self.data)
  

    
class Kepseismic_token_unsupervised(Dataset):

  '''Full Kepseismic dataset used for unsupervised learning. Also providing customized data format for Transformer'''
  def __init__(self, paths):
    lc_dir = paths[0]
    lc_paths = sorted(glob(lc_dir+'*.npy'))
    data_kids = np.sort(np.array([int(path[-13:-4]) for path in lc_paths]))
    
    with mp.Pool(mp.cpu_count()*6) as pool:
        results = pool.map(tokenize_path, enumerate(lc_paths))

    results = [r for r in results if r is not None]
    idx, data, stds, num_tokens = map(list, zip(*results))
    
    idx = np.array(idx)
    self.start_time = longcadence['t_first_cadence'].min()
    self.end_time = longcadence['t_last_cadence'].max()
    self.max_len = max(num_tokens)
    self.time_scale = (self.end_time-self.start_time)
    self.maxstd = max(stds)
    
    for i, (tokens, std) in enumerate(zip(data, stds)):
      tokens[:, 0] = (tokens[:, 0]-self.start_time)/self.time_scale-0.5
      
      tokens[:, 1:] = tokens[:, 1:]/(np.log(self.maxstd/std)+1)
      
      data[i] = torch.from_numpy(tokens).float()
        
    self.data = data
    self.kids = data_kids[idx]
    self.info = print(f'{sum(num_tokens)} tokens from {len(data)} light curves from 207607 candidate stars.\n')
    print(self.info)

  def __getitem__(self, idx):
    return self.data[idx], self.kids[idx]
  
  def __len__(self):
    return len(self.data)

class TimeSeriesDataset(Dataset):
  def __init__(self, paths,  labels=['Inclination', 'Period'], t_samples=360*48, norm='std', num_classes=2, transform=False,
                noise=False, spectrogram=False, n_fft=1000, p_norm=False):
      print("***************creating dataset object**************", flush=True)
      self.lc_path, self.labels_path = paths
      self.Nlc = len(glob(self.lc_path+'/*.pqt'))
      self.Nlc = self.Nlc // 3 # for debugging
      print(f"{self.Nlc} light curves found in {self.lc_path}.")
      self.idx_list = [f'{idx:d}'.zfill(int(np.log10(self.Nlc))+1) for idx in range(self.Nlc)]
      self.target = ', '.join(labels)
      self.mean_label=None
      self.labels = labels
      self.seq_len = t_samples
      self.norm=norm
      self.num_classes = num_classes
      self.transforms = None
      if transform:
        print("creating transforms object", flush=True)
        if not p_norm:
          self.transforms = Compose([Detrend(), RandomCrop(t_samples)])
        else:
          self.transforms = Compose([Detrend()])
        print("transform object: ", self.transforms, flush=True)
      # else:
      #   print("SELF.TRANSFORM SHOULD BE NONE", flush=True)
      #   self.transforms = None
      self.p_norm = p_norm

      # self.transform = Compose([Detrend(), RandomCrop(int(dur/cad*DAY2MIN))])

      self.noise = noise
      self.n_fft = n_fft
      self.spec = spectrogram
      self.info = f"{self.Nlc} stars from {self.Nlc} candidates and {self.Nlc} labels.\n"
      print("transform object2: ", self.transforms, flush=True)

      stds = []
      self.maxstd = 0.15
      self.weights = torch.ones(self.Nlc)
      self.samples = []
      self.prepare_data()
     
  def prepare_data(self):
    stds = []
    all_inclinations = np.arange(0, 91)
    counts = np.zeros_like(all_inclinations)
    incl = (np.arcsin(np.random.uniform(0, 1, self.Nlc))*180/np.pi).astype(np.int16)
    unique, unique_counts = np.unique(incl, return_counts=True)
    counts[unique] = unique_counts
    weights = torch.tensor([])
    pbar = tqdm(range(len(self.idx_list)))
    for idx in pbar:
      sample_idx = self.remove_leading_zeros(self.idx_list[idx])
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      x = x[int(0.4*len(x)):,:]
      
      y = pd.read_csv(self.labels_path, skiprows=range(1,sample_idx+1), nrows=1)
      w = 1/counts[(y['Inclination']*180/np.pi).astype(np.int16)]
      weights = torch.cat((weights, torch.tensor(w)), dim=0)
      flux = self.fill_nan_np(x[:,1])
      stds.append(flux.std())

      # if self.labels is None:
      #   self.samples.append((x, y))
      #   return 
      # y = torch.tensor([y[label] for label in self.labels])
      # if 'Inclination' in self.labels:
      #    y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
      # for i,label in enumerate(self.labels):
      #   if label in boundary_values_dict.keys():
      #     y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
      # self.samples.append((x, y, sample_idx))    
    self.maxstd = max(stds)
    self.weights = weights
    print("maxstd: ", self.maxstd)

  def remove_leading_zeros(self, s):   
    s = s.lstrip('0')
    if not s:
        return 0
    return int(s) 
  
  def __len__(self):
      return self.Nlc
  
  def fill_nan_np(self, x, interpolate=True):
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
  

  def __getitem__(self, idx):
      # s = time.time()
      info = {'idx': idx}
      sample_idx = self.remove_leading_zeros(self.idx_list[idx])
      x = pd.read_parquet(os.path.join(self.lc_path, f"lc_{self.idx_list[idx]}.pqt")).values
      y = self.get_labels(sample_idx)
      x = x[int(0.4*len(x)):,:]
      # x, y, sample_idx = self.samples[idx]
      if self.p_norm:
        row = pd.read_csv(self.labels_path, skiprows=range(1,sample_idx+1), nrows=1)
        period = row['Period'].values[0]
        x = preprocess_single_mock(x, self.maxstd, transforms=self.transforms, p_norm=period)
      else:
        x = preprocess_single_mock(x, self.maxstd, transforms=self.transforms)
      x = torch.tensor(self.fill_nan_np(x))
      return x.float(), y, torch.tensor(sample_idx)
  
  def get_labels(self, sample_idx):
      y = pd.read_csv(self.labels_path, skiprows=range(1,sample_idx+1), nrows=1)
      if self.labels is None:
        return y
      y = torch.tensor([y[label] for label in self.labels])
      if 'Inclination' in self.labels:
         y[self.labels.index('Inclination')] = np.sin(y[self.labels.index('Inclination')])
      for i,label in enumerate(self.labels):
        y[i] = (y[i] - boundary_values_dict[label][0])/(boundary_values_dict[label][1]-boundary_values_dict[label][0])
      
      if len(self.labels) == 1:
        return y.squeeze(-1).float()
      return y.squeeze(0).squeeze(-1).float()
  
def get_dataloader(named_dataset, collate_fn, args, rank=0, world_size=1):
  '''Generate case-aware dataloader for given tr/val/test indices, which use drop_last=True and shuffle=True for 
  training dataloader. Note that the input should be a dict, which could preserve the order of the input item after
  python 3.7.'''
  loaders = []

  for key, dataset in named_dataset.items():
    print("dataset transforms: ", dataset.dataset.transform, flush=True)
    if dataset is None:
      loaders.append(None)
      continue

    drop_last, shuffle = (True, True) if 'tr' in key else (False, False)
    if args.distributed and key != 'test':
      # weights = dataset.dataset.weights
      # sampler = DistributedSamplerWrapper(sampler=WeightedRandomSampler(weights, len(weights)),
      #                                         num_replicas=world_size, rank=rank)
      sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
      loaders.append(DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, sampler=sampler, \
                        num_workers=args.n_workers, pin_memory=True, drop_last=drop_last))
    else:
      loaders.append(DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, \
                         num_workers=args.n_workers, pin_memory=True, drop_last=drop_last, shuffle=shuffle))
  return loaders

def multi_quarter_random_collate(x):
  batch = []
  num_quarters = [len(x[i][0]) for i in range(len(x))]

  data = torch.stack([x[i][0][torch.randint(0, num_quarter, (1,)).item()] for i, num_quarter in enumerate(num_quarters)], 0)
  batch.append(data)

  label = torch.stack([x[i][1] for i in range(len(x))])
  batch.append(label)
  
  kid = np.array([x[i][2] for i in range(len(x))])
  batch.append(kid)

  quarter = np.array([x[i][3] for i in range(len(x))])
  batch.append(quarter)

  return batch

def multi_quarter_collate(x):
  '''Collect all available quarters for each star, and repeat labels according to the number of quarters. Might be\
    flawed on sample diversity.'''
  batch = []
  num_quarters = [len(x[i][0]) for i in range(len(x))]
  
  data = torch.cat([x[i][0] for i in range(len(x))], 0)
  batch.append(data)
  
  label = torch.stack([x[i][1] for i in range(len(x))])
  batch.append(label.repeat_interleave(torch.tensor(num_quarters), dim=0))

  kid = np.array([x[i][2] for i in range(len(x))])
  batch.append(np.repeat(kid, num_quarters))

  return batch

def semisupervision_collate(x):
  '''Collect all available quarters for each star, and repeat labels according to the number of quarters. Might be\
    flawed on sample diversity.'''
  batch = []
  num_quarters = [len(x[i][0]) for i in range(len(x))]
  sample_indices = [np.random.choice(num_quarter, 2).item() for num_quarter in num_quarters]
  
  data = [x[i][0][sample_indices[i]].unsqueeze(0) for i in range(len(x))]
  batch.append(data)
  
  label = torch.stack([x[i][1] for i in range(len(x))])
  batch.append(label.repeat_interleave(torch.tensor(num_quarters), dim=0))

  kid = np.array([x[i][2] for i in range(len(x))])
  batch.append(np.repeat(kid, num_quarters))

  return batch

def pad_mask_collate(x):
  batch = []
  
  sequence_list = [x[i][0] for i in range(len(x))]
  padded_sequence, mask = pad_and_mask(sequence_list)
  batch.append(padded_sequence)
  batch.append(mask)
  
  label = torch.FloatTensor([x[i][1] for i in range(len(x))]).unsqueeze(1)
  batch.append(label)
  
  kid = np.concatenate([x[i][2] for i in range(len(x))])
  batch.append(kid)
      
  return batch

ref = ['/g/data/y89/jp6476/hlsp_kg-radii_kepler-gaia_multi_all_multi_v1_star-cat.fits', \
       ]
data_dict = {
    'Seismic_logg': [['/g/data/y89/jp6476/seismic_npy/', '/g/data/y89/jp6476/Maryum2021.csv'],\
       Seismic_set, None],
    'Seismic_numax': [['/g/data/y89/jp6476/seismic_npy/', '/g/data/y89/jp6476/yu18_numax.csv'],\
       Seismic_set, None],
    'PDCSAP_MQ_numax': [['/g/data/y89/jp6476/PDCSAP/', '/g/data/y89/jp6476/yu18_numax.csv', \
                   '/g/data/y89/jp6476/mathur_output.csv'],\
       PDCSAP_MQ, multi_quarter_collate],
    'PDCSAP_MQ_logg': [['/g/data/y89/jp6476/PDCSAP/', '/g/data/y89/jp6476/Maryum2021.csv', \
                   '/g/data/y89/jp6476/mathur_output.csv'],\
       PDCSAP_MQ, multi_quarter_collate],
    'Kepseismic_maryum': [['/g/data/y89/jp6476/Kepseismic_full/', '/g/data/y89/jp6476/Maryum2021.csv'],\
       Kepseismic, multi_quarter_collate],
    'Kepseismic_MQ': [['/g/data/y89/jp6476/Kepseismic_full/', '/g/data/y89/jp6476/Kepler_sample_ast.csv'],\
       Kepseismic, multi_quarter_collate],
    'Kepseismic_normed': [['/g/data/y89/jp6476/Kepseismic_full/', '/g/data/y89/jp6476/yu18_logg.csv',\
                          '/g/data/y89/jp6476/hlsp_kg-radii_kepler-gaia_multi_all_multi_v1_star-cat.fits'],\
       Kepseismic_normed, multi_quarter_collate],
    'Kepseismic_token': [['/g/data/y89/jp6476/Kepseismic_full/', '/g/data/y89/jp6476/Kepler_sample_ast.csv'],\
       Kepseismic_token, pad_mask_collate],
    'TimeSeriesDataset': [['/data/butter/data2/simulations', '/data/butter/data2/simulation_properties.csv'],\
      TimeSeriesDataset, None],

}



def data_provider(args):
  paths, data_construct, collate_fn = data_dict[args.dataset]
  data_set = data_construct(paths, labels=args.labels, transform=args.transform)
  if 'token' in args.dataset:
    args.input_shape = (args.batch_size, data_set.max_len, 81)
  elif 'MQ' in args.dataset:
    args.input_shape = (10*args.batch_size, 4000)
  else:
    args.input_shape = (args.batch_size, 4000)

  with open(args.log_dir+'log.txt', 'a') as fp:
    fp.write(f"[Info]: Finish loading {args.dataset}!\n")
  if 'token' in args.dataset:
    args.sequence_scale = data_set.max_len
  if 'unsupervised' in args.dataset:
    args.target = 'unsupervised'
  else:
    args.target = data_set.target
    args.mean_label = data_set.mean_label
  data_set.transform = args.transform
  
  return data_set, collate_fn
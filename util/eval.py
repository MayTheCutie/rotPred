from collections import OrderedDict
import json
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import torch.optim as optim
import yaml
import warnings
import sys

from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from dataset.dataloader import *
from util.utils import *
from nn.train import *
import yaml
import glob
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore")


def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1

  # Extract unique coordinates and their counts
  coords = np.array(list(coord_counts.keys()))
  counts = np.array(list(coord_counts.values()))
  return coords, counts/np.sum(counts)

def sample_subset(array, probabilities):
    # Normalize probabilities to sum up to 1
    probabilities /= np.sum(probabilities)
    
    # Sample indices from the array based on probabilities
    subset_indices = np.random.choice(len(array), size=1, p=probabilities)
    
    # Create the subset array by selecting values from the original array
    subset_array = array[subset_indices]
    
    return subset_array

    

def eval_results(output, target, conf, data_dir, model_name, boundaries, cls=False, labels=['Inclination', 'Period'], num_classes=2, only_one=False, test_df=None,
                scale_target=True, scale_output=True, kepler=False, cos_inc=False):
    """_summary_
    evaluate results of a model and plot
    Args:
    
    """
    num_classes = len(labels) if not cls else num_classes
    if scale_target:
        print("boundaries: ", boundaries)
        max_p, min_p = boundaries['Period']
        max_i, min_i = boundaries['Inclination']
        print("scaling target with cos_inc: ", cos_inc)
        print("target before ", target[:,1].max(), target[:,1].min())
        print("output before ", output[:,1].max(), output[:,1].min())
        for i in range(len(labels)):
            if labels[i] == 'Inclination' and cos_inc:
                target[:,i] = np.arccos(target[:,i])
            else:
                target[:,i] = target[:,i] * (boundaries[labels[i]][1] - boundaries[labels[i]][0]) + boundaries[labels[i]][0]
        
    if scale_output:
        print("scaling output")
        for i in range(len(labels)):
            print(labels[i], "before ", output[:,i].max(), output[:,i].min())
            if labels[i] == 'Inclination' and cos_inc:
                output[:,i] = np.arccos(np.clip(output[:,i], a_min=0, a_max=1))
            else:
                output[:,i] = output[:,i] * (boundaries[labels[i]][1] - boundaries[labels[i]][0]) + boundaries[labels[i]][0]
            print(labels[i], "after ", output[:,i].max(), output[:,i].min())

    df, diff =calc_diff(target, output, conf, test_df=test_df, cls=cls, labels=labels, num_classes=num_classes)
    for i in range(len(labels)):            
        thresholds = [0] if not len(conf) else [0,0.8,0.83, 0.85,0.87,0.88,0.89,0.9,0.91, 0.92, 0.94,0.96,0.98]
        for thresh in thresholds:
            high_conf = np.where(1 - np.abs(conf[:,i]) > thresh)[0] if len(conf) else np.arange(len(target))
            name = f'{model_name}-{labels[i]}'
            if len(high_conf > 1):
                plot_results(data_dir, name, target[:,i][high_conf], output[:,i][high_conf], conf=thresh)
    if not kepler:
        plot_df(df, data_dir, labels=labels)
    else:
        plot_kepler_df(df, data_dir)


def plot_results(data_dir, name, target, output, conf=''):
    
    # df, diff = calc_diff(target, output)
    diff = np.abs(output - target)

    acc_6 = (diff < 6).sum()/len(diff)
    acc_10 = (diff < 10).sum()/len(diff)
    acc_10p = (diff < target/10).sum()/len(diff)
    mean_error = np.mean(diff/(target+1)*100)

    print("num correct: ", (diff < target/10).sum())
    plt.figure(figsize=(12, 8))

    plt.scatter(target, output, label=f'confidence > {conf} ({len(target)} points)')
    plt.plot(target, 0.9*target, color='red')
    plt.plot(target, 1.1*target, color='red')
    plt.title(f"{name} acc6={acc_6:.2f} acc10={acc_10:.2f} acc10p={acc_10p:.2f}, mean error(%)={mean_error:.2f}")
    plt.xlabel("True ")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{data_dir}/{name}_eval_{conf}.png")
    plt.clf()

def plot_kepler_df(df, save_dir):
    plt.figure(figsize=(12, 7))
    plt.subplot2grid((1, 3), (0, 0))
    plt.hist(df['Prot'], 20, color="C0")
    plt.xlabel("Rotation Period (days")
    plt.xlim(0, 60)
    plt.ylabel("N")
    # if "predicted period" in df.columns:
    plt.subplot2grid((1, 3), (0, 1))
    plt.hist(df['predicted period'], 20, color="C0")
    plt.xlabel("Predicted Period (days")
    plt.ylabel("N")
    plt.subplot2grid((1, 3), (0, 2))
    plt.hist(df['predicted inclination'], 20, color="C3")
    plt.xlabel("Predicted Inclination (deg)")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distributions_kepler.png", dpi=150)
    df.to_csv(f"{save_dir}/kepler_eval.csv")
    plt.clf()

def plot_df(df, save_dir, labels):
    num_cols = len(df.columns)
    plt.figure(figsize=(12, 7))   
    # if "predicted period" in df.columns:
    for i in range(len(labels)):
        row = 0 
        col = i 
        plt.subplot2grid((2, 4), (row, col))
        plt.hist(df[f'predicted {labels[i]}'], 20, color="C0")
        plt.xlabel(f"Predicted {labels[i]}")
        plt.ylabel("N")
        row = 1 
        plt.subplot2grid((2, 4), (row, col))
        plt.hist(df[f'{labels[i]}'], 20, color="C1")
        plt.xlabel(f"True {labels[i]}")
        plt.ylabel("N")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/distributions.png", dpi=150)
    df.to_csv(f"{save_dir}/eval.csv")
    plt.clf()

def calc_diff(target, output, conf, test_df=None, cls=False, labels=['Inclination', 'Period'], num_classes=2):
    if test_df is not None:
        df = test_df
    else:
        df = pd.DataFrame()
    for i in range(len(labels)):
        df[f"predicted {labels[i]}"] = output[:,i] if not cls else np.argmax(output[:,i*num_classes:(i+1)*num_classes], axis=1)
        df[labels[i]] = target[:,i]
        if not cls and len(conf):
            df[f'{labels[i]} confidence'] = conf[:,i]
    diff = np.abs(output[:,:len(labels)] - target[:,:len(labels)]) if not cls else None
    return df,diff


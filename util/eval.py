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
    
def eval_results(output, target, conf, quantiles, data_dir, model_name, boundaries, cls=False, labels=['Inclination', 'Period'], num_classes=2, only_one=False, test_df=None,
                scale_target=True, scale_output=True, kepler=False, cos_inc=False):
    """_
    evaluate results of a model and plot    
    """
    num_classes = len(labels) if not cls else num_classes
    if scale_target:
        print("scaling target with cos_inc: ", cos_inc)
        for i, label in enumerate(labels):
            max_val = boundaries[f'max {label}']
            min_val = boundaries[f'min {label}']
            if label == 'Inclination' and cos_inc:
                target[...,i] = np.arccos(target[...,i])
            else:
                target[...,i] = target[:,i] * (max_val - min_val) + min_val
        
    if scale_output:
        print("scaling output")
        for i, label in enumerate(labels):
            max_val = boundaries[f'max {label}']
            min_val = boundaries[f'min {label}']
            print(label, "before ", output[...,i].max(), output[...,i].min())
            if label == 'Inclination' and cos_inc:
                output[...,i] = np.arccos(np.clip(output[...,i], a_min=0, a_max=1))
            else:
                output[...,i] = output[...,i] * (max_val - min_val) + min_val
            print(label, "after ", output[...,i].max(), output[...,i].min())

    df, diff =calc_diff(target, output, conf, quantiles, test_df=test_df, cls=cls, labels=labels, num_classes=num_classes)
    df.to_csv(f"{data_dir}/eval.csv")
    if conf is not None:
        if not kepler:
            plot_df(df, data_dir, labels=labels)
        else:
            plot_kepler_df(df, data_dir)
        for i, label in enumerate(labels):            
            thresholds = [0] if not len(conf) else [0,0.8,0.83, 0.85,0.87,0.88,0.89,0.9,0.91, 0.92, 0.94,0.96,0.98]
            for thresh in thresholds:
                name = f'{model_name}-{label}'
                high_conf = np.where(1 - np.abs(conf[...,i]) > thresh)[0] if len(conf) else np.arange(len(target))
                if len(high_conf > 1):
                    plot_results(data_dir, name, target[...,i][high_conf], output[...,i][high_conf], conf=thresh)
    if quantiles is not None:
        for i, label in enumerate(labels):
            name = f'{model_name}-{label}'
            plot_quantile_predictions(data_dir, name, output[...,i], target[...,i], quantiles)

def plot_quantile_predictions(data_dir, name, quantile_preds, true_values, quantiles):
    """
    Plot true vs predicted results with confidence intervals.
    
    Args:
    quantile_preds (np.array): Predictions of shape (b, q) where:
        b is batch size, q is number of quantiles
    true_values (np.array): True values of shape (b, l)
    quantiles (list): List of quantiles used in the prediction
    """
    assert quantile_preds.shape[1] == len(quantiles), "Number of quantiles doesn't match predictions"
    assert quantile_preds.shape[0] == true_values.shape[0], "Batch sizes don't match"
    # assert quantile_preds.shape[2] == true_values.shape[1], "Number of predictions doesn't match"
    
    # Flatten the arrays
    true_flat = true_values.flatten()
    pred_flat = quantile_preds.reshape(-1, len(quantiles))
    
    # Sort by true values for a cleaner plot
    sort_idx = np.argsort(true_flat)
    true_sorted = true_flat[sort_idx]
    pred_sorted = pred_flat[sort_idx]
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    diff = np.abs(quantile_preds[:, median_idx] - true_values)

    acc_6 = (diff < 6).sum()/len(diff)
    acc_10 = (diff < 10).sum()/len(diff)
    acc_10p = (diff < true_values/10).sum()/len(diff)
    mean_error = np.mean(diff/(true_values+1)*100)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    
    # Plot confidence intervals
    for i in range(len(quantiles) // 2):
        lower = pred_sorted[:, i]
        upper = pred_sorted[:, -(i+1)]
        error = np.clip(np.concatenate([lower.reshape(1,-1), upper.reshape(1,-1)], axis=0), 0, None)
        plt.errorbar(true_values, quantile_preds[:, median_idx], alpha=0.1, yerr=error,
                         label=f'{quantiles[i]*100:.0f}%-{quantiles[-(i+1)]*100:.0f}% CI')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{data_dir}/{name}_eval_quantiles.png")
    plt.clf()
        
    # Plot true values
    plt.scatter(true_values, quantile_preds[:, median_idx])
    
    plt.plot(true_values, 0.9*true_values, color='red')
    plt.plot(true_values, 1.1*true_values, color='red')
    plt.title(f"{name} acc6={acc_6:.2f} acc10={acc_10:.2f} acc10p={acc_10p:.2f}, mean error(%)={mean_error:.2f}")
    plt.xlabel("True ")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{data_dir}/{name}_eval.png")
    plt.clf()

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

def calc_diff(target, output, conf, quantiles=[0.5], test_df=None, cls=False, labels=['Inclination', 'Period'], num_classes=2):
    if test_df is not None:
        df = test_df
    else:
        df = pd.DataFrame()
    for i in range(len(labels)):
        if quantiles is not None:
            for q in range(len(quantiles)):
                df[f"predicted {labels[i]} {quantiles[q]}"] = output[:,q,i]
                df[labels[i]] = target[:,i]            
        if conf is not None:
            df[f'{labels[i]} confidence'] = conf[:,i]
    if quantiles is not None:
        diff = np.abs(output[:,len(quantiles)//2, :len(labels)] - target[:,:len(labels)]) if not cls else None
    else:
        diff = np.abs(output[:, :len(labels)] - target[:,:len(labels)]) if not cls else None
    return df,diff


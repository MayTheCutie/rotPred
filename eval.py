from collections import OrderedDict
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import warnings

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from lightPred.dataloader import *
from lightPred.model import *
from lightPred.utils import *
from lightPred.train import *
import yaml
import glob
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore")


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

test_data_folder = "/data/butter/test2"

max_p, min_p = 60, 0
max_i, min_i = np.pi/2, 0
min_cycle, max_cycle = 1, 10
min_tau, max_tau = 1,10
min_lat, max_lat = 0, 80

boundary_values_dict = {'Period': (min_p, max_p), 'Inclination': (min_i, max_i),
 'Decay Time': (min_tau, max_tau), 'Cycle Length': (min_cycle, max_cycle), 'Spot Max': (min_lat, max_lat),}

# filtered_idx = filter_p( os.path.join(data_folder, "simulation_properties.csv"), max_p)


Nlc = 5000
idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

b_size = 256


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

def plot_eval(output, target, ten_perc, twenty_perc, xlabel, ylabel, title, data_dir):
    unique_values, counts = np.unique(output, return_counts=True)
    counts = np.pad(counts, (0,len(output) - len(counts)), 'constant')
    # counts = np.concatenate([counts, [0]], axis=0)
    cmap = plt.cm.get_cmap('viridis', len(unique_values))
    color = cmap(counts / np.max(counts))
    # color = np.append(color, [0])
    try:
        plt.scatter(target, output,cmap=cmap, c=color)
        plt.colorbar()
    except ValueError as e:
        print(e)
        plt.scatter(target, output)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} acc10={ten_perc}, acc20={twenty_perc}')

# Define the colormap based on the frequency
    plt.savefig(f'{data_dir}/{title}_eval.png')
    plt.clf()

def eval_acf(data_folder):
    # init_wandb(group='model selection', name=f'test-acf-p60')
    loss, target, output = evaluate_acf(data_folder, idx_list)
    print(output)
    df = pd.read_csv(os.path.join(data_folder, "simulation_properties.csv"))
    df['predicted period'] = output
    df.to_csv("/data/logs/acf/acf_eval.csv")
    print("df saved")
    plot_results("/data/logs/acf", "acf", target, output)
    

def eval_acf_kepler(kepler_path, kepler_df, name='acf_kepler'):
    # init_wandb(group='model selection', name=f'test-acf-p60')
    loss, target, output = evaluate_acf_kepler(kepler_path, kepler_df)
    kepler_df['predicted period'] = output
    kepler_df.to_csv("/data/logs/acf/acf_kepler_eval.csv")
    print("df saved")
    plot_results("/data/logs/acf", name, target, output, conf=0)

    # wandb.log({"acf+gwp acc10" : ten_perc_p, "acf+gwp acc20" : twenty_perc_p})
    # table = wandb.Table(dataframe=df)
    # wandb.log({"results table": table})
    # wandb.finish()

def eval_combine(data_dir, model, test_ds,
                trained_on_ddp=True, cls=False, norm='std', run_name='acf+',
                group='combine'):
    model, net_params, model_name = load_model(data_dir, model, trained_on_ddp)

    init_wandb(group, name=f'test-{run_name}{model_name}-p60')

    loss_fn = nn.MSELoss()
    test_dataset = test_ds(data_folder, idx_list, t_samples=net_params['t_samples'], norm=norm)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    loss, target, output, outliers = combine_evaluation_p(model, data_folder, idx_list, device='cpu')
    # target, output = target.cpu().detach().numpy(), output.cpu().detach().numpy()

    df = pd.read_csv(os.path.join(data_folder, "simulation_properties.csv"))

    df = df[df['Period'] <= max_p]
    df['predicted period'] = output
    out_arr = np.zeros(len(df))
    out_arr[np.array(outliers).astype(np.int64)] = 1
    # print("outliers shape:" , out_arr.shape, "outliers type", out_arr.dtype)
    # print("outliers:" , outliers)
    df["outliers"] = out_arr
    df.to_csv(f"/data/logs/acf+{model_name}.csv")
    # print("df saved")
    diff = np.abs(df['Period'] - df['predicted period'])
    target = df['Period'].values
    output = df['predicted period'].values

    ten_perc_p = (diff < (target/10)).sum()/len(diff)

    twenty_perc_p = (diff < (target/5)).sum()/len(diff)
    plt.scatter(target, output)
    plt.plot(target, 0.9*target, color='red')
    plt.plot(target, 1.1*target, color='red')
    plt.xlabel("true period")
    plt.ylabel("predicted period")
    plt.title(f'acf acc10={ten_perc_p:.2f}, acc20={twenty_perc_p:.2f}')
    plt.ylim(0, 60)
    plt.savefig(f"/data/logs/acf/acf+{model_name}.png")

    table = wandb.Table(dataframe=df)
    wandb.log({"results table": table})
    wandb.finish()

# Load the modified state dict into your model
    


def kepler_inference(dl, model, device, conf=False):
    model.eval()
    output = []
    confs = []
    with torch.no_grad():
        for x, y in tqdm(dl):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            output.append(out)
            if conf:
                confs.append(model.confidence(x))
    output = torch.cat(output, dim=0)
    if conf:
        confs = torch.cat(confs, dim=0)
    return target, output, confs   


def eval_model(data_dir, model, test_dl, data_folder=None, scale_target=True, scale_output=True, load=True,
                distribute=True, cls=False, only_one=False, num_classes=2, conf=None, run_name='regression',
                  group='model selection', kepler=False, kepler_df=None):
    # init_wandb(group, name=f'test-{run_name}-work')
    if load:
        model, net_params, model_name = load_model(data_dir, model, distribute=distribute, device=DEVICE)
    else:
        model_name = model.__class__.__name__
        

    loss_fn = nn.CrossEntropyLoss() if cls else nn.MSELoss() 
    # test_dataset = test_ds(data_folder, idx_list, t_samples=net_params['t_samples'], norm=norm)
    # test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    # print("dataset length: ", len(test_dl.dataset))
    if not kepler:
        loss, _, target, output, conf = evaluate_model(model, test_dl, loss_fn, DEVICE, cls=cls, only_one=only_one, num_classes=num_classes, conf=conf)
    else:
        loss, _, target, output, conf = evaluate_kepler(model, test_dl, loss_fn, DEVICE, cls=cls, only_one=only_one, num_classes=num_classes, conf=conf)
    # print('conf', conf)
    print("loss on test set : ", loss)
    target, output = target.cpu().detach().numpy(), output.cpu().detach().numpy()
    conf = conf.cpu().detach().numpy()
    # print("target shape: ", target.shape, "output shape: ", output.shape)
    test_folder = test_data_folder if data_folder is None else data_folder

    df, diff =calc_diff(target, output, test_folder, conf=conf, kepler_df=kepler_df, cls=cls, num_classes=num_classes)


    if not cls:
        if scale_target:
            print("scaling target")
            # target[:,2] = target[:,2] * (max_tau - min_tau) + min_tau
            target[:,1] = target[:,1] * (max_p - min_p) + min_p
            target[:,0] = (target[:,0] * (max_i - min_i) + min_i)*180/np.pi

            
        if scale_output:
            print("scaling output")
            # output[:,2] = output[:,2] * (max_tau - min_tau) + min_tau
            output[:,1] = output[:,1] * (max_p - min_p)+ min_p
            output[:,0] = (output[:,0] * (max_i - min_i)+ min_i)*180/np.pi

        plot_diffs(data_dir, model_name, target, output, diff, conf)

        if not kepler:
            # print("plotting results:")
            thresholds = [0] if not len(conf) else [0,0.8,0.83, 0.85,0.87,0.88,0.89,0.9,0.91, 0.92]
            for thresh in thresholds:
                print('inc-thresh', thresh)
                high_conf = np.where(1 - conf[:,0] > thresh)[0] if len(conf) else np.arange(len(target))
                name = f'{model_name}-Inclination'
                if len(high_conf > 1):
                    plot_results(data_dir, name, target[:,0][high_conf], output[:,0][high_conf], conf=thresh)
            thresholds = [0] if not len(conf) else [0,0.8,0.85,0.92,0.94, 0.96, 0.98]
            for thresh in thresholds:
                print('period-thresh', thresh)
                high_conf = np.where(1 - conf[:,1] > thresh)[0] if len(conf) else np.arange(len(target))
                name = f'{model_name}-Period'
                if len(high_conf > 1):
                    plot_results(data_dir, name, target[:,1][high_conf], output[:,1][high_conf], conf=thresh)
            # for thresh in thresholds:
            #     print('tau-thresh', thresh)
            #     high_conf = np.where(1 - conf[:,2] > thresh)[0] if len(conf) else np.arange(len(target))
            #     name = f'{model_name}-tau'
            #     if len(high_conf > 1):
            #         plot_results(data_dir, name, target[:,2][high_conf], output[:,2][high_conf], conf=thresh)
            
    else:
        out, tar = output[:,:num_classes//2], target[:,:num_classes//2]
        out_, tar_, correct, correct_thresh, len_thresh = eval_predictions_thresholded(torch.tensor(out), torch.tensor(tar))
        plot_thresholded_acc(correct_thresh, len_thresh, len(test_dl.dataset), data_dir, model_name, "Inclination")
        plt.close('all')
        out, tar = out.argmax(axis=1), tar.argmax(axis=1)
        num_correct = (out == tar).sum().item() 
        test_acc = num_correct / len(target)
        plot_results_cls(data_dir, model_name, tar, out, test_acc, "Inclinaiton")
        plt.close('all')
        plot_confusion_mat(tar, out, data_dir, model_name, "Inclination", data_dir)
        plt.close('all')


        if not only_one:
            out, tar = output[:,num_classes//2:], target[:,num_classes//2:]
            out_, tar_, correct, correct_thresh, len_thresh = eval_predictions_thresholded(torch.tensor(out), torch.tensor(tar))
            plot_thresholded_acc(correct_thresh, len_thresh, len(test_dl.dataset), data_dir, model_name, "Period")
            out, tar = out.argmax(axis=1), tar.argmax(axis=1)
            num_correct = (out == tar).sum().item() 
            test_acc = num_correct / len(target)
            plot_results_cls(data_dir, model_name, tar, out, test_acc, "Period")
            plot_confusion_mat(tar, out, data_dir, model_name, "Period", data_dir)

    if not kepler:
            plot_df(df, data_dir)
    else:
        plot_kepler_df(df, data_dir)


def eval_results(output, target, conf, data_dir, model_name, cls=False, labels=['Inclination', 'Period'], num_classes=2, only_one=False, test_df=None,
                scale_target=True, scale_output=True, kepler=False, cos_inc=False):
    num_classes = len(labels) if not cls else num_classes
    if not cls:
        if scale_target:
            print("scaling target with cos_inc: ", cos_inc)
            for i in range(len(labels)):
                if labels[i] == 'Inclination' and cos_inc:
                    target[:,i] = np.pi/2 - np.arccos(target[:,i])
                else:
                    target[:,i] = target[:,i] * (boundary_values_dict[labels[i]][1] - boundary_values_dict[labels[i]][0]) + boundary_values_dict[labels[i]][0]
            
        if scale_output:
            print("scaling output")
            for i in range(len(labels)):
                print(labels[i], "before ", output[:,i].max(), output[:,i].min())
                if labels[i] == 'Inclination' and cos_inc:
                    output[:,i] = np.pi/2 - np.arccos(output[:,i])
                else:
                    output[:,i] = output[:,i] * (boundary_values_dict[labels[i]][1] - boundary_values_dict[labels[i]][0]) + boundary_values_dict[labels[i]][0]
                print(labels[i], "after ", output[:,i].max(), output[:,i].min())

        df, diff =calc_diff(target, output, conf, test_df=test_df, cls=cls, labels=labels, num_classes=num_classes)

        # plot_diffs(data_dir, model_name, target, output, diff, conf)

        # print("plotting results:")
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
            
    else:
        out, tar = output[:,:num_classes//2], target[:,:num_classes//2]
        out_, tar_, correct, correct_thresh, len_thresh = eval_predictions_thresholded(torch.tensor(out), torch.tensor(tar))
        plot_thresholded_acc(correct_thresh, len_thresh, len(tar), data_dir, model_name, "Inclination")
        plt.close('all')
        out, tar = out.argmax(axis=1), tar.argmax(axis=1)
        num_correct = (out == tar).sum().item() 
        test_acc = num_correct / len(tar)
        plot_results_cls(data_dir, model_name, tar, out, test_acc, "Inclinaiton")
        plt.close('all')
        plot_confusion_mat(tar, out, data_dir, model_name, "Inclination", data_dir)
        plt.close('all')

        if not only_one:
            out, tar = output[:,num_classes//2:], target[:,num_classes//2:]
            out_, tar_, correct, correct_thresh, len_thresh = eval_predictions_thresholded(torch.tensor(out), torch.tensor(tar))
            plot_thresholded_acc(correct_thresh, len_thresh, len(tar), data_dir, model_name, "Period")
            out, tar = out.argmax(axis=1), tar.argmax(axis=1)
            num_correct = (out == tar).sum().item() 
            test_acc = num_correct / len(tar)
            plot_results_cls(data_dir, model_name, tar, out, test_acc, "Period")
            plot_confusion_mat(tar, out, data_dir, model_name, "Period", data_dir)


def eval_lat_results(output, target, data_dir, model_name):
    # out_, tar_, correct, correct_thresh, len_thresh = eval_predictions_thresholded(torch.tensor(output), torch.tensor(target))
    # plot_thresholded_acc(correct_thresh, len_thresh, len(tar), data_dir, model_name, "Inclination")
    # plt.close('all')
    out, tar = np.round(output), target
    num_correct = (out == tar).sum().item() 
    test_acc = num_correct / len(tar)
    print("test acc: ", test_acc)
    # plot_results_cls(data_dir, model_name, tar, out, test_acc, "Inclinaiton")
    # plt.close('all')
    plot_confusion_mat(tar, out, data_dir, model_name, "Inclination", data_dir)
    plt.close('all')
    

def eval_quantiled_results(output, target, qs, data_dir, model_name, cls=False, num_classes=2, test_df=None,
                scale_target=True, scale_output=True, kepler=False):

    if scale_target:
        print("scaling target")
        # target[:,2] = target[:,2] * (max_tau - min_tau) + min_tau
        target[:,1] = target[:,1] * (max_p - min_p) + min_p
        target[:,0] = (target[:,0] * (max_i - min_i) + min_i)*180/np.pi
        print('after', target[:,1], target[:,0])

        
    if scale_output:
        print("scaling output")
        # output[:,2] = output[:,2] * (max_tau - min_tau) + min_tau
        print('before', output[:,1], output[:,0])
        output[:,1] = output[:,1] * (max_p - min_p)+ min_p
        output[:,0] = (output[:,0] * (max_i - min_i)+ min_i)*180/np.pi
        print('after', output[:,1], output[:,0])

    idx = qs.index(0.5)
    print("idx", idx)
    df, diff =calc_diff(target, output[:,:,len(qs)//2], conf=torch.zeros((0)), test_df=test_df, cls=cls, num_classes=num_classes)

    plot_diffs(data_dir, model_name, target, output, diff, conf=torch.zeros((0)))

    # print("plotting results:")
    for i in range(len(qs)):
        plot_results(data_dir, f'{model_name}-Inclination', target[:,0], output[:,0,i], conf=qs[i])
        plot_results(data_dir, f'{model_name}-Period', target[:,1], output[:,1,i], conf=qs[i])

    if not kepler:
            plot_df(df, data_dir)
    else:
        plot_kepler_df(df, data_dir)
    
def plot_diffs(data_dir, model_name, target, output, diff, conf):
    fig, subplot = plt.subplots(2,3)
    bins = np.linspace(0, 90, 7)
    for i in range(1,7):
        idx = np.where((target >= bins[i-1]) & (target < bins[i]))
        subplot[(i-1)//3, (i-1)%3].scatter(diff[idx,0], diff[idx,1])
        subplot[(i-1)//3, (i-1)%3].set_title(f"{int(bins[i-1])}-{int(bins[i])} deg")
        subplot[(i-1)//3, (i-1)%3].set_xlabel("inclination difference")
        subplot[(i-1)//3, (i-1)%3].set_ylabel("period difference")
    fig.tight_layout()
    plt.savefig(f"{data_dir}/diffs.png")
    plt.clf()
    plt.scatter(target[:,0], diff[:,1])
    plt.xlabel("Inclination")
    plt.ylabel("Period difference")
    plt.title(f"{model_name} difference")
    plt.savefig(f"{data_dir}/diff_period.png")
    plt.clf()
    plt.scatter(target[:,1], diff[:,0])
    plt.xlabel("Period")
    plt.ylabel("Inclination difference")
    plt.title(f"{model_name} difference")
    plt.savefig(f"{data_dir}/diff_inclination.png")
    plt.clf()
    if len(conf):
        # print(diff[:,0].shape, conf[:,0].shape)
        plt.scatter(1-conf[:,0], diff[:,0])
        plt.xlabel("Inclination confidence")
        plt.ylabel("Inclination difference")
        plt.title(f"{model_name} difference")
        plt.savefig(f"{data_dir}/conf_inclination.png")
        plt.clf()
    
        plt.scatter(1-conf[:,1], diff[:,1])
        plt.xlabel("Period confidence")
        plt.ylabel("Period difference")
        plt.title(f"{model_name} difference")
        plt.savefig(f"{data_dir}/conf_period.png")
        plt.clf()
       

def plot_thresholded_acc(correct_thresh, len_thresh, dataset_len, data_dir, model_name, name):
    threshs = np.linspace(0, 1, 20)
    
    acc_thresh = correct_thresh/len_thresh
    plt.scatter(threshs, acc_thresh, label='acc_thresh')
    plt.scatter(threshs, len_thresh/dataset_len, label='dataset_ratio')
    plt.plot(threshs, (acc_thresh*len_thresh)/dataset_len, '--', label='acc_thresh*dataset_ratio')
    plt.title(f"{model_name} {name} threshold test")
    plt.xlabel("threshold value")
    plt.ylabel("accuracy(dataset_ratio)")
    plt.legend()
    plt.savefig(f"{data_dir}/{name}_cls_thresh.png")
    plt.clf()


def plot_results(data_dir, name, target, output, conf=''):
    
    # df, diff = calc_diff(target, output)
    diff = np.abs(output - target)

    acc_6 = (diff < 6).sum()/len(diff)
    acc_10 = (diff < 10).sum()/len(diff)
    acc_10p = (diff < target/10).sum()/len(diff)
    mean_error = np.mean(diff/(target+1)*100)

    print("num correct: ", (diff < target/10).sum())

    plt.scatter(target, output, label=f'confidence > {conf} ({len(target)} points)')
    plt.plot(target, 0.9*target, color='red')
    plt.plot(target, 1.1*target, color='red')
    # if max(target) < 100:
    #     plt.xlim(0, max(target) + 5)
    #     plt.ylim(0, max(target) + 5)
    # else:
    #     plt.xlim(0, 60)
    #     plt.ylim(0, 60)
    plt.title(f"{name} acc6={acc_6:.2f} acc10={acc_10:.2f} acc10p={acc_10p:.2f}, mean error(%)={mean_error:.2f}")
    plt.xlabel("True ")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{data_dir}/{name}_eval_{conf}.png")
    plt.clf()


def plot_results_cls(data_dir, model_name, pred, target, test_acc, name):
    coords, counts = count_occurence(target, pred)
    plt.scatter(coords[:, 0], coords[:, 1], c=counts, cmap='viridis')
    plt.title(f"{model_name} {name}  acc={test_acc:.2f}")
    plt.xlabel(f"true {name}")
    plt.ylabel(f"predicted {name}")
    plt.colorbar(label='points frequency')
    plt.savefig(f"{data_dir}/{name}_cls.png")
    plt.close()

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

    # plt.subplot2grid((2, 4), (0, 1))
    # plt.hist(df['predicted period'], 20, color="C0")
    # plt.xlabel("Predicted Period (days")
    # plt.ylabel("N")
    # plt.subplot2grid((2, 4), (1, 0))
    # plt.hist(df['Inclination'], 20, color="C3")
    # plt.xlabel("Stellar inclincation (deg)")
    # plt.ylabel("N")
    # if "predicted inclination" in df.columns:
    # plt.subplot2grid((2, 4), (1, 1))
    # plt.hist(df['predicted inclination'], 20, color="C3")
    # plt.xlabel("Predicted Inclination (deg)")
    # plt.ylabel("N")

    # if num_cols > 6:
    #     plt.subplot2grid((2, 4), (0, 2))
    #     plt.hist(df['Decay Time'], 20, color="C1")
    #     plt.xlabel("Spot lifetime (Prot)")
    #     plt.ylabel("N")
    #     plt.subplot2grid((2, 4), (1, 2))
    #     plt.hist(df['Activity Rate'], 20, color="C4")
    #     plt.xlabel("Stellar activity rate (x Solar)")
    #     plt.ylabel("N")
    #     plt.subplot2grid((2, 4), (0, 3))
    #     plt.hist(df['Shear'], 20, color="C5")
    #     plt.xlabel(r"Differential Rotation Shear $\Delta \Omega / \Omega$")
    #     plt.ylabel("N")
    #     plt.subplot2grid((2, 4), (1, 3))
    #     plt.hist(df['Spot Max'] - df['Spot Min'], 20, color="C6")
    #     plt.xlabel("Spot latitude range")
    #     plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distributions.png", dpi=150)
    df.to_csv(f"{save_dir}/eval.csv")
    plt.clf()

def eval_predictions_thresholded(predictions, targets):
    print("predictions shape: ", predictions.shape, "targets shape: ", targets.shape)
    correct_thresh = np.zeros(20)
    len_thresh = np.zeros(20)
    pred = []
    target = []
    threshs =  np.linspace(0, 1, 20)
    for i, t in enumerate(threshs):
        pred_soft = predictions.softmax(dim=1)
        topk = pred_soft.topk(2, dim=1)
        thresh = topk.values[:, 0] - topk.values[:, 1]
        pred_t = predictions[thresh > t]
        y_t = targets[thresh > t]
        correct_thresh[i] = (pred_t.argmax(dim=1) == y_t.argmax(dim=1)).sum().item()
        len_thresh[i] = len(pred_t)
    num_correct = (predictions.argmax(dim=1) == targets.argmax(dim=1)).sum().item() 
    pred = predictions.argmax(dim=1).cpu().detach().tolist()
    target = targets.argmax(dim=1).cpu().detach().tolist()         
    return pred, target, num_correct, correct_thresh, len_thresh
        
def plot_confusion_mat(y, y_pred, data_dir, model_name, name, save_dir):
    print(y[:10], y_pred[:10])
    cm = confusion_matrix(y, y_pred, normalize='true')
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(f"{model_name} - {name} confusion matrix")
    plt.colorbar()
    plt.savefig(f"{save_dir}/{name}_confusion.png")
    plt.clf()

def calc_diff_and_log(model_name, target, output, test_folder,conf, kepler_df, cls):
    df, diff = calc_diff(target, output, test_folder,conf=conf, kepler_df=kepler_df, cls=cls)
    if not cls:
        ten_perc_p = (diff[:,0] < (target[:,0]/10)).sum()/len(diff)
        ten_perc_i = (diff[:,1] < (target[:,1]/10)).sum()/len(diff)

        twenty_perc_p = (diff[:,0] < (target[:,0]/5)).sum()/len(diff)
        twenty_perc_i = (diff[:,1] < (target[:,1]/5)).sum()/len(diff)

    # wandb.log({f"{model_name} acc10_p" : ten_perc_p, f"{model_name} acc20_p" : twenty_perc_p})
    # wandb.log({f"{model_name} acc10_i" : ten_perc_i, f"{model_name} acc20_i" : twenty_perc_i})

    # data = []
    # for i in range(len(diff)):
    #         wandb.log({f"{model_name} Period" : output[i, 0], f"{model_name} Inclination" : output[i, 1],
    #          "True Period": target[i, 0], "True Inclination": target[i,1]})
    #         data.append([output[i, 0], output[i, 1], target[i, 0], target[i,1]])
    # table = wandb.Table(dataframe=df)
    # wandb.log({"results table": table})
    # wandb.finish()
    return df, diff

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

    # out1 = output[:,0] if not cls else np.argmax(output[:,:num_classes//2], axis=1)
    # out2 = output[:,1] if not cls else np.argmax(output[:,num_classes//2:], axis=1)
    # tar1 = target[:,0] if not cls else np.argmax(target[:,:num_classes//2], axis=1)
    # tar2 = target[:,1] if not cls else np.argmax(target[:,num_classes//2:], axis=1)
    # df['predicted inclination'] = out1
    # df['predicted period'] = out2
    # df['Inclination'] = tar1
    # df['Period'] = tar2
    # if not cls:
    #     if len(conf):
    #         df['inclination confidence'] = conf[:, 0]
    #         df['period confidence'] = conf[:, 1]
        # if output[:,0].max() <= 1:
        #     output[:,0] = np.sin(output[:,0]) * 180/np.pi
        #     target[:,0] = np.sin(target[:,0]) * 180/np.pi
    #     diff = np.abs(output[:,:2] - target[:,:2])
    # else:
    #     diff = None
    # return df,diff
    # table = wandb.Table(data=data, columns = ["Predicted Period (days)", "Predicted Inclination (deg)", "True Period (days)", "True Inclination (deg)"])
    # wandb.log({f"{model_name}-Period" : wandb.plot.scatter(table, "True Period (days)", "Predicted Period (days)",
    #                              title=f"{model_name}-Period acc10: {ten_perc_p}, acc20: {twenty_perc_p}")})
    # wandb.log({f"{model_name}-Inclination" : wandb.plot.scatter(table, "True Inclination (deg)",
    #                                                              "Predicted Inclination (deg)",
    #                                                                  title=f"{model_name}-Inclination acc10: {ten_perc_i}, acc20: {twenty_perc_i}")})
    


    # plot_eval(output[:,0], target[:,0], ten_perc_p, twenty_perc_p, 'period (days)', 'prediction (days)', 'Period', data_dir)
    # plot_eval(output[:,1], target[:,1], ten_perc_i, twenty_perc_i, 'inclination (deg)', 'prediction (deg)', 'Inclination', data_dir)

def acf_on_train():
    train_data_folder = "/data/butter/data"
    loss, target, output = evaluate_acf(train_data_folder, idx_list)
    df = pd.read_csv(os.path.join(train_data_folder, "simulation_properties.csv"))
    df = df[df['Period'] <= max_p]
    df['predicted period'] = output
    df.to_csv("/data/logs/acf.csv")
    print("finished acf on train")


if __name__ == '__main__': 
    # eval_model('/data/logs/bert/exp5',model=BertRegressor, test_ds=TimeSeriesDataset, norm='minmax')
    # eval_model('/data/logs/freqcnn/exp18',model=CNN_B, test_ds=TimeSeriesDataset)
    # eval_model('/data/logs/freqcnn/exp15',model=CNN_B, test_ds=TimeSeriesDataset)
    # eval_model('/data/logs/lstm/exp12',model=LSTM, test_ds=ACFDataset)
    # eval_model('/data/logs/lstm_attn/exp2',model=LSTM_ATTN, test_ds=ACFDataset)
    eval_acf(test_data_folder)
    # test_dataset = TimeSeriesDataset(data_folder, idx_list, t_samples=1024, num_classes=4, noise=True , noise_factor=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=2048, \
    #                               num_workers=8)


    # eval_model('/data/logs/lstm_attn/exp15',model=LSTM_ATTN, test_dl=test_dataloader, num_classes=2, conf=True) 

    # eval_model('/data/logs/lstm/exp11',model=LSTM, test_ds=ACFDataset)
    # eval_model('/data/logs/lstm/exp12',model=LSTM, test_ds=ACFDataset)


    # eval_combine('/data/logs/freqcnn/exp20',model=CNN_B, test_ds=TimeSeriesDataset)
    # acf_on_train()
    # eval_model_cls('/data/logs/lstm/exp14',model=LSTM, test_ds=ACFClassifierDataset)
    # data_folder = "/data/lightPred/data"
    # table_path  = "/data/lightPred/kois.csv"
    # kepler_df = create_kepler_df(data_folder, table_path)
    # eval_acf_kepler(data_folder, kepler_df, name='acf_kepler_kois')

    


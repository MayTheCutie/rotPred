import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import yaml
import warnings
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.distributed as dist
import optuna
from sklearn.metrics import mean_absolute_percentage_error

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)   

from util.utils import *
from transforms import *
from util.classical_analysis import analyze_lc
from dataset.dataloader import *

torch.manual_seed(1234)
np.random.seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 19

log_path = '/data/logs/acf'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

data_folder = "/data/butter/data_aigrain2"
dataset_name = data_folder.split('/')[-1]
kepler_data_folder = "/data/lightPred/data"

Nlc = 50000

test_Nlc = 5000

idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
train_list, test_list = train_test_split(idx_list, test_size=0.1, random_state=1234)


CUDA_LAUNCH_BLOCKING='1'


cad = 30

DAY2MIN = 24*60


freq_rate = 1/48

if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)

num_qs = 8
dur = num_qs*90

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def kepler_sanity_check():
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])

    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)

    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    print(f"rank: {rank}, local_rank: {local_rank}")

    print("logdir ", f'{log_path}')
    
    num_qs = 12
    dur = 90*num_qs
    max_lag_day = 70
    kepler_df = get_all_samples_df(num_qs)
    filtered_df = pd.read_csv('/data/lightPred/tables/Table_1_Periodic.txt')
    kepler_df = kepler_df.merge(filtered_df, on='KID')
    print(f"all samples with at least {num_qs} consecutive qs:  {len(kepler_df)}")

    transform = Compose([Slice(0 , int(dur / cad * DAY2MIN) ),
                             Normalize('med'),
                             MovingAvg(kernel_size=5),
                             ACF(prom=0.2, max_lag=max_lag_day, calc_p=True, only_acf=True, distance=10,),
                              ToTensor()])
    full_dataset = KeplerDataset(kepler_data_folder, path_list=None, df=kepler_df, t_samples=int(max_lag_day / cad * DAY2MIN),
                                    skip_idx=0, num_qs=num_qs, transforms=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)
    full_dl = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                            collate_fn=kepler_collate_fn, pin_memory=True, sampler=sampler)
    pbar = tqdm(full_dl, total=len(full_dl))
    all_info = np.zeros((0, 6))
    for i, (x, y, _, _, info, _) in enumerate(pbar):
            batch_info = np.array([[d['KID'],d['acf_p'], d['period'], d['Teff'],d['R'],d['logg']] for d in info])
            diff = batch_info[:,1].squeeze() - batch_info[:,2].squeeze()
            # print(diff.shape, batch_info.shape, batch_info[:,1].shape, batch_info[:,2].shape)
            # print(batch_info[:,1])
            # print(batch_info[:,2])
            print(diff.min(), diff.max(), diff.mean())
            all_info = np.concatenate((all_info, batch_info), axis=0)
            # if i == 10:
            #     break
    df = pd.DataFrame(all_info, columns=['KID', 'acf_p', 'period', 'Teff', 'R', 'logg'])
    df.to_csv(f'{log_path}/exp{exp_num}/acf_kepler_sanity_check.csv')
    acc = np.sum(np.abs(all_info[:,1] - all_info[:,2]) < 0.1*all_info[:,2])/len(all_info)
    print("acc10p: ", acc)
    plt.scatter(df['period'], df['acf_p'])
    plt.xlabel('McQ14 period')
    plt.ylabel('Predicted period')
    plt.title("acc10p: {:.2f}".format(acc))
    plt.savefig(f'{log_path}/exp{exp_num}/acf_kepler_sanity_check.png')
    plt.close('all')


def kepler_prediction_gpu(max_lag_day=50, prom=0.07093, distance=20):
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])

    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)
    
    # save params to ymal
    with open(f'{log_path}/exp{exp_num}/params.yaml', 'w') as f:
        yaml.dump({'max_lag_day': max_lag_day, 'prom': prom, 'distance': distance}, f)

    setup(rank, world_size)

    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    print(f"rank: {rank}, local_rank: {local_rank}")

    print("logdir ", f'{log_path}')
    
    num_qs = 8
    dur = 90*num_qs
    max_lag_day = max_lag_day
    kepler_df = get_all_samples_df(num_qs=num_qs)
    # filtered_df = pd.read_csv('/data/lightPred/tables/astroconf_exp45_ssl_no_thresh.csv')
    # kepler_df = kepler_df[kepler_df['KID'].isin(filtered_df['KID'])]
    print(f"all samples with at least {num_qs} qs:  {len(kepler_df)}")
    for q in range(15 - num_qs):
        pred_ps = []
        teff = []
        kids = []
        radius = []
        logg = []
        qs = []
        step = int(q * int(90 / cad * DAY2MIN))
        transform = Compose([Slice(0 + step, int(dur / cad * DAY2MIN) + step),
                            #  Normalize('med'),
                            #  MovingAvg(kernel_size=5),
                             ACF(prom=prom, max_lag=max_lag_day, calc_p=True, only_acf=True, distance=distance,),
                              ToTensor()])
        full_dataset = KeplerDataset(kepler_data_folder, path_list=None, df=kepler_df, t_samples=int(max_lag_day / cad * DAY2MIN),
                                        skip_idx=q, num_qs=num_qs, transforms=transform)
        sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)
        full_dl = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
                             collate_fn=kepler_collate_fn, pin_memory=True, sampler=sampler)
        all_info = np.zeros((0, 6))
        pbar = tqdm(full_dl, total=len(full_dl))
        for i, (x, y, _, _, info, _) in enumerate(pbar):
            batch_info = np.array([[d['KID'],d['predicted acf_p'], d['Teff'],d['R'],d['logg'], d['double_peaked']] for d in info])
            all_info = np.concatenate((all_info, batch_info), axis=0)
        df = pd.DataFrame(all_info, columns=['KID', 'predicted acf_p', 'Teff', 'R', 'logg', 'double_peaked'])
        df.to_csv(f'{log_path}/exp{exp_num}/acf_kepler_q_{q}_rank_{rank}.csv')



def objective(trial):
    # Define the hyperparameters to optimize
    max_lag_day = trial.suggest_float('max_lag_day', 50, 500, step=50)
    prom = trial.suggest_float('prom', 0.001, 0.1, log=True)
    distance = trial.suggest_int('distance', 5, 20)

    # Load ground truth data
    ground_truth = pd.read_csv('/data/lightPred/tables/Table_1_Periodic.txt')

    # Load Kepler data
    kepler_df = get_all_samples_df(num_qs=None)
    
    # Merge dataframes and select a subset
    merged_df = pd.merge(kepler_df, ground_truth[['KID', 'Prot']], on='KID', how='inner')
    subset_df = merged_df.sample(n=5000, random_state=42)

    transform = Compose([
        Slice(0, int(dur / cad * DAY2MIN)),
        # Normalize('med'),
        # MovingAvg(kernel_size=5),
        ACF(prom=prom, max_lag=max_lag_day, calc_p=True, only_acf=True, distance=distance),
        ToTensor()
    ])

    dataset = KeplerDataset(kepler_data_folder, path_list=None, df=subset_df, 
                            t_samples=int(max_lag_day / cad * DAY2MIN),
                            skip_idx=0, num_qs=num_qs, transforms=transform)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                            collate_fn=kepler_collate_fn, pin_memory=True)

    predicted_periods = []
    true_periods = []

    for x, y, _, _, info, _ in tqdm(dataloader):
        predicted_periods.extend([d['predicted acf_p'] for d in info])
        true_periods.extend([d['period'] for d in info])

    # Calculate mean absolute percentage error
    mape = mean_absolute_percentage_error(true_periods, predicted_periods)
    acc = np.sum(np.abs(np.array(true_periods) - np.array(predicted_periods)) < 0.1*np.array(true_periods))/len(true_periods)
    print(f"MAPE: {mape}, acc10p: {acc}")

    return mape

def optimize_parameters():
    # Get the Slurm job ID and array task ID
    job_id = os.environ.get('SLURM_JOB_ID', '0')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    
    study_name = f"acf_parameter_optimization_job_{job_id}_task_{task_id}"
    
    # Use in-memory storage
    study = optuna.create_study(
        study_name=study_name,
        storage=None,  # This uses in-memory storage
        direction='minimize'
    )
    study.optimize(objective, n_trials=100)  # Adjust n_trials as needed


    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Update the global variables with the best parameters
    global max_lag_day, prom, distance
    max_lag_day = trial.params['max_lag_day']
    prom = trial.params['prom']
    distance = trial.params['distance']

def kepler_prediction():
    kepler_df = get_all_samples_df(num_qs)
    ps = []
    std_ps = []
    kids = []
    qs = []
    pbar = tqdm(kepler_df.iterrows(), total=len(kepler_df))
    for i, row in pbar:
        q_sequence_idx = row['longest_consecutive_qs_indices']
        if q_sequence_idx is np.nan:
              print("skip")
              continue
        try:
            if q_sequence_idx[1] > q_sequence_idx[0]:
                for j in range(q_sequence_idx[0], q_sequence_idx[1]):
                    x,time,meta = read_fits(row['data_file_path'][j])
                    x /= x.median()
                    x = fill_nan_np(np.array(x), interpolate=True)
                    if j == q_sequence_idx[0]:
                        x_tot = x.copy()
                    else:
                        x_tot = np.concatenate((x_tot, x))
                effective_qs = row['qs'][q_sequence_idx[0]: q_sequence_idx[1]]
            else:
                continue
        except (TypeError, ValueError, FileNotFoundError) as e:
            print("ERROR: ", e)
            continue
        if len(x_tot) < int(dur/freq_rate):
            x_tot= np.pad(x_tot, (0, int(dur/freq_rate) - len(x_tot)))
        sample_len_qs = int(len(x_tot)*freq_rate/90)
        sample_preds = []
        for q in range(sample_len_qs - num_qs):
            x_sample = x_tot[int(q*90/freq_rate):int((q+num_qs)*90/freq_rate)]
            pred_p, lags, xcf, peaks = analyze_lc(x_sample, prom=0.005)
            sample_preds.append(pred_p)
        std_ps.append(np.std(sample_preds))
        kids.append(row['KID'])
        qs.append(row['qs'])
        ps.append(sample_preds)
    df_full = pd.DataFrame({'KID': kids, 'predicted period': ps, 'std period': std_ps, 'qs': qs})
    df_full.to_csv(f'{log_path}/exp{exp_num}/acf_results_{dataset_name}.csv')
    print(df_full)

def split_dataframe(df, n_chunks):
    return np.array_split(df, n_chunks)

def process_chunk(kepler_df_chunk, chunk_idx):
    results = []

    for q in range(15 - num_qs):
        pred_ps = []
        teff = []
        kids = []
        radius = []
        logg = []
        qs = []

        step = int(q * int(90 / cad * DAY2MIN))
        transform = Compose([Slice(0 + step, int(dur / cad * DAY2MIN) + step), ToTensor()])
        full_dataset = KeplerDataset(kepler_data_folder, path_list=None, df=kepler_df_chunk, t_samples=int(dur / cad * DAY2MIN),
                                     skip_idx=q, num_qs=num_qs, transforms=transform)
        for i, (x, y, _, _, info, _) in enumerate(full_dataset):
            if i % 1000 == 0:
                print(chunk_idx, i, flush=True)
            x = x.numpy().squeeze()
            pred_p, lags, xcf, peaks, lph = analyze_lc(x, prom=0.005, max_period=50)
            pred_ps.append(pred_p)
            teff.append(info['Teff'])
            kids.append(info['KID'])
            radius.append(info['R'])
            logg.append(info['logg'])
            qs.append(info['qs'])
        
        df = pd.DataFrame({'KID': kids, 'Teff': teff, 'R': radius, 'logg': logg, 'qs': qs, 'predicted period': pred_ps})
        df['start_idx'] = q
        df['duration(days)'] = dur
        output_file = f'{log_path}/exp{exp_num}/acf_results_kepler_chunk_{chunk_idx}_q_{q}.csv'
        df.to_csv(output_file)
        results.append(output_file)
    
    return results

def kepler_sequence_prediction():
    kepler_df = get_all_samples_df(num_qs)
    filtered_df = pd.read_csv('/data/lightPred/tables/astroconf_exp45_ssl_no_thresh.csv')
    kepler_df = kepler_df[kepler_df['KID'].isin(filtered_df['KID'])]
    print(f"all samples with at least {num_qs} consecutive qs:  {len(kepler_df)}")

    # Define the number of chunks you want to split the dataframe into
    n_chunks = 4  # or any other number based on your system's capabilities and size of the dataframe
    kepler_df_chunks = split_dataframe(kepler_df, n_chunks)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk, idx) for idx, chunk in enumerate(kepler_df_chunks)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                print(f"Completed processing for chunk results: {result}")
            except Exception as e:
                print(f"Error processing chunk: {e}")

def simulation_prediction(prom=0.005, distance=5, max_lag_day=50):
    ps = []
    pred_ps = []
    lphs = []
    phrs = []
    peak_diffs = []
    double_peaks = []
    print(f"running acf on {dataset_name} exp{exp_num}")

    dur = 720
    max_lag_day = max_lag_day
    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
    test_transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                              Normalize('med'),
                              MovingAvg(kernel_size=5),
                              ACF(prom=prom, max_lag=max_lag_day, calc_p=True, only_acf=True, distance=distance),
                              ToTensor(),
                              ])
    test_dataset = TimeSeriesDataset(data_folder, test_list, transforms=test_transform, t_samples=int(max_lag_day / cad * DAY2MIN),
                                    init_frac=0.2,   dur=dur, freq_rate=freq_rate)
    import inspect
    boundaries_dict = test_dataset.boundary_values_dict
    for i, (x,y,_,info) in enumerate(test_dataset):
        x = x.numpy().squeeze()
        p = y[1]*(boundaries_dict['max Period'] - boundaries_dict['min Period']) + boundaries_dict['min Period']
        # pred_p, lags, xcf, peaks, lph, phr, p_diff, d_peaked = analyze_lc(x, prom=0.005, max_period=50)
        pred_p, lags, xcf, peaks, lph, phr, p_diff, d_peaked = info['predicted acf_p'],None, None,None, info['lph'], info['phr'], info['peak_diff'], info['double_peaked']
        ps.append(p.item())
        pred_ps.append(pred_p)
        double_peaks.append(d_peaked)
        lphs.append(lph)
        phrs.append(phr)
        peak_diffs.append(p_diff)
        if i % 1000 == 0:
            print(f"sample {i} of {len(test_dataset)}")
            # fig , ax = plt.subplots(1,2)
            # ax[0].plot(x)
            # ax[1].plot(lags, xcf)
            # ax[1].plot(lags[peaks], xcf[peaks], 'o')
            # plt.savefig(f'{log_path}/exp{exp_num}/acf_results_{dataset_name}_sample_{i}.png')
            # plt.close()
    print(len(ps), len(pred_ps), len(double_peaks), len(peak_diffs), len(lphs), len(phrs))
    df_full = pd.DataFrame({'period': ps, 'predicted period': pred_ps,
     'double_peaked': double_peaks, 'peak_diff': peak_diffs, 'lph': lphs, 'phr': phrs})
    df_full.to_csv(f'{log_path}/exp{exp_num}/acf_results_{dataset_name}_clean.csv')
    plt.scatter(ps, pred_ps, c=double_peaks)
    acc10p = np.sum(np.abs(np.array(ps) - np.array(pred_ps)) < 0.1*np.array(ps))/len(ps)
    plt.xlabel('True period')
    plt.ylabel('Predicted period')
    plt.title(f'acc10p={acc10p:.2f}')
    plt.ylim(0,120)
    plt.savefig(f'{log_path}/exp{exp_num}/acf_results_{dataset_name}.png')
    plt.close('all')


        



if __name__ == '__main__':
    # optimize_parameters()
    simulation_prediction(prom=0.02260122394403307, distance=5, max_lag_day=50)
    # kepler_prediction_gpu(prom=0.02260122394403307, distance=5, max_lag_day=50)
    
    
    

    
   
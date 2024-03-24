



import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import pandas as pd
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
from matplotlib import pyplot as plt
import torchvision
# from pytorch_forecasting.metrics.quantile import QuantileLoss


import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)   

from lightPred.datasets.simulations import TimeSeriesDataset
from lightPred.transforms import Compose, StandardScaler, Mask, RandomCrop, DownSample, AddGaussianNoise

from lightPred.timeDetr import TimeSeriesDetrModel
from lightPred.timeDetrLoss import SetCriterion, HungarianMatcher, cxcy_to_cxcywh
from lightPred.dataloader import *
from lightPred.utils import *
from lightPred.train import *
from lightPred.transforms import *
from lightPred.sampler import DistributedSamplerWrapper
from lightPred.eval import eval_model, eval_results, eval_quantiled_results
from lightPred.optim import WeightedMSELoss, QuantileLoss
print(f"python path {os.sys.path}")

torch.manual_seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 1

log_path = '/data/logs/timeDetr'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
# checkpoint_path = '/data/logs/astroconf/exp14'
data_folder = "/data/butter/data_cos"

test_folder = "/data/butter/test_cos"

Nlc = 50000

test_Nlc = 5000

CUDA_LAUNCH_BLOCKING='1'

DAY2MIN = 24*60

dur = 360

b_size = 16

num_epochs = 1000

cad = 30

# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
samples = os.listdir(os.path.join(data_folder, 'simulations'))
idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]

train_list, val_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

test_idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc))+1) for idx in range(test_Nlc)]

# class_labels = ['Period', 'Decay Time', 'Cycle Length']
class_labels = ['Inclination']

if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


if __name__ == '__main__':


    
    optim_params = {
    # "lr": 0.0096, "weight_decay": 0.0095
    'lr': 0.00024222641190709926,
     'weight_decay': 0.009840196207898829
    }


    new_params = {'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
        'dropout': 0.14091365738520598,
        'num_queries': 600,
    }
    
    weight_dict = {'loss_ce': 0.822321274651144,
     'loss_bbox': 0.10110399527930165, 'loss_giou': 1}
    eos = 1644644781327425
    losses = ['labels', 'boxes', 'cardinality']
       

      
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    setup(rank, world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")


   
    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                        ])
    test_transform = Compose([Slice(0, int(dur/cad*DAY2MIN)),
                            ])


    train_dataset = TimeSeriesDataset(data_folder, train_list, transforms=transform, prepare=False, acf=False,
                                        spots=True, init_frac=0, freq_rate=1, period_norm=True)
    val_dataset = TimeSeriesDataset(data_folder, val_list, transforms=transform, prepare=False, acf=False,
                                        spots=True, init_frac=0, freq_rate=1,  period_norm=True)
    test_dataset = TimeSeriesDataset(test_folder, test_idx_list, transforms=test_transform, prepare=False, acf=False,
                                        spots=True, init_frac=0, freq_rate=1,  period_norm=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, sampler=val_sampler, \
                                 num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

    test_dataloader = DataLoader(test_dataset, batch_size=b_size, \
    num_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) 

    model = TimeSeriesDetrModel(input_dim=1, num_classes=2, num_angles=4, **new_params)
    matcher = HungarianMatcher()
    spots_loss = SetCriterion(1, matcher, weight_dict, eos, losses=losses, device=DEVICE)
    att_loss = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(),**optim_params)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print("number of params:", count_params(model))

    data_dict = {'dataset': train_dataset.__class__.__name__,
                   'transforms': transform,  'batch_size': b_size,
     'num_epochs':num_epochs, 'checkpoint_path': f'{log_path}/exp{exp_num}', 'loss_fn':
      att_loss.__class__.__name__,
     'model': model.module.__class__.__name__, 'optimizer': optimizer.__class__.__name__,
     'data_folder': data_folder, 'test_folder': test_folder, 'class_labels': class_labels}

    with open(f'{log_path}/exp{exp_num}/data_params.yml', 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)
    print("logdir: ", f'{log_path}/exp{exp_num}')
    print("data params: ", data_dict)

    trainer = SpotsTrainer(model=model, optimizer=optimizer, spots_loss=spots_loss,
                        criterion=att_loss, num_classes=len(class_labels),
                       scheduler=None, train_dataloader=train_dataloader, optim_params=optim_params,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=exp_num, log_path=log_path,
                        exp_name="timeDetr") 
    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank,
                           early_stopping=40, only_p=False, best='loss', conf=True) 
    output_filename = f'{log_path}/exp{exp_num}/astroconf.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)
    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fit.png")
    plt.clf()

    print("Evaluation on test set:")

    preds, targets, confs, spots_boxes, target_boxes, spots_labels = trainer.predict(test_dataloader, device=local_rank,
                                             conf=True, load_best=False)

    eval_results(preds, targets, confs, labels=class_labels, data_dir=f'{log_path}/exp{exp_num}',
                  model_name=model.module.__class__.__name__,  num_classes=len(class_labels), cos_inc=False)




                                                                        
    

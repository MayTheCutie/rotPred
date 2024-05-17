import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)    

from dataset.dataloader import *
from nn.models import *
from nn.train import *
from nn.optim import QuantileLoss
from util.utils import *
from util.eval import eval_model, eval_results
from transforms import *
from Astroconf.Train.utils import init_train
from Astroconf.utils import Container


warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

if torch.cuda.is_available():
    print("gpu number: ", torch.cuda.current_device())
    
exp_num = 51

local = False

root_dir = '/data' if not local else '../'

log_path = f'{root_dir}/logs/astroconf'

yaml_dir = '/data/lightPred/Astroconf'
# yaml_dir = 'Astroconf/'



if (not torch.cuda.is_available()) or torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(f'{log_path}/exp{exp_num}'):
        os.makedirs(f'{log_path}/exp{exp_num}')
    if not os.path.exists(f'{log_path}/exp{exp_num}/fine_tune2'):
        os.makedirs(f'{log_path}/exp{exp_num}/fine_tune2')
    # if not os.path.exists(f'{log_path}/exp{exp_num}_koi'):
    #     os.makedirs(f'{log_path}/exp{exp_num}_koi')

# chekpoint_path = '/data/logs/lstm_attn/exp52'
root_data_folder =  f"{root_dir}/lightPred/data"
table_path  =  f"{root_dir}/lightPred/tables/Table_1_Periodic.txt"
kois_table_path =  f"{root_dir}/lightPred/tables/kois_no_fp.csv"
inc_path = f"{root_dir}/lightPred/tables/all_incs.csv"

class_labels = ['Inclination', 'Period']

b_size = 16

num_epochs = 30

# min_p, max_p = 0, 60
# min_i, max_i = 0, 90

cad = 30

DAY2MIN = 24*60

dur = 720

# labels = ['Inclination', 'Period']
    
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def aggregate_results_from_gpus(y_pred, conf_pred, kic, teff, r, g, qs):
        torch.distributed.gather(y_pred, [torch.zeros_like(y_pred) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(conf_pred, [torch.zeros_like(conf_pred) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(kic, [torch.zeros_like(kic) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(teff, [torch.zeros_like(teff) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(r, [torch.zeros_like(r) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(g, [torch.zeros_like(g) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(qs, [torch.zeros_like(qs) for _ in range(torch.distributed.get_world_size())], dst=0)
        return y_pred, conf_pred, kic, teff, r, g, qs


if __name__ == '__main__':

    torch.manual_seed(42)

    # optim_params = {"betas": (0.7191221416723297, 0.9991147816604715),
    # "lr": 2.4516572028943392e-05,
    # "weight_decay": 3.411877716394279e-05}
    optim_params = {
    # "lr": 0.0096, "weight_decay": 0.0095
    "lr": 2e-4, "weight_decay": 1e-5
    }

    net_params = {
         'in_channels':1,
 'dropout': 0.35,
 'hidden_size': 64,
 'num_layers': 5,
 'seq_len': int(dur/cad*DAY2MIN), 
 "num_classes": len(class_labels)*2,
    'stride': 4,
    'kernel_size': 4}

    backbone= {'in_channels':1,
 'dropout': 0.35,
 'hidden_size': 64,
 'num_layers': 4,
 'seq_len': 1024,
 'num_classes': 4, 
    'stride': 4,
    'kernel_size': 4}

    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)

    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    print(f"rank: {rank}, local_rank: {local_rank}")

    print("logdir ", f'{log_path}/exp{exp_num}')

    num_qs = dur//90
    kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')
    refs = pd.read_csv('/data/lightPred/tables/all_refs.csv')
    refs.dropna(subset=['i', 'prot'], inplace=True)
    # kepler_df = multi_quarter_kepler_df('data/', table_path=None, Qs=np.arange(3,17))
    kepler_df = get_all_samples_df(num_qs)
    # kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]
    kepler_df = kepler_df.merge(refs, on='KID', how='right')
    kepler_df.to_csv('/data/lightPred/tables/ref_merged.csv', index=False)
    kepler_df.dropna(subset=['i', 'longest_consecutive_qs_indices'], inplace=True)
    segments_df = break_samples_to_segments(num_qs=8)
    print(f"all samples:  {len(segments_df)}")
    
    data_params = yaml.safe_load(open(f'{log_path}/{exp_num}/data_params.yaml', 'r'))
    transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)), MovingAvg(13), Detrend(),
                                ACF(), Normalize('std'), ToTensor()])

    full_dataset = KeplerLabeledDataset(root_data_folder, path_list=None, boundaries_dict=data_params['boundaries']
                                    df=kepler_df, t_samples=int(dur/cad*DAY2MIN),
                                    skip_idx=0, num_qs=num_qs,cos_inc=False, transforms=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)

    full_dataloader = DataLoader(full_dataset, batch_size=b_size, \
                                    num_workers=num_workers,
                                    collate_fn=kepler_collate_fn, pin_memory=True, sampler=sampler)


    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
    conf_model, _, scheduler, scaler = init_train(args, local_rank)
    conf_model.pred_layer = nn.Identity()

    # print("number of params:", count_params(model))
    
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    folds_accs_val = []
    folds_losses_val = []
    folds_accs_train = []
    folds_losses_train = []
    folds_models = []
    

    for i in range(num_folds):
        print('Processing fold: ', i + 1)
        """%%%% Initiate new model %%%%""" #in every fold
        valid_idx = np.arange(len(full_dataset))[i * num_val_samples:(i + 1) * num_val_samples]
        train_idx = np.concatenate([np.arange(len(full_dataset))[:i * num_val_samples], np.arange(len(full_dataset))[(i + 1) * num_val_samples:]], axis=0)
        train_dataset = Subset(full_dataset, train_idx)
        valid_dataset = Subset(full_dataset, valid_idx)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=1)
        model = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim, lstm_args=net_params)
        state_dict = torch.load(f'{log_path}/exp{exp_num}/astroconf.pth', map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                while key.startswith('module.'):
                    key = key[7:]
            new_state_dict[key] = value
        state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
        optimizer = optim.AdamW(model.parameters(), **optim_params)

        trainer = KeplerTrainer(model=model, optimizer=optimizer, criterion=loss_fn,
                    scheduler=None, train_dataloader=train_loader, val_dataloader=valid_loader,
                        device=local_rank, optim_params=optim_params, net_params=net_params,
                          exp_num=f'exp{exp_num}/fine_tune2',
                          log_path=log_path,
                        exp_name="lstm_attn", num_classes=len(class_labels),
                        eta=0.8)
        fit_res = trainer.fit(100, device=local_rank, early_stopping=10, conf=True, best='loss')
        folds_accs_val.extend(fit_res['val_acc'])
        folds_losses_val.extend(fit_res['val_loss'])
        folds_accs_train.extend(fit_res['train_acc'])
        folds_losses_train.extend(fit_res['train_loss'])
        folds_models.append(trainer.best_state_dict)
    global_fit = {'val_acc': folds_accs_val, 'val_loss': folds_losses_val, 'train_acc': folds_accs_train, 'train_loss': folds_losses_train}
    fig, axes = plot_fit(global_fit, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fine_tune2/fit.png")
    plt.clf()
    best_model = folds_models[-1]
    torch.save(best_model, f'{log_path}/exp{exp_num}/fine_tune2/astroconf_finetune_best.pth')

    
    

        

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import pandas as pd
import numpy as np
import torch.optim as optim
import optuna
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


import sys
from os import path

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)

from lightPred.models import LSTM_ATTN, LSTM_DUAL
from lightPred.Astroconf.Train.utils import init_train
from lightPred.Astroconf.utils import Container, same_seeds
from lightPred.dataloader import *
from lightPred.utils import *
from lightPred.train import *
from lightPred.transforms import *

print(f"python path {os.sys.path}")

torch.manual_seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

local = True

exp_num = 40

log_path = '/data/logs/astroconf' if not local else './logs/astroconf'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

root_dir = './' if local else '/data/lightPred'
# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
# checkpoint_path = '/data/logs/astroconf/exp14'
data_folder = "/data/butter/data_cos_old" if not local else '../butter/data_cos_old'

test_folder = "/data/butter/test_cos_old" if local else 'butter/test_cos_old'

yaml_dir = f'{root_dir}/Astroconf/'

Nlc = 50000

test_Nlc = 5000

CUDA_LAUNCH_BLOCKING = '1'

# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
samples = os.listdir(os.path.join(data_folder, 'simulations'))
idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]

train_list, val_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

test_idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc)) + 1) for idx in range(test_Nlc)]

b_size = 2

num_epochs = 1

max_iter = 500
val_iter = max_iter//5

cad = 30

DAY2MIN = 24 * 60

dur = 720

freq_rate = 1 / 48

# class_labels = ['Period', 'Decay Time', 'Cycle Length']
class_labels = ['Inclination', 'Period']

if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)

encoder_architectures = [
        ["mhsa_pro", "conv", "mhsa_pro"],
        ["mhsa_pro", "conv", "conv"],
        ["mhsa_pro", "mhsa_pro", "conv"],
        ["mhsa_pro", "conv"]
    ]

lstm_params = {
    'dropout': 0.35,
    'hidden_size': 64,
    'image': False,
    'in_channels': 1,
    'kernel_size': 4,
    'num_classes': len(class_labels) * 2,
    'num_layers': 5,
    'seq_len': int(dur / cad * DAY2MIN),
    'stride': 4}



def setup(rank, world_size):
    if world_size > 1:
        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def objective(trial):
    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])

    # Sample one of the model architectures
    encoder_idx = trial.suggest_int('encoder_idx', 0, len(encoder_architectures) - 1)
    encoder = encoder_architectures[encoder_idx]
    encoder_dim = trial.suggest_int("hidden_dim", 64, 128, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    num_heads = trial.suggest_int("num_heads", 4, 8, step=4)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    stride = trial.suggest_int("stride", 12, 24, step=4)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    args.encoder = encoder
    args.encoder_dim = encoder_dim
    args.num_layers = num_layers
    args.num_heads = num_heads
    args.dropout_p = dropout
    args.stride = stride
    args.kernel_size = kernel_size

    lstm_hidden = trial.suggest_int("lstm_hidden_dim", 64, 128, step=64)
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1,5)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.4)
    lstm_stride = trial.suggest_int("lstm_stride", 4, 24, step=4)
    lstm_kernel_size = trial.suggest_int("lstm_kernel_size", 3, 7,step=2)
    lstm_params["hidden_size"] = lstm_hidden
    lstm_params["num_layers"] = lstm_num_layers
    lstm_params["dropout"] = lstm_dropout
    lstm_params["stride"] = lstm_stride
    lstm_params["kernel_size"] = lstm_kernel_size

    phr = trial.suggest_int("phr", 0, 1)
    norm = trial.suggest_categorical("norm", ["std", "minmax", "none"])
    detrend = trial.suggest_int("detrend", 0, 1)

    predict_size = trial.suggest_int("predict_size", 64, 128, step=64)

    conf_model, _, scheduler, scaler = init_train(args, DEVICE)
    conf_model.pred_layer = nn.Identity()
    model = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim,
                      lstm_args=lstm_params, predict_size=predict_size, num_classes=4)

    kep_transform = RandomCrop(int(dur /cad *DAY2MIN))
    if detrend:
        transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)),
                         KeplerNoiseAddition(noise_dataset=None, noise_path=f'{root_dir}/data/noise',
                                             transforms=kep_transform),
                         MovingAvg(49), Detrend(), ACF(calc_phr=phr), Normalize(norm), ToTensor(), ])
    else:
        transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)),
                             KeplerNoiseAddition(noise_dataset=None, noise_path=f'{root_dir}/data/noise',
                                                 transforms=kep_transform),
                             MovingAvg(49), ACF(calc_phr=phr), Normalize(norm), ToTensor(), ])
    train_dataset = TimeSeriesDataset(data_folder, train_list, labels=class_labels, transforms=transform,
    init_frac=0.2,  prepare=False, dur=dur, freq_rate=freq_rate, period_norm=False)
    val_dataset = TimeSeriesDataset(data_folder, val_list, labels=class_labels,  transforms=transform,
     init_frac=0.2, prepare=False, dur=dur, freq_rate=freq_rate, period_norm=False)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size)

    lr = trial.suggest_float("lr", 1e-5 ,1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

    # ce_weight = trial.suggest_float("ce_weight", 0.1, 1)
    # bbox_weight = trial.suggest_float("bbox_weight", 0.1, 1)
    # eos_val = trial.suggest_float("eos_val", 0.1, 0.5)
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(DEVICE)
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        model.train()
        t_loss = 0
        v_loss = 0
        pbar = tqdm(train_dataloader, total=max_iter)
        for i, (x, y, _, info) in enumerate(pbar):
            if i > max_iter:
                break
            # print("x: ", x.shape, "y: ", y.shape)
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            if 'acf_phr' in info:
                y_pred = model(x1.float(), x2.float(), acf_phr=info['acf_phr'])
            else:
                y_pred = model(x1.float(), x2.float())
            y_pred, conf_pred = y_pred[:, :2], y_pred[:, 2:]
            conf_y = torch.abs(y - y_pred)
            loss = loss_fn(y_pred, y)
            loss += loss_fn(conf_pred, conf_y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            pbar.set_description(f"train_loss:  {loss.item()}")
        train_loss.append(t_loss / i)
        model.eval()
        pbar = tqdm(val_dataloader, total=val_iter)
        for i, (x ,y ,_ ,info) in enumerate(pbar):
            with torch.no_grad():
                if i > val_iter:
                    break
                x1, x2 = x[:, 0, :], x[:, 1, :]
                x1 = x1.to(DEVICE)
                x2 = x2.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                if 'acf_phr' in info:
                    y_pred = model(x1.float(), x2.float(), acf_phr=info['acf_phr'])
                else:
                    y_pred = model(x1.float(), x2.float())
                y_pred, conf_pred = y_pred[:, :2], y_pred[:, 2:]
                conf_y = torch.abs(y - y_pred)
                loss = loss_fn(y_pred, y)
                loss += loss_fn(conf_pred, conf_y)
                v_loss += loss.item()
                pbar.set_description(f"val_loss:  {loss.item():.2f}")
        trial.report(v_loss, epoch)
        val_loss.append(v_loss/ i)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    torch.cuda.empty_cache()
    return val_loss[-1]

if __name__ == "__main__":

    study = optuna.create_study(study_name='astroconf', storage='sqlite:///../optuna/astroconf.db', load_if_exists=True)
    study.optimize(lambda trial: objective(trial), n_trials=100)
    print('Device: ', DEVICE)
    print("Best trial:")
    trial = study.best_trial

    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))
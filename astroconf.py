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

from lightPred.models import LSTM_ATTN, LSTM_DUAL
from lightPred.dataloader import *
from lightPred.Astroconformer.Astroconformer.Astroconformer.Train.utils import init_train
from lightPred.Astroconformer.Astroconformer.Astroconformer.utils import Container, same_seeds
from lightPred.utils import *
from lightPred.train import *
from lightPred.eval import eval_model, eval_results, eval_quantiled_results
from lightPred.optim import WeightedMSELoss, QuantileLoss
from lightPred.transforms import *
from lightPred.sampler import DistributedSamplerWrapper
print(f"python path {os.sys.path}")

torch.manual_seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 12

log_path = '/data/logs/astroconf'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
checkpoint_path = '/data/logs/astroconf/exp5'
data_folder = "/data/butter/data_cos"

test_folder = "/data/butter/test_cos"

yaml_dir = '/data/lightPred/Astroconformer/Astroconformer/'

Nlc = 50000

test_Nlc = 5000

CUDA_LAUNCH_BLOCKING='1'


idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

train_list, val_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

test_idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc))+1) for idx in range(test_Nlc)]

b_size = 16

num_epochs = 1000

cad = 30

DAY2MIN = 24*60

dur = 900

# class_labels = ['Period', 'Decay Time', 'Cycle Length']
class_labels = ['Inclination', 'Period']

if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


if __name__ == '__main__':


    # optim_params = {"betas": (0.7191221416723297, 0.9991147816604715),
    # "lr": 2.4516572028943392e-05,
    # "weight_decay": 3.411877716394279e-05}
    optim_params = {
    # "lr": 0.0096, "weight_decay": 0.0095
    "lr": 5e-4
    }

    lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_classes': len(class_labels),
        'num_layers': 5,
        'predict_size': 128,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}
    # 'num_att_layers':2,
    # 'n_heads': 4,}

    backbone= { 'in_channels':1,
 'dropout': 0.35,
 'hidden_size': 64,
 'num_layers': 5,
 'seq_len': int(dur/cad*DAY2MIN), 
 "num_classes": 4,
    'stride': 4,
    'kernel_size': 4}
      
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


   
    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
    print("args : ", args)

    # transform_train = Compose([ AddGaussianNoise(sigma=0.005),
    #                     ])

    kepler_data_folder = "/data/lightPred/data"
    non_ps = pd.read_csv('/data/lightPred/tables/non_ps.csv')
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5,6,7,8,9,10,11,12,13])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==10]
    kepler_df.to_csv('/data/lightPred/tables/kepler_noise_4567.csv', index=False)
    # kepler_df = pd.read_csv('/data/lightPred/tables/kepler_noise_4567.csv')
    print(kepler_df.head())
    kep_transform = Compose([RandomCrop(int(dur/cad*DAY2MIN))])
    merged_df = pd.merge(kepler_df, non_ps, on='KID', how='inner')
    noise_ds = KeplerDataset(kepler_data_folder, path_list=None, df=merged_df,
    transforms=kep_transform, acf=False, norm='none')

    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                        #   KeplerNoise(noise_ds, min_ratio=0.02, max_ratio=0.05), 
                          KeplerNoiseAddition(noise_ds),                         
     moving_avg(49), Detrend()])
    test_transform = Compose([Slice(0, int(dur/cad*DAY2MIN)),
                            # KeplerNoise(noise_ds, min_ratio=0.02, max_ratio=0.05), 
                            KeplerNoiseAddition(noise_ds),
     moving_avg(49), Detrend()])

#     image_transform = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(p=0.5),
#     # torchvision.transforms.ToDtype(torch.float32, scale=True),
#     torchvision.transforms.Normalize(mean=[0.445], std=[0.269]),
# ])

#     image_test_transform = torchvision.transforms.Compose([
#     # torchvision.transforms.ToDtype(torch.float32, scale=True),
#     torchvision.transforms.Normalize(mean=[0.445], std=[0.269]),
# ])
    
    # train_dataset = ACFImageDataset(data_folder, train_list, t_samples=t_samples, transforms=transform, image_transform=image_transform)
    # val_dataset = ACFImageDataset(data_folder, val_list, t_samples=t_samples, transforms=transform, image_transform=image_transform)
    # test_dataset = ACFImageDataset(data_folder, val_list, t_samples=t_samples, transforms=test_transform, image_transform=image_test_transform)

    # train_dataset = ACFDataset(data_folder, train_list,labels=class_labels, t_samples=None, transforms=transform, return_raw=False)
    # val_dataset = ACFDataset(data_folder, val_list, labels=class_labels, t_samples=None, transforms=transform, return_raw=False)
    # test_dataset = ACFDataset(test_folder, test_idx_list, labels=class_labels, t_samples=None, transforms=test_transform, return_raw=False)

    # train_dataset = FourierDataset(data_folder, train_list,labels=class_labels, n_days=dur*2, )
    # val_dataset = FourierDataset(data_folder, val_list, labels=class_labels, n_days=dur*2,)
    # test_dataset = FourierDataset(test_folder, test_idx_list, labels=class_labels, n_days=dur*2, )

    # train_dataset = ACFDataset(data_folder, train_list, labels=class_labels, t_samples=None, transforms=transform, return_raw=False)
    # val_dataset = ACFDataset(data_folder, val_list, labels=class_labels, t_samples=None, transforms=transform, return_raw=False)
    # test_dataset = ACFDataset(test_folder, test_idx_list, labels=class_labels, t_samples=None, transforms=transform, return_raw=False)
   
    train_dataset = TimeSeriesDataset(data_folder, train_list, transforms=transform,
    init_frac=0.05, acf=True, return_raw=True, prepare=False, dur=dur)
    val_dataset = TimeSeriesDataset(data_folder, val_list,  transforms=transform,
     init_frac=0.05, acf=True, return_raw=True, prepare=False, dur=dur)
    test_dataset = TimeSeriesDataset(test_folder, test_idx_list, transforms=test_transform,
    init_frac=0.05, acf=True, return_raw=True, prepare=False, dur=dur)

    # train_dataset = TimeSeriesDataset2(data_folder, train_list, labels=class_labels,
    #  t_samples=None, transforms=transform, spectrogram=False, prepare=False, p_norm=False)
    # val_dataset = TimeSeriesDataset2(data_folder, val_list, labels=class_labels,
    #  t_samples=None, transforms=transform, spectrogram=False, prepare=False, p_norm=False)
    # test_dataset = TimeSeriesDataset2(test_folder, test_idx_list, labels=class_labels,
    #  t_samples=None, transforms=test_transform, spectrogram=False, prepare=False, p_norm=False)
  

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)

    # print("calculating weights...")
    # if os.path.exists(f'{log_path}/exp{exp_num}/train_weights.npy'):
    #     train_weights = np.load(f'{log_path}/exp{exp_num}/train_weights.npy')
    # else:
    #     train_weights = dataset_weights(train_dataloader, Nlc)
    #     os.makedirs(f'{log_path}/exp{exp_num}', exist_ok=True)
    #     np.save(f'{log_path}/exp{exp_num}/train_weights.npy', train_weights)

    # train_sampler_weighted = DistributedSamplerWrapper(sampler=WeightedRandomSampler(train_weights, len(train_weights)),
    #                                             num_replicas=world_size, rank=rank)
    # train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler_weighted,\
    #                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True) 

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, sampler=val_sampler, \
                                 num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
    

    test_dataloader = DataLoader(test_dataset, batch_size=b_size, \
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) 

    print("dataset length: ", len(train_dataset), len(val_dataset), len(test_dataset))

    # for i in range(10):
    #     idx = np.random.randint(0, len(train_dataset))
    #     x,y,_,_ = train_dataset[idx]

    # print("check weights...")
    # incs = torch.zeros(0)
    # for i, (x,y,_,_) in enumerate(test_dataloader):
    #     print(i)
    #     incs = torch.cat((incs, y[:,0]*90), dim=0)
    # plt.hist(incs.squeeze(), 80)
    # plt.savefig('/data/tests/incs_hist_test_lstm_attn.png')
    # plt.clf()
    # print("done")
    # train_dataset = TimeSeriesDataset(data_folder, train_list, t_samples=net_params['seq_len'])
    # val_dataset = TimeSeriesDataset(data_folder, val_list, t_samples=net_params['seq_len'])
    # test_dataset = TimeSeriesDataset(test_folder, test_idx_list, t_samples=net_params['seq_len'])


    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, \
    #                                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=b_size, \
    #                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

   

    # model, net_params, _ = load_model(f'{log_path}/exp{exp_num}', LSTM_ATTN, distribute=True, device=local_rank, to_ddp=True)
    
    conf_model, _, scheduler, scaler = init_train(args, local_rank)
    conf_model.pred_layer = nn.Identity()
    model = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim, **lstm_params)

    # model = conf_model

    # state_dict = torch.load(f'{log_path}/exp5/astroconf.pth')
    # new_state_dict = OrderedDict()
    # for key, value in state_dict.items():
    #     if key.startswith('module.'):
    #         while key.startswith('module.'):
    #             key = key[7:]
    #     new_state_dict[key] = value
    # state_dict = new_state_dict
    # model.load_state_dict(new_state_dict)

    


    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print("number of params:", count_params(model))
    
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), **optim_params)

    # loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss(beta=0.0005)
    # loss_fn = WeightedMSELoss(factor=1.2)
    # loss_fn = nn.GaussianNLLLoss()

    data_dict = {'dataset': train_dataset.__class__.__name__, 'transforms': transform, 'batch_size': b_size,
     'num_epochs':num_epochs, 'checkpoint_path': f'{log_path}/exp{exp_num}', 'loss_fn': loss_fn.__class__.__name__,
     'model': model.module.__class__.__name__, 'optimizer': optimizer.__class__.__name__,
     'data_folder': data_folder, 'test_folder': test_folder, 'class_labels': class_labels}

    with open(f'{log_path}/exp{exp_num}/data_params.yml', 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

    print("data params: ", data_dict)

    
    trainer = DoubleInputTrainer(model=model, optimizer=optimizer, criterion=loss_fn, num_classes=len(class_labels),
                       scheduler=None, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        device=local_rank, optim_params=optim_params, net_params=lstm_params, exp_num=exp_num, log_path=log_path,
                        exp_name="astroconf")
    

    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank, early_stopping=40, only_p=False, best='loss', conf=False)

    
    output_filename = f'{log_path}/exp{exp_num}/astroconf.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)
    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fit.png")
    plt.clf()

    print("Evaluation on test set:")

    preds, targets, confs = trainer.predict(test_dataloader, device=local_rank, conf=False, load_best=True)

    eval_results(preds, targets, confs, labels=class_labels, data_dir=f'{log_path}/exp{exp_num}', model_name=model.module.__class__.__name__,
     num_classes=len(class_labels))


    # eval_model(f'{log_path}/exp{exp_num}',model=LSTM_ATTN, test_dl=val_dataloader,
    #                 data_folder=test_folder, conf=True, num_classes=net_params['num_classes']//2) 

    
    
    
   
      
    
    

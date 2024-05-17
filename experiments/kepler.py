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

from lightPred.dataloader import *
from lightPred.models import *
from lightPred.utils import *
from lightPred.train import *
from lightPred.eval import eval_model, eval_results
from lightPred.optim import QuantileLoss
from lightPred.transforms import *
from lightPred.utils import collate as my_collate, convert_to_list, extract_qs, consecutive_qs, kepler_collate_fn
from lightPred.Astroconf.Train.utils import init_train
from lightPred.Astroconf.utils import Container


warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

if torch.cuda.is_available():
    print("gpu number: ", torch.cuda.current_device())

exp_num = 51


local = False

root_dir = '/data' if not local else '../'

log_path = f'{root_dir}/logs/astroconf/exp{exp_num}/fine_tune2'
yaml_dir = '/data/lightPred/Astroconf'
# yaml_dir = 'Astroconf/'



# if (not torch.cuda.is_available()) or torch.cuda.current_device() == 0:
#     if not os.path.exists(log_path):
#         os.makedirs(log_path)
#     if not os.path.exists(f'{log_path}/exp{exp_num}'):
#         os.makedirs(f'{log_path}/exp{exp_num}')
    # if not os.path.exists(f'{log_path}/exp{exp_num}_koi'):
    #     os.makedirs(f'{log_path}/exp{exp_num}_koi')

# chekpoint_path = '/data/logs/lstm_attn/exp52'
# chekpoint_path = f'{root_dir}/logs/astroconf/exp45/fine_tune'
# data_folder =  f"{root_dir}/lightPred/data/Q4"
root_data_folder =  f"{root_dir}/lightPred/data"
table_path  =  f"{root_dir}/lightPred/tables/Table_1_Periodic.txt"
# kois_table_path =  f"{root_dir}/lightPred/tables/kois_no_fp.csv"
# inc_path = f"{root_dir}/lightPred/tables/all_incs.csv"

class_labels = ['Inclination', 'Period']

# all_samples_list = [file_name for file_name in glob.glob(os.path.join(data_folder, '*')) if not os.path.isdir(file_name)]


b_size = 32

num_epochs = 200

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

    try:
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

        print("logdir ", f'{log_path}/exp{exp_num}')
        # print("checkpoint path ", chekpoint_path)
    except:
        world_size = 1
        rank = 0
        local_rank = 0
        num_workers = 1
        print("running locally")
        print("logdir ", f'{log_path}/exp{exp_num}')
        # print("checkpoint path ", chekpoint_path)

    # q_list = [[3,4,5,6,7,8,9,10],
    # [4,5,6,7,8,9,10,11], [5,6,7,8,9,10,11,12], [6,7,8,9,10,11,12,13],
    #     [7,8,9,10,11,12,13,14],[8,9,10,11,12,13,14,15], [9,10,11,12,13,14,15,16]]
    # for Q in q_list:
    # kepler_df = create_kepler_df(data_folder, table_path)
    # kepler_df = multi_quarter_kepler_df(root_data_folder, ta
    # ble_path=None, Qs=np.arange(3,17))
    # kepler_df = kepler_df.sample(frac=1)
    # kepler_df = kepler_df[kepler_df['number_of_quarters'] == len(Q)]
    num_qs = dur//90
    kepler_df = get_all_samples_df(num_qs)
    print(f"all samples with at least {num_qs} consecutive qs:  {len(kepler_df)}")
    for q in range(15-num_qs):
        tic = time.time()
        print("i: ", q)
        step = int(q*int(90/cad*DAY2MIN))
        transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)), MovingAvg(49), Detrend(),
                              ACF(), Normalize('std'), ToTensor()])
        test_transform = Compose([Slice(0 + step, int(dur / cad * DAY2MIN) + step), MovingAvg(49), Detrend(),
                                  ACF(), Normalize('std'), ToTensor()])

        full_dataset = KeplerDataset(root_data_folder, path_list=None,
                                      df=kepler_df, t_samples=int(dur/cad*DAY2MIN), skip_idx=q, num_qs=num_qs,
            transforms=test_transform)
        sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)

        full_dataloader = DataLoader(full_dataset, batch_size=b_size, \
                                        num_workers=num_workers,
                                        collate_fn=kepler_collate_fn, pin_memory=True, sampler=sampler)
        data_dict = {'dataset': full_dataset.__class__.__name__, 'batch_size': b_size,
         'num_epochs':num_epochs}

        with open(f'{log_path}/data_kepler.json', 'w') as fp:
            json.dump(data_dict, fp)

        args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
        args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
        conf_model, _, scheduler, scaler = init_train(args, local_rank)
        conf_model.pred_layer = nn.Identity()
        model = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim, lstm_args=net_params,
                           num_classes=len(class_labels)*2)

        print(f"loading model from {log_path}")
        state_dict = torch.load(f'{log_path}/astroconf_finetune_best.pth', map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            while key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        model = model.to(local_rank)

        # model, net_params, _ = load_model(chekpoint_path, LSTM_ATTN, distribute=True, device=local_rank, to_ddp=True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)


        # print("number of params:", count_params(model))
        
        loss_fn = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), **optim_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.1)
    
        trainer = KeplerTrainer(model=model, optimizer=optimizer, criterion=loss_fn, 
                        scheduler=scheduler, train_dataloader=None, val_dataloader=None,
                            device=local_rank, optim_params=optim_params, net_params=net_params, exp_num=exp_num, log_path=log_path,
                            exp_name="kepler_inference", num_classes=len(class_labels), eta=0.5)

        preds_f, conf_f, kids_f, teff_f, radius_f, logg_f, qs_f = trainer.predict(full_dataloader,
                                                                                device=local_rank, conf=True, only_p=False)
        # if dist.is_available() and dist.is_initialized():
        #         if torch.distributed.get_rank() == 0:
        #             preds_f, conf_f, kids_f, teff_f, radius_f, logg_f, qs_f = aggregate_results_from_gpus(
        #             preds_f.contiguous().float(), conf_f.contiguous(), kids_f.contiguous(), teff_f.contiguous(),
        #               radius_f.contiguous(), logg_f.contiguous(), qs_f.contiguous())
        # if preds_f[:,0].max() <= 1:
        print("number of predictions: ", len(preds_f))
        print("inc range ", preds_f[:, 0].max(), preds_f[:, 0].min())

        # preds_f[:,0] = np.arcsin(preds_f[:,0])*180/np.pi
        # target_f = np.arcsin(target_f)*180/np.pi
        print("predictions shapes: ", preds_f.shape, conf_f.shape, kids_f.shape, teff_f.shape, radius_f.shape, logg_f.shape, qs_f.shape)
        df_full = pd.DataFrame({'KID': kids_f,  'Teff': teff_f, 'R': radius_f,
         'logg': logg_f, 'qs': qs_f.tolist()})
        df_full['start_idx'] = q
        df_full['duration(days)'] = dur

        # preds_cls = np.argmax(preds_f, axis=-1)               #classifiacation
        # preds_p = np.exp(preds_f)
        # df_full['predicted inclination class'] = preds_cls
        # df_full['predicted inclination probability'] = list(preds_p)

        for i,label in enumerate(class_labels):               # regression
            print("label: ", label)
            print("range: ", preds_f[:, i].max(), preds_f[:, i].min())
            print("boundary values: ", boundary_values_dict[label][0], boundary_values_dict[label][1])
            if label == 'Inclination':
                new_pred = np.arccos(np.clip(preds_f[:,i], a_min=0, a_max=1))
            else:
                new_pred = (preds_f[:, i]*(boundary_values_dict[label][1]-boundary_values_dict[label][0]) + boundary_values_dict[label][0])
            df_full[f'predicted {label}'] = new_pred
            df_full[f'{label} confidence'] = conf_f[:, i]


        print("df shape: ", df_full.shape)
        df_full.to_csv(f'{log_path}/kepler_inference_full_detrend_{q}_rank_{rank}.csv', index=False)
        toc = time.time()
        print("time: ", toc-tic)
        

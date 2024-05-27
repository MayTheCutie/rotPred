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
    
exp_num = 7

local = False

root_dir = '/data' if not local else '../'

log_path = f'{root_dir}/logs/astroconf_cls'

yaml_dir = 'Astroconf'
# yaml_dir = 'Astroconf/'



if (not torch.cuda.is_available()) or torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(f'{log_path}/exp{exp_num}'):
        os.makedirs(f'{log_path}/exp{exp_num}')
    if not os.path.exists(f'{log_path}/exp{exp_num}/fine_tune'):
        os.makedirs(f'{log_path}/exp{exp_num}/fine_tune')
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

    try:
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
    except:
        world_size = 1
        rank = 0
        local_rank = 0
        num_workers = 1
        print("running locally")
        print("logdir ", f'{log_path}/exp{exp_num}')
    num_qs = dur//90
    kepler_df = pd.read_csv('tables/all_kepler_samples.csv')
    refs = pd.read_csv('tables/all_refs.csv')
    refs.dropna(subset=['i', 'prot'], inplace=True)
    refs['err_i'] = refs['err_i'].apply(convert_floats_to_list)
    # kepler_df = multi_quarter_kepler_df('data/', table_path=None, Qs=np.arange(3,17))
    kepler_df = get_all_samples_df(num_qs=2)
    # kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]
    kepler_df = kepler_df.merge(refs, on='KID', how='right')
    kepler_df.to_csv('tables/ref_merged.csv', index=False)
    kepler_df.dropna(subset=['i', 'longest_consecutive_qs_indices'], inplace=True)
    print(f"all samples:  {len(kepler_df)}")
    global_t_loss = []
    global_t_acc = []
    global_best_model = None
    best_loss = np.inf
    best_acc = 0
    for q in range(15-num_qs):
        kepler_df['is_in_batch'] = (kepler_df['longest_consecutive_qs_indices'].
                                       apply(lambda x: (x[0] <= q) and (x[1] >= q)))
        sample_df = kepler_df[kepler_df['is_in_batch']]
        tic = time.time()
        print("***********q: ", q, "number of samples: ", len(sample_df), "***********")
        step = int(q*int(90/cad*DAY2MIN))
        # transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)), MovingAvg(13), Shuffle(),
        #                       Detrend(),
        #                       ACF(), Normalize('std'), ToTensor()])
        transform = Compose([Slice(0 + step, int(dur / cad * DAY2MIN) + step),
                             MovingAvg(13),PeriodNorm(num_ps=10), Detrend(),
                                  ACF(), Normalize('std'), ToTensor()])

        full_dataset = KeplerLabeledDataset(root_data_folder, path_list=None, classification=True,
                                      df=sample_df, t_samples=int(dur/cad*DAY2MIN), num_classes=10,
                                       skip_idx=q, num_qs=num_qs,cos_inc=False, transforms=transform)
        sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)

        full_dataloader = DataLoader(full_dataset, batch_size=b_size, \
                                        num_workers=num_workers, 
                                        collate_fn=kepler_collate_fn, pin_memory=True, sampler=sampler)
    

        args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
        args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
        conf_model, _, scheduler, scaler = init_train(args, local_rank)
        conf_model.pred_layer = nn.Identity()
        model = LSTM_DUAL_CLS(conf_model, encoder_dims=args.encoder_dim, lstm_args=net_params, num_classes=10)

        # state_dict = torch.load(f'{log_path}/exp{exp_num}/astroconf.pth', map_location=torch.device('cpu'))
        # new_state_dict = OrderedDict()
        # for key, value in state_dict.items():
        #     if key.startswith('module.'):
        #         while key.startswith('module.'):
        #             key = key[7:]
        #     new_state_dict[key] = value
        # state_dict = new_state_dict
        # model.load_state_dict(state_dict)
        model = model.to(local_rank)

        # model, net_params, _ = load_model(chekpoint_path, LSTM_ATTN, distribute=True, device=local_rank, to_ddp=True)
        # model = DDP(model, device_ids=[local_rank])


        # print("number of params:", count_params(model))
        
        # loss_fn = nn.MSELoss()
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.AdamW(model.parameters(), **optim_params)
    
        trainer = KeplerTrainer(model=model, optimizer=optimizer, criterion=loss_fn,
                        scheduler=None, train_dataloader=full_dataloader, val_dataloader=full_dataloader,
                            device=local_rank, optim_params=optim_params, net_params=net_params, exp_num=exp_num, log_path=log_path,
                            exp_name="lstm_attn", num_classes=len(class_labels),
                            eta=-1)
        epoch_loss = []
        epoch_acc = []
        epoch_best_loss = np.inf
        epoch_best_acc = 0
        epoch_best_model = None
        for epoch in range(num_epochs):
            loss, acc = trainer.train_epoch(device=local_rank, conf=False)
            print(f"epoch: {epoch} loss: {np.mean(loss)} acc: {acc}")
            epoch_loss.extend(loss)
            epoch_acc.append(acc.tolist())
            if  np.mean(loss) < epoch_best_loss:
                epoch_best_loss = np.mean(loss)
                epoch_best_acc = acc
                epoch_best_model = model.state_dict() 
        global_t_loss.extend(epoch_loss)
        global_t_acc.extend(epoch_acc)
        if epoch_best_loss < best_loss:
            best_loss = epoch_best_loss
            best_acc = epoch_acc
            global_best_model = epoch_best_model
    fit_res = {"train_loss": global_t_loss, "train_acc": global_t_acc, "val_loss": global_t_loss, "val_acc": global_t_acc}
    output_filename = f'{log_path}/exp{exp_num}/astroconf_fine_tune.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)
    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=False)
    plt.savefig(f"{log_path}/exp{exp_num}/fine_tune/fit.png")
    plt.clf()

    
    print("finshed all qs! saving results...")
    with open(f'{log_path}/exp{exp_num}/fine_tune/fit_res.json', 'w') as fp:
        json.dump(fit_res, fp)
    print("best global loss: ", best_loss, "best loss last epoch: ", epoch_best_loss)
    print("saving last model...")
    torch.save(epoch_best_model, f'{log_path}/exp{exp_num}/fine_tune/astroconf_finetune_last.pth')
    print("done")
    print("saving best global model...")
    torch.save(global_best_model, f'{log_path}/exp{exp_num}/fine_tune/astroconf_finetune_global.pth')
    print("done")
    # print("saving model")
    # torch.save(trainer.best_state_dict, f'{log_path}/exp{exp_num}/fine_tune/astroconf_finetune.pth')

        

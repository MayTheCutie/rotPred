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

print("gpu number: ", torch.cuda.current_device())

exp_num = 53

log_path = '/data/logs/kepler'

yaml_dir = '/data/lightPred/Astroconf/'


if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(f'{log_path}/exp{exp_num}'):
        os.makedirs(f'{log_path}/exp{exp_num}')
    # if not os.path.exists(f'{log_path}/exp{exp_num}_koi'):
    #     os.makedirs(f'{log_path}/exp{exp_num}_koi')

# chekpoint_path = '/data/logs/simsiam/exp8'
# chekpoint_path = '/data/logs/lstm_attn/exp47'
# chekpoint_path = '/data/logs/lstm_attn/exp52'
chekpoint_path = '/data/logs/astroconf/exp31'
data_folder = "/data/lightPred/data/Q4"
root_data_folder = "/data/lightPred/data"
table_path  = "/data/lightPred/tables/Table_1_Periodic.txt"
kois_table_path = "/data/lightPred/tables/kois_no_fp.csv"
inc_path = "/data/lightPred/tables/all_incs.csv"

class_labels = ['Inclination', 'Period']

all_samples_list = [file_name for file_name in glob.glob(os.path.join(data_folder, '*')) if not os.path.isdir(file_name)]


# kepler_df = create_kepler_df(data_folder, table_path)
# kepler_df = multi_quarter_kepler_df(root_data_folder, table_path=None, Qs=[4,5,6,7])
# kepler_df = kepler_df.sample(frac=1)
# kepler_df = kepler_df[kepler_df['number_of_quarters'] == 4]
# print("reduced df: ", len(kepler_df))

# mazeh_df = multi_quarter_kepler_df(root_data_folder, table_path=table_path, Qs=[4,5,6,7])
# mazeh_df = mazeh_df.sample(frac=1)
# mazeh_df = mazeh_df[mazeh_df['number_of_quarters'] == 4]

# inc_df = multi_quarter_kepler_df(root_data_folder, table_path=inc_path, Qs=[4,5])
# inc_df = inc_df.sample(frac=1)
# inc_df = inc_df[inc_df['number_of_quarters'] == 2]


# merged_df = pd.concat([df1, df2]).groupby('KID')['data_file_path'].apply(list).reset_index()

# kois_df = multi_quarter_kepler_df(root_data_folder, table_path=kois_table_path, Qs=[4,5])
# kois_df = kois_df.sample(frac=1)
# kois_df = kois_df[kois_df['number_of_quarters'] == 2]

# train_df, test_df = train_test_split(mazeh_df, test_size=0.1, random_state=42, shuffle=True)
# train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
# print("df shapes: ", train_df.shape, val_df.shape, test_df.shape)
# print("df columns: ", train_df.columns)
b_size = 128

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
        'predict_size':128,
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
    print(f"rank: {rank}, local_rank: {local_rank}")

    print("logdir ", f'{log_path}/exp{exp_num}')
    print("checkpoint path ", chekpoint_path)

    # q_list = [[3,4,5,6,7,8,9,10],
    # [4,5,6,7,8,9,10,11], [5,6,7,8,9,10,11,12], [6,7,8,9,10,11,12,13],
    #     [7,8,9,10,11,12,13,14],[8,9,10,11,12,13,14,15], [9,10,11,12,13,14,15,16]]
    # for Q in q_list:
    # kepler_df = create_kepler_df(data_folder, table_path)
    # kepler_df = multi_quarter_kepler_df(root_data_folder, table_path=None, Qs=np.arange(3,17))
    # kepler_df = kepler_df.sample(frac=1)
    # kepler_df = kepler_df[kepler_df['number_of_quarters'] == len(Q)]
    num_qs = dur//90
    kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')
    kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
    kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
    kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
    kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]
    print(f"all samples with at least {num_qs} consecutive qs:  {len(kepler_df)}")
    for q in range(15-num_qs):
        tic = time.time()
        print("i: ", q)
        step = int(q*int(90/cad*DAY2MIN))
        transform = Compose([moving_avg(49), Detrend(), RandomCrop(int(dur/cad*DAY2MIN))])
        test_transform = Compose([moving_avg(49), Detrend(), Slice(0+step, int(dur/cad*DAY2MIN) + step)])

        full_dataset = KeplerDataset(data_folder, path_list=None, df=kepler_df, t_samples=int(dur/cad*DAY2MIN),
            transforms=test_transform, acf=True, return_raw=True)
        sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)

        full_dataloader = DataLoader(full_dataset, batch_size=b_size, \
                                        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                        collate_fn=kepler_collate_fn, pin_memory=True, sampler=sampler)
        data_dict = {'dataset': full_dataset.__class__.__name__, 'batch_size': b_size, 'num_epochs':num_epochs, 'checkpoint_path': chekpoint_path}

        with open(f'{log_path}/exp{exp_num}/data.json', 'w') as fp:
            json.dump(data_dict, fp)

        args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
        args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
        conf_model, _, scheduler, scaler = init_train(args, local_rank)
        conf_model.pred_layer = nn.Identity()
        model = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim, **net_params)

        state_dict = torch.load(f'{chekpoint_path}/astroconf.pth')
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                while key.startswith('module.'):
                    key = key[7:]
            new_state_dict[key] = value
        state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model = model.to(local_rank)

        # model, net_params, _ = load_model(chekpoint_path, LSTM_ATTN, distribute=True, device=local_rank, to_ddp=True)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)


        # print("number of params:", count_params(model))
        
        # loss_fn = nn.MSELoss()
        loss_fn = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), **optim_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.1)
    
        trainer = KeplerTrainer(model=model, optimizer=optimizer, criterion=loss_fn,
                        scheduler=scheduler, train_dataloader=None, val_dataloader=None,
                            device=local_rank, optim_params=optim_params, net_params=net_params, exp_num=exp_num, log_path=log_path,
                            exp_name="lstm_attn", num_classes=len(class_labels))

        preds_f, target_f, conf_f, kids_f, teff_f, radius_f, logg_f, qs_f = trainer.predict(full_dataloader, device=local_rank, conf=True, only_p=False)
        # if preds_f[:,0].max() <= 1:
        print("number of predictions: ", len(preds_f))
        print("inc range ", preds_f[:, 0].max(), preds_f[:, 0].min())

        # preds_f[:,0] = np.arcsin(preds_f[:,0])*180/np.pi
        # target_f = np.arcsin(target_f)*180/np.pi
        df_full = pd.DataFrame({'KID': kids_f,  'Teff': teff_f, 'R': radius_f,
         'logg': logg_f, 'qs': np.array(qs_f) })
        df_full['start_idx'] = q
        df_full['duration(days)'] = dur
        for i,label in enumerate(class_labels):
            print("label: ", label)
            print("range: ", preds_f[:, i].max(), preds_f[:, i].min())
            print("boundary values: ", boundary_values_dict[label][0], boundary_values_dict[label][1])
            new_pred = (preds_f[:, i]*(boundary_values_dict[label][1]-boundary_values_dict[label][0]) + boundary_values_dict[label][0])
            df_full[f'predicted {label}'] = new_pred
            df_full[f'{label} confidence'] = conf_f[:, i]
        print("df shape: ", df_full.shape)
        df_full.to_csv(f'{log_path}/exp{exp_num}/kepler_inference_full_detrend_{q}.csv', index=False)
        toc = time.time()
        print("time: ", toc-tic)

    # print("Evaluation on test set:")

    # eval_model(f'{log_path}/exp{exp_num}',model=LSTM_ATTN, test_dl=test_dataloader, 
    #   num_classes=2, conf=True, kepler=True, kepler_df=test_df)

    # df = pd.read_csv(f'{log_path}/exp{exp_num}/kepler_eval.csv')
    # df.to_csv(f'{log_path}/exp{exp_num}/kepler_eval_small.csv', index=False)

    # print("Evaluation on full test set:")

    # eval_model(f'{log_path}/exp{exp_num}',model=LSTM_ATTN, test_dl=full_mazeh_dataloader, 
    #   num_classes=2, conf=True, kepler=True, kepler_df=mazeh_df)

    # print("Evaluation on test set koi:")
    # eval_model(f'{log_path}/exp{exp_num}_koi',model=LSTM_ATTN, test_dl=test_dataloader_koi,
    #   num_classes=2, conf=True, kepler=True, kepler_df=kois_df)
     


    # print("loading best model...")

    # model, net_params, _ = load_model(f'{log_path}/exp{exp_num}', LSTM_ATTN, distribute=True, device=local_rank, to_ddp=True)
    # model = DDP(model, device_ids=[local_rank])

    # print("model loaded")

    # print("Kepler inference on entire dataset:")


    # output, conf, kids, teff = kepler_inference(model, full_dataloader, device=local_rank, conf=True)
    # out_df = pd.DataFrame({'KID': kids, 'predicted period': output[:,1]*max_p, 'period confidence': conf[:, 1], 
    #                         'predicted inclination': output[:,0]*max_i, 'inclination confidence': conf[:, 0],
    #                         'Teff': teff,
    #                         })
    # out_df.to_csv(f'{log_path}/exp{exp_num}/kepler_inference_mazeh.csv', index=False)

    # plt.hist(output[:,0]*max_i, bins=100)
    # plt.title('inclination distribution')
    # plt.savefig(f'{log_path}/exp{exp_num}/full_inc_dist_mazeh.png')
    # plt.clf()

    # plt.hist(output[:,1]*max_p, bins=100)
    # plt.title('period distribution')
    # plt.savefig(f'{log_path}/exp{exp_num}/full_p_dist_mazeh.png')
    # plt.clf()
        

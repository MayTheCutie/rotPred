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
from lightPred.Astroconf.Train.utils import init_train
from lightPred.Astroconf.utils import Container, same_seeds
from lightPred.Astroconf.Model.models import AstroDecoder
from lightPred.loss import MaskedMSELoss
from lightPred.eval import eval_model
from lightPred.optim import QuantileLoss
from lightPred.transforms import *
from lightPred.utils import collate as my_collate

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)
if DEVICE == 'cuda':
    print("gpu number: ", torch.cuda.current_device())

local = False

root_dir = '.' if local else '/data/lightPred'

exp_num = 1

log_path = '../logs/masked_ssl' if local else '/data/logs/masked_ssl'
# print(os.listdir('lightPred/Astroconf'))
yaml_dir = f'{root_dir}/Astroconf'
if DEVICE.type == 'cpu' or torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(f'{log_path}/exp{exp_num}'):
        os.makedirs(f'{log_path}/exp{exp_num}')
    if not os.path.exists(f'{log_path}/exp{exp_num}_koi'):
        os.makedirs(f'{log_path}/exp{exp_num}_koi')

# chekpoint_path = '/data/logs/simsiam/exp8'
# chekpoint_path = '/data/logs/lstm_attn/exp29'
# chekpoint_path = '/data/logs/kepler/exp13'
data_folder = "/data/lightPred/data/Q4"
root_data_folder = "/data/lightPred/data"
table_path  = "/data/lightPred/Table_1_Periodic.txt"
kois_table_path = "/data/lightPred/kois_no_fp.csv"

all_samples_list = [file_name for file_name in glob.glob(os.path.join(data_folder, '*')) if not os.path.isdir(file_name)]


# kepler_df = create_kepler_df(data_folder, table_path)
# kepler_df = multi_quarter_kepler_df(root_data_folder, table_path=None, Qs=[4])
# kepler_df = kepler_df.sample(frac=1)
# kepler_df = kepler_df[kepler_df['number_of_quarters'] == 2]
# print("reduced df: ", len(kepler_df))


b_size = 32

num_epochs = 200

min_p, max_p = 0, 60
min_i, max_i = 0, 90

cad = 30

DAY2MIN = 24*60

dur = 720

num_qs = 8
    
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
    "lr": 1e-5
    }

    lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_classes': 2,
        'num_layers': 5,
        'predict_size': 128,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}

    if DEVICE == 'cuda':
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
    else:
        local_rank = DEVICE
        world_size = 1
        rank = 0

    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
    print("args : ", vars(args))


    kepler_df = pd.read_csv(f'{root_dir}/tables/all_kepler_samples.csv').iloc
    # kepler_df = multi_quarter_kepler_df('data/', table_path=None, Qs=np.arange(3, 17))
    try:
        kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
    except TypeError:
        pass
    kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
    kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
    kepler_df['longest_consecutive_qs_indices'] = kepler_df['qs'].apply(find_longest_consecutive_indices)

    # kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]
    train_df, val_df = train_test_split(kepler_df, test_size=0.2, random_state=42)
    print("df shapes: ", train_df.shape, val_df.shape)
    # kepler_df = multi_quarter_kepler_df('data/', table_path=None, Qs=np.arange(3,17))


    transform = Compose([MovingAvg(kernel_size=49), RandomCrop(int(dur / cad * DAY2MIN)),
                         Mask(0.1, value=-1), Normalize(norm='minmax')])

    target_transform = Compose([MovingAvg(kernel_size=49), RandomCrop(int(dur / cad * DAY2MIN)),
                                Normalize(norm='minmax')])

    
    train_dataset = KeplerDataset(data_folder, path_list=None, df=train_df,
     t_samples=int(dur/cad*DAY2MIN), transforms=transform, target_transforms=target_transform,
                                  acf=True, return_raw=True, mask_prob=0.1)
    val_dataset = KeplerDataset(data_folder, path_list=None, df=val_df, t_samples=int(dur/cad*DAY2MIN),
     transforms=transform, target_transforms=target_transform,
                                acf=True, return_raw=True, mask_prob=0.1)

    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    slurm_cpus_per_task = int(slurm_cpus_per_task)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                                                    shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, sampler=train_sampler, collate_fn=kepler_collate_fn, \
                                               num_workers=slurm_cpus_per_task, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=True, collate_fn=kepler_collate_fn, \
                                 num_workers=slurm_cpus_per_task)



    # for i in range(20):
    #     x, y, mask, info = train_dataset[i]
    #     print(x.shape, y.shape, mask.shape, info)
    #     x_masked = x.clone()
    #     # fig ,axes = plt.subplots(1,2)
    #     # axes[0].plot(x[0])
    #     # axes[1].plot(y[0])
    #     # plt.show()
    #     print("number of differnt values: ", len(np.where((x[0] - y[0])!=0)[0]))
    #     plt.scatter(np.arange(len(x[0])), x[0] - y[0], s=1)
    #     plt.show()
    # for i, (x,x_masked, inv_mask, info) in enumerate(train_dataloader):
    #     print(i, x.shape, x_masked.shape, inv_mask.shape)
    #     plt.plot(x[0,1])
    #     # plt.plot(x_masked[0,0])
    #     plt.show()
    #     if i == 5:
    #         break


    data_dict = {'dataset': train_dataset.__class__.__name__, 'batch_size': b_size, 'num_epochs':num_epochs, 'dur': dur}

    with open(f'{log_path}/exp{exp_num}/data.json', 'w') as fp:
        json.dump(data_dict, fp)

    # model, net_params, _ = load_model(chekpoint_path, LSTM_ATTN, distribute=True, device=local_rank, to_ddp=True)

    conf_model, _, scheduler, scaler = init_train(args, local_rank)
    conf_model.pred_layer = nn.Identity()

    decoder = NaiveDecoder(args.encoder_dim, args.stride)
    model = EncoderDecoder(conf_model, decoder)
    print(model)
    model = model.to(local_rank)
    if DEVICE == 'cuda':
        model = DDP(model, device_ids=[local_rank])

    # print("number of params:", count_params(model))
    
    loss_fn = MaskedMSELoss()
    optimizer = optim.AdamW(model.parameters(), **optim_params)
 
    trainer = MaskedSSLTrainer(model=model, optimizer=optimizer, criterion=loss_fn,
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        device=local_rank, optim_params=optim_params, net_params=lstm_params, exp_num=exp_num, log_path=log_path,
                        exp_name="conformer_ssl", max_iter=1000)
    

    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank, early_stopping=15, only_p=False, best='loss', conf=True)

    
    output_filename = f'{log_path}/exp{exp_num}/lstm_attn.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)

    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fit.png") 


    
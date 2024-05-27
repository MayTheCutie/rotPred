import sys
from os import path
import warnings
import torch
import torch.optim as optim

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)

from dataset.dataloader import *
from nn.models import *
from nn.train import *
from nn.lr_scheduler import WarmupLRScheduler
from util.utils import *
from transforms import *
from Astroconf.Train.utils import init_train
from Astroconf.utils import Container, same_seeds


from sklearn.model_selection import train_test_split
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import glob

local = False
root_dir = '.' if local else '/data/lightPred'

data_folder = f"{root_dir}/data"

log_path = '../logs/simsiam' if local else '/data/logs/simsiam'

yaml_dir = f'{root_dir}/Astroconf/'

exp_num = 18

num_epochs = 400

b_size = 128

dur = 720

cad = 30

num_qs = 8

DAY2MIN = 24*60

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    os.mkdir(f'{log_path}/exp{exp_num}')

# net_params = {"backbone": {'seq_len': 4370,
#     'd_model': 128,
#     'n_heads': 8,
#    'e_layers': 5,
#     'enc_in':1,
#     'dropout':0.3,
#      'c_out':4,
#      'ssl': True,}}
lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_classes': 4,
        'num_layers': 5,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}

optim_params = {"lr": 5e-5, 'weight_decay': 1e-4}


kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')

try:
    kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
except TypeError:
    pass
kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
kepler_df['longest_consecutive_qs_indices'] = kepler_df['qs'].apply(find_longest_consecutive_indices)
kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]


train_df, val_df = train_test_split(kepler_df, test_size=0.1, random_state=1234)

prot_df = pd.read_csv('/data/lightPred/tables/kepler_inference_astroconf_exp45.csv')

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":
    print(DEVICE)
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    gpus_per_node = torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)

    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")

    print(f"logdir: ,{log_path}/exp{exp_num}")
    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
    print("args : ", vars(args))

    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)), MovingAvg(13), Shuffle(), PeriodNorm(num_ps=10), 
                        RandomTransform([AddGaussianNoise(sigma=0.0001),Mask(0.2), Identity()]),
                             Detrend(), ACF(), Normalize('std'), ToTensor(), ])
    train_ds = KeplerDataset(data_folder, path_list=None, df=train_df, prot_df=prot_df, t_samples=10000,
     transforms=transform, target_transforms=transform)
    val_ds = KeplerDataset(data_folder, path_list=None, df=val_df, prot_df=prot_df, t_samples=10000,
     transforms=transform, target_transforms=transform)

    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    slurm_cpus_per_task = int(slurm_cpus_per_task)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=b_size, sampler=train_sampler, collate_fn=kepler_collate_fn,
                                                num_workers=slurm_cpus_per_task, pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_dl = DataLoader(val_ds,  batch_size=b_size, sampler=val_sampler,  collate_fn=kepler_collate_fn,
                                                num_workers=slurm_cpus_per_task, pin_memory=True)

    for i in range(4):
        fig, axes = plt.subplots(1, 2)
        x,y,mask, mask_y, info, info_y = train_ds[i]

        print(x.shape, y.shape, info['random_transform'], info_y['random_transform'])

    # for i, (x1,x2) in enumerate(val_dl):
    #     print(x1.shape, x2.shape)
    #     if i == 100:
    #         break

    print("train size: ", len(train_ds), "val size: ", len(val_ds))
    conf_model, _, scheduler, scaler = init_train(args, local_rank)
    conf_model.pred_layer = nn.Identity()
    backbone = LSTM_DUAL_CLS(conf_model, encoder_dims=args.encoder_dim, lstm_args=lstm_params,
                         ssl=True)
    backbone.pred_layer = nn.Identity()
    # backbone.conf_layer = nn.Identity()


    model = SimSiam(backbone)
    # state_dict = torch.load(f'{log_path}/exp{exp_num}/simsiam_astroconf.pth', map_location=torch.device('cpu'))
    # new_state_dict = OrderedDict()
    # for key, value in state_dict.items():
    #     while key.startswith('module.'):
    #         key = key[7:]
    #     new_state_dict[key] = value
    # state_dict = new_state_dict
    # model.load_state_dict(state_dict)
    
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # print(model)
    print("number of params:", sum(p.numel() for p in model.parameters() if p.requires_grad))




    optimizer = optim.Adam(model.parameters(), **optim_params)
    # print("optimizer: ", optimizer)

    scheduler = WarmupLRScheduler(optimizer, warmup_steps=1000)

    # print("scheduler: ", scheduler)

    data_dict = {'dataset': train_ds.__class__.__name__,
                   'transforms': transform,  'batch_size': b_size,
     'num_epochs':num_epochs, 'checkpoint_path': f'{log_path}/exp{exp_num}', 'loss_fn':'siamse loss',
     'model': 'astroconf', 'optimizer': optimizer.__class__.__name__,
     'data_folder': data_folder, }
    
    with open(f'{log_path}/exp{exp_num}/data_params.yml', 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


    trainer = SiameseTrainer(model=model, optimizer=optimizer, criterion =None,
                        scheduler=None, train_dataloader=train_dl, val_dataloader=val_dl,
                            device=local_rank, optim_params=optim_params, net_params=args, exp_num=exp_num, log_path=log_path,
                            exp_name="simsiam_astroconf", max_iter=150)
    print("trainer: ", trainer)
    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank, early_stopping=15)

    output_filename = f'{log_path}/exp{exp_num}/sims_astroconf.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)

    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fit.png")


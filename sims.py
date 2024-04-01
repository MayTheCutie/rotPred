import sys
from os import path
import warnings
import torch
import torch.optim as optim

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)    

from lightPred.augmentations import *
from lightPred.dataloader import *
from lightPred.models import *
from lightPred.train import *
from lightPred.utils import *
from lightPred.transforms import *
from lightPred.Astroconf.Train.utils import init_train
from lightPred.Astroconf.utils import Container, same_seeds


from sklearn.model_selection import train_test_split
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import glob


data_folder = "/data/lightPred/data"

log_path = '/data/logs/simsiam'

yaml_dir = '/data/lightPred/Astroconf/'

exp_num = 13

num_epochs = 400

b_size = 16

dur = 720

cad = 30 

num_qs = 8

DAY2MIN = 24*60

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        'num_classes': 2,
        'num_layers': 5,
        'predict_size': 128,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}

optim_params = {"lr": 5e-5, 'weight_decay': 1e-4}

# samples_list = list_files_in_directory(data_folder)
samples_list = [file_name for file_name in glob.glob(os.path.join(data_folder, '*')) if not os.path.isdir(file_name)]

# samples_list = os.listdir(data_folder)
# print("before split: ", samples_list[:10000])
kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')
try:
    kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
except TypeError:
    pass
kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]


train_df, val_df = train_test_split(kepler_df, test_size=0.2, random_state=1234)
# print("after split:" ,train_list[:1000], val_list[:10])

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

if __name__ == "__main__":

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

    args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])
    print("args : ", vars(args))

    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)), moving_avg(49)])
    train_ds = TimeSsl(data_folder, path_list=None, df=train_df, acf=True, return_raw=True,
     ssl_tf=DataTransform_TD_bank, t_samples=None, transforms=transform,)
    val_ds = TimeSsl(data_folder, path_list=None, df=val_df, acf=True, return_raw=True,
     ssl_tf=DataTransform_TD_bank, t_samples=None, transforms=transform)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=b_size, sampler=train_sampler, \
                                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True)
    val_dl = DataLoader(val_ds,  batch_size=b_size, shuffle=True, \
                                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
                                    
    for i in range(4):
        x,y = train_ds[i]
        print(x.shape, y.shape)  

    # for i, (x1,x2) in enumerate(val_dl):
    #     print(x1.shape, x2.shape)
    #     if i == 100:
    #         break

    print("train size: ", len(train_ds), "val size: ", len(val_ds))
    conf_model, _, scheduler, scaler = init_train(args, local_rank)
    conf_model.pred_layer = nn.Identity()
    backbone = LSTM_DUAL(conf_model, encoder_dims=args.encoder_dim, **lstm_params)
    backbone.pred_layer = nn.Identity()


    model = SimSiam(backbone)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print(model)
    print("number of params:", sum(p.numel() for p in model.parameters() if p.requires_grad))


    optimizer = optim.Adam(model.parameters(), **optim_params)
    # print("optimizer: ", optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, verbose=True, factor=0.1)

    # print("scheduler: ", scheduler)


    trainer = SiameseTrainer(model=model, optimizer=optimizer, criterion =None,
                        scheduler=scheduler, train_dataloader=train_dl, val_dataloader=val_dl,
                            device=local_rank, optim_params=optim_params, net_params=args, exp_num=exp_num, log_path=log_path,
                            exp_name="simsiam_astroconf", max_iter=2000)
    print("trainer: ", trainer)
    fit_res = trainer.fit(num_epochs=num_epochs, device=local_rank, early_stopping=15)

    output_filename = f'{log_path}/exp{exp_num}/sims_astroconf.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)

    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{log_path}/exp{exp_num}/fit.png") 
    


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim
import sys
from os import path
import torch.distributed as dist

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)   
from lightPred.timeDetr import TimeSeriesDetrModel
from lightPred.timeDetrLoss import SetCriterion, HungarianMatcher, cxcy_to_cxcywh
from lightPred.Astroconf.Train.utils import init_train
from lightPred.Astroconf.utils import Container, same_seeds
from lightPred.Astroconf.Model.models import AstroDecoder
from lightPred.dataloader import *
from lightPred.utils import *
# from lightPred.train import *
from lightPred.transforms import *
from lightPred.models import LSTM_ATTN, SpotNet
import optuna
import yaml
from tqdm import tqdm

# torch.manual_seed(1234)

# warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

exp_num = 0

log_path = '/data/logs/spotNet'

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

yaml_dir = '/data/lightPred/Astroconf/'

Nlc = 50000

test_Nlc = 5000

CUDA_LAUNCH_BLOCKING='1'


# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
samples = os.listdir(os.path.join(data_folder, 'simulations'))
idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]
train_list, val_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

test_idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc))+1) for idx in range(test_Nlc)]

b_size = 8

num_epochs = 2

cad = 30

DAY2MIN = 24*60

dur = 360

max_iter = 400

val_iter = 100

# class_labels = ['Period', 'Decay Time', 'Cycle Length']
class_labels = ['Inclination']

if torch.cuda.current_device() == 0:
    if not os.path.exists(log_path):
        os.makedirs(log_path)


lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_classes': len(class_labels)*2,
        'num_layers': 5,
        'predict_size': 128,
        'seq_len': int(dur/cad*DAY2MIN),
        'stride': 4}

encoder_architectures = [
        ["mhsa_pro", "conv", "mhsa_pro"],
        ["mhsa_pro", "conv", "conv"],
        ["mhsa_pro", "mhsa_pro", "conv"],
        ["mhsa_pro", "conv"]
    ]

decoder_architectures = [
    ["mhsa_decoder", "conv", "mhsa_decoder"],
        ["mhsa_decoder", "conv", "conv"],
        ["mhsa_decoder", "mhsa_decoder", "conv"],
        ["mhsa_decoder", "conv"],
        ["mhsa_decoder"]
    ]
    

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def get_spot_dict(spot_arr):
        spot_arr = spot_arr*180/np.pi
        bs, _,_ = spot_arr.shape
        idx = [spot_arr[b,0,:] != 0 for b in range(bs)]
        res = []
        for i in range(bs):
            spot_dict = {'boxes': cxcy_to_cxcywh(spot_arr[i, :, idx[i]], 1, 1).transpose(0,1).to(spot_arr.device),
                        'labels': torch.ones((spot_arr[i, :, idx[i]].shape[-1]), device=spot_arr.device).long()}
            res.append(spot_dict)
        return res

def objective(trial):
        args = Container(**yaml.safe_load(open(f'{yaml_dir}/default_config.yaml', 'r')))
        args.load_dict(yaml.safe_load(open(f'{yaml_dir}/model_config.yaml', 'r'))[args.model])

         # Sample one of the model architectures
        encoder_idx = trial.suggest_int('encoder_idx', 0, len(encoder_architectures) - 1)
        encoder = encoder_architectures[encoder_idx]
        decoder_idx = trial.suggest_int('decoder_idx', 0, len(decoder_architectures) - 1)
        decoder = decoder_architectures[decoder_idx]
        encoder_dim = trial.suggest_int("hidden_dim", 64, 256, 64)
        num_layers = trial.suggest_int("num_layers", 1, 6)
        num_heads = trial.suggest_int("num_heads", 4, 8, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        stride = trial.suggest_int("stride", 4, 24, 4)
        kernel_size = trial.suggest_int("kernel_size", 3, 7, 2)
        args.encoder = encoder
        args.decoder = decoder
        args.encoder_dim = encoder_dim
        args.decoder_dim = encoder_dim
        args.num_layers = num_layers
        args.num_heads = num_heads
        args.dropout_p = dropout
        args.stride = stride
        args.kernel_size = kernel_size

        # num_queries = trial.suggest_int("num_queries", 300, 600, 100)
        num_queries = 250
        conformer_enc, _, scheduler, scaler = init_train(args, DEVICE)
        conformer_enc.pred_layer = nn.Identity()
        conformer_dec = AstroDecoder(args)
        lstm_model = LSTM_ATTN(**lstm_params)
        model = SpotNet(conformer_enc, args.encoder_dim, conformer_dec,
         num_queries, lstm_model, num_classes=lstm_model.num_classes)

        kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
        transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                          KeplerNoiseAddition(noise_dataset=None, noise_path='/data/lightPred/data/noise',
                          transforms=kep_transform),                         
                            moving_avg(49)])
        train_dataset = TimeSeriesDataset(data_folder, train_list, transforms=transform, prepare=False, acf=True,
                                        spots=True, return_raw=True, init_frac=0.2, labels=class_labels)
        val_dataset = TimeSeriesDataset(data_folder, val_list, transforms=transform, prepare=False, acf=True,
                                        spots=True, return_raw=True, init_frac=0.2, labels=class_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=b_size)
        val_dataloader = DataLoader(val_dataset, batch_size=b_size)

        lr = trial.suggest_float("lr", 1e-5,1e-3)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

        # ce_weight = trial.suggest_float("ce_weight", 0.1, 1)
        # bbox_weight = trial.suggest_float("bbox_weight", 0.1, 1)
        # eos_val = trial.suggest_float("eos_val", 0.1, 0.5)
        ce_weight = 1
        bbox_weight = 5
        eos_val = 0.5
        weight_dict = {'loss_ce': ce_weight, 'loss_bbox': bbox_weight, 'loss_giou': 1}
        eos = eos_val
        losses = ['labels', 'boxes', 'cardinality']
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        spots_loss = SetCriterion(1, matcher, weight_dict, eos, losses=losses, device=DEVICE)
        att_loss = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        eta = 0.5

        
        model.to(DEVICE)
        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            model.train()
            t_loss = 0
            v_loss = 0
            pbar = tqdm(train_dataloader, total=max_iter)
            for i, (x,y,_,_) in enumerate(pbar):
                # print(i, x.shape, y.shape)
                if i > max_iter:
                    break
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y = y.unsqueeze(-1)
                x, spots_arr = x[:, :-2, :], x[:, -2:, :]
                tgt_spots = get_spot_dict(spots_arr)
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    out_spots, y_pred = model(x1, x2)
                else:
                    out_spots, y_pred = model(x.unsqueeze(-1))
                y_pred, conf_pred = y_pred[:, :len(class_labels)], y_pred[:, len(class_labels):]
                conf_y = torch.abs(y - y_pred)
                att_loss_val = att_loss(y_pred, y)
                att_loss_val += att_loss(conf_pred, conf_y)

                spots_loss_dict = spots_loss(out_spots, tgt_spots)
                weight_dict = spots_loss.weight_dict
                spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
                # print(out_att.shape, y[:,0].shape)
                loss = eta*spot_loss_val + (1-eta)*att_loss_val
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
                pbar.set_description(f"train loss: {spot_loss_val.item(), att_loss_val.item()}")          
            train_loss.append(t_loss / max_iter)

            model.eval()
            pbar = tqdm(val_dataloader, total=val_iter)
            for i, (x,y,_,_) in enumerate(pbar):
                with torch.no_grad():
                    if i > val_iter:
                        break
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    y = y.unsqueeze(-1)
                    x, spots_arr = x[:, :-2, :], x[:, -2:, :]
                    tgt_spots = get_spot_dict(spots_arr)
                    if x.shape[1] == 2:
                        x1, x2 = x[:, 0, :], x[:, 1, :]
                        out_spots, y_pred = model(x1, x2)
                    else:
                        out_spots, y_pred = model(x.unsqueeze(-1))
                    y_pred, conf_pred = y_pred[:, :len(class_labels)], y_pred[:, len(class_labels):]
                    conf_y = torch.abs(y - y_pred)
                    att_loss_val = att_loss(y_pred, y)
                    att_loss_val += att_loss(conf_pred, conf_y)

                    tgt_spots = get_spot_dict(spots_arr)
                    spots_loss_dict = spots_loss(out_spots, tgt_spots)
                    weight_dict = spots_loss.weight_dict
                    spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
                    v_loss += eta*spot_loss_val + (1-eta)*att_loss_val
                    pbar.set_description(f"val loss: {spot_loss_val.item(), att_loss_val.item()}")
            v_loss /= val_iter
            val_loss.append(v_loss)
            trial.report(v_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        torch.cuda.empty_cache()
        return val_loss[-1]

if __name__ == "__main__":

    study = optuna.create_study(study_name='spotnet', storage='sqlite:////data/optuna/spotnet.db', load_if_exists=True)
    study.optimize(lambda trial: objective(trial), n_trials=100)
    print('Device: ', DEVICE)
    print("Best trial:")
    trial = study.best_trial

    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))


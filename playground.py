import os
import sys
from os import path
import math
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf as A
from collections import OrderedDict
from pyts.image import GramianAngularField
import shutil
from scipy.signal import find_peaks, peak_prominences, peak_widths
from matplotlib.lines import Line2D
import butterpy as bp
from scipy.signal import savgol_filter as savgol





import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)
# from lightPred.datasets.simulations import TimeSeriesDataset
from lightPred.dataloader import *
# from lightPred.augmentations import *
from lightPred.utils import *
# from lightPred.optim import NoamOpt, ScheduledOptim
# from lightPred.Informer2020.models.attn import HwinAttentionLayer, patchify, FullAttention
# from lightPred.Informer2020.models.encoder import HwinEncoderLayer, Encoder, ConvLayer

from lightPred.models import *

from lightPred.train import *
from lightPred.sampler import DistributedSamplerWrapper
from lightPred.transforms import *
from lightPred.period_analysis import analyze_lc, analyze_lc_kepler
from lightPred.timeDetr import TimeSeriesDetrModel
from lightPred.timeDetrLoss import TimeSeriesDetrLoss, SetCriterion, HungarianMatcher
from lightPred.analyze_results import read_csv_folder
from lightPred.augmentations import DataTransform_TD_bank


import sys
from butterpy import Surface


# from lightPred.autoformer.layers import series_decomp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device is ', DEVICE)


Nlc = 50000

test_Nlc = 5000

max_p, min_p = 60, 0

max_inc, min_inc = 90, 0

dur = 1000 # Duration in days

cad = 30 # cadence in minutes

DAY2MIN = 24 * 60

time = np.arange(0, dur, cad / DAY2MIN)

data_folder_local = r"C:\Users\ilaym\Desktop\kepler/data/butter/data_cos"
data_folder = "/data/butter/data_cos"
table_path  = "/data/lightPred/Table_1_Periodic.txt"
kois_table_path = "/data/lightPred/kois.csv"
kepler_data_folder = "/data/lightPred/data"

# kepler_df = create_kepler_df(data_folder, table_path)
# kois_df = create_kepler_df(data_folder, kois_table_path)
# train_df, test_df = train_test_split(kepler_df, test_size=0.1, random_state=42, shuffle=True)

idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc))+1) for idx in range(test_Nlc)]

all_samples_list = [file_name for file_name in glob.glob(os.path.join(kepler_data_folder, '*')) if not os.path.isdir(file_name)]



def test_time_augmentations():
    ssl_tf = DataTransform_TD_bank
    dur = 90
    kepler_df = multi_quarter_kepler_df('lightPred/data/', table_path=None, Qs=np.arange(3, 17))
    try:
        kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
    except TypeError:
        pass
    kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
    kepler_df['consecutive_qs'] = kepler_df['qs'].apply(
        consecutive_qs)  # Calculate length of longest consecutive sequence
    kepler_df['longest_consecutive_qs_indices'] = kepler_df['qs'].apply(find_longest_consecutive_indices)
    transform = Compose([MovingAvg(kernel_size=49), RandomCrop(int(dur / cad * DAY2MIN)),
                         Normalize(norm='median')])
    dataset = KeplerDataset(data_folder, path_list=None, df=kepler_df,
                            t_samples=int(dur / cad * DAY2MIN), transforms=transform)
    for i in range(10):
        print(i)
        fig, ax = plt.subplots(1, 2)
        x, y, _, _ = dataset[i]
        print(x.shape)
        x_tf = ssl_tf(x.unsqueeze(0))
        ax[0].plot(x)
        ax[1].plot(x_tf)
        plt.savefig(f'/data/tests/augment_{i}.png')
        plt.show()
        if i == 10:
            break

def test_peak_height_ratio(data_folder, num_samples):
    lc_path = os.path.join(data_folder, 'simulations')
    csv_path = os.path.join(data_folder, 'simulation_properties.csv')
    labels_df = pd.read_csv(csv_path)
    idx_list = [f'{idx:d}'.zfill(int(np.log10(test_Nlc)) + 1) for idx in range(test_Nlc)]
    ratios = []
    ps = []
    incs = []
    max_lats = []
    for i, idx_s in enumerate(idx_list[:num_samples]):
        clean_idx = remove_leading_zeros(idx_s)
        row = labels_df.iloc[clean_idx]
        p, inc = row['Period'], row['Inclination']*180/np.pi
        max_lats.append(row['Spot Max'])
        x = pd.read_parquet(os.path.join(lc_path, f"lc_{idx_s}.pqt")).values
        p , lags, xcf, peaks = analyze_lc_kepler(x[:,1], prom=0.01)
        if len(peaks) > 2:
            peak_height_ratio = xcf[peaks[0]] / xcf[peaks[1]]
        else:
            peak_height_ratio = 0
        if i % 100 == 0:
            plt.plot(lags, xcf)
            plt.plot(lags[peaks], xcf[peaks], 'o')
            plt.title(f'peak_height_ratio: {peak_height_ratio:.2f}, p: {p:.2f}, i: {inc:.2f}')
            plt.savefig(f'/data/tests/peaks_{i%100}.png')
            plt.close()
            print(peak_height_ratio, p, inc)
        ratios.append(peak_height_ratio)
        ps.append(p)
        incs.append(inc)
    plt.hist(ratios, 100)
    plt.xlabel('peak height ratio')
    plt.ylabel('count')
    plt.savefig('/data/tests/peak_height_ratio_hist.png')
    plt.close()
    plt.scatter(ps, np.log(ratios))
    plt.xlabel('period')
    plt.ylabel('peak height ratio')
    plt.savefig('/data/tests/peak_height_ratio_vs_period.png')
    plt.close()
    plt.scatter(incs, np.log(ratios))
    plt.xlabel('inclination')
    plt.ylabel('peak height ratio')
    plt.savefig('/data/tests/peak_height_ratio_vs_inclination.png')
    plt.close()
    plt.scatter(max_lats, np.log(ratios))
    plt.xlabel('max latitude')
    plt.ylabel('peak height ratio')
    plt.savefig('/data/tests/peak_height_ratio_vs_max_lat.png')
    plt.close()

    

def sun_like_analysis():
    T_sun = 5777
    dur = 720
    kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')
    try:
        kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
    except TypeError:
        pass
    kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
    kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
    kepler_df = kepler_df[kepler_df['consecutive_qs'] >= 8]
    results_df = read_csv_folder('/data/logs/kepler/exp52', filter_thresh=6)
    merged_df = pd.merge(kepler_df, results_df, on='KID', how='inner')
    print(len(merged_df))
    print(merged_df.columns)
    merged_df['T_sun'] = merged_df['Teff']/T_sun
    df_sun_like_M = merged_df[(merged_df['R'] > 0.9) & (merged_df['R'] < 1.1) & (merged_df['Teff'] > T_sun - 100)
     & (merged_df['Teff'] < T_sun + 100)]
    print(len(df_sun_like_M))
    transform = Compose([RandomCrop(int(dur / cad * DAY2MIN))])

    ds = KeplerDataset(data_folder, path_list=None,  df=df_sun_like_M, t_samples=int(dur/cad*DAY2MIN),\
                             skip_idx=0, num_qs=8,  transforms=transform)

    for i, (x,y,mask,info) in enumerate(ds):
        print(i)
        df_row = df_sun_like_M.iloc[info['idx']]
        predicted_p = df_row['predicted period']
        x = savgol(x, predicted_p*48 , 1, mode='mirror', axis=0)
        other_p, lags, xcf = analyze_lc_kepler(x.squeeze())
        print('x shape', x.shape, 'xcf shape', xcf.shape, "number of nans: ",
         np.sum(np.isnan(x.numpy())), np.sum(np.isnan(xcf)))
        if i % 1000 == 0:
            fig , axis = plt.subplots(1,2)
            axis[0].plot(x)
            axis[1].plot(lags, xcf)
            plt.title(f"predicted period: {predicted_p:.2f}, other periods: {other_p:.2f}")
            plt.savefig(f'/data/tests/sun_like_{i}.png')
            plt.close()
        if i >= 100:
            break

        





def create_period_normalized_samples(data_folder, num_samples, num_ps=20):
    lc_path = os.path.join(data_folder, 'simulations')
    csv_path = os.path.join(data_folder, 'simulation_properties.csv')
    norm_lc_path = os.path.join(data_folder, 'simulations_norm')
    if not os.path.exists(norm_lc_path):
        os.makedirs(norm_lc_path)
    labels_df = pd.read_csv(csv_path)
    idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc)) + 1) for idx in range(Nlc)]
    for i, idx_s in enumerate(idx_list[:num_samples]):
        if i % 1000 == 0:
            print(i)
        clean_idx = remove_leading_zeros(idx_s)
        row = labels_df.iloc[clean_idx]
        p = row['Period']
        x = pd.read_parquet(os.path.join(lc_path, f"lc_{idx_s}.pqt")).values
        norm_t, norm_f = period_norm(x, period=p, num_ps=num_ps)
        norm_df = pd.DataFrame(np.c_[norm_t, norm_f], columns=["time", "flux"])
        norm_df.to_parquet(os.path.join(norm_lc_path, f"lc_{idx_s}.pqt"))

def show_samples(num_samples):
    idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc)) + 1) for idx in range(Nlc)]
    dataset = TimeSeriesDataset(data_folder, idx_list, t_samples=None, norm='none', prepare=False,
                                spots=True, init_frac=0.2)
    time = np.arange(0, 800, cad / DAY2MIN)
    n_spots = []
    for i in range(num_samples):
        if i % 1000 == 0:
            print(i)
        x, y, _, _ = dataset[i]
        # print('xshape', x.shape)
        # ax[0].plot(time, x[:,0])
        spot_idx = torch.where(x[:,1] != 0)[0]
        n_spots.append(len(spot_idx))
        spots_arr = x[:, 1:]*180/np.pi
        # print("spots max day ", time[spot_idx[-1]])
        # spots_arr = spots_arr[spots_arr[:,0] > 90]
        spots_arr[:, 0] -= 90
        # ax[1].scatter(spots_arr[:, 1], spots_arr[:, 0])
        # ax[1].scatter(time, spots_arr[:, 0])
        # ax[0].set_title(f"p: {y[1].item():.2f}, inc: {y[0].item():.2f} nspots : {n_spots}")
        # ax[0].set_xlabel('time(days)')
        # ax[0].set_ylabel('flux')
        # ax[1].set_xlabel('time(days)')
        # ax[1].set_ylabel('spots latitude (deg)')
        # ax[1].set_ylim(0, 100)
        # plt.tight_layout()
        # plt.savefig(rf'C:\Users\ilaym\Desktop\kepler\/data/tests/sample_{i}.png')
        # plt.close()
    print(np.mean(n_spots), np.std(n_spots), np.max(n_spots), np.min(n_spots))
    plt.hist(n_spots, 100)
    plt.xlabel('number of spots')
    plt.ylabel('count')
    plt.savefig('/data/tests/n_spots_hist.png')



def cxcy_to_cxcywh(arr, w, h):
    x, y = arr[0], arr[1]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=1)

def get_spot_dict(spot_arr):
    bs, _,_ = spot_arr.shape
    idx = [spot_arr[b,:, 0] != 0 for b in range(bs)]
    res = []
    for i in range(bs):
        spot_dict = {'boxes': cxcy_to_cxcywh(spot_arr[i, idx[i], :], 1, 1),
                     'labels': torch.ones(len(spot_arr[i, idx[i], :])).long()}
        res.append(spot_dict)
    return res

def test_timeDetr():
    dur = 90
    print(os.getcwd())
    data_folder = "../butter/data_cos"
    transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    train_dataset = TimeSeriesDataset(data_folder, idx_list, transforms=transform, prepare=False, acf=False,
                                      spots=True, init_frac=0.2)

    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    model = TimeSeriesDetrModel(input_dim=1, hidden_dim=64, num_layers=4, num_heads=4,
     dropout=0.3, num_classes=2, num_angles=4, num_queries=500)
    model = model.to(DEVICE)
    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    eos = 0.2
    losses = ['labels', 'boxes', 'cardinality']
    spots_loss = SetCriterion(1, matcher, weight_dict, eos,  losses = losses)
    att_loss = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,)
    pbar = tqdm(train_dl)
    model.train()
    for i, (x,y,_,_) in enumerate(pbar):
        # print(i, x.shape, y.shape)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x, spots_arr = x[:,:, 0], x[:,:, 1:3]
        tgt_spots = get_spot_dict(spots_arr)
        out_spots, out_att = model(x.unsqueeze(-1))
        spots_loss_dict = spots_loss(out_spots, tgt_spots)
        weight_dict = spots_loss.weight_dict
        spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
        # print(out_att.shape, y[:,0].shape)
        att_loss_val = att_loss(out_att.squeeze(), y[:, 0])
        tot_loss = spot_loss_val + att_loss_val
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        pbar.set_description(f'spot loss: {spot_loss_val.item()}, att loss: {att_loss_val.item()} tot loss: {tot_loss.item()}')
        # if i == 3:
        #     break

def get_spot_dict(spot_arr):
    bs, _,_ = spot_arr.shape
    idx = [spot_arr[b,0,:] != 0 for b in range(bs)]
    res = []
    for i in range(bs):
        spot_dict = {'boxes': cxcy_to_cxcywh(spot_arr[i, :, idx[i]], 1/360, 1/360).to(spot_arr.device),
                    'labels': torch.ones((spot_arr[i, :, idx[i]].shape[-1]), device=spot_arr.device).long()}
        res.append(spot_dict)
    return res

def get_spots_from_dict(spot_dict):
    res = []
    for i in range(len(spot_dict)):
        spots_box = spot_dict[i]['boxes']
        res.append(spots_box[:, :2].cpu().numpy())
    return np.array(res)
def test_spots_dataset():
    dur = 360
    data_folder = "../butter/data_cos"
    transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    train_dataset = TimeSeriesDataset(data_folder, idx_list, transforms=transform, prepare=False, acf=True,
                                      spots=True, return_raw=True, init_frac=0.2)
    for i in range(10):
        print(i)
        x, y, _, _ = train_dataset[i]
        x, spots_arr = x[:-2, :], x[-2:, :]
        spots_dict = get_spot_dict(spots_arr[None, :, :])
        spots_arr = get_spots_from_dict(spots_dict).squeeze()
        fig, ax = plt.subplots(1,2)
        ax[0].plot(x[0, :])
        ax[1].scatter(spots_arr[:, 1], spots_arr[:, 0])
        plt.savefig(f'../tests/spots_{i}.png')
        print(x.shape, y)
        spots_arr = x[:, 1:]
        spot_idx = torch.where(spots_arr[:,0] != 0)
        spots_arr = spots_arr[spot_idx]
        # spots_arr = spots_arr[spots_arr != 0, :]
        print(spots_arr.shape)


def read_spots_and_lightcurve(idx, data_folder):
    spots_dir  = os.path.join(data_folder, 'spots')
    lc_dir = os.path.join(data_folder, 'simulations')
    sample_idx = remove_leading_zeros(idx)
    x_lc = pd.read_parquet(os.path.join(data_folder, f"simulations/lc_{idx}.pqt")).values
    # spots is N,4 array with columns: [nday, lat, lon, bmax]
    x_spots = pd.read_parquet(os.path.join(data_folder, f"spots/spots_{idx}.pqt")).values
    y = pd.read_csv(os.path.join(data_folder, 'simulation_properties.csv'), skiprows=range(1,sample_idx+1), nrows=1)
    print(x_lc.shape, y.shape, x_spots.shape)
    print(x_spots.columns)

def init_distritubuted_mode():
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size, gpus_per_node

def inference(model, src_input, max_len, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # 1. Encode the input sequence
        enc_output = model.encode(src_input)

        # 2. Initialize the decoder input (start with <SOS>)
        decoder_input = torch.tensor([[1]], device=device)  # Replace <SOS> with the actual start token index

        # 3. Decode iteratively
        for _ in range(max_len):
            # 4. Generate output at the current time step
            dec_output = model.decode(enc_output, decoder_input)

            # 5. Use argmax to get the most likely token
            top_token = torch.argmax(F.softmax(dec_output, dim=-1), dim=-1)[:, -1]

            # 6. Check for the end-of-sequence token
            # if top_token.item() == <EOS>:  # Replace <EOS> with the actual end token index
            #     break

            # 7. Append the generated token to the decoder input
            decoder_input = torch.cat([decoder_input, top_token.unsqueeze(1)], dim=-1)

        # 8. Post-process the output (remove special tokens)
        generated_sequence = decoder_input.squeeze().tolist()

    return generated_sequence

def test_denoiser():
    dur = 90
    kepler_data_folder = "/data/lightPred/data"
    non_ps = pd.read_csv('/data/lightPred/tables/non_ps.csv')
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5,6,7])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==4]
    kep_transform = Compose([RandomCrop(int(dur/cad*DAY2MIN))])
    merged_df = pd.merge(kepler_df, non_ps, on='KID', how='inner')
    noise_ds = KeplerDataset(kepler_data_folder, path_list=None, df=merged_df,
    transforms=kep_transform, acf=False, norm='none')
    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN))])

    data_folder = "/data/butter/data_cos"
    train_dataset = CleanDataset(data_folder, idx_list, transforms=transform, noise_ds=noise_ds)
    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    denoiser_params = {
        'enc_in': 1,
        'dec_in': 1,
         'c_out': 1,
          'out_len': int(dur/cad*DAY2MIN),
          'dropout': 0.2}
    # denoiser = Informer(**denoiser_params)
    denoiser = EncoderDecoder()
    state_dict = torch.load('/data/logs/clean/exp2/encoder_decoder.pth')
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
        
    denoiser.load_state_dict(new_state_dict)
    denoiser = denoiser.to(DEVICE)
    denoiser.eval()
    denoiser_length = int(90/cad*DAY2MIN)
    for i, (x_noies,y,_,_) in enumerate(train_dl):
        print(i)
        x_clean = denoiser(x_noies.to(DEVICE)).squeeze().detach().cpu()
        plt.plot(x_noies[0])
        plt.plot(x_clean[0])
        plt.savefig(f'/data/tests/denoiser_{i}.png')
        plt.close()
        if i == 10:
            break

def create_noise_dataset():
    kepler_data_folder = "/data/lightPred/data"
    non_ps = pd.read_csv('/data/lightPred/tables/non_ps.csv')
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5,6,7,8,9,10,11,12,13,14,15,16])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==13]
    kepler_df.to_csv('/data/lightPred/tables/kepler_noise_4567.csv', index=False)
    # kepler_df = pd.read_csv('/data/lightPred/tables/kepler_noise_4567.csv')
    kep_transform = Compose([RandomCrop(int(dur/cad*DAY2MIN))])
    merged_df = pd.merge(kepler_df, non_ps, on='KID', how='inner')
    noise_ds = KeplerDataset(kepler_data_folder, path_list=None, df=merged_df,
    transforms=kep_transform, acf=False, norm='none')
    print("creating noise dataset from ", len(noise_ds), " samples")
    for i in range(len(noise_ds)):
        print(i)
        x,y,_,info = noise_ds[i]
        kid = info['KID']
        x = x/x.median() - 1
        # print(x.shape, info)
        np.save(f'/data/lightPred/data/noise/{kid}.npy', x.squeeze())
        # plt.plot(x.squeeze())
        # plt.savefig(f'/data/tests/noise_{i}.png')
        # plt.close()
        # if i == 10:
        #     break

def add_noise_mult(x, noise_dataset, min_ratio, max_ratio):
    std = x.std()
    idx = np.random.randint(0, len(noise_dataset))
    x_noise,_,_,noise_info = noise_dataset[idx]
    noise_std = np.random.uniform(std*min_ratio, std*max_ratio)
    x_noise = (x_noise - x_noise.mean()) 
    x = x*x_noise.squeeze().numpy()
    return x, noise_std, std 

def add_noise(x, noise_dataset):
    idx = np.random.randint(0, len(noise_dataset))
    x_noise,_,_,noise_info = noise_dataset[idx]
    x_noise = x_noise/x_noise.median() - 1
    x = x/x.median()
    x = x + x_noise.squeeze().numpy()
    return x, x_noise.squeeze()




def test_kepler_noise():
    rank, world_size, gpus_per_node = init_distritubuted_mode()
    non_ps = pd.read_csv('/data/lightPred/tables/non_ps.csv')
    dur = 90
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5,6,7])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==4]
    merged_df = pd.merge(kepler_df, non_ps, on='KID', how='inner')
    kep_transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    print(len(merged_df), len(kepler_df))
    noise_ds = KeplerDataset(kepler_data_folder, path_list=None, df=merged_df,
     t_samples=None, transforms=kep_transform, acf=False, norm='none')

    # denoiser_params = {
    #     'enc_in': 1,
    #     'dec_in': 1,
    #      'c_out': 1,
    #       'out_len': int(dur/cad*DAY2MIN),
    #       'dropout': 0.2}
    # denoiser = Informer(**denoiser_params)
    # state_dict = torch.load('/data/logs/clean/exp2/encoder_decoder.pth')
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
    #     new_key = k.replace('module.', '')
    #     new_state_dict[new_key] = v
        
    # denoiser.load_state_dict(new_state_dict)
    # denoiser = denoiser.to(DEVICE)
    # denoiser.eval()
    # denoiser_length = int(10/cad*DAY2MIN)
    
    # transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)), KeplerNoise(noise_ds, min_std=0.0005, max_std=0.01),
    #                      moving_avg(kernel_size=49)])
    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)),
                          ])

    data_folder = "/data/butter/data_cos"
    train_dataset = TimeSeriesDataset(data_folder, idx_list[:1000], t_samples=None, norm='none',
                                       transforms=transform, acf=False, wavelet=False, dur=dur, kep_noise=noise_ds,
                                       freq_rate=1/48, init_frac=0.4, prepare=False)
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    stds_noise = []
    stds = []
    ratios = []
    steps = []
    for i, (x,y,_,info) in enumerate(train_dl):
        print(i)
        # x_noised, noise_std, std = add_noise_mult(x[0], noise_ds, 0.02, 0.05)
        x_noised_add, noise_add = add_noise(x, noise_ds) 
        stds_noise.append(noise_add.std()+1)
        stds.append(x.std())
        time = np.arange(0, dur, cad / DAY2MIN)
        fig, ax = plt.subplots(1,2)
        ax[0].plot(time, x[0], label='raw')
        ax[1].plot(time, x_noised_add[0], label='noisy_mult')
        plt.title(f"noise std:  {x_noised_add.std()}, orig std: , {x[0].std()} ")
        # ax[1].set_title(f"noise_mult - noise ratio: {noise_std/std:.4f}")
        # ax[2].plot(time, noise_add)
        # ax[2].set_title(f"addition noise")
        # ax[3].plot(time, x_noised_add)
        # ax[3].set_title(f"noise_add")
        plt.tight_layout()


        plt.savefig(f'/data/tests/kepler_noise_{i}_x_norm.png')
        plt.clf()
        # if i % 10 == 0:
        #     plt.figure()
        #     plt.plot(x[0])
        #     plt.title(f"noise ratio: {info['noise_std'][0]/info['std'][0]:.4f}")
        #     plt.savefig(f'/data/tests/kepler_noise_2_{i}.png') 
        #     plt.close('all')       
        # stds_noise.extend(info['noise_std'].tolist())
        # stds.extend(info['std'].tolist())
        # ratios.extend((info['noise_std']/info['std']).tolist())
        # print(info['noise_std']/info['std'])
        # steps.append(train_dataset.step)
        # train_dataset.step += 1
        # x= savgol(x, 5, 1, mode='mirror')
        # x[0] = x[1]
        # ax[0].plot(x, label='raw')
        # # ax[0].plot(x_noise, label='noise')
        # plt.title(f"noise std: {info['noise_std']:.4f}, orig std: {info['std']:.4f}")
        # plt.savefig(f'/data/tests/kepler_noise_{i}.png')
        # plt.close()

        # x_clean = x.clone()
        # tgt = torch.ones(1, denoiser_length, 1).to(DEVICE)
        # x_clean = denoiser(x.unsqueeze(0).unsqueeze(-1).to(DEVICE), x.unsqueeze(0).unsqueeze(-1).to(DEVICE)).squeeze().detach().cpu()
        # ax[1].plot(x_clean, label='denoised')
        # plt.legend()
        # plt.savefig(f'/data/tests/kepler_noise_denoised_{i}.png')
        # out = torch.zeros(1,0,1)
        # for s in range(dur//10):
        #     patch = x[s*denoiser_length:(s+1)*denoiser_length]
        #     patch = torch.tensor(patch).unsqueeze(0).unsqueeze(-1)
        #     patch = patch.to(DEVICE)
        #     tgt = torch.zeros(1, denoiser_length, 1).to(DEVICE)
        #     print(patch.shape, tgt.shape)
        #     out = denoiser(patch, tgt)
        #     out = out.squeeze().detach().cpu()
        #     print(out.shape, x_clean.shape)
        #     # plt.plot(patch)
        #     # plt.plot(out)
        #     # plt.savefig(f'/data/tests/kepler_noise_denoised_{i}_patch_{s}.png')
        #     # plt.close()            
        #     x_clean[s*denoiser_length:(s+1)*denoiser_length] = out
        #     if s:
        #         x_clean[s*denoiser_length] = x_clean[s*denoiser_length-1]
        # ax[1].plot(x_clean, label='denoised')
        # plt.legend()
        # plt.savefig(f'/data/tests/kepler_noise_denoised_{i}.png')
        # plt.close()
    fig, ax = plt.subplots(1,2)
    ax[0].hist(stds_noise, 100, label='noise std', histtype='step')
    ax[0].hist(stds, 100, label='orig std', histtype='step')
    ax[0].legend()
    # ax[1].hist(ratios, 100, label='noise std/orig std')
    # ax[1].legend()
    # plt.savefig('/data/tests/samples_std_hist_2.png')
    # plt.close()
    # plt.plot(steps, stds, 'o')
    # plt.xlabel('step')
    # plt.ylabel('noise std')
    plt.savefig('/data/tests/kepler_noise_std.png')


def test_wavelet():
    non_ps = pd.read_csv('/data/lightPred/tables/non_ps.csv')
    dur = 180
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5,6,7])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==4]
    merged_df = pd.merge(kepler_df, non_ps, on='KID', how='inner')
    kep_transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    print(len(merged_df), len(kepler_df))
    noise_ds = KeplerDataset(kepler_data_folder, path_list=None, df=merged_df,
     t_samples=None, transforms=kep_transform, acf=False, norm='none')
    
    transform = Compose([RandomCrop(int(dur/cad*DAY2MIN)), KeplerNoise(noise_ds, min_std=0.0005, max_std=0.01),
                         Moving(kernel_size=49)])
    data_folder = "/data/butter/data_cos"
    train_dataset = TimeSeriesDataset(data_folder, idx_list[:10], t_samples=None, norm='minmax',
                                       transforms=transform, acf=True, wavelet=True, dur=dur, kep_noise=noise_ds,
                                       freq_rate=1/4, init_frac=0.2)
    for i in range(10):
        print(i)
        idx = np.random.randint(0, len(train_dataset))
        x,y,_,_ = train_dataset[idx] 
        print(x.shape)
        period = np.arange(0, dur, 0.25)
        plt.plot(period,x[0])
        plt.plot(period,x[1])
        # ax[1].plot(A(x[0], nlags=len(x[0])))
        plt.title(f"p: {y[1].item()*60:.2f}, inc: {y[0].item()*np.pi/2:.2f}")
        plt.savefig(f'/data/tests/wavelet_{i}.png')
        plt.clf()
        if i == 10:
            break

def diffs():
    dur = 180
    data_folder = "/data/butter/data2/simulations"
    labels_folder = "/data/butter/data2"
    transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    Nlc = len(os.listdir(data_folder))
    idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

    incs = []
    lats = []
    mses = []
    fluxes = []
    for i in range(1000):
        inc = np.arccos(np.random.uniform(0, 1))
        p = np.random.uniform(1,50)
        max_lat = np.random.uniform(20,40)
        tau = 10 ** np.random.uniform(low=0, high=np.log10(10))
        print(tau)
        s = Surface(nlat=32)
        s.emerge_regions(max_lat=max_lat)
        # print(s.regions['bmax'].sum())
        lc = s.evolve_spots(incl=inc, period=p, tau_evol=tau)
        flux, time = lc.flux, lc.time
        flux = 1 - flux
        lc_arr = np.array([lc.time, flux]).T[int(0.4*len(lc.time)):]

        # idx = np.random.randint(0, len(idx_list))
        # sample_idx = remove_leading_zeros(idx_list[idx])
        # x = pd.read_parquet(os.path.join(data_folder, f"lc_{idx_list[idx]}.pqt")).values
        # x = x[int(0.4*len(x)):,:]
        # time, flux = x[:,0], x[:,1]
        # y = pd.read_csv(os.path.join(labels_folder, 'simulation_properties.csv'), skiprows=range(1,sample_idx+1), nrows=1)
        # p = y['Period'].values[0]
        # max_lat = y['Spot Max'].values[0]
        # inc = y['Inclination'].values[0]
        # df = 1 - flux
        # flux = 1 - df
        # lc_arr = np.array([time, flux]).T[int(0.4*len(time)):]

        lc, _, _ = transform(lc_arr, info=dict())
        t, flux = period_norm(lc, p, 10)
        flux_shifted = np.roll(flux, 1)
        diff = flux - flux_shifted
        diff[0] = diff[1]
        mse = np.sum((diff**2)/flux**2)
        mses.append(mse)
        incs.append(inc)
        lats.append(max_lat)
        fluxes.append(flux.mean())
    
    plt.scatter(incs, np.log(mses), c=lats)
    plt.colorbar(label='max_lat', )
    plt.ylabel(r'$\log{\sum{(\frac{(L(t)-L(t+1)}{L(t)})^2}}$')
    plt.xlabel('Inclination (Radians)')
    plt.savefig('/data/tests/period_vs_inc_log.png')
        
def period_norm(lc, period, num_ps, orig_freq=1/48):
    if len(lc.shape) == 1:
        flux = lc
        time = np.arange(0, len(flux)*orig_freq, orig_freq)
    else:
        time, flux = lc[:, 0], lc[:, 1]
    time = time - time[0]

    # Define the new time points and the desired sampling rate
    new_sampling_rate = period / 1000  
    new_time = np.arange(0, period * num_ps, new_sampling_rate)
    new_flux = np.interp(new_time, time, flux)
    t_norm = np.linspace(0, num_ps, num=len(new_flux))
    return t_norm, new_flux

def filter_samples():
    data_dir = "/data/lightPred/LightCurves/data/simulations"
    csv_path = "/data/lightPred/LightCurves/data/simulation_properties.csv"
    df = pd.read_csv(csv_path)
    filtered_df = pd.DataFrame(columns=df.columns)
    incs = []
    for sample in os.listdir(data_dir):
        # print(sample)
        try:
            simulation_number = int(sample.split('.')[0])
            print(simulation_number)
            
            row = df[(df['Simulation Number']==simulation_number).astype(bool)]
            inc = row['Inclination'].values[0]
            incs.append(inc)
        except:
            continue
    plt.hist(incs, 100)
    plt.savefig('/data/tests/incs_hist_lightcurves_data.png')
    #     if len(row) > 0:
    #         filtered_df = pd.concat([filtered_df, row])
    # filtered_df.to_csv("/data/lightPred/LightCurves/data2/simulation_properties_filtered.csv", index=False)
    # print(len(filtered_df), len(os.listdir(data_dir)))

def get_dispersion():
    import matplotlib as mpl
    np.random.seed(1234)
    time = np.arange(0, 1000, cad / DAY2MIN)
    fluxes = []
    areas = []
    images = []
    transform = Moving(49)
    incs = np.arange(0,100, 10)
    ratios = []
    mpl.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize=(10,8))
    for i in incs:
        spot_properties = bp.regions()
        print(spot_properties)
        spots = bp.Spots(
                spot_properties,
                incl=i)
        dF = spots.calc(time)
        flux = dF + 1
        flux =transform(torch.tensor(flux), info=dict())[0].numpy()
        ax.plot(time, flux, label=f'inc={i}', linewidth=3)
    plt.legend()
    plt.xlabel('Time(Days)', fontdict={'fontsize': 18})
    plt.ylabel('flux', fontdict={'fontsize': 18})
    ax.tick_params(axis='both', which='both', labelsize=12, width=2, length=6, direction='out', pad=5)
    plt.xlim(0,180)
    plt.legend(prop={'size': 14})
    plt.savefig('/data/tests/inclination_effect.png')
    plt.close()

def test_astroconformer():
    dur = 180
    transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    train_ds = TimeSeriesDataset(data_folder, idx_list[:1000], labels=['Spot Max'], t_samples=None, norm='std',
     transforms=transform, spectrogram=False)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    model = AstroConformerEncoder()
    model = model.to(DEVICE)
    for i, (x,y,_,_) in enumerate(train_dl):
        print(y)
        x = x.to(DEVICE)
        out = model(x)
        if i == 10:
            break

def test_spectrogram():
    spec = torchaudio.transforms.Spectrogram(n_fft=1000, win_length=4, hop_length=4)
    dur = 180
    time = np.arange(0, dur, cad / DAY2MIN)
    transform = Compose([Detrend(), RandomCrop(width=int(dur/cad*DAY2MIN))])
    train_ds = TimeSeriesDataset(data_folder, idx_list[:1000], t_samples=None, norm='std',
     transforms=transform, spectrogram=False)

    # kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5])
    # kepler_df = kepler_df[kepler_df['number_of_quarters']==2]
    # full_dataset = KeplerDataset(data_folder, path_list=None, df=kepler_df, t_samples=None, transforms=transform, acf=False, return_raw=False)
    for i, (x_spec,y,x_t,_) in enumerate(train_ds):
        print(x_spec.shape, x_t.shape)
        # spectrogram = spec(x)
        # print(spectrogram.shape, x.shape)
        # fig, ax = plt.subplots(1,2)
        # ax[0].plot(time, x)
        # ax[1].imshow(spectrogram.squeeze())
        # plt.title(f"p: {y[1].item():.2f}, inc: {y[0].item():.2f}")
        # plt.savefig(f'/data/tests/spectrogram_{i}.png')
        # plt.clf()
        if i == 20:
            break
    # for i, (x,y,_,_) in enumerate(full_dataset):
    #     spectrogram = spec(x)
    #     print(spectrogram.shape, x.shape)
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].plot(time, x[0])
    #     ax[1].imshow(spectrogram.squeeze())
    #     plt.title(f"p: {y[1].item():.2f}, inc: {y[0].item():.2f}")
    #     plt.savefig(f'/data/tests/spectrogram_kepler_{i}.png')
    #     plt.clf()
    #     if i == 5:
    #         break


def test_lagp_dataset():
    dur = 360
    time = np.arange(0, dur, cad / DAY2MIN)
    transform = Compose([Detrend(), Moving(kernel_size=49), RandomCrop(width=int(dur / cad * DAY2MIN))])
    train_ds = PlagDataset(data_folder, idx_list[:1000], dur=dur,lag_len=512,
     labels=None, t_samples=None, norm='std', transforms=transform, return_raw=True)
    for i, (x,y,_,_) in enumerate(train_ds):
        print(x.shape)
        # fig, ax = plt.subplots(1,2)
        # ax[0].plot(time, x[0])
        # ax[1].plot(time, x[1])
        # plt.title(f"p: {y['Period'].item()}, inc: {y['Inclination'].item()}")
        # plt.savefig(f'/data/tests/lagp_{i}.png')
        # plt.clf()
        if i == 10:
            break

def test_hdiff():
    def find_peaks_in_lag(x, p):
        peaks= find_peaks(x, prominence=0.1)[0]
        valleys = find_peaks(-x, prominence=0.1)[0]
        peaks_valleys = np.concatenate([peaks, valleys])
        if len(peaks) == 0:
            return None, None, None, None
        first_peak = peaks[0]
        # print(first_peak, len(x), p, p//2*DAY2MIN//(cad))
        p1_indices = np.arange(first_peak, len(x), int(p*DAY2MIN/cad))
        p2_indices = np.arange(first_peak, len(x), int(p/2*DAY2MIN//(cad)))
        dist_p1 = peaks[None] - p1_indices[:,None]
        matches_p1 = np.argmin(np.abs(dist_p1), axis=1) 

        dist_p2 = peaks_valleys[None] - p2_indices[:,None] 
        matches_p2 = np.argmin(np.abs(dist_p2), axis=1)
        # print(matches_p1, time[matches_p1])
        p1 = x[peaks[matches_p1]]
        p2 = x[peaks_valleys[matches_p2]]
        t1 = time[peaks[matches_p1]]
        t2 = time[peaks_valleys[matches_p2]]
        return p1, p2, t1, t2
    dur = 360
    time = np.arange(0, dur, cad / DAY2MIN)
    transform = Compose([Detrend(), Moving(kernel_size=49), RandomCrop(width=int(dur / cad * DAY2MIN))])
    train_ds = ACFDataset(data_folder, idx_list[:1000], labels=None, t_samples=None, norm='std', transforms=transform, return_raw=True)
    # train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    diffs_t = []
    diffs_a = []
    incs_t = []
    incs_a = []
    for i, (x,y,_,_) in enumerate(train_ds):
        print(i)
        # print(x.shape)
        fig, axes = plt.subplots(1,2)
        p = y['Period'].item()
        # difference of x in half period
        p1, p2, t1, t2 = find_peaks_in_lag(x[0,:], p)
        if p1 is None:
            continue
        diffs1_t = p1[:-1] - p1[1:]
        diffs2_t = p2[:-1] - p2[1:]
        diffs_t.append([diffs1_t.mean(), diffs2_t.mean()])
        incs_t.append(y['Inclination'].item())
        p1_a, p2_a, t1_a, t2_a = find_peaks_in_lag(x[1,:], p)
        if p1_a is None:
            continue
        diffs1_a = p1_a[:-1] - p1_a[1:]
        diffs2_a = p2_a[:-1] - p2_a[1:]
        diffs_a.append([diffs1_a.mean(), diffs2_a.mean()])
        incs_a.append(y['Inclination'].item())
        # if i % 5 == 0:
        # axes[0].plot(diffs1_t, label='p1')
        axes[0].plot(diffs2_t, label='p2')
        # axes[0].plot(time,x[0])
        # axes[0].scatter(t1, p1, c='r', label='p')
        # axes[0].scatter(t2, p2, c='b', marker='*', label='p//2')
        # axes[0].scatter(time[peaks_t], x[0,peaks_t], marker='*',c='g', label='peaks')
        # axes[0].scatter(time[valleys_t], x[0,valleys_t], marker='*',c='y', label='valleys')
        axes[0].set_title("lightcurve")

        
        # axes[1].plot(diffs1_a, 'o', label='p1')
        axes[1].plot(diffs2_a,  label='p2')
        # axes[1].plot(time,x[1])
        # axes[1].scatter(t1_a, p1_a, c='r', label='p')
        # axes[1].scatter(t2_a, p2_a, c='b', marker='*', label='p//2')
        # axes[1].scatter(time[peaks_a], x[1,peaks_a], marker='*',c='g', label='peaks')
        # axes[1].scatter(time[valleys_a], x[1,valleys_a], marker='*',c='y', label='valleys')
        axes[1].set_title("acf")
        plt.legend()
        plt.suptitle(f"p: {y['Period'].item()}, inc: {y['Inclination'].item()}")
        plt.savefig(f'/data/tests/hdiff_{i}.png')
        plt.close("all")
        # hdiff = np.abs(p1 - p2)
        # print(hdiff.shape)
        if i >= 10:
            break
    # diffs_t = np.array(diffs_t)
    # diffs_a = np.array(diffs_a)
    # plt.plot(incs_t, diffs_t[:,0], 'o', label='t1')
    # plt.plot(incs_a, diffs_a[:,0], 'o', label='a1')
    # plt.plot(incs_t, diffs_t[:,1], 'o', label='t2')
    # plt.plot(incs_a, diffs_a[:,1], 'o', label='a2')
    # plt.legend()
    # plt.savefig('/data/tests/hdiff.png')
    # plt.close()

        
def test_depth_width(): 
    dur = 180
    data_folder = "/data/butter/data7"
    transform = Compose([Detrend(), RandomCrop(width=int(dur/cad*DAY2MIN))])                 
    train_dataset = TimeSeriesDataset(data_folder, idx_list[:10000], t_samples=None, norm='minmax',
     transforms=transform, labels=None)
    ratios = []
    incs = []
    max_lat = []
    cycle_lengths = []
    for i in range(400):
        idx = np.random.randint(0, len(train_dataset))
        x,y,_,info = train_dataset[idx]
        peaks = find_peaks(x, prominence=0.1)[0]
        prominences = peak_prominences(x, peaks)[0]
        highest_idx = np.argsort(prominences)[-10:]
        prominences = prominences[highest_idx]
        widths = peak_widths(x, peaks[highest_idx], rel_height=0.1)[0]
        ratio = prominences / widths
        ratios.append(ratio.mean())
        incs.append(y['Inclination'].values[0])
        max_lat.append(y['Spot Max'].values[0])
        cycle_lengths.append(y['Cycle Length'].values[0])
        if i % 100 == 0:
            print(i)
            fig, ax = plt.subplots(1,2)
            ax[0].plot(x)
            ax[1].plot(A(x, nlags=len(x)))
            plt.title(f"p: {y['Period'].values[0]*60:.2f}, inc: {y['Inclination'].values[0]*90:.2f}")
            plt.savefig(f'/data/tests/depth_width_{i}.png')
    plt.close('all')
    print(len(incs), len(ratios), len(max_lat), type(incs[0]), type(ratios[0]), type(max_lat[0]))
    shapes = ['o' if max_lat[i] > 45 else 'x' for i in range(len(max_lat))]
    labels = ['>45' if max_lat[i] > 45 else '<45' for i in range(len(max_lat))]
    plt.scatter(incs, ratios, c=max_lat)
    # for i in range(len(shapes)):
    #     print(cycle_lengths[i])
    #     plt.scatter(incs[i], ratios[i], c=cycle_lengths[i],  marker=shapes[i], alpha=0.7)
    plt.xlabel('inclination')
    plt.ylabel('prominence/width')
    # marker_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='max_lat > 45'),
    #                           Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=10,
    #                                  label='max_lat < 45')]
    # plt.legend(handles=marker_legend_elements) 
    plt.colorbar(label='max_lat')
    plt.savefig('/data/tests/depth_width.png')    
        
def read_fits(filename):
    # print("reading fits file: ", filename)
    with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          meta = hdulist[0].header
          # print(header)
    df = pd.DataFrame(data=binaryext)
    x = df['PDCSAP_FLUX']
    time = df['TIME'].values
    # teff = meta['TEFF']
    return x,time, meta

def acf_on_winn():
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path='/data/lightPred/tables/win2017_acf.csv', Qs=[4,5])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==2]
    print(len(kepler_df))
    df = pd.read_csv('/data/lightPred/tables/win2017.csv')
    acf_periods = []
    for index, row in kepler_df.iterrows():
        # print(row)
        # print(row['data_file_path'][0])
        x, time, meta = read_fits(row['data_file_path'][0])
        x = fill_nan_np(x, interpolate=True)
        # print(len(x))
        p, lags, xcf = analyze_lc_kepler(x, len(x))
        if index % 10 == 0:
            fig, ax = plt.subplots(1,2)
            ax[0].plot(time, x)
            ax[1].plot(lags, xcf)
            plt.title(f"p: {p:.2f}")
            plt.savefig(f'/data/tests/win_acf_{index}.png')
        acf_periods.append(p)
        # print(p)
        df.at[index, 'acf_period_2q'] = p
    print(df.columns)
    df.to_csv('/data/lightPred/tables/win2017_acf.csv', index=False)


def test_koi_sample(kids, names, df_path=None):
    if df_path is not None:
        kepler_df = pd.read_csv(df_path)
        kids = kepler_df.dropna()['KID'].values
        names = kepler_df['kepler_name'].values
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    # kepler_df = kepler_df[kepler_df['number_of_quarters']==7]
    for kid, name in zip(kids, names):
        row = kepler_df[kepler_df['KID']==int(kid)]
        if len(row['data_file_path'].values):
            shutil.copyfile(row['data_file_path'].values[0][0], f'/data/tests/{name}.fits')
            print(row)

def test_tfc():
    dur = 180
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==2]
    transform = Compose([Moving(kernel_size=49), RandomCrop(int(dur / cad * DAY2MIN))])
    ds = TFCKeplerDataset(data_folder, path_list=None, df=kepler_df.iloc[:1000], ssl_tf=DataTransform_TD_bank,
     t_samples=transform, transforms=None, acf=True, return_raw=False)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    for i, (x_t,x_f,info) in enumerate(dl):
        print(x_t.shape, x_f.shape)
        sampling_rate = 1/30   # Adjust the sampling rate based on your data
        frequency_axis = torch.fft.fftfreq(x_t.size(-1), d=sampling_rate).numpy()
        period_axis = 1 / frequency_axis  

        fig, ax = plt.subplots(1,2)
        ax[0].plot(x_t[0].squeeze())
        ax[0].set_title(f"time domain")
        ax[1].plot(period_axis, x_f[0].squeeze())
        ax[1].set_title(f"frequency domain")
        # plt.suptitle(f"p: {info['Period'].item()*60:.2f}")
        plt.savefig(f'/data/tests/tfc_{i}.png')
        if i == 10:
            break


def test_gaf():
    dur = 180
    transform = Compose([Detrend()])
                        
    train_dataset = TimeSeriesDataset(data_folder, idx_list[:1000], t_samples=1024, norm='std', transforms=transform)
    for i in range(5):
        idx = np.random.randint(0, len(train_dataset))
        fig, ax = plt.subplots(1,2)
        x,y,_,info = train_dataset[i]
        # t= np.linspace(0,dur,len(x.squeeze()))
        # f = interp1d(x, t, fill_value="extrapolate")
        # new_t = np.linspace(0, dur, 1024)
        # x = f(new_t)
        print(x.shape)
        print(x.shape)
        gaf = GramianAngularField()
        x_gram = gaf.transform(x.reshape(1,-1)).squeeze()
        # print(x_gram.shape)
        ax[0].imshow(x_gram, cmap='gray')
        ax[1].plot(x)
        plt.title(f"p: {y[1]*60:.2f}, inc: {y[0]*90:.2f}")
        plt.savefig(f'/data/tests/gaf_{i}.png')



def test_sims():
    dur = 180
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==2]
    train_df, val_df = train_test_split(kepler_df, test_size=0.01, random_state=1234)
    transform = Compose([Moving(kernel_size=25), RandomCrop(int(dur / cad * DAY2MIN))])
    # train_ds = TimeSsl(data_folder, path_list=None, df=train_df, ssl_tf=DataTransform_TD_bank, t_samples=None, transforms=transform, acf=True, return_raw=False)
    val_ds = TimeSsl(data_folder, path_list=None, df=val_df, ssl_tf=DataTransform_TD_bank, t_samples=transform, transforms=None, acf=True, return_raw=False)

    val_dl = DataLoader(val_ds, batch_size=4, \
                                    )
    x,x2 = val_ds[0]
    print(x.shape)
    for i, (x1,x2) in enumerate(val_dl):
        print(x1.shape, x2.shape)
        if i == 100:
            break

def test_conv_block():
    x = torch.randn(10, 1, 9600)
    in_channel = 1
    conv_list = nn.Sequential(ConvBlock(1,32, kernel_size=3, stride=2, padding=1, dropout=0.1),
                                ConvBlock(32,128, kernel_size=3, stride=2, padding=1, dropout=0.1),
                                ConvBlock(128,256, kernel_size=3, stride=2, padding=1, dropout=0.1))

    # conv_block = ConvBlock(1, 256, kernel_size=3, stride=2, padding=1, dropout=0.1)
    out = conv_list(x)
    print(out.shape)

def test_quantiles():
    model = LSTM_ATTN_QUANT()
    model = model.to(DEVICE)
    dummy_input = torch.randn(10, 1024, 1)
    dummy_input = dummy_input.to(DEVICE)
    quantiles = model(dummy_input)
    print(quantiles.shape)

def test_sampler():
    # /
    # print(unique, counts)
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    

    train_dataset = TimeSeriesDataset(data_folder, idx_list[:5000], t_samples=None, labels=['Inclination', 'Period'],
                                       norm='std', transforms=None, acf=False, prepare=True) 
    # train_dl = DataLoader(train_dataset, batch_size=128, num_workers=0)
    # incs = torch.zeros(0)
    weights = train_dataset.weights
    # weights = torch.ones(len(train_dataset))
    plt.hist(weights, 100)
    plt.savefig('/data/tests/weights_hist.png')
    plt.clf()
    # for i, (x,y,_,_) in enumerate(train_dl):
    #     print(i)
    #     incs = torch.cat((incs, y[:,0]*90), dim=0)
    #     w = 1/counts[(y[:,0]*90).squeeze().numpy().astype(np.int16)]
    #     weights = torch.cat((weights, torch.tensor(w)), dim=0)
    # print(weights.shape)
    # plt.hist(incs.squeeze(), 80)
    # plt.savefig('/data/tests/incs_hist.png')
    # plt.clf()

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    sampler = DistributedSamplerWrapper(sampler=WeightedRandomSampler(weights, len(weights)))
    train_dl_weighted = DataLoader(train_dataset, batch_size=128, num_workers=0, sampler=sampler)
    incs = torch.zeros(0)
    for i, (x,y,_,_) in enumerate(train_dl_weighted):
        print(i)
        incs = torch.cat((incs, y[:,0]*180/np.pi), dim=0)
    plt.hist(incs.squeeze(), 80)
    plt.savefig('/data/tests/incs_hist_weighted.png')

def test_consistency():
    _dur = 180
    model, net_params, _ = load_model('/data/logs/lstm_attn/exp29',
     LSTM_ATTN, distribute=True, device=DEVICE, to_ddp=False)
    model.eval()
    transform = Compose([Moving(49), Slice(0, int(_dur / cad * DAY2MIN))])
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=table_path, Qs=[4,5])
    kepler_df_2 = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5])
    # kepler_df = kepler_df.sample(frac=1)
    kepler_df = kepler_df[kepler_df['number_of_quarters']==2]
    kepler_df_2 = kepler_df_2[kepler_df_2['number_of_quarters']==2]
    kepler_df.reset_index(drop=True, inplace=True)
    kepler_df_2.reset_index(drop=True, inplace=True)
    print(kepler_df_2.iloc[-5:]['KID'])
    print(len(kepler_df), len(kepler_df_2))
    print(kepler_df_2.where(kepler_df_2['KID']==757280).dropna().index[0])


    train_dataset = KeplerLabeledDataset(data_folder, path_list=None, df=kepler_df,
     t_samples=None, norm='std', transforms=None, mask_prob=0.1)
    train_dataset_2 = KeplerDataset(data_folder, path_list=None, df=kepler_df_2,
        t_samples=None, norm='std', transforms=None, mask_prob=0.1)
    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    diffs_i = []
    diffs_p = []
    x2,_,_,info = train_dataset_2[3]
    print(info['KID'])
    print(len(train_dataset), len(train_dataset_2))

    for i in range(100):
        idx1 = np.random.randint(0, len(train_dataset))
        x1,y, _,_,info = train_dataset[idx1]
        # try:
        idx2 = kepler_df_2.where(kepler_df_2['KID']==info['KID']).dropna().index[0]
        # if len(idx2) > 0:
        #     pass
        #     # print(f"sample {info['KID']} found in both datasets - idx1: {idx1}, idx2: {idx2.index[0]}")
        # else:
        #     print(f"sample {info['KID']} not found in both datasets")
        print(idx2)
        x1 = x1.unsqueeze(0).to(DEVICE)
        y1 = model(x1)
        x2,_,_,_ = train_dataset_2[idx2]
        x2 = x2.unsqueeze(0).to(DEVICE)
        y2 = model(x2)
        diff = torch.abs(y1-y2)
        print(diff)
        # diffs_i.append(diff[:,0].item()*90)
        # diffs_p.append(diff[:,1].item()*60)
        # except:
        #     continue
    # plt.hexbin(np.arange(len(diffs_i)), diffs_i, gridsize=100, cmap='inferno', mincnt=1)
    # plt.colorbar(label='Density')
    # plt.savefig('/data/tests/consistency_i.png')
    # plt.hexbin(np.arange(len(diffs_p)), diffs_p, gridsize=100, cmap='inferno', mincnt=1)
    # plt.colorbar(label='Density')
    # plt.savefig('/data/tests/consistency_p.png')

        # print(idx, info['KID'], kepler_df_2.iloc[idx2]['KID'])
      

    


def test_patchify():
    win_size = 10
    x = torch.randn(10, 100, 50)
    plt.plot(x[0,:,0])
    for i in range(5):
        x_patches, indices = patchify(x,win_size)
        # print( torch.arange(0,x.shape[1], win_size).view(1,-1,1,1))
        indices += torch.arange(0,x.shape[1], win_size).view(1,-1,1,1)
        # print(indices[0,:,:,0])
        plt.plot(indices[0,:,0,0], x_patches[0,:,0], 'o')
        plt.savefig('/data/tests/patchify.png')
    for i in range(0, 100, 10):
        plt.axvline(i)

def test_hwin():
    x = torch.randn(10, 9600).to(DEVICE)
    # encdoer_layer = HwinEncoderLayer(HwinAttentionLayer(FullAttention(),128, 4, 24, 4),
    # 128, 512, 0.1).to(DEVICE)
    e_layers = 3
    window_size = 6
    n_windows = 4
    base_seq_len = 9600
    shrink_factor =base_seq_len/(np.sum([base_seq_len/(window_size*2**i) for i in range(n_windows)]))
    print("shrink factor: ", shrink_factor)
    base_d_model = 32
    model = HwinEncoder(enc_in=1, c_out=4, seq_len=base_seq_len, d_model=base_d_model, e_layers=e_layers, d_ff=512,
     predict_size=64, window_size=window_size).to(DEVICE)
    # encoder = Encoder(
    #         [
    #             HwinEncoderLayer(
    #                 HwinAttentionLayer(FullAttention(False), d_model=base_d_model*2**l, n_heads=4, window_size=window_size,
    #                  n_windows=n_windows),
    #                 base_d_model*2**l,
    #                 512,
    #                 dropout=0.1,
    #                 activation='gelu'
    #             ) for l in range(e_layers)
    #         ],
    #         [
    #             ConvLayer(
    #                 base_d_model*2**(l), c_out=base_d_model*2**(l+1)
    #             ) for l in range(e_layers-1)
    #         ],
    #         norm_layer=torch.nn.LayerNorm(base_d_model*2**(e_layers-1))
    #     ).to(DEVICE)
    # out, attn = encoder(x)
    # attn = HwinAttentionLayer(9600, FullAttention(),128, 4, 48, 4).to(DEVICE)
    # out, _, indices = attn(x,x,x)
    out = model(x)
    print(out.shape)

def test_masked_ssl():
    _dur = 180

    transform = Compose([Moving(49), RandomCrop(width=int(_dur / cad * DAY2MIN))])
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5])
    # kepler_df = kepler_df.sample(frac=1)
    kepler_df = kepler_df[kepler_df['number_of_quarters']==2]


    train_dataset = KeplerDataset(data_folder, path_list=None, df=kepler_df,
     t_samples=None, norm='std', transforms=transform, mask_prob=0.1)
    train_dl = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    loss_fn = nn.MSELoss()
    model = Informer(enc_in=1, dec_in=1, c_out=int(_dur/cad*DAY2MIN), seq_len=int(_dur/cad*DAY2MIN),
                label_len=int(_dur/cad*DAY2MIN), out_len=1)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

    trainer = MaskedSSLTrainer(model=model, optimizer=optimizer, criterion=loss_fn,
                    train_dataloader=train_dl, device=DEVICE)
    for i in range(1):
        loss, acc = trainer.train_epoch(device=DEVICE)                
    # for i in range(5):
    #     idx = i
    #     x,masked_x,inv_mask, info = train_dataset[idx]
    #     print(x.shape, masked_x.shape, inv_mask.shape)
    #     fig, ax = plt.subplots(1,3)
    #     ax[0].plot(x)
    #     ax[1].plot(masked_x)
    #     ax[1].set_ylim(x.min(), x.max())
    #     ax[2].plot(inv_mask, 'o')
    #     ax[0].set_title("original")
    #     ax[1].set_title("masked")
    #     ax[2].set_title("mask")
    #     plt.tight_layout()
    #     plt.savefig(f'/data/tests/masked_ssl_{i}.png')
    #     plt.clf()

def test_kepler():
    kepler_df = multi_quarter_kepler_df(kepler_data_folder, table_path=None, Qs=[4,5])
    kepler_df = kepler_df[kepler_df['number_of_quarters']==2]
    transforms = Compose([Moving(kernel_size=49), RandomCrop(int(dur / cad * DAY2MIN))])
    # kepler_df = kepler_df.sample(frac=1)
    full_data = KeplerDataset(data_folder, path_list=None, t_samples=None, transforms=transforms, df=kepler_df)
    # dl = DataLoader(full_data, batch_size=4, shuffle=False, num_workers=0)
    # idx = 19955
    # x,x_masked,_,info = full_data[idx]
    # print(x.shape, x_masked.shape, info)
    for i in range(10):
        x,x_masked,_,info = full_data[i]
        print(x.isnan().sum())
        plt.plot(x.squeeze())
        plt.title(f"{info['KID']}")
        plt.savefig(f'/data/tests/kepler_q_{i}.png')
        plt.clf()
        print(x.shape)
    # try:
    #     for i, (x,y,_,info) in enumerate(dl):
    #         print(x.shape, y.shape)
    #         if i == 1000:
    #             break
    # except Exception as e:
    #     print(e)
    #     print(info)
    #     print(i)

    # for i in range(5):
    #     x,x_masked,_,info = full_data[i]
    #     print(x.isnan().sum())
    #     plt.plot(x)
    #     plt.title(f"{info['KIC']}")
    #     plt.savefig(f'/data/tests/kepler_nulti_q_{i}.png')
    #     plt.clf()
    #     print(x.shape)
    # full_dataset = KeplerDataset(data_folder, all_samples_list, t_samples=None, transforms=None, acf=True)
    # for i in range(10):
    #     x = full_dataset[i]
    


def test_decomp():
    dur = 180
    transform = Compose([Detrend(), RandomCrop(width=int(dur/cad*DAY2MIN))
                        ])
    decomp = series_decomp(25)
    ds = KeplerDataset(test_df, t_samples=None)
    # ds = TimeSeriesDataset(data_folder2, idx_list, t_samples=None, transforms=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    for i, (x,y) in enumerate(dl):
        fig, ax = plt.subplots(1,3)
        res, mean = decomp(x.unsqueeze(-1))
        # print(y['Period'])
        ax[0].plot(np.linspace(0,91,len(x.squeeze())),x.squeeze())
        ax[1].plot(np.linspace(0,91,len(x.squeeze())),res.squeeze())
        ax[2].plot(np.linspace(0,91,len(x.squeeze())),mean.squeeze())
        ax[0].set_title("original")
        ax[1].set_title("residual")
        ax[2].set_title("mean")
        plt.suptitle(f"p: {y['Period'].item()*60:.2f}")
        plt.tight_layout()
        plt.savefig(f'/data/tests/decomp_{i}.png')
        plt.clf()
        if i == 10:
            break
    


def remove_leading_zeros(s):
    """
    Remove leading zeros from a string of numbers and return as integer
    """
    # Remove leading zeros
    s = s.lstrip('0')
    # If the string is now empty, it means it was all zeros originally
    if not s:
        return 0
    # Convert the remaining string to an integer and return
    return int(s)

def test_butter():
    sims = os.listdir(f'{data_folder}/simulations')
    props = pd.read_csv(f'{data_folder}/simulation_properties.csv')
    incs = []
    periods = []
    samples = set()
    print("number of files: ", len(sims))
    for s in sims:
        sample_num = remove_leading_zeros(s.split('_')[1].split('.')[0])
        if sample_num in samples:
            print("duplicate sample number: ", sample_num)
        samples.add(sample_num)
        p = props.iloc[sample_num]
        incs.append(p['Inclination'])
        periods.append(p['Period'])
    incs = np.array(incs)
    periods = np.array(periods)
    plt.hist(incs, 80)
    plt.savefig('/data/tests/incs_data2.png')
    plt.clf()
    plt.hist(periods, 60)
    plt.savefig('/data/tests/periods_data2.png')
    plt.clf()



def test_butter2():
    data_folder = "/data/butter/data_cos"
    dur = 180
    transform = Compose([RandomCrop(width=int(dur/cad*DAY2MIN))])
    train_ds = TimeSeriesDataset(data_folder, idx_list[:1000], t_samples=None, norm='none',
    transforms=transform, spectrogram=False)
    for i in range(10):
        idx = np.random.randint(0, len(train_ds))
        x,y,_,_ = train_ds[idx]
        time = np.arange(0,dur,cad/DAY2MIN)
        print(x.shape, time.shape)
        fig, ax = plt.subplots(1,2)
        ax[0].plot(time, x)
        ax[1].plot(time, A(x, nlags=len(x)))
        plt.title(f"p: {y[1]*60:.2f}, inc: {y[0]*90:.2f}")
        plt.savefig(f'/data/tests/butter_{i}.png')
        plt.clf()
    # for idx in samples_list:
    #     x = pd.read_parquet(os.path.join(lc_path, f"lc_{idx}.pqt")).values
    #     print(x.shape, x[:,1].max(), x[:,1].min())
    #     xcf = A(x[:,1], nlags=len(x[:,1]))
    #     fig, ax = plt.subplots(1,2)
    #     ax[0].plot(x[:,0], x[:,1])
    #     ax[1].plot(xcf)
    #     plt.title(f"sample {idx}")
    #     plt.savefig(f'/data/tests/sample_{idx}.png')
    #     plt.clf()


def test_warmup():
    model = torch.nn.Linear(128, 10)
    # optimizer = NoamOpt(512, 2000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 1, 128, 2000)
    lr_arr = []
    for i in range(10000):
        if i % 1000 == 0:
            print(i)
        optimizer.zero_grad()
        loss = torch.sum(model(torch.randn(100, 128)).pow(2))
        loss.backward()
        optimizer.step()
        lr_arr.append(optimizer._optimizer.param_groups[0]['lr'])
    plt.plot(lr_arr)
    plt.savefig('/data/tests/warmup2.png')


def plot_sample(ax, sample, t, inc, p):
        amp = mean_amp(sample)
        inc = inc * (max_inc - min_inc) + min_inc
        p = p * (max_p - min_p) + min_p
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t, sample)
        ax.set_title(f"inc: {inc:.2f}, p: {p:.2f}, amp: {amp:.2f}")
        return ax

def mean_amp(x, k=3):
    peaks, _ = find_peaks(x, distance=48)  # `distance` parameter sets the minimum separation between peaks

    minima, _ = find_peaks(-x, distance=48)  

    if len(peaks) == 0 or len(minima) == 0:
        return np.nan

    # Find the indices of the nearest minimum for each peak
    peak_indices = np.repeat(peaks, len(minima))
    minima_indices = np.tile(minima, len(peaks))
    peak_distances = np.abs(peak_indices - minima_indices)
    minima_indices_for_peaks = minima_indices[np.argmin(peak_distances.reshape(len(peaks), -1), axis=1)]

    # Calculate the relative amplitudes using vectorized operations
    relative_amplitudes = x[peaks] - (x[minima_indices_for_peaks])

    top_k = np.argsort(relative_amplitudes)[-k:]
    bottom_k = np.argsort(relative_amplitudes)[:k]

    # Calculate the mean amplitude
    mean_amplitude = np.nanmean(relative_amplitudes)
    min_amplitude = np.nanmin(relative_amplitudes[bottom_k])
    max_amplitude = np.nanmax(relative_amplitudes[top_k])
    return mean_amplitude, min_amplitude, max_amplitude


def test_amp():
    idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

    # train_list, val_list = train_test_split(idx_list, test_size=0.2, random_state=1234

    transform = Compose([ Detrend() ])

    train_dataset = TimeSeriesDataset(data_folder, idx_list, seq_len=None, norm='std', transforms=transform)

    num_samples = 5000

    incs = []

    amps = []
    min_amps = []
    max_amps = []

    k=10

    for i in range(num_samples):
        if i % 1000 == 0:
            print(i)
        # idx = np.random.randint(0, len(train_dataset))
        x,y,_,_ = train_dataset[i]
        inc = y[0]*90
        p = y[1]
        incs.append(inc)
        amp, min_amp, max_amp = mean_amp(x[:,0], k=k)
        amps.append(amp)
        min_amps.append(min_amp)
        max_amps.append(max_amp)
    plt.scatter(incs, amps)
    plt.savefig(f'/data/tests/mean_amp.png')
    plt.clf()
    plt.scatter(incs, min_amps)
    plt.savefig(f'/data/tests/min_amp.png')
    plt.clf()
    plt.scatter(incs, max_amps)
    plt.savefig(f'/data/tests/max_amp.png')
    plt.clf()

def test_clen():
    idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
    target_path = f'{data_folder}/simulation_properties.csv'
    lc_path = f'{data_folder}/simulations'
    for i in range(10):
        idx = np.random.randint(0, len(idx_list))
        sample_idx = remove_leading_zeros(idx_list[idx])
        x = pd.read_parquet(os.path.join(lc_path, f"lc_{idx_list[idx]}.pqt")).values
        y = pd.read_csv(target_path, skiprows=range(1,sample_idx+1), nrows=1)
        clen = y['Cycle Length'].item()
        print(clen)
        plt.plot(x[:,0], x[:,1])
        plt.title(f"spots cycle (years): {clen}")
        plt.savefig(f'/data/tests/clen_{i}.png')
        plt.clf()
    



def butter():

    idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]

    # train_list, val_list = train_test_split(idx_list, test_size=0.2, random_state=1234

    transform = Compose([ Detrend() ])

    train_dataset = TimeSeriesDataset(data_folder, idx_list, seq_len=None, norm='std', transforms=transform)


    num_samples = 1000
    incs = np.zeros(num_samples)
    ps = np.zeros(num_samples)
    xs = np.zeros((num_samples, int(dur*DAY2MIN/cad)))
    for i in range(num_samples):
        x,y,_,_ = train_dataset[i]
        incs[i] = y[0]
        ps[i] = y[1]
        xs[i] = x[:,0]
    amps = []
    incs_thres = np.arange(0,1,0.1)
    for i in range(len(incs_thres) - 1):
        idx = np.where(np.logical_and(incs > incs_thres[i], incs < incs_thres[i+1]))[0]
        samples = xs[idx]
        fig, ax = plt.subplots(2,3, figsize=(10,10))
        fig2, ax2 = plt.subplots(2,3, figsize=(10,10))
        amps.append([mean_amp(x) for x in samples])
        for j in range(6):
            sample_idx = np.random.randint(0, len(samples))
            sample = samples[sample_idx]
            acf = A(sample, nlags=len(sample))
            plot_sample(ax[j//3, j%3], sample, time, incs[idx[sample_idx]], ps[idx[sample_idx]])
            plot_sample(ax2[j//3, j%3], acf, time, incs[idx[sample_idx]], ps[idx[sample_idx]])
        fig.suptitle(f"incs between {incs_thres[i]:.2f} and {incs_thres[i+1]:.2f}")
        fig.tight_layout()
        fig.savefig(f'/data/tests/samples_{incs_thres[i]:.2f}_{incs_thres[i+1]:.2f}.png')
        fig2.suptitle(f"incs between {incs_thres[i]:.2f} and {incs_thres[i+1]:.2f} - ACF")
        fig2.tight_layout()
        fig2.savefig(f'/data/tests/samples_acf_{incs_thres[i]:.2f}_{incs_thres[i+1]:.2f}.png')
        print(incs_thres[i], samples.shape)
        plt.close()
    plt.close('all')
    min_value = min([min(arr) for arr in amps])
    max_value = max([max(arr) for arr in amps])
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_amps = [np.nanmean(amp) for amp in amps]
    for i, amp in enumerate(amps):
        ax.hist(amp, 20, label=f"{incs_thres[i]:.2f} - {incs_thres[i+1]:.2f}: {amp}", histtype='step', range=(min_value, max_value))
        ax.legend()
        ax.set_xlim(min_value, max_value)
    plt.savefig(f'/data/tests/inc_amps.png')
    plt.clf()
    print(len(incs_thres), len(mean_amps))
    plt.scatter(incs_thres[:-1], mean_amps)
    plt.savefig(f'/data/tests/inc_amps_scatter.png')
    plt.clf()


def koi():
    kepler_df = create_kepler_df(data_folder, kois_table_path)
    ds = KeplerDataset(kepler_df, t_samples=None)
    for i in range(10):
        x,y = ds[i]
        xcf = A(x, nlags=len(x))
        fig, ax = plt.subplots(1,2)
        ax[0].plot(np.linspace(0,len(x)*cad/DAY2MIN,len(x)),x)
        ax[1].plot(np.linspace(0,len(x)*cad/DAY2MIN,len(x)),xcf)
        plt.title(f"p: {y['Period']*60:.2f}")
        plt.savefig(f'/data/tests/koi_{i}.png')
        plt.clf()

if __name__ == "__main__":
    # butter()
    # koi()
    # test_warmup()
    # test_butter2()
    # test_decomp()
    # test_amp()
    # test_clen()
    # test_kepler()
    # test_patchify()
    # test_hwin()
    # test_butter2()
    # test_masked_ssl()
    # test_consistency()
    # test_sampler()
    # test_quantiles()
    # test_conv_block()
    # test_sims()
    # test_gaf()
    # test_tfc()
    # test_koi_sample(kids=['1160684', '1164584', '1995168', '2010137'], names=['noise1', 'noise2', 'noise3', 'noise4'])
    # test_koi_sample(kids=None, names=None, df_path='/data/lightPred/tables/tables.csv')
    # acf_on_winn()
    # test_depth_width()
    # test_hdiff()
    # test_lagp_dataset()
    # test_spectrogram()
    # test_astroconformer()
    # get_dispersion()
    # filter_samples()
    # diffs()
    # test_wavelet()
    # test_kepler_noise()
    # test_denoiser()
    # create_noise_dataset()
    # read_spots_and_lightcurve('00010', '/data/butter/data2')
    # test_spots_dataset()
    # show_samples(48000)
    # create_period_normalized_samples('/data/butter/data_cos_old', 50000, num_ps=20)
    # create_period_normalized_samples('/data/butter/data_sun_like', 50000, num_ps=20)
    # sun_like_analysis()
    # test_peak_height_ratio('/data/butter/test_cos_old', 1000,)
    test_time_augmentations()

    # test_timeDetr()





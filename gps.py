import numpy as np
from tqdm import tqdm

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from lightPred.wavelet import wavelet as wvl
from lightPred.transforms import *
from lightPred.utils import *
from lightPred.dataloader import *
from scipy.stats import linregress
torch.manual_seed(1234)
np.random.seed(1234)

# warnings.filterwarnings("ignore")

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cpu')

print('device is ', DEVICE)

print("gpu number: ", torch.cuda.current_device())

local = False

exp_num = 2

log_path = '/data/logs/gps' if not local else './logs/gps'

if not os.path.exists(f'{log_path}/exp{exp_num}'):
    try:
        print("****making dir*******")
        os.makedirs(f'{log_path}/exp{exp_num}')
    except OSError as e:
        print(e)

root_dir = 'lightPred' if local else '/data/lightPred'
# chekpoint_path = '/data/logs/simsiam/exp13/simsiam_lstm.pth'
# checkpoint_path = '/data/logs/astroconf/exp14'

data_folder = "/data/butter/data_aigrain2"

# dataset_name = 'kepler'
dataset_name = data_folder.split('/')[-1]


yaml_dir = f'{root_dir}/Astroconf/'

Nlc = 50000

test_Nlc = 5000

freq_rate = 1/48

alpha = 0.332


CUDA_LAUNCH_BLOCKING='1'


# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
# samples = os.listdir(os.path.join(data_folder, 'simulations'))
# idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]
idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
train_list, test_list = train_test_split(idx_list, test_size=0.1, random_state=1234)

dur=720


def complex_gradient(arr):
    # Separate real and imaginary parts
    real_part = arr.real
    imag_part = arr.imag

    # Compute gradients for real and imaginary parts separately
    gradient_real = np.gradient(real_part)
    gradient_imag = np.gradient(imag_part)

    # Combine real and imaginary gradients into complex gradient
    gradient = gradient_real + 1j * gradient_imag

    return np.abs(gradient)

def find_maxima(data):
    peaks, _ = find_peaks(data, prominence=1e-7)
    # Find the maximum that has a minimum after it
    if len(peaks) > 0:
        # idx = np.argsort(data[peaks])
        # peaks = peaks[idx[::-1]]
        for peak_idx in peaks:
            # Find the next local minimum after the current peak
            next_min_idx = np.where(data[peak_idx:] <= data[peak_idx])[0]
            if len(next_min_idx) > 0:
                return peak_idx
    print("no peaks")
    return np.argmax(data)

def choose_p(p_acf, p_gps, lph):
    chosen = 'acf'
    if (0 < p_acf <= 10) & (lph > 0.1):
        return p_acf, chosen
    if 10 < p_acf <= 20:
        if (np.abs(p_acf - p_gps) / p_acf < 0.1) & (lph > 0.1):
            return p_acf, chosen
    chosen = 'gps'
    return p_gps, chosen

def simulation_prediction():
    alpha = 0.72
    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
    transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)),
                         KeplerNoiseAddition(noise_dataset=None, noise_path=f'/data/lightPred/data/noise',
                                             transforms=kep_transform),
                         MovingAvg(13), Wavelet(gps=True, num_scales=1024),
                         ToTensor()])
    dataset = TimeSeriesDataset(data_folder, test_list, transforms=transform,
    init_frac=0.2,  prepare=False, dur=dur, wavelet=True)
    t = np.linspace(0, dur, int(dur / cad * DAY2MIN))
    # samples = os.listdir('data/samples')
    pred_ps = []
    ps = []
    lphs = []
    methods = []
    acf_ps = 0
    gps_ps = 0
    pbar = tqdm(dataset, total=len(dataset))
    for i, (x,y,mask,info) in enumerate(pbar):
        # if i % 100 == 0:
        print(i)
        p_acf, lags, acf, peaks, lph = analyze_lc(x[1])
        freqs = info['wavelet_freqs']
        grad_w = x[0][:len(freqs)]
        periods = 1/freqs
        # gsp_idx = np.argmax(grad_w)
        gsp_idx = find_maxima(grad_w)
        p_gps, val = periods[gsp_idx], grad_w[gsp_idx]
        p, chosen = choose_p(p_acf, p_gps, lph)
        if chosen == 'acf':
            pred_ps.append(p)
            acf_ps += 1
        else:
            pred_ps.append(p/alpha)
            gps_ps += 1
        ps.append(y[1].item()*50)
        lphs.append(lph)
        methods.append(chosen)
        #
        if i % 250==0:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(t, x[1])
            ax[1].plot(periods, grad_w)
            ax[1].scatter(p, val, c='r')
            fig.suptitle(f'period={y[1].item() * 50:2f}, gps={(p):.2f}')
            plt.savefig(f'{log_path}/exp{exp_num}/gps_{i}.png')
            plt.close()
        # if i == 100:
        #     break
    ps = np.array(ps).squeeze()
    pred_ps = np.array(pred_ps).squeeze()
    df_full = pd.DataFrame({'period': ps, 'predicted period': pred_ps, 'lph': lphs, 'method': methods})
    df_full.to_csv(f'{log_path}/exp{exp_num}/gps_results_{dataset_name}_dual.csv')
    plt.scatter(ps, pred_ps)
    acc10p = np.sum(np.abs(np.array(ps) - np.array(pred_ps)) < 0.1*np.array(ps))/len(ps)
    plt.xlabel('True period')
    plt.ylabel('Predicted period')
    plt.title(f'acc10p={acc10p:.2f}')
    plt.ylim(0,60)
    plt.savefig(f'{log_path}/exp{exp_num}/gps_results_{dataset_name}_dual.png')
    plt.close()
    print("total of acf predictions: ", acf_ps, "total of gps predictions: ", gps_ps)

if __name__ == '__main__':
    simulation_prediction()
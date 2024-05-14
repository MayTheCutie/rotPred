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

exp_num = 1

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

data_folder = "/data/butter/test_cos_old"

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
test_samples = os.listdir(os.path.join(data_folder, 'simulations'))
test_idx_list = [sample.split('_')[1].split('.')[0] for sample in test_samples if sample.startswith('lc_')]

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
def simulation_prediction():
    alpha = 0.72
    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
    transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)),
                         KeplerNoiseAddition(noise_dataset=None, noise_path=f'data/noise',
                                             transforms=kep_transform),
                         MovingAvg(49), Wavelet(gps=True, num_scales=1024),
                         ToTensor()])
    dataset = TimeSeriesDataset(data_folder, test_idx_list, transforms=transform,
    init_frac=0.2,  prepare=False, dur=dur, wavelet=True)
    t = np.linspace(0, dur, int(dur / cad * DAY2MIN))
    # samples = os.listdir('data/samples')
    pred_ps = []
    ps = []
    pbar = tqdm(dataset, total=len(dataset))
    for i, (x,y,mask,info) in enumerate(pbar):
        if i % 100 == 0:
            print(i)
        p, lags, acf, peaks, lph = analyze_lc(x[1])
        print(lph)
        if i == 10:
            exit()
        freqs = info['wavelet_freqs']
        grad_w = x[0][:len(freqs)]
        periods = 1/freqs
        # gsp_idx = np.argmax(grad_w)
        gsp_idx = find_maxima(grad_w)
        p, val = periods[gsp_idx], grad_w[gsp_idx]
        # x = x.numpy().squeeze()
        # w, _, scale, coi = wvl(x, freq_rate, pad=1)
        # grad_w = complex_gradient(w[-1])
        # pred_p = t[np.argmax(grad_w)]
        pred_ps.append(p/alpha)
        ps.append(y[1].item()*50)
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
    df_full = pd.DataFrame({'true period': ps, 'predicted period': pred_ps})
    df_full.to_csv(f'{log_path}/exp{exp_num}/gps_results_{dataset_name}_clean.csv')
    plt.scatter(ps, pred_ps)
    acc10p = np.sum(np.abs(np.array(ps) - np.array(pred_ps)) < 0.1*np.array(ps))/len(ps)
    plt.xlabel('True period')
    plt.ylabel('Predicted period')
    plt.title(f'acc10p={acc10p:.2f}')
    plt.ylim(0,60)
    plt.savefig(f'{log_path}/exp{exp_num}/gps_results_{dataset_name}_clean.png')


if __name__ == '__main__':
    simulation_prediction()
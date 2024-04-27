import numpy as np

from wavelet import wavelet as wvl
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

local = True

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
data_folder = "/data/butter/data_cos_old" if not local else '../butter/data_cos'

test_folder =  "/data/butter/data_cos_old" if not local else '../butter/test_cos'

yaml_dir = f'{root_dir}/Astroconf/'

Nlc = 50000

test_Nlc = 5000

freq_rate = 1/48

CUDA_LAUNCH_BLOCKING='1'


# idx_list = [f'{idx:d}'.zfill(int(np.log10(Nlc))+1) for idx in range(Nlc)]
# samples = os.listdir(os.path.join(data_folder, 'simulations'))
# idx_list = [sample.split('_')[1].split('.')[0] for sample in samples if sample.startswith('lc_')]
test_samples = os.listdir(os.path.join(test_folder, 'simulations'))
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
    kep_transform = RandomCrop(int(dur/cad*DAY2MIN))
    transform = Compose([RandomCrop(int(dur / cad * DAY2MIN)),
                         KeplerNoiseAddition(noise_dataset=None, noise_path=f'./data/noise',
                                             transforms=kep_transform),
                         MovingAvg(49), Wavelet(gsp=True, num_scales=1024),
                         ToTensor()])
    dataset = TimeSeriesDataset(test_folder, test_idx_list, transforms=transform,
    init_frac=0.2,  prepare=False, dur=dur, wavelet=True)
    t = np.linspace(0, dur, int(dur / cad * DAY2MIN))
    # samples = os.listdir('data/samples')
    pred_ps = []
    ps = []
    for i, (x,y,mask,info) in enumerate(dataset):
        print(i, x.shape)
        if i % 100 == 0:
            print(i)
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
        pred_ps.append(p)
        ps.append(y[1].item()*60)
        #
        if i % 10==0:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(t, x[1])
            ax[1].plot(periods, grad_w)
            ax[1].scatter(p, val, c='r')
            fig.suptitle(f'period={y[1].item() * 50:2f}, gps={p:.2f}')
            plt.show()
        if i == 100:
            break
    alpha = 0.8
    ps = np.array(ps).squeeze()
    pred_ps = np.array(pred_ps).squeeze()
    acc10 = np.sum((np.abs(pred_ps - ps)) < ps / 10)/len(pred_ps)
    acc20 = np.sum((np.abs(pred_ps - ps)) < ps / 5)/len(pred_ps)
    print(acc10, acc20)
    plt.scatter(pred_ps, ps)

    pred_ps_bounded_idx = np.where([np.logical_and(pred_ps >= 0.5, pred_ps <= 60)])
    res = linregress(pred_ps[pred_ps_bounded_idx[1]], ps[pred_ps_bounded_idx[1]])
    plt.plot(pred_ps, np.array(pred_ps)*res.slope + res.intercept, color='r')
    plt.plot(ps, ps, color='r')
    plt.title(f'slope - {res.slope:.2f} - intercept - {res.intercept:.2f}')
    plt.xlim(0, 65)
    plt.show()


if __name__ == '__main__':
    simulation_prediction()
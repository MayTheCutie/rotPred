import copy
from itertools import product

import numpy as np
from astropy.table import Table
import torch
from torch.optim import AdamW, Adam
# from torchinfo import summary
# from nvitop import Device
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import same_seeds
from ..Model.Modules.mhsa_pro import MHA_rotary
from ..Model.models import model_dict
from ..Model.utils import deepnorm_init
from .lr_scheduler import get_scheduler

def init_train(args, rank):
  # model initialization
  same_seeds(args.randomseed)
  # if args.model == 'Astroconformer':
  #   args.stride = int(20/args.sample_rates[0]**0.5)
  if args.deepnorm and args.num_layers >= 10:
    layer_coeff = args.num_layers/5.0
    args.alpha, args.beta = layer_coeff**(0.5), layer_coeff**(-0.5)
    
  model = model_dict[args.model](args)
  deepnorm_init(model, args)
  # summary(model, args.input_shape, depth=10)
  # if args.distributed:
  #   model = DDP(model, device_ids=[rank])
  # if args.fold == 0:
  #   with open(args.log_dir+'log.txt', 'a') as fp:
  #     fp.write(f"[Info]: Finish initializing {args.model}, summary of the model:\n{summary(model, args.input_shape, depth=10)}\n")
  
  # optimizer initialization
  # args.lr = args.batch_size/512*args.basic_lr if 'Kepseismic' not in args.dataset else args.basic_lr
  args.lr = args.basic_lr
  optimizer_dict = {'adamw': AdamW, 'adam': Adam}
  if 'NCE' in args.model:
    model.embedding.load_state_dict(torch.load('/g/data/y89/jp6476/best_model.pth'))
    for param in model.embedding.parameters():
      param.requires_grad = False
    parameter = []
    for name, param in model.named_parameters():
      if 'embedding' not in name:
        parameter.append(param)
  else:
    parameter = model.parameters()
  optimizer = optimizer_dict[args.optimizer](parameter, lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

  # scheduler initialization
  scheduler = get_scheduler(optimizer, args)

  # scaler initialization
  scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

  return model, optimizer, scheduler, scaler


def save_checkpoint(model, optimizer, scheduler, scaler, step, null_step, snapshot, args, name):
  torch.save({'net': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'scaler': scaler.state_dict(),
              'fold': args.fold,
              'step': step,
              'null_step': null_step,
              'snapshot': snapshot,
              }, args.log_dir + f'{name}.ckpt')
  
def check_gpu_use(args):

  devices = Device.all()  # or Device.cuda.all()

  with open(args.log_dir + 'log.txt', 'a') as fp:
    for device in devices:
      processes = device.processes()  # type: Dict[int, GpuProcess]
      sorted_pids = sorted(processes)

      fp.write(str(device) + '\n')
      fp.write(f'  - Fan speed:       {device.fan_speed()}%\n')
      fp.write(f'  - Temperature:     {device.temperature()}C\n')
      fp.write(f'  - GPU utilization: {device.gpu_utilization()}%\n')
      fp.write(f'  - Total memory:    {device.memory_total_human()}\n')
      fp.write(f'  - Used memory:     {device.memory_used_human()}\n')
      fp.write(f'  - Free memory:     {device.memory_free_human()}\n')
      fp.write(f'  - Processes ({len(processes)}): {sorted_pids}\n')
      for pid in sorted_pids:
          fp.write(f'    - {processes[pid]}\n')
      fp.write('-' * 120 + '\n')

def register_leaf_hooks(module, hook_fn):
    if not list(module.children()):
        # This is a leaf node
        print(f"Registering hook for {module}")
        module.register_forward_hook(hook_fn)
    else:
        # This is not a leaf node, so recurse on children
        for child in module.children():
            register_leaf_hooks(child, hook_fn)

def nan_detector(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in output of {module.__class__.__name__}")

def log_parameters_gradients_in_model(model, logger, step):
  for tag, value in model.named_parameters():
    logger.add_histogram(tag+"/param", value.data.cpu(), step)
    if value.grad is not None:
        logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

def uniform_sample(dataloader, args):
  # Get subset of data
  indices = dataloader.dataset.indices
  dataset = [dataloader.dataset.dataset.data[indice] for indice in indices]
  label = dataloader.dataset.dataset.label[indices]
  kids = dataloader.dataset.dataset.kids[indices]
  
  # Get sample indices
  sample_indices = []
  gap = (max(label)-min(label))/args.num_sample
  for i in range(0,args.num_sample):
    mask = (label>=min(label)+gap*i)&(label<=min(label)+gap*(i+1))
    sample_indice = np.where(mask)[0][0]
    sample_indices.append(sample_indice)
  sample_indices = torch.LongTensor(sample_indices)
  sample_labels = label[sample_indices]
  sample_kids = kids[sample_indices]
  if 'MQ' in args.dataset:
    sample_lcs = [dataset[indice][0][:4000] for indice in sample_indices]
  else:
    sample_lcs = [dataset[indice][:4000] for indice in sample_indices]
  sample_lcs = torch.stack(sample_lcs)
  sample_labels = sample_labels.float()

  return sample_lcs, sample_labels, sample_kids

def outlier_detection(dataloader, pred, label, kids, args):

  # 15 largest outliers
  outlier_idx = np.argsort(np.abs(pred - label))[-args.num_sample:]
  outlier_pred, outlier_label, outlier_kid = pred[outlier_idx], label[outlier_idx], kids[outlier_idx]
  if 'MQ' in args.dataset:
    outlier_quarter = np.array([(kids == kid).cumsum()[idx] for kid, idx in zip(outlier_kid, outlier_idx)])

  # sort by label
  outlier_idx = np.argsort(outlier_label)
  outlier_pred, outlier_label, outlier_kid = outlier_pred[outlier_idx], outlier_label[outlier_idx], outlier_kid[outlier_idx]
  if 'MQ' in args.dataset:
    outlier_quarter = outlier_quarter[outlier_idx]

  # get outlier light curves
  outlier_idx = [np.argwhere(dataloader.dataset.dataset.kids == kid)[0][0] for kid in outlier_kid]
  if 'MQ' in args.dataset:
    outlier_lc = torch.stack([dataloader.dataset.dataset.data[idx][outlier_quarter] for idx, outlier_quarter in zip(outlier_idx, outlier_quarter)])
  else:
    outlier_lc = torch.stack([dataloader.dataset.dataset.data[idx] for idx in outlier_idx])

  return outlier_lc, outlier_pred, outlier_label, outlier_kid

def inspect_snapshot(step, snapshot, args):

  # Comparison scatter between train, val, test
  prefix = ['tr', 'val', 'test']
  plt.figure(figsize=(6,6))
  plt.xlabel('label')
  plt.ylabel('pred')
  for i, (loss, pred, label, _) in enumerate(snapshot):
    if i == 1:
      lim = label.min()-1e-2, label.max()+1e-2
    if loss is None:
      continue
    plt.scatter(label, pred, s=1.5, label=f'{prefix[i]}_loss={loss:.5f}')
  plt.plot(lim, lim, c='k', ls='--')
  plt.xlim(lim)
  plt.ylim(lim)
  plt.legend()
  plt.savefig(args.log_dir+f'figures/scatter/fold{args.fold}_step{str(step).zfill(5)}.png')
  plt.close()

  # Colormap test results by stellar properties
  # catalogue_dir = "/g/data/y89/jp6476/hlsp_kg-radii_kepler-gaia_multi_all_multi_v1_star-cat.fits"
  # catalogue = Table.read(catalogue_dir).to_pandas().sort_values(by='KIC_ID')
  # test_kids = snapshot[-1][-1]
  # plt.figure(figsize=(6,6))
  # plt.xlabel('label')
  # plt.ylabel('pred')


def visualize_attention(step, model, samples, args):
  model.eval()
  sample_lcs, sample_labels, sample_kids = samples

  for mod in model.modules():
    if isinstance(mod, MHA_rotary):
      mod.collect_attention_map = True

  with torch.no_grad():
    sample_preds = model(sample_lcs.to(args.device)).detach().cpu()

  attention_maps = []
  for mod in model.modules():
    if isinstance(mod, MHA_rotary):
      attention_maps.append(mod.attention_map.detach().cpu())
  
  layer, (batch, num_head, seq_len) = len(attention_maps), attention_maps[0].shape[:-1]
  h, w = 3, 5
  figsize = w*4, h*4
  for l in range(layer):
    layer_attention_map = attention_maps[l]
    for head in range(num_head):
      head_attention_map = layer_attention_map[:, head, :]
      fig, axs = plt.subplots(h, w, sharex='all', sharey='all', figsize=figsize) 
      fig.subplots_adjust(hspace=0.1, wspace=0)
      fig.suptitle(f'layer {l}, head {head}')
      for i, j in product(range(h), range(w)):
        order = i*w+j
        ax = axs[i, j]
        ax.imshow(head_attention_map[order].reshape(seq_len, seq_len))
        ax.text(0, 0.85, f'logg: {sample_labels[order]:.2f}\npred: {sample_preds[order]:.2f}', fontsize=12, weight='bold', color='C1',\
                transform=ax.transAxes)
        ax.set_title(f'{sample_kids[order]}')
      plt.savefig(args.log_dir+f'figures/attention/fold{args.fold}_step{str(step).zfill(5)}_layer{l}_head{head}.png')
      plt.close()
  model.train()
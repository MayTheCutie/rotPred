from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import yaml
from matplotlib import pyplot as plt
import glob
from collections import OrderedDict
from tqdm import tqdm
import torch.distributed as dist
from util.classical_analysis import analyze_lc
import umap
# from lightPred.dataloader import boundary_values_dict

# import NamedTuple
# import List

# class FitResult(NamedTuple):
#     """
#     Represents the result of fitting a model for multiple epochs given a
#     training and test (or validation) set.
#     The losses and the accuracies are per epoch.
#     """

#     num_epochs: int
#     train_loss: List[float]
#     test_loss: List[float]


def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1


class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, world_size=1, num_classes=2,
                 scheduler=None, val_dataloader=None,  optim_params=None, max_iter=np.inf, net_params=None,
                  scaler=None, grad_clip=False, exp_num=None, log_path=None, exp_name=None, plot_every=None,
                   cos_inc=False, range_update=None, quantiles=1,
                  ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.quantiles = quantiles
        self.cos_inc = cos_inc
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.train_sampler = self.get_sampler_from_dataloader(train_dataloader)
        self.val_sampler = self.get_sampler_from_dataloader(val_dataloader)
        self.max_iter = max_iter
        self.device = device
        self.world_size = world_size
        self.optim_params = optim_params
        self.net_params = net_params
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        self.logger = None
        self.range_update = range_update
        self.nc_err_i = np.zeros((1,quantiles))
        self.nc_err_p = np.zeros((1,quantiles))
        if log_path is not None:
            self.logger =SummaryWriter(f'{self.log_path}/exp{self.exp_num}')
            # print(f"logger path: {self.log_path}/exp{self.exp_num}")
            if not os.path.exists(f'{self.log_path}/exp{self.exp_num}'):
                os.makedirs(f'{self.log_path}/exp{self.exp_num}')
            with open(f'{self.log_path}/exp{exp_num}/net_params.yml', 'w') as outfile:
                yaml.dump(self.net_params, outfile, default_flow_style=False)
            with open(f'{self.log_path}/exp{exp_num}/optim_params.yml', 'w') as outfile:
                    yaml.dump(self.optim_params, outfile, default_flow_style=False)

        print("logger is: ", self.logger)

    def plot_pred_vs_true(self, output, target, epoch, data='train'):
        plt.scatter(target[:, 1], output[:, 1])
        plt.plot(target, 0.9*target, color='red')
        plt.plot(target, 1.1*target, color='red')
        plt.xlim(0, max(target[:,1]) + 5)
        plt.ylim(0, max(target[:,1]) + 5)
        plt.title(f"{self.exp_name}")
        plt.xlabel("true Period")
        plt.ylabel("predicted Period")
        plt.savefig(f"{self.log_path}/exp{self.exp_num}/Period_{data}_epoch{epoch}.png")
        plt.clf()

        plt.scatter(target[:, 0], output[:, 0])
        plt.title(f"{self.exp_name} Inclination")
        plt.xlabel("true Inclination")
        plt.ylabel("predicted Inclination")
        plt.savefig(f"{self.log_path}/exp{self.exp_num}/Inc_{data}_epoch{epoch}.png")

    def plot_pred_vs_true_cls(self, output, target, epoch, data='train'):
        coords, counts = count_occurence(target, output)
        plt.scatter(coords[:, 0], coords[:, 1], c=counts, cmap='viridis')
        plt.title(f"{self.exp_name} Inclination")
        plt.xlabel("true")
        plt.ylabel("predicted")
        plt.colorbar(label='points frequency')
        plt.savefig(f"{self.log_path}/exp{self.exp_num}/Inc_{data}_epoch{epoch}.png")
        plt.close()
    
    def get_sampler_from_dataloader(self, dataloader):
        if hasattr(dataloader, 'sampler'):
            if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                return dataloader.sampler
            elif hasattr(dataloader.sampler, 'sampler'):
                return dataloader.sampler.sampler
        
        if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'sampler'):
            return dataloader.batch_sampler.sampler
        
        return None

    def fit(self, num_epochs, device,  early_stopping=None, only_p=False, best='loss', conf=False):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        # self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        main_proccess = (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or self.device == 'cpu'

        print(f"Starting training for {num_epochs} epochs with parameters: {self.optim_params}, {self.net_params}")
        print("is main process: ", main_proccess, flush=True)
        global_time = time.time()
        for epoch in range(num_epochs):
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch, only_p=only_p, plot=plot, conf=conf)
            t_loss_mean = np.mean(t_loss)
            train_loss.extend(t_loss)
            global_train_accuracy, global_train_loss = self.process_loss(t_acc, t_loss_mean)
            if main_proccess:  # Only perform this on the master GPU
                train_acc.append(global_train_accuracy.mean().item())
                
            v_loss, v_acc = self.eval_epoch(device, epoch=epoch, only_p=only_p, plot=plot, conf=conf)
            v_loss_mean = np.mean(v_loss)
            val_loss.extend(v_loss)
            global_val_accuracy, global_val_loss = self.process_loss(v_acc, v_loss_mean)
            if main_proccess:  # Only perform this on the master GPU                
                val_acc.append(global_val_accuracy.mean().item())
                
                current_objective = global_val_loss if best == 'loss' else global_val_accuracy.mean()
                improved = False
                
                if best == 'loss':
                    if current_objective < min_loss:
                        min_loss = current_objective
                        improved = True
                else:
                    if current_objective > best_acc:
                        best_acc = current_objective
                        improved = True
                
                if improved:
                    print("saving model...")
                    torch.save(self.model.state_dict(), f'{self.log_path}/exp{self.exp_num}/{self.exp_name}.pth')
                    self.best_state_dict = self.model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                print(f'Epoch {epoch}: Train Loss: {global_train_loss:.6f}, Val Loss:'\
                f'{global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, '\
                f'Val Acc: {global_val_accuracy.round(decimals=4).tolist()}, Time: {time.time() - start_time:.2f}s', flush=True)
                if epoch % 10 == 0:
                    print(os.system('nvidia-smi'))

                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
                if time.time() - global_time > (23.83 * 3600):
                    print("time limit reached")
                    break 

        self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']
        # self.optim_params['lr_history'].append(self.optim_params['lr'])
        with open(f'{self.log_path}/exp{self.exp_num}/optim_params.yml', 'w') as outfile:
            yaml.dump(self.optim_params, outfile, default_flow_style=False)

        # if self.logger is not None:
        #     self.logger.add_scalar('time', time.time() - start_time, epoch)
        #     self.logger.close()
        return {"num_epochs":num_epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc}

    def process_loss(self, acc, loss_mean):
        if  torch.cuda.is_available() and torch.distributed.is_initialized():
            global_accuracy = torch.tensor(acc).cuda()  # Convert accuracy to a tensor on the GPU
            torch.distributed.reduce(global_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss = torch.tensor(loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss /= torch.distributed.get_world_size()
        else:
            global_loss = torch.tensor(loss_mean)
            global_accuracy = torch.tensor(acc)
        return global_accuracy, global_loss

    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            loss, acc = self.train_batch(batch, device, conf)
            train_loss.append(loss.item())
            all_accs = all_accs + acc
            pbar.set_description(f"train_acc: {all_accs}, train_loss:  {loss.item()}")      
            if i > self.max_iter:
                break
        print("number of train_accs: ", train_acc)
        return train_loss, all_accs/len(self.train_dl.dataset)
    
    def train_batch(self, batch, device, conf):
        x,y,_,_ = batch
        if x.shape[-1] == 2:
            x = x[...,0].unsqueeze(-1).permute(0,2,1)
        x = x.to(device)
        y = y.to(device)
        self.optimizer.zero_grad()
        y_pred = self.model(x.float())
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        if conf:
            y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
            conf_y = torch.abs(y - y_pred) 
        
        loss = self.criterion(y_pred, y)
        if conf:
            loss += self.criterion(conf_pred, conf_y)
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        diff = torch.abs(y_pred - y)
        acc = (diff < (y/10)).sum(0)
        return loss, acc

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        for i,batch in enumerate(pbar):
            loss, acc = self.eval_batch(batch, device, conf)
            val_loss.append(loss)
            all_accs = all_accs + acc
            pbar.set_description(f"val_acc: {all_accs}, val_loss:  {loss.item()}")
        return val_loss, all_accs/len(self.val_dl.dataset)

    def eval_batch(self, batch, device, conf):
        x,y,_,_ = batch
        if x.shape[-1] == 2:
            x = x[...,0].unsqueeze(-1).permute(0,2,1)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = self.model(x)
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                conf_y = torch.abs(y - y_pred)
            else:
                conf_pred = torch.ones_like(y_pred)
        loss = self.criterion(y_pred, y)
        if conf:
            loss += self.criterion(conf_pred, conf_y)
        diff = torch.abs(y_pred - y)
        acc = (diff < (y/10)).sum(0)
        return loss, acc

    def load_best_model(self, to_ddp=True, from_ddp=True):
        data_dir = f'{self.log_path}/exp{self.exp_num}'
        # data_dir = f'{self.log_path}/exp29' # for debugging

        state_dict_files = glob.glob(data_dir + '/*.pth')
        print("loading model from ", state_dict_files[-1])
        
        state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=self.device)
    
        if from_ddp:
            print("loading distributed model")
            # Remove "module." from keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    while key.startswith('module.'):
                        key = key[7:]
                new_state_dict[key] = value
            state_dict = new_state_dict
        # print("state_dict: ", state_dict.keys())
        # print("model: ", self.model.state_dict().keys())

        self.model.load_state_dict(state_dict, strict=False)

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        tot_kic = []
        tot_teff = []
        for i,(x, y,_,info) in enumerate(test_dataloader):
            x = x.to(device)
            with torch.no_grad():
                y_pred, memory = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.num_classes:
                targets = np.concatenate((targets, y.cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, confs
    
class DoubleInputTrainer(Trainer):
    def __init__(self, num_classes=2, eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eta = eta
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            loss, acc,_ = self.train_batch(batch, device, conf)
            train_loss.append(loss.item())
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item()}")
            all_accs = all_accs + acc
            if i > self.max_iter:
                break
            if self.range_update is not None and (i % self.range_update == 0):
                self.train_dl.dataset.expand_label_range()
                print("range: ", y.min(dim=0).values, y.max(dim=0).values)
        return train_loss, all_accs/len(self.train_dl.dataset)
    
    def train_batch(self, batch, device, conf):
        x,y,_,info = batch
        x1, x2 = x[...,0], x[...,1]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        self.optimizer.zero_grad()
        y_pred = self.model(x1.float(), x2.float())
        if conf:
            y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            conf_y = torch.abs(y - y_pred) 
        if self.cos_inc:
            inc_idx = 0
            y_pred[:, inc_idx] = torch.cos(y_pred[:, inc_idx]*np.pi/2)
            y[:, inc_idx] = torch.cos(y[:, inc_idx]*np.pi/2)
        if self.num_classes > 1:
            loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
            loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
            loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
            if conf:
                loss_conf_i = self.criterion(conf_pred[:, 0], conf_y[:, 0])
                loss_conf_p = self.criterion(conf_pred[:, 1], conf_y[:, 1])
                loss += (self.eta * loss_conf_i) + ((1-self.eta) * loss_conf_p)
        else:
            loss = self.criterion(y_pred, y)
            if conf:
                loss += self.criterion(conf_pred, conf_y)
        loss.backward()
        self.optimizer.step()
        diff = torch.abs(y_pred - y)
        acc = (diff < (y/10)).sum(0)
        return loss, acc, y_pred
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        if self.val_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        pbar = tqdm(self.val_dl)
        targets = np.zeros((0, self.num_classes))
        for i, batch in enumerate(pbar):
            loss, acc,_ = self.eval_batch(batch, device, conf)
            val_loss.append(loss.item())
            all_accs = all_accs + acc  
            pbar.set_description(f"val_acc: {acc}, val_loss:  {loss.item()}")
            if i > self.max_iter:
                break
            if self.range_update is not None and (i % self.range_update == 0):
                self.train_dl.dataset.expand_label_range()
        return val_loss, all_accs/len(self.val_dl.dataset)
    
    def eval_batch(self, batch, device, conf):
        x,y,_,info = batch
        x1, x2 = x[...,0], x[...,1]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = self.model(x1.float(), x2.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                conf_y = torch.abs(y - y_pred) 
            if self.cos_inc:
                inc_idx = 0
                y_pred[:, inc_idx] = torch.cos(y_pred[:, inc_idx]*np.pi/2)
                y[:, inc_idx] = torch.cos(y[:, inc_idx]*np.pi/2)
            if self.num_classes > 1:
                loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
                loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
                loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
            if conf:
                loss_conf_i = self.criterion(conf_pred[:, 0], conf_y[:, 0])
                loss_conf_p = self.criterion(conf_pred[:, 1], conf_y[:, 1])
                loss += (self.eta * loss_conf_i) + ((1-self.eta) * loss_conf_p)
            else:
                loss = self.criterion(y_pred, y)
                if conf:
                    loss += self.criterion(conf_pred, conf_y)
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum(0)
        return loss, acc, y_pred

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        tot_kic = []
        tot_teff = []
        test_loss = []
        pbar = tqdm(test_dataloader)
        for i,(x,y,_,info) in enumerate(pbar): 
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            with torch.no_grad():
                if 'acf_phr' in info:
                    y_pred = self.model(x1.float(), x2.float(), acf_phr=info['acf_phr'])
                else:
                    y_pred = self.model(x1.float(), x2.float())
                # print(i, " y_pred: ", y_pred.shape, "y: ", y.shape)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            loss = self.criterion(y_pred, y)
            diff = torch.abs(y_pred - y)
            all_acc = (diff < (y/10)).sum(0)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.num_classes:
                targets = np.concatenate((targets, y.cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
            if i >= self.max_iter:
                break
            pbar.set_description(f"test_acc: {all_acc}, test_loss:  {loss.item()}")
            if i > self.max_iter:
                break
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        
        return preds, targets, confs, []

class MultiTrainer(Trainer):
    def __init__(self, num_classes=2, eta=0.5, reg_lambda=0.1, start_epoch_reg=0, temperature=10,
                 kl_lambda=0.1, kl_update_freq=5, start_epoch_kl=30, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.start_epoch_reg = start_epoch_reg
        self.temperature = temperature
        self.kl_lambda = kl_lambda
        self.kl_update_freq = kl_update_freq
        self.accumulated_predictions = []
        self.accumulated_targets = []
        self.start_epoch_kl = start_epoch_kl
        self.cur_kl_lambda = 0
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)

    def calculate_kl_loss(self):
        if not self.accumulated_predictions:
            return torch.tensor(0.0, device=self.device)
        
        all_preds = torch.cat(self.accumulated_predictions, dim=0)
        
        # Calculate mean prediction for the first label
        mean_pred = all_preds.mean(dim=0)
        
        # Calculate KL divergence for the first label using KLDivLoss
        kl_div = self.kl_loss(F.log_softmax(all_preds, dim=-1),
                              F.softmax(mean_pred.unsqueeze(0).expand_as(all_preds), dim=-1))
        
        return kl_div

    def train_epoch(self, device, epoch=None, only_p=False, plot=False, conf=False):
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)

        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        if epoch >= self.start_epoch_reg:
            k = 0.7
            self.reg_lambda = 1 / (1 + np.exp(-k * (epoch - 35)))
        else:
            self.reg_lambda = 0
        
        if epoch >= self.start_epoch_kl:
            self.cur_kl_lambda = self.kl_lambda
        else:
            self.cur_kl_lambda = 0
        
        pbar = tqdm(self.train_dl)
        for i, (x, y, _, info) in enumerate(pbar):
            self.optimizer.zero_grad()  # Zero gradients at the start of each batch
            
            loss, acc, y_pred = self.train_batch(x, y,  i, device, conf)
            train_loss.append(loss.item())
            all_accs += acc
            
            pbar.set_description(f"train_acc: {acc}, train_loss: {loss.item()}")
            
            if self.max_iter and i >= self.max_iter:
                break

            if self.range_update is not None and (i % self.range_update == 0):
                self.train_dl.dataset.expand_label_range()

        return train_loss, all_accs / len(self.train_dl.dataset)

    def train_batch(self, x, y,  idx, device, conf):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        b, c, l, h = x.shape  # batch size, number of copies, sequence length, dimension
        
        x_reshaped = x.reshape(b*c, l, h)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        self.optimizer.zero_grad()
        
        y_pred = self.model(x1.float(), x2.float())
        y_pred = y_pred.reshape(b, c, -1, self.num_classes)

        if conf:
            y_pred, conf_pred = y_pred[..., :self.num_classes], y_pred[..., self.num_classes:]
            conf_y = torch.abs(y.unsqueeze(1) - y_pred)
        # Calculate the main loss (average across copies)
        if self.num_classes > 1:
            loss_i = self.criterion(y_pred[..., 0].mean(dim=1), y[..., 0])
            loss_p = self.criterion(y_pred[..., 1].mean(dim=1), y[..., 1])
            loss = (self.eta * loss_i) + ((1 - self.eta) * loss_p)
            if conf:
                loss_conf_i = self.criterion(conf_pred[..., 0].mean(dim=1), conf_y[..., 0].mean(dim=1))
                loss_conf_p = self.criterion(conf_pred[..., 1].mean(dim=1), conf_y[..., 1].mean(dim=1))
                loss += (self.eta * loss_conf_i) + ((1 - self.eta) * loss_conf_p)
        else:
            loss = self.criterion(y_pred.mean(dim=1), y)
            if conf:
                loss += self.criterion(conf_pred.mean(dim=1), conf_y.mean(dim=1))
        if y_pred.shape[1] > 1:
            # Calculate regularization loss (standard deviation across copies)
            reg_loss = y_pred.std(dim=1).mean()
            
            # Combine losses
            total_loss = (1 - self.reg_lambda) * loss + self.reg_lambda * reg_loss
        else:
            total_loss = loss
        
        if (idx + 1) % self.kl_update_freq == 0:
                kl_loss = self.calculate_kl_loss()
                total_loss += self.cur_kl_lambda * kl_loss
                self.accumulated_predictions = []  # Clear accumulated predictions
        total_loss.backward()
        self.optimizer.step()

        # Calculate accuracy (using mean prediction across copies)
        y_pred = y_pred.mean(dim=1)
        # Accumulate predictions for KL loss (only first label)
        self.accumulated_predictions.append(y_pred[:, :, 0].detach())  # Only first label
        mean_pred = y_pred[:, self.quantiles//2, :]
        diff = torch.abs(mean_pred - y)
        acc = (diff < (y/10)).sum(0)

        return total_loss, acc, y_pred

    def eval_epoch(self, device, epoch=None, only_p=False, plot=False, conf=False):
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_preds = np.zeros((0, self.quantiles, self.num_classes))
        all_ys = np.zeros((0, self.num_classes))
        all_accs = torch.zeros(self.num_classes, device=device)

        if self.val_sampler is not None:
            try:
                self.val_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        if epoch >= self.start_epoch_reg:
            k = 0.7
            self.reg_lambda = self.reg_lambda / (1 + np.exp(-k * (epoch - 35)))
        else:
            self.reg_lambda = 0
        
        if epoch >= self.start_epoch_kl:
            self.cur_kl_lambda = self.kl_lambda
        else:
            self.cur_kl_lambda = 0

        pbar = tqdm(self.val_dl)
        for i, (x, y, _, info) in enumerate(pbar):
            loss, acc, preds = self.eval_batch(x, y, i, device, conf)
            val_loss.append(loss.item())
            all_accs += acc
            pbar.set_description(f"val_acc: {acc}, val_loss: {loss.item()}")

            all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
            all_ys = np.concatenate((all_ys, y.cpu().numpy()))
            
            if self.max_iter and i >= self.max_iter:
                break
        if self.quantiles > 1: # CQR calibration
            self.nc_err_i = self.criterion.calibrate(all_preds[...,0], all_ys[...,0])
            self.nc_err_p = self.criterion.calibrate(all_preds[...,1], all_ys[...,1])
        return val_loss, all_accs / len(self.val_dl.dataset)

    def eval_batch(self, x, y, idx, device, conf):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        b, c, l, h = x.shape

        x_reshaped = x.reshape(b*c, l, h)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = self.model(x1.float(), x2.float())
            y_pred = y_pred.reshape(b, c, -1, self.num_classes)

            if conf:
                y_pred, conf_pred = y_pred[..., :self.num_classes], y_pred[..., self.num_classes:]
                conf_y = torch.abs(y.unsqueeze(1) - y_pred)

            # Calculate loss (average across copies)
            if self.num_classes > 1:
                loss_i = self.criterion(y_pred[..., 0].mean(dim=1), y[:, 0])
                loss_p = self.criterion(y_pred[..., 1].mean(dim=1), y[:, 1])
                loss = (self.eta * loss_i) + ((1 - self.eta) * loss_p)
                if conf:
                    loss_conf_i = self.criterion(conf_pred[..., 0].mean(dim=1), conf_y[..., 0].mean(dim=1))
                    loss_conf_p = self.criterion(conf_pred[..., 1].mean(dim=1), conf_y[..., 1].mean(dim=1))
                    loss += (self.eta * loss_conf_i) + ((1 - self.eta) * loss_conf_p)
            else:
                loss = self.criterion(y_pred[...,0].mean(dim=1), y)
                if conf:
                    loss += self.criterion(conf_pred.mean(dim=1), conf_y.mean(dim=1))
            if y_pred.shape[1] > 1:
                # Calculate regularization loss (standard deviation across copies)
                reg_loss = y_pred.std(dim=1).mean()
                
                # Combine losses
                total_loss = (1 - self.reg_lambda) * loss + self.reg_lambda * reg_loss
            else:
                total_loss = loss
            
            if (idx + 1) % self.kl_update_freq == 0:
                kl_loss = self.calculate_kl_loss()
                total_loss += self.cur_kl_lambda * kl_loss
                self.accumulated_predictions = []  # Clear accumulated predictions

            # Calculate accuracy (using mean prediction across copies)
            mean_pred = y_pred.mean(dim=1)
            self.accumulated_predictions.append(mean_pred[:, :, 0].detach())  # Only first label
            diff = torch.abs(mean_pred[:, self.quantiles//2] - y)
            acc = (diff < (y/10)).sum(0)

        return loss, acc, mean_pred

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        reducer = umap.UMAP()
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.quantiles, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        embs = np.zeros((0, 2))
        pbar = tqdm(test_dataloader)
        for i, (x, y, _, info) in enumerate(pbar):
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            b, c, l, h = x.shape
            x_reshaped = x.reshape(b*c, l, h)
            x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            with torch.no_grad():
                if 'acf_phr' in info:
                    y_pred, features = self.model(x1.float(), x2.float(), acf_phr=info['acf_phr'], return_features=True)
                else:
                    y_pred, features = self.model(x1.float(), x2.float(), return_features=True)
                y_pred = y_pred.reshape(b, c, -1, self.num_classes)
                features = features.reshape(b, c, -1)
                if conf:
                    y_pred, conf_pred = y_pred[..., :self.num_classes], y_pred[..., self.num_classes:]
            mean_pred = y_pred.mean(dim=1)
            features = features.mean(dim=1)
            preds = np.concatenate((preds, mean_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
            # embedding = reducer.fit_transform(features.detach().cpu().numpy())
            # embs = np.concatenate((embs, embedding))
            if conf:
                mean_conf = conf_pred.mean(dim=1)
                confs = np.concatenate((confs, mean_conf.cpu().numpy()))
            
            if self.max_iter and i >= self.max_iter:
                break
            
            if self.num_classes > 1:
                loss_i = self.criterion(y_pred[..., 0].mean(dim=1), y[:, 0])
                loss_p = self.criterion(y_pred[..., 1].mean(dim=1), y[:, 1])
                loss = (self.eta * loss_i) + ((1 - self.eta) * loss_p)
            else:
                loss = self.criterion(y_pred[...,0].mean(dim=1), y)
            diff = torch.abs(mean_pred[:, self.quantiles//2,:] - y)
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"test_acc: {all_acc}, test_loss: {loss.item()}")
        if self.quantiles > 1:
            print(self.nc_err_i, self.nc_err_p)
            preds[...,0] = self.criterion.predict(preds[...,0], self.nc_err_i)
            preds[...,1] = self.criterion.predict(preds[...,1], self.nc_err_p)

        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        
        return preds, targets, confs, embs

class KeplerTrainer(Trainer):
    def __init__(self, eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
    
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i, (x, y, _, _, info, _) in enumerate(pbar):
            y = y.to(device)
            self.optimizer.zero_grad()
            if x.shape[1] == 2:
                x1, x2 = x[:, 0, :], x[:, 1, :]
                x1 = x1.to(device)
                x2 = x2.to(device)
                y_pred = self.model(x1.float(), x2.float())
            else:
                x = x.to(device)
                y_pred = self.model(x.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                conf_y = torch.abs(y - y_pred)
            if self.eta >= 0:
                loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
                loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
                loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
            else:
                loss = self.criterion(y_pred, y)
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff < (y/10)).sum(0)
            pbar.set_description(f"train_acc: {train_acc}, train_loss:  {loss.item()}")
        # print("number of train_accs: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        pbar = tqdm(self.val_dl)
        for i, (x, y, _, _, info, _) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            if x.shape[1] == 2:
                x1, x2 = x[:, 0, :], x[:, 1, :]
                x1 = x1.to(device)
                x2 = x2.to(device)
                y_pred = self.model(x1.float(), x2.float())
            else:
                x = x.to(device)
                y_pred = self.model(x.float())
            # y_val = y['Period'] if only_p else y['i']
            # pred_idx = 1 if only_p else 0
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                conf_y = torch.abs(y - y_pred)
            loss = self.criterion(y_pred, y) 
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            val_acc += (diff < (y/10)).sum(0)
            pbar.set_description(f"val_acc: {val_acc}, val_loss:  {loss.item()}")
        return val_loss, val_acc/len(self.val_dl.dataset)

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.quantiles, self.num_classes))
        targets = np.zeros((0))
        confs = np.zeros((0, self.num_classes))
        tot_kic = np.zeros((0))
        tot_teff = np.zeros((0))
        tot_r = np.zeros((0))
        tot_g = np.zeros((0))
        tot_qs = np.zeros((0, 14))

        print(os.system('nvidia-smi'))

        print("len test dataset: ", len(test_dataloader.dataset))
        pbar = tqdm(test_dataloader)
        for i,(x, y, _, _, info, _) in enumerate(pbar):
            if i > self.max_iter:
                break
            # print("batch: ", i, "x: ", x.shape, "y: ", y.shape, flush=True)
            x = x.to(device)
            y = y.to(device)
            kic = torch.tensor([d['KID'] for d in info]).to(device)
            teff = torch.tensor([d['Teff'] for d in info]).to(device)
            r = torch.tensor([d['R'] for d in info]).to(device)
            g = torch.tensor([d['logg'] for d in info]).to(device)
            qs = []
            for d in info:
                _qs = torch.tensor(d['qs'])
                _qs = torch.nn.functional.pad(_qs, (0, 14 - _qs.shape[0]), value=-1)
                qs.append(_qs)
            qs = torch.stack(qs).to(device)
            with torch.no_grad():
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y_pred = self.model(x1.float(), x2.float())
                else:
                    x = x.to(device)
                    y_pred = self.model(x.float())
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            preds = np.concatenate((preds, y_pred.squeeze().cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
            tot_kic = np.concatenate((tot_kic, kic.cpu().numpy()))
            tot_teff = np.concatenate((tot_teff, teff.cpu().numpy()))
            tot_r = np.concatenate((tot_r, r.cpu().numpy()))
            tot_g = np.concatenate((tot_g, g.cpu().numpy()))
            tot_qs = np.concatenate((tot_qs, qs.cpu().numpy()))
        return preds, confs, tot_kic, tot_teff, tot_r, tot_g, tot_qs
    
    def aggregate_results_from_gpus(self, y_pred, y, conf_pred, kic, teff, r, g, qs):
        torch.distributed.gather(y_pred, [torch.zeros_like(y_pred) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(y, [torch.zeros_like(y) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(conf_pred, [torch.zeros_like(conf_pred) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(kic, [torch.zeros_like(kic) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(teff, [torch.zeros_like(teff) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(r, [torch.zeros_like(r) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(g, [torch.zeros_like(g) for _ in range(torch.distributed.get_world_size())], dst=0)
        torch.distributed.gather(qs, [torch.zeros_like(qs) for _ in range(torch.distributed.get_world_size())], dst=0)
        return y_pred, y, conf_pred, kic, teff, r, g, qs


class SpotsTrainer(Trainer):
    def __init__(self, spots_loss, num_classes=2, eta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.spots_loss = spots_loss
        self.eta = eta
        self.bbox_loss_log = []
        self.class_loss_log = []
        self.giou_loss_log = []
    
    def get_spot_dict(self, spot_arr):
        bs, _, _ = spot_arr.shape
        idx = [spot_arr[b, 0, :] != 0 for b in range(bs)]
        res = []
        for i in range(bs):
            spot_dict = {'boxes': cxcy_to_cxcywh(spot_arr[i, :, idx[i]], 1 / 360, 1 / 360).transpose(0,1).to(spot_arr.device),
                         'labels': torch.ones((spot_arr[i, :, idx[i]].shape[-1]), device=spot_arr.device).long()}
            res.append(spot_dict)
        return res

    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, (x,y, _,_) in enumerate(pbar):
            tic = time.time()
            x, spots_arr = x[:, :-2, :], x[:, -2:, :]
            x = x.to(device)
            y = y.to(device)
            spots_arr = spots_arr.to(device)
            tgt_spots = self.get_spot_dict(spots_arr)
            shapes = [(tgt_spots[i]['boxes'].shape, tgt_spots[i]['labels'].shape) for i in range(len(tgt_spots))]
            if x.shape[1] == 2:
                x1, x2 = x[:, 0, :], x[:, 1, :]
                out_spots, y_pred = self.model(x1, x2)
            else:
                out_spots, y_pred = self.model(x)
            t1 = time.time()
            src_shapes = (out_spots['pred_boxes'].shape, out_spots['pred_logits'].shape)
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                conf_y = torch.abs(y - y_pred)
            att_loss_val = self.criterion(y_pred, y.view(-1, self.num_classes))
            if conf:
                att_loss_val += self.criterion(conf_pred, conf_y.view(-1, self.num_classes))
            t2 = time.time()
            spots_loss_dict = self.spots_loss(out_spots, tgt_spots)
            weight_dict = self.spots_loss.weight_dict
            spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
            loss = att_loss_val
            t3 = time.time()
            loss = self.eta*spot_loss_val + (1-self.eta)*att_loss_val
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            spots_acc = (100 - spots_loss_dict['class_error'])
           
            diff = torch.abs(y_pred - y)
            # train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            att_acc = (diff < (y/10)).sum(0)
            all_accs = all_accs + att_acc
            # print("spots dict: ", spots_loss_dict)
            # if i % 10 == 0:
            pbar.set_description(f"train_acc: {att_acc, spots_acc}, train_loss:  {loss.item():.2f}, "\
             f"spot_loss: {self.eta*spot_loss_val:.5f}, att_loss: {(1-self.eta)*att_loss_val:.5f}")
            if i > self.max_iter:
                break
            toc = time.time()
            if i % 100 == 0:
                print(f"train time: {toc-tic}, forward time: {t1-tic}, loss time: {t2-t1}, spot loss time: {t3-t2}, step time: {toc-t3}")
        return train_loss, all_accs/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        with torch.no_grad():
            for i, (x,y,_,_) in enumerate(pbar):
                x, spots_arr = x[:, :-2, :], x[:, -2:, :]
                x = x.to(device)
                y = y.to(device)
                spots_arr = spots_arr.to(device)
                tgt_spots = self.get_spot_dict(spots_arr)
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    out_spots, y_pred = self.model(x1, x2)
                else:
                    out_spots, y_pred = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                    conf_y = torch.abs(y - y_pred)
                att_loss_val = self.criterion(y_pred, y.view(-1, self.num_classes))
                if conf:
                    att_loss_val += self.criterion(conf_pred, conf_y.view(-1, self.num_classes))
                spots_loss_dict = self.spots_loss(out_spots, tgt_spots)
                weight_dict = self.spots_loss.weight_dict
                spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
                loss = att_loss_val

                loss = self.eta*spot_loss_val + (1-self.eta)*att_loss_val
                val_loss.append(loss.item())
                spots_acc = (100 - spots_loss_dict['class_error'])
                diff = torch.abs(y_pred - y)
                # val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
                att_acc = (diff < (y/10)).sum(0)
                all_accs = all_accs + att_acc
                pbar.set_description(f"val_acc: {att_acc, spots_acc}, val_loss:  {loss.item()}")
        return val_loss, all_accs/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=True, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        spots_bbox = np.zeros((0, 4))
        spots_labels = np.zeros((0))
        spots_target_bbox = np.zeros((0, 4))
        for i,(x, y,_,info) in enumerate(test_dataloader):
            x, spots_arr = x[:, :-2, :], x[:, -2:, :]
            x = x.to(device)
            tgt_spots = self.get_spot_dict(spots_arr)
            with torch.no_grad():
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    out_spots, y_pred = self.model(x1, x2)
                else:
                    out_spots, y_pred = self.model(x.unsqueeze(-1))
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                    conf_y = torch.abs(y - y_pred)
                spots_loss_dict = self.spots_loss(out_spots, tgt_spots)
                weight_dict = self.spots_loss.weight_dict
                spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
                # print(out_att.shape, y[:,0].shape)
                att_loss_val = self.criterion(y_pred, y)
                if conf:
                    att_loss_val += self.criterion(conf_pred, conf_y)
                loss = self.eta*spot_loss_val + (1-self.eta)*att_loss_val
            src_boxes, target_boxes = self.spots_loss.post_process(out_spots, tgt_spots)
            spots_bbox = np.concatenate((spots_bbox, src_boxes.cpu().numpy()))
            spots_target_bbox = np.concatenate((spots_target_bbox, target_boxes.cpu().numpy()))
            labels = out_spots['pred_labels'].argmax().cpu().numpy()
            spots_labels = np.concatenate((spots_labels,labels))
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.num_classes:
                targets = np.concatenate((targets, y.cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
            
        return preds, targets, confs, spots_bbox, spots_target_bbox, spots_labels

class ClassifierTrainer(Trainer):
    def __init__(self, num_classes=10,
                  regression_loss=torch.nn.MSELoss(),
                   eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.regression_loss = regression_loss
        self.eta = eta
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i,(x, y, _, info)  in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            print(info)
            y_gt = info['y_val'].float().to(device)
            self.optimizer.zero_grad()
            y_cls = self.model(x1.float(), x2.float())            
            loss = self.criterion(y_cls, y)
            # loss_val = self.regression_loss(y_val, y_gt)
            # loss = self.eta*loss_cls + (1-self.eta)*loss_val
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            # print("y_val: ", y_val.shape, "y_gt: ", y_gt.shape, "y_cls: ", y_cls.shape, "y: ", y.shape)
            acc = (y_cls.argmax(dim=1) == y.argmax(dim=1)).sum().item()   
            # acc = (torch.abs(y_val - y_gt) < y_gt/10).sum().item()
            train_acc += acc
            pbar.set_description(f"train_loss:  {loss.item()}, train_acc: {acc}")
            if i > self.max_iter:
                break
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        pbar = tqdm(self.val_dl)
        for i, (x, y, _,info) in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_gt = info['y_val'].to(device)
            with torch.no_grad():
                y_cls = self.model(x1.float(), x2.float())
            
            loss= self.criterion(y_cls, y)
            # loss_val = self.regression_loss(y_val, y_gt)
            # loss = self.eta*loss_cls + (1-self.eta)*loss_val
            acc = (y_cls.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            val_loss.append(loss.item())
            # acc = (torch.abs(y_val - y_gt) < y_gt/10).sum().item()
            val_acc += acc
            pbar.set_description(f"val_loss:  {loss.item()}, val_acc: {acc}")
            if i > self.max_iter:
                break
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds_cls = np.zeros((0, self.num_classes))
        preds_val = np.zeros((0))
        targets_cls = np.zeros((0, self.num_classes))
        targets_val = np.zeros((0))
        pbar = tqdm(test_dataloader)
        for i, (x, y, _, info) in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_gt = info['y_val'].to(device)
            with torch.no_grad():
                y_cls = self.model(x1.float(), x2.float())
            preds_cls = np.concatenate((preds_cls, y_cls.cpu().numpy()))
            # preds_val = np.concatenate((preds_val, y_val.cpu().numpy()))
            targets_cls = np.concatenate((targets_cls, y.cpu().numpy()))
            targets_val = np.concatenate((targets_val, y_gt.cpu().numpy()))
        return preds_cls, targets_cls, targets_val

class ClassifierKeplerTrainer(Trainer):
    def __init__(self, num_classes=10,
                  regression_loss=torch.nn.MSELoss(),
                    **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.regression_loss = regression_loss
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i,(x, y, _, _, _, info)  in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_gt = torch.tensor([v['y_val'] for v in info]).to(device)
            self.optimizer.zero_grad()
            y_cls = self.model(x1.float(), x2.float())            
            loss = self.criterion(y_cls, y)
            # loss_val = self.regression_loss(y_val, y_gt)
            # loss = self.eta*loss_cls + (1-self.eta)*loss_val
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            # print("y_val: ", y_val.shape, "y_gt: ", y_gt.shape, "y_cls: ", y_cls.shape, "y: ", y.shape)
            acc = (y_cls.argmax(dim=1) == y.argmax(dim=1)).sum().item()   
            # acc = (torch.abs(y_val - y_gt) < y_gt/10).sum().item()
            train_acc += acc
            pbar.set_description(f"train_loss:  {loss.item()}, train_acc: {acc}")
            if i > self.max_iter:
                break
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        pbar = tqdm(self.val_dl)
        for i, (x, y, _,_,_,info) in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_gt = torch.tensor([v['y_val'] for v in info]).to(device)
            with torch.no_grad():
                y_cls = self.model(x1.float(), x2.float())
            
            loss= self.criterion(y_cls, y)
            # loss_val = self.regression_loss(y_val, y_gt)
            # loss = self.eta*loss_cls + (1-self.eta)*loss_val
            acc = (y_cls.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            val_loss.append(loss.item())
            # acc = (torch.abs(y_val - y_gt) < y_gt/10).sum().item()
            val_acc += acc
            pbar.set_description(f"val_loss:  {loss.item()}, val_acc: {acc}")
            if i > self.max_iter:
                break
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds_cls = np.zeros((0, self.num_classes))
        preds_val = np.zeros((0))
        targets_cls = np.zeros((0, self.num_classes))
        targets_val = np.zeros((0))
        pbar = tqdm(test_dataloader)
        for i, (x, y, _, info) in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_gt = info['y_val'].to(device)
            with torch.no_grad():
                y_cls = self.model(x1.float(), x2.float())
            preds_cls = np.concatenate((preds_cls, y_cls.cpu().numpy()))
            # preds_val = np.concatenate((preds_val, y_val.cpu().numpy()))
            targets_cls = np.concatenate((targets_cls, y.cpu().numpy()))
            targets_val = np.concatenate((targets_val, y_gt.cpu().numpy()))
        return preds_cls, targets_cls, targets_val

class ContrastiveTrainer(Trainer):
    def __init__(self, temperature=1, stack_pairs=False, **kwargs):
        super().__init__(**kwargs)
        self.stack_pairs = stack_pairs
        self.temperature = temperature
        
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=None):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        pbar = tqdm(self.train_dl)
        for i, (x1, x2, _, _, info1, info2) in enumerate(pbar):
            x1, x2 = x1.to(device), x2.to(device)
            if self.stack_pairs:
                x = torch.cat((x1, x2), dim=0)
                out = self.model(x, temperature=self.temperature)
            else:
                out = self.model(x1, x2)
            self.optimizer.zero_grad()
            loss = out['loss']
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())  
            if i > (self.max_iter//self.world_size):
                break   
            pbar.set_description(f"train_loss:  {loss.item()}")
        return train_loss, 0.

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        pbar = tqdm(self.val_dl)
        for i, (x1, x2, _, _, info1, info2) in enumerate(pbar):
            x1, x2 = x1.to(device), x2.to(device)
            with torch.no_grad():
                if self.stack_pairs:
                    x = torch.cat((x1, x2), dim=0)
                    out = self.model(x, temperature=self.temperature)
                else:
                    out = self.model(x1, x2)
            loss = out['loss']
            val_loss.append(loss.item())  
            if i > (self.max_iter//self.world_size/5):
                break  
            pbar.set_description(f"val_loss:  {loss.item()}")     
        return val_loss, 0.
    
class HybridTrainer(Trainer):
    def __init__(self, temperature=1, stack_pairs=False, eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.stack_pairs = stack_pairs
        self.temperature = temperature
        self.eta = eta
        
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=None):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i, (x1, x2, y, _, info1, info2) in enumerate(pbar):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            if self.stack_pairs:
                x = torch.cat((x1, x2), dim=0)
                out = self.model(x, temperature=self.temperature)
            else:
                out = self.model(x1, x2)
            self.optimizer.zero_grad()
            loss_ssl = out['loss']
            y_pred = out['predictions']
            y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            conf_y = torch.abs(y - y_pred)
            loss_pred = self.criterion(y_pred, y) 
            loss_pred += self.criterion(conf_pred, conf_y)
            loss = loss_ssl*self.eta + loss_pred*(1-self.eta)

            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())  
            if i > (self.max_iter//self.world_size):
                break
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum(0)
            train_acc += acc
            pbar.set_description(f"train_loss:  {loss.item()}, train_acc: {acc}")
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        pbar = tqdm(self.val_dl)
        for i, (x1, x2, y, _, info1, info2) in enumerate(pbar):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            with torch.no_grad():
                if self.stack_pairs:
                    x = torch.cat((x1, x2), dim=0)
                    out = self.model(x, temperature=self.temperature)
                else:
                    out = self.model(x1, x2)
            loss_ssl = out['loss']
            y_pred = out['predictions']
            y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            conf_y = torch.abs(y - y_pred)
            loss_pred = self.criterion(y_pred, y) 
            loss_pred += self.criterion(conf_pred, conf_y)
            loss = loss_ssl*self.eta + loss_pred*(1-self.eta)

            val_loss.append(loss.item())  
            if i > (self.max_iter//self.world_size/5):
                break  
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum(0)
            val_acc += acc
            pbar.set_description(f"val_loss:  {loss.item()}, val_acc: {acc}")     
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        pbar = tqdm(self.val_dl)
        for i, (x1, x2, y, _, info1, info2) in enumerate(pbar):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            with torch.no_grad():
                if self.stack_pairs:
                    x = torch.cat((x1, x2), dim=0)
                    out = self.model(x, temperature=self.temperature)
                else:
                    out = self.model(x1, x2)
            y_pred = out['predictions']
            y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            targets = np.concatenate((targets, y.cpu().numpy()))
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            confs = np.concatenate((confs, conf_pred.cpu().numpy()))
           
            if i > (self.max_iter//self.world_size/5):
                break          
        return preds, targets, confs


class MaskedSSLTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        predicted_tokens = 0
        pbar = tqdm(self.train_dl)
        for i, (x, y, mask, info) in enumerate(pbar):
            y, mask, x = y.to(device), mask.to(device), x.to(device)
            out = self.model(x)
            loss = self.criterion(out, y, mask)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            train_acc += self.mask_accuracy(out, y, mask).item()
            predicted_tokens += mask.sum().item()
            pbar.set_description(f"train_loss:  {loss.item()}, train_acc: {train_acc/predicted_tokens}")
            if i > self.max_iter:
                break
        print("number of bad samples: ", self.train_dl.dataset.num_bad_samples)
        return train_loss, train_acc/predicted_tokens

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        predicted_tokens = 0
        pbar = tqdm(self.val_dl)
        for i, (x, y, mask, info) in enumerate(pbar):
            y, mask, x = y.to(device), mask.to(device), x.to(device)
            with torch.no_grad():
                out = self.model(x)
            loss = self.criterion(out, y, mask)
            val_loss.append(loss.item())
            val_acc += self.mask_accuracy(out, y, mask).item()
            predicted_tokens += mask.sum().item()
            pbar.set_description(f"val_loss:  {loss.item()}, val_acc: {val_acc/predicted_tokens}")
            if i > self.max_iter/5:
                break
        return val_loss, val_acc/predicted_tokens
    
    def mask_accuracy(self, result, target, inverse_token_mask, epsilon=1e-6):
        # print(inverse_token_mask.shape, result.shape, target.shape)
        r = result.masked_select(inverse_token_mask)
        t = target.masked_select(inverse_token_mask)
        s = (torch.abs(r - t) < epsilon).sum()
        return s

class CQRTrainer(object):
    def __init(self, trainer):
        self.trainer = trainer
    def fit(self, device, epochs=1):
        for epoch in range(epochs):
            self.trainer.train_epoch(device, epoch)
    def calibrate(self, device):
        self.trainer.eval_epoch
    def predict(self, device, test_dl):
        for i, (x, y, _, info) in enumerate(test_dl):
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            b, c, l, h = x.shape
            x_reshaped = x.reshape(b*c, l, h)
            x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            with torch.no_grad():              
                y_pred, features = self.trainer.model(x1.float(), x2.float(), return_features=True)
                y_pred = y_pred.reshape(b, c, -1, self.trainer.num_classes)
            mean_pred = y_pred.mean(dim=1)
            preds = np.concatenate((preds, mean_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))         
        preds[...,0] = self.trianer.criterion.predict(preds[...,0], self.trainer.nc_err_i)
        preds[...,1] = self.trainer.criterion.predict(preds[...,1], self.trainer.nc_err_p)
        return preds, targets

# class ACFPredictorKepler():
#     def __init__(self):
#         pass

#     def predict(self, test_dataset, prom=0.005):
#         pbar = tqdm(enumerate(test_dataset), total=len(test_dataset))
#         ps = []
#         pred_ps = []
#         for i, (x,target,_,_,info, info_y) in pbar:
#             p = target[1] = target[1] * (boundary_values_dict['Period'][1]
#             - boundary_values_dict['Period'][0]) + boundary_values_dict['Period'][0]
#             x_np = x.numpy().squeeze()
#             pred_p, lags, xcf, pe
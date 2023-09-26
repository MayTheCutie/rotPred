from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
import os
import yaml
from matplotlib import pyplot as plt
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
    def __init__(self, model, optimizer, criterion, train_dataloader, device,scheduler=None, val_dataloader=None,
                 optim_params=None, net_params=None, exp_num=None, log_path=None, exp_name=None, plot_every=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.optim = optimizer
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.device = device
        self.optim_params = optim_params
        self.net_params = net_params
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        self.logger = None
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

    def fit(self, num_epochs, device,  early_stopping=None, only_p=False, best='loss', conf=False):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        self.optim_params['lr_history'] = []
        epochs_without_improvement = 0

        print(f"Starting training for {num_epochs} epochs with parameters: {self.optim_params}, {self.net_params}")
        for epoch in range(num_epochs):
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch, only_p=only_p, plot=plot, conf=conf)
            t_loss_mean = np.mean(t_loss)
            self.logger.add_scalar('train_acc', t_acc, epoch)
            train_acc.append(t_acc)
            train_loss.extend(t_loss)

            v_loss, v_acc = self.eval_epoch(device, epoch=epoch, only_p=only_p, plot=plot, conf=conf)
            v_loss_mean = np.mean(v_loss)
            self.logger.add_scalar('validation_acc', v_acc, epoch)
            val_acc.append(v_acc)
            val_loss.extend(v_loss)
            if self.scheduler is not None:
                self.scheduler.step(v_loss_mean)
            criterion = min_loss if best == 'loss' else best_acc
            mult = 1 if best == 'loss' else -1
            objective = v_loss_mean if best == 'loss' else v_acc
            if mult*objective < mult*criterion:
                print("saving model...")
                if best == 'loss':
                    min_loss = v_loss_mean
                else:
                    best_acc = v_acc
                torch.save(self.model.state_dict(), f'{self.log_path}/exp{self.exp_num}/{self.exp_name}.pth')
                self.best_state_dict = self.model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
            if self.logger is not None:
                self.logger.add_scalar('time', time.time() - start_time, epoch)
                self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']
                self.optim_params['lr_history'].append(self.optim_params['lr'])
                print(f'Epoch {epoch}: Train Loss: {t_loss_mean:.6f}, Val Loss: {v_loss_mean:.6f}, Train Acc: {t_acc:.6f}, Val Acc: {v_acc:.6f}, Time: {time.time() - start_time:.2f}s')
                with open(f'{self.log_path}/exp{self.exp_num}/optim_params.yml', 'w') as outfile:
                    yaml.dump(self.optim_params, outfile, default_flow_style=False)

            if epoch % 10 == 0:
                print(os.system('nvidia-smi'))
        if self.logger is not None:
            self.logger.close()
        return {"num_epochs":num_epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc}
    
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        train_acc2 = 0
        for i, (x, y,_,_) in enumerate(self.train_dl):
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :2], y_pred[:, 2:]
                conf_y = torch.abs(y - y_pred) 
            # print(f"inclination range: {y_pred[:,1].min()} - {y_pred[:,1].max()}")
            # print(f"true y: {y[:10,:]}")
            # print(f"pred y: {y_pred[:10,:]}")

            # print("y_pred: ", y_pred.shape, "y: ", y.shape)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item() 
            # mean_acc = (diff[:,0]/(y[:,0])).sum().item()
            # train_acc2 += (diff[:,1] < 0.1).sum().item()
        print("number of train_accs: ", train_acc, train_acc2)
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i,(x, y,_,_) in enumerate(self.val_dl):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :2], y_pred[:, 2:]
                    conf_y = torch.abs(y - y_pred)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()  
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)

class ClassifierTrainer(Trainer):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        for i,(x, y, _,_)  in enumerate(self.train_dl):
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            # print("y_pred: ", y_pred.shape, "y: ", y.shape)
            
            loss1 = self.criterion(y_pred[:, :self.num_classes], y[:, :self.num_classes].argmax(dim=1))
            loss2 = self.criterion(y_pred[:, self.num_classes:], y[:, self.num_classes:].argmax(dim=1))
            loss = loss1 + loss2
            # print("y_pred: ", y_pred.argmax(1), "y: ", y.argmax(1))
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))

            # if not only_p:            
            #     # loss1 = self.criterion(y_pred_i, y[:, self.num_classes:])
            #     # loss += loss1
            #     loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            acc = (y_pred[:, :self.num_classes].argmax(dim=1) == y[:, :self.num_classes].argmax(dim=1)).sum().item()
            # if plot:
            #     print(y_pred.argmax(dim=1), y.argmax(dim=1))
                # self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss.append(loss.item())
            train_acc += acc
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, _, _) in enumerate(self.val_dl):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            
            loss1 = self.criterion(y_pred[:, :self.num_classes], y[:, :self.num_classes].argmax(dim=1))
            loss2 = self.criterion(y_pred[:, self.num_classes:], y[:, self.num_classes:].argmax(dim=1))
            loss = loss1 + loss2
            self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))

            # if not only_p:            
            #     # loss1 = self.criterion(y_pred_i, y[:, self.num_classes:])
            #     # loss += loss1
            #     loss = self.criterion(y_pred, y)
            acc = (y_pred[:, :self.num_classes].argmax(dim=1) == y[:, :self.num_classes].argmax(dim=1)).sum().item()
            # if plot:
            #     print(y_pred.argmax(dim=1), y.argmax(dim=1))
                # self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss.append(loss.item())
            val_acc += acc
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = np.zeros((0, self.num_classes*2))
        targets = np.zeros((0, self.num_classes*2))
        for x, y in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
                # y_pred = torch.cat((y_pred_p, y_pred_i), dim=1)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets


class SiameseTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=None):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        for x1, x2 in self.train_dl:
            # print("x1: ", x1.shape, "x2: ", x2.shape)
            x1, x2 = x1.to(device), x2.to(device)
            out = self.model(x1, x2)
            loss = out['loss']
            loss.backward()
            self.optim.step()
            train_loss.append(loss.item())          
        return train_loss, 0

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        for x1, x2 in self.val_dl:
            x1, x2 = x1.to(device), x2.to(device)
            with torch.no_grad():
                out = self.model(x1, x2)
            loss = out['loss']
            val_loss.append(loss.item())         
        return val_loss, 0


class EncoderTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y  in self.train_dl:
            # print("grads :", len(g), "last grad: ", g[-1])
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y) 
            loss.backward()
            self.optimizer.step()
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)

            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss += loss.item()
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of train_acc: ", train_acc)
        return train_loss/len(self.train_dl), train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for x, y in self.val_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch, data='val')
            
            acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss += loss.item()
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of val_acc: ", val_acc)
        return val_loss/len(self.val_dl), val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dl, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.to(device)
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
        preds = np.zeros((0, 2))
        targets = np.zeros((0, 2))
        for x, y in test_dl:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x, return_logits=True)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dl.dataset))
        return preds, targets

class EncoderDecoderTrainer(Trainer):
    def __init__(self, recon_loss, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.recon_loss = recon_loss
        self.alpha = alpha


    def train_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y  in self.train_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred, logits = self.model(x, return_logits=True)
            loss_pred = self.criterion(y_pred, y)
            x_norm = x[:,0,:] / x[:,0,:].max(dim=-1).values.unsqueeze(-1)
            logits = logits.argmax(dim=-1)
            logits_norm = logits / logits.max(dim=-1).values.unsqueeze(-1)
            loss_recon = self.recon_loss(logits_norm, x_norm)
            # loss = loss_pred + self.alpha * loss_recon
            loss = loss_pred
            loss.backward()
            self.optimizer.step()
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)

            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss += loss.item()
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of train_acc: ", train_acc)
        return train_loss/len(self.train_dl), train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for x, y in self.val_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred, logits = self.model(x, return_logits=True)
            loss_pred = self.criterion(y_pred, y)
            loss_recon = self.recon_loss(logits.argmax(dim=-1), x[:,0,:])
            # print(f"val - loss_pred: {loss_pred}, loss_recon: {loss_recon}")
            # loss = loss_pred + self.alpha * loss_recon
            loss = loss_pred
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch, data='val')
            
            acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss += loss.item()
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of val_acc: ", val_acc)
        return val_loss/len(self.val_dl), val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dl, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.to(device)
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
        preds = np.zeros((0, 2))
        targets = np.zeros((0, 2))
        for x, y in test_dl:
            x = x.to(device)
            with torch.no_grad():
                y_pred, _ = self.model(x, return_logits=True)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dl.dataset))
        return preds, targets
    

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
        for i, (x, x_masked, mask, _) in enumerate(self.train_dl):
            x_masked, mask, x = x_masked.to(device), mask.to(device), x.to(device)
            out = self.model(x_masked, x).squeeze()
            # print(mask.shape, out.shape, x.shape)
            # tm = mask.expand_as(out)  
            out = out.masked_fill(mask, 0)
            x_target = x.squeeze().masked_fill(mask, 0)
            # print(x_target.shape, out.shape ) 
            loss = self.criterion(out, x_target)
            if self.logger is not None:
                self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            loss.backward()
            self.optim.step()
            train_loss.append(loss.item())
            train_acc += self.mask_accuracy(out, x_target, mask).item()
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, x_masked, mask, _) in enumerate(self.val_dl):
            x_masked, mask, x = x_masked.to(device), mask.to(device), x.to(device)
            with torch.no_grad():
                out = self.model(x_masked, x).squeeze()
            # print(mask.shape, out.shape, x.shape)
            # tm = mask.expand_as(out)  
            out = out.masked_fill(mask, 0)
            x_target = x.squeeze().masked_fill(mask, 0)
            loss = self.criterion(out, x_target)
            if self.logger is not None:
                self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            if torch.isnan(loss).any():
                print("loss is nan")
                print("out: ", out, "x_target: ", x_target, "mask: ", mask)
            val_loss.append(loss.item())
            val_acc += self.mask_accuracy(out, x_target, mask).item()
            # print("val_loss: ",loss.item(), "val_acc: ", self.mask_accuracy(out, x_target, mask))
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def mask_accuracy(self, result, target, inverse_token_mask):
        # print(inverse_token_mask.shape, result.shape, target.shape)
        r = result.masked_select(~inverse_token_mask)  
        t = target.masked_select(~inverse_token_mask)  
        s = (r == t).sum()  
        return s
    
class KeplerTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        for i, (x, y) in enumerate(self.train_dl):
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            self.optimizer.zero_grad()
            y_pred = self.model(x.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :2], y_pred[:, 2:]
                conf_y = torch.abs(y['Period'] - y_pred[:,1]) 
            loss = self.criterion(y_pred[:,1].float(), y['Period'].float()) 
            if conf:
                loss += self.criterion(conf_pred[:,1].float(), conf_y.float())
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            loss.backward()
            self.optimizer.step()
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            train_loss.append(loss.item())
            diff = torch.abs(y_pred[:,1] - y['Period'])
            train_acc += (diff < (y['Period']/10)).sum().item() 
        print("number of train_accs: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y) in enumerate(self.val_dl):
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            self.optimizer.zero_grad()
            y_pred = self.model(x.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :2], y_pred[:, 2:]
                conf_y = torch.abs(y['Period'] - y_pred[:,1]) 
            loss = self.criterion(y_pred[:,1].float(), y['Period'].float()) 
            if conf:
                loss += self.criterion(conf_pred[:,1].float(), conf_y.float())
            self.logger.add_scalar('val_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            val_loss.append(loss.item())
            diff = torch.abs(y_pred[:,1] - y['Period'])
            val_acc += (diff < (y['Period']/10)).sum().item() 
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)


class DenoisingTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        self.model.train()
        train_loss = []
        for x, y  in self.train_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            if y_pred.isnan().any():
                print("y_pred is nan")
            # print(y_pred - y)
            loss = self.criterion(y_pred.squeeze(1), y.squeeze(1)) 
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
        return train_loss, 0 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        self.model.eval()
        val_loss = []
        for x, y  in self.val_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred.squeeze(1), y.squeeze(1)) 
            val_loss.append(loss.item())
        return val_loss, 0
    


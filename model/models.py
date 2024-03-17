import math
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as opt
import pytorch_lightning as pl

# from .loss import MaskedMSELoss, MaskedL1Loss, MaskedHuberLoss, IQRLoss
from util.stats import estimate_noise
# from neptune.types import File
import pandas as pd
import matplotlib.pyplot as plt

max_p, min_p = 0, 60
max_i, min_i = 0, np.pi/2


class LSTMFeatureExtractor(pl.LightningModule):
    def __init__(self, seq_len=1024, hidden_size=256, num_layers=4, num_classes=4,
                 in_channels=1, channels=256, dropout=0.2, kernel_size=4 ,stride=4):
        super(LSTMFeatureExtractor, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1, stride=2)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=channels, kernel_size=kernel_size, padding=1, stride=4)
        self.skip = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
        
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout1d(p=dropout)
        # self.batchnorm1 = nn.BatchNorm1d(64)
        # self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm1 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.num_features = self._out_shape()
        self.output_dim = self.num_features

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======       
            dummy_input = torch.randn(2,self.in_channels, self.seq_len)
            x = self.conv1(dummy_input)
            x = torch.swapaxes(x, 1,2)
            x_f,(h_f,_) = self.lstm(x)
            h_f = h_f.transpose(0,1).transpose(1,2)
            h_f = h_f.reshape(h_f.shape[0], -1)
            return h_f.shape[1] 
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        skip = self.skip(x)
        # x = self.conv(x)
        x = self.drop(self.activation(self.batchnorm1(self.conv1(x))))
        # x = self.drop(self.activation(self.batchnorm2(self.conv2(x))))
        # x = self.drop(self.activation(self.batchnorm3(self.conv3(x))))
        x = x + skip
        x = torch.swapaxes(x, 1,2)
        x_f,(h_f,c_f) = self.lstm(x)
        if return_cell:
            return x_f, h_f, c_f
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        return h_f

class LSTM(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4,  seq_len=1024, hidden_size=256, num_layers=4, num_classes=2,
                 in_channels=1, predict_size=256, channels=256, dropout=0.2, kernel_size=4,stride=4, log_path=None,
                 keep_ratio=0., random_ratio=1., uniform_bound=2., token_ratio=0., train_unit='standard',
                 task='ssl', train_loss='mse', **kwargs):
        super(LSTM, self).__init__()
        self.activation = nn.GELU()
        self.feature_extractor = LSTMFeatureExtractor(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,in_channels=in_channels,
                                                      channels=channels, dropout=dropout, kernel_size=kernel_size)
        self.num_classes = num_classes
        self.out_shape = self.feature_extractor.num_features
        self.predict_size = predict_size
        self.fc1 = nn.Linear(self.out_shape, predict_size)
        self.fc2 = nn.Linear(predict_size, num_classes)
        self.recons_head = nn.Linear(self.out_shape, seq_len)
        self.msk_token_emb = nn.Parameter(torch.randn(1, 1, self.out_shape))
        self.lr = lr
        self.weight_decay = weight_decay
        self.keep_ratio = keep_ratio
        self.random_ratio = random_ratio
        self.uniform_bound = uniform_bound
        self.token_ratio = token_ratio
        self.train_unit = train_unit
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_outputs = []
        self.task = task
        if self.task == 'ssl':
            if train_loss == 'mse':
                self.criterion = MaskedMSELoss()  # masked or not masked
            elif train_loss == 'mae':
                self.criterion = MaskedL1Loss()
            elif train_loss == 'huber':
                self.criterion = MaskedHuberLoss()
        else:
            self.criterion = nn.MSELoss()

        
        self.log_path = log_path

        # self.fc3 = nn.Linear(predict_size, num_classes)
    def apply_mask(self, x, mask):
        if mask is None:
            out = x
            out[torch.isnan(out)] = 0.
            return out, torch.zeros_like(x)

        r = torch.rand_like(x)
        keep_mask = (~mask | (r <= self.keep_ratio)).to(x.dtype)
        random_mask = (mask & (self.keep_ratio < r)
                       & (r <= self.keep_ratio+self.random_ratio)).to(x.dtype)
        token_mask = (mask & ((1-self.token_ratio) < r)).to(x.dtype)
        xm, xM = -self.uniform_bound, self.uniform_bound
        out = x * keep_mask + (torch.rand_like(x)*(xM-xm)+xm) * random_mask
        out[torch.isnan(out)] = 0.
        return out, token_mask

    def forward(self, x, mask=None):
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(1)
        x, token_mask = self.apply_mask(x, mask)
        # out = self.msk_token_emb * token_mask + (1-token_mask) * x
        h_f = self.feature_extractor(x)
        out = self.fc2(self.activation(self.fc1(h_f))) if self.task != 'ssl' else self.recons_head(h_f)
        # out2 = self.fc3(self.fc1(x_f))
        return out

    def configure_optimizers(self):
        optimiser = opt.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimiser

    def training_step(self, batch, batch_index):
        if self.task == 'ssl':
            return self.masked_training_step(batch, batch_index)
        x, y, m, info = batch
        m = None
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        out = dict()
        
        diff = torch.abs(pred - y)
        acc = (diff[:,0] < (y[:,0]/10)).sum().float()
        out.update({'loss': loss, 'train_acc': acc})
        print("loss", loss, "train accs in step", acc, "number of samples in step", y.shape[0])
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.train_step_outputs.append(out)
        # print("train accs in step", acc, "number of samples in step", y.shape[0])
        return out

    def masked_training_step(self, batch, batch_index):
        x, y, m, info = batch
        pred = self.forward(x, m)
        if self.train_unit == 'standard':
            loss = self.criterion(pred, y, m)
        elif self.train_unit == 'noise':
            noise = estimate_noise(y)
            loss = self.criterion(pred/noise, y/noise, m)
        elif self.train_unit == 'star':
            y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
            pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
            y_d = detrend(y_o, pred_o)
            loss = self.criterion(y_d, torch.ones_like(y_d), m)
        if torch.isnan(loss):
            print('Pred has nans?', torch.isnan(pred).sum().item())
            print('Y has nans?', torch.isnan(
                y).sum().item(), f' shape({y.shape})')
            print('M has fully masked items?',
                  ((m.int()-1).sum((1, 2)) == 0).sum().item())
            print('mu has nans?', torch.isnan(info['mu']).sum().item())
            raise ValueError('Nan Loss found during training')
        return {'loss': loss}

    def on_training_epoch_end(self):
        outputs = self.train_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)
        if self.task != 'ssl':
            avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
            self.log('train_acc', avg_acc)
        # print("number of train accs", self.train_acc, "number of samples", self.n_samples)
        # print(f"Epoch {self.current_epoch} train loss: {avg_loss}, train acc: {avg_acc}")
      

    def validation_step(self, batch, batch_index, dataloader_idx=None):
        if self.task == 'ssl':
            return self.masked_validation_step(batch, batch_index, dataloader_idx)
        x, y, m, info = batch
        m = None
        pred = self.forward(x)
        out = dict()
        loss = self.criterion(pred, y)
        diff = torch.abs(pred - y)
        acc = (diff[:,0] < (y[:,0]/10)).sum().float()
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        out.update({'val_loss': loss, 'val_acc': acc})
        self.validation_step_outputs.append(out)
        return out

    def masked_validation_step(self, batch, batch_index, dataloader_idx=None):
        variable_noise = 0.5
        x, y, m, info = batch
        pred = self.forward(x, m)
        
        noise = estimate_noise(y)
        variable = (noise <= variable_noise).squeeze()
        n_variables = variable.sum()
        pred_noise = pred / noise
        y_noise = y / noise

        # star normalised unit space
        y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
        pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
        y_d = detrend(y_o, pred_o)

        out = dict()
        if dataloader_idx is None or dataloader_idx == 0:  # Imputing
            # Imputation
            rmse = torch.sqrt(self.mse_loss(pred, y, m))
            
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise, m)) 
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d, m)) 
            mae = self.mae_loss(pred, y, m)
            mae_noise = self.mae_loss(pred_noise, y_noise, m) 
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d, m) 

            out.update({'val_mrmse': rmse, 'val_mmae': mae,
                        'val_mrmse_noise': rmse_noise, 'val_mmae_noise': mae_noise,
                        'val_mrmse_star': rmse_star, 'val_mmae_star': mae_star
                        })


    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.task == 'ssl':
            for dataloader_idx in range(len(outputs)):
                # print(outputs[dataloader_idx])
                for name, val in outputs[dataloader_idx].items():
                    if name in results:
                        results[name] += val.item()
                    else:
                        results[name] = val.item()
                    self.log(name, val, prog_bar=True)
            results = {k: v/len(outputs) for k, v in results.items()}
        else:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
            self.log('val_loss', avg_loss)
            self.log('val_acc', avg_acc)
        
    
    def test_step(self, batch, batch_index):
        x, y, m, info = batch
        m = None
        pred = self.forward(x)
        out = dict()
        loss = self.criterion(pred, y)
        out = dict()
        diff = torch.abs(pred - y)
        acc = (diff[:,0] < (y[:,0]/10)).sum().float()
        out['test_loss'] = loss
        out['test_acc'] = acc
        out['predictions'] = pred
        out['targets'] = y
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        shapes = [x['predictions'].shape[0] for x in outputs]
        print(f"Test loss: {avg_loss}, test acc: {avg_acc}, test shapes: {shapes}")
        tot_preds = torch.cat([x['predictions'] for x in outputs], dim=0)
        tot_targets = torch.cat([x['targets'] for x in outputs], dim=0)
        if self.log_path is not None:
            tot_preds[:,1] = tot_preds[:,1] * (max_p - min_p)+ min_p
            tot_preds[:,0] = (tot_preds[:,0] * (max_i - min_i)+ min_i)*180/np.pi
            tot_targets[:,1] = tot_targets[:,1] * (max_p - min_p)+ min_p
            tot_targets[:,0] = (tot_targets[:,0] * (max_i - min_i)+ min_i)*180/np.pi
            self.plot_results('inc', tot_targets[:,0].detach().cpu().numpy(), tot_preds[:,0].detach().cpu().numpy())
            self.plot_results('period', tot_targets[:,1].detach().cpu().numpy(), tot_preds[:,1].detach().cpu().numpy())

        pred_df = pd.DataFrame({'inc_preds': tot_preds[:,0].detach().cpu().numpy(), 'period_preds': tot_preds[:,1].detach().cpu().numpy(),
                                 'inc_targets': tot_targets[:,0].detach().cpu().numpy(), 'period_targets': tot_targets[:,1].detach().cpu().numpy()})
        self.logger.experiment["pred_df"].upload(File.as_html(pred_df))
        # self.logger.experiment["targets"].append(File.as_html(tot_targets.detach().cpu()))

        self.log('test_loss', avg_loss)
        self.log('test_acc', avg_acc)

    def plot_results(self, name, target, output, conf=''):
    
        diff = np.abs(output - target)
        print(diff)
        acc_6 = (diff < 6).sum()/len(diff)
        acc_10 = (diff < 10).sum()/len(diff)
        acc_10p = (diff < target/10).sum()/len(diff)

        plt.scatter(target, output)
        plt.plot(target, 0.9*target, color='red')
        plt.plot(target, 1.1*target, color='red')
        # plt.xlim(0, max(target) + 5)
        # plt.ylim(0, max(target) + 5)
        plt.title(f"{name} acc6={acc_6:.2f} acc10={acc_10:.2f} acc10p={acc_10p:.2f}")
        plt.xlabel("True ")
        plt.ylabel("Predicted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.log_path}/{name}_prediction.png")
        print(f"Saved {name} prediction plot on {self.log_path}/{name}_prediction.png")
        plt.clf()

class LSTM_ATTN(LSTM):
    def __init__(self, **kwargs):
        super(LSTM_ATTN, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        self.recons_head = nn.Linear(self.feature_extractor.hidden_size*2, self.feature_extractor.seq_len)
        self.msk_token_emb = nn.Parameter(torch.randn(1, 1, self.feature_extractor.hidden_size*2))

        

    def attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

    def forward(self, x, mask=None):
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(1)      
        x, token_mask = self.apply_mask(x, mask)
        # out = self.msk_token_emb * token_mask + (1-token_mask) * x 
        x_f, h_f, c_f = self.feature_extractor(x, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        attn, values = self.attention(c_f, x_f, x_f) 
        out = self.fc2(self.activation(self.fc1(values))) if self.task != 'ssl' else self.recons_head(values)
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return out


class Encoder(pl.LightningModule):
    def __init__(self,
                 n_dim=1,
                 d_model=64,
                 nhead=8,
                 dim_feedforward=128,
                 eye=0,
                 dropout=0.1,
                 num_layers=3,
                 lr=0.001,
                 weight_decay = 1e-4,
                 learned_pos=False,
                 norm='batch',
                 attention='full',
                 seq_len=1024,
                 keep_ratio=0.,
                 random_ratio=1.,
                 token_ratio=0.,
                 uniform_bound=2.,
                 train_unit='standard',
                 train_loss='mae',
                 task='ssl',
                 n_outputs=2,
                 **kwargs
                 ):
        """Instanciate a Lit TPT imputer module

        Args:
            n_dim (int, optional): number of input dimensions. Defaults to 1.
            d_model (int, optional): Encoder latent dimension. Defaults to 128.
            nhead (int, optional): Number of heads. Defaults to 8.
            dim_feedforward (int, optional): number of feedforward units in the encoder.
                Defaults to 256.
            dropout (float, optional): Encoder dropout. Defaults to 0.1.
            num_layers (int, optional): Number of encoder layer(s). Defaults to 3.
            lr (float, optional): AdamW earning rate. Defaults to 0.001.
        """
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.n_dim = n_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.norm = norm
        # self.zero_ratio =  zero_ratio
        self.keep_ratio = keep_ratio
        self.random_ratio = random_ratio
        self.uniform_bound = uniform_bound
        self.token_ratio = token_ratio
        self.train_unit = train_unit
        self.validation_step_outputs = []
        self.train_step_outputs = []
        assert train_unit in ['standard', 'noise', 'flux', 'star']

        self.ie = nn.Linear(n_dim, d_model)
        self.pe = PosEmbedding(d_model, learned=learned_pos)
        # self.ea = EyeAttention(eye)
        if attention == 'linear':
            self.encoder = Linformer(
                dim=d_model,
                seq_len=seq_len,
                depth=num_layers,
                heads=nhead,
                k=32,
                one_kv_head=True,
                share_kv=True
            )
        else:
            encoder_layer = TransformerEncoderLayer(d_model,
                                                    nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=0.1,
                                                    batch_first=True,
                                                    norm=norm, seq_len=seq_len,
                                                    attention=attention
                                                    )
            self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.recons_head = nn.Linear(d_model, n_dim)
        self.msk_token_emb = nn.Parameter(torch.randn(1, 1, d_model))
        if task == 'ssl':
            if train_loss == 'mse':
                self.criterion = MaskedMSELoss()  # masked or not masked
            elif train_loss == 'mae':
                self.criterion = MaskedL1Loss()
            elif train_loss == 'huber':
                self.criterion = MaskedHuberLoss()
            else:
                raise NotImplementedError
            self.mae_loss = MaskedL1Loss()
            self.mse_loss = MaskedMSELoss()
            self.iqr_loss = IQRLoss()
        else:
            if train_loss == 'mse':
                self.criterion = nn.MSELoss()
            elif train_loss == 'mae':
                self.criterion = nn.L1Loss()
            elif train_loss == 'huber':
                self.criterion = nn.SmoothL1Loss()
            else:
                raise NotImplementedError
        self.task = task
        self.cls_head = nn.Linear(d_model*seq_len, n_outputs)
        print('task', task)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Encoder")
        parser.add_argument("--n_dim", type=int)
        parser.add_argument("--d_model", type=int)
        parser.add_argument("--nhead", type=int)
        parser.add_argument("--dim_feedforward", type=int)
        parser.add_argument("--eye", type=int)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--num_layers", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--learned_pos", action='store_true')
        parser.add_argument("--norm", type=str)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--seq_len", type=int)
        parser.add_argument("--keep_ratio", type=float)
        parser.add_argument("--random_ratio", type=float)
        parser.add_argument("--token_ratio", type=float)
        parser.add_argument("--uniform_bound", type=float)
        parser.add_argument("--train_unit", type=str,
                            choices=['standard', 'noise', 'star'])
        parser.add_argument("--train_loss", type=str,
                            choices=['mae', 'mse', 'huber'])
        return parent_parser

    def configure_optimizers(self):
        optimiser = opt.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimiser

    def apply_mask(self, x, mask):
        if mask is None:
            out = x
            out[torch.isnan(out)] = 0.
            return out, torch.zeros_like(x)

        r = torch.rand_like(x)
        keep_mask = (~mask | (r <= self.keep_ratio)).to(x.dtype)
        random_mask = (mask & (self.keep_ratio < r)
                       & (r <= self.keep_ratio+self.random_ratio)).to(x.dtype)
        token_mask = (mask & ((1-self.token_ratio) < r)).to(x.dtype)
        xm, xM = -self.uniform_bound, self.uniform_bound
        out = x * keep_mask + (torch.rand_like(x)*(xM-xm)+xm) * random_mask
        out[torch.isnan(out)] = 0.
        return out, token_mask

    def forward(self, x, mask=None):
        x, token_mask = self.apply_mask(x, mask)
        out = self.ie(x)
        out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        # attention_mask = self.ea(x)
        out = self.encoder(out, mask=None)
        out = self.recons_head(out) if self.task == 'ssl' else self.cls_head(out.reshape(out.shape[0], -1))
        return out

    def get_attention_maps(self, x, mask=None):
        out, token_mask = self.apply_mask(x, mask)
        out = self.ie(out)
        if self.token_ratio:
            out = self.msk_token_emb * token_mask + (1-token_mask) * out

        out = out + self.pe(out)

        attention_mask = self.ea(x)
        out = self.encoder.get_attention_maps(out, mask=attention_mask)
        return out

    def training_step(self, batch, batch_index):
        out = dict()
        if self.task == 'ssl':
            return self.masked_training_step(batch, batch_index)
        x, y, m, info = batch
        m = None
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        out = dict()
        
        diff = torch.abs(pred - y)
        acc = (diff[:,0] < (y[:,0]/10)).sum().float()
        out.update({'loss': loss, 'train_acc': acc})
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.train_step_outputs.append(out)
        return out
        
    def masked_training_step(self, batch, batch_index):
        x, y, m, info = batch
        pred = self.forward(x, m)
        if self.train_unit == 'standard':
            loss = self.criterion(pred, y, m)
        elif self.train_unit == 'noise':
            noise = estimate_noise(y)
            loss = self.criterion(pred/noise, y/noise, m)
        elif self.train_unit == 'star':
            y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
            pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
            y_d = detrend(y_o, pred_o)
            loss = self.criterion(y_d, torch.ones_like(y_d), m)
        if torch.isnan(loss):
            print('Pred has nans?', torch.isnan(pred).sum().item())
            print('Y has nans?', torch.isnan(
                y).sum().item(), f' shape({y.shape})')
            print('M has fully masked items?',
                  ((m.int()-1).sum((1, 2)) == 0).sum().item())
            print('mu has nans?', torch.isnan(info['mu']).sum().item())
            raise ValueError('Nan Loss found during training')
        return {'loss': loss}

    def on_training_epoch_end(self):
        outputs = self.train_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)
        if self.task != 'ssl':
            avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
            self.log('train_loss', avg_loss)
        print(f"Epoch {self.current_epoch} train loss: {avg_loss}, train acc: {avg_acc}")


    def validation_step(self, batch, batch_index, dataloader_idx=None):
        if self.task == 'ssl':
            return self.masked_validation_step(batch, batch_index, dataloader_idx)       

        x, y, m, info = batch
        m = None
        pred = self.forward(x, m)
        out = dict()
        loss = self.criterion(pred, y)
        diff = torch.abs(pred - y)
        acc = (diff[:,0] < (y[:,0]/10)).sum()
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        out.update({'val_loss': loss, 'val_acc': acc / y.shape[0]})
        self.validation_step_outputs.append(out)
        return out

    def masked_validation_step(self, batch, batch_index, dataloader_idx=None):
        variable_noise = 0.5
        x, y, m, info = batch
        pred = self.forward(x, m)
        
        noise = estimate_noise(y)
        variable = (noise <= variable_noise).squeeze()
        n_variables = variable.sum()
        pred_noise = pred / noise
        y_noise = y / noise

        # star normalised unit space
        y_o = inverse_standardise_batch(y, info['mu'], info['sigma'])
        pred_o = inverse_standardise_batch(pred, info['mu'], info['sigma'])
        y_d = detrend(y_o, pred_o)

        out = dict()
        if dataloader_idx is None or dataloader_idx == 0:  # Imputing
            # Imputation
            rmse = torch.sqrt(self.mse_loss(pred, y, m))
            
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise, m)) 
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d, m)) 
            mae = self.mae_loss(pred, y, m)
            mae_noise = self.mae_loss(pred_noise, y_noise, m) 
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d, m) 

            out.update({'val_mrmse': rmse, 'val_mmae': mae,
                        'val_mrmse_noise': rmse_noise, 'val_mmae_noise': mae_noise,
                        'val_mrmse_star': rmse_star, 'val_mmae_star': mae_star
                        })

        if dataloader_idx is None or dataloader_idx == 1:
            # Bias
            rmse = torch.sqrt(self.mse_loss(pred, y))
            rmse_noise = torch.sqrt(self.mse_loss(pred_noise, y_noise)) 
            rmse_star = torch.sqrt(self.mse_loss(torch.ones_like(y_d), y_d)) 
            mae = self.mae_loss(pred, y)
            mae_noise = self.mae_loss(pred_noise, y_noise)
            mae_star = self.mae_loss(torch.ones_like(y_d), y_d) 

            out.update({'val_rmse': rmse, 'val_mae': mae,
                        'val_rmse_noise': rmse_noise, 'val_mae_noise': mae_noise,
                        'val_rmse_star': rmse_star, 'val_mae_star': mae_star
                        })
            # Denoising
            iqr = self.iqr_loss(pred, y)
            iqr_variable = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable = self.iqr_loss((pred-y)[variable])
            iqr_noise = self.iqr_loss(pred_noise, y_noise)
            iqr_variable_noise = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable_noise = self.iqr_loss(
                    (pred_noise-y_noise)[variable])
            iqr_star = self.iqr_loss(y_d)
            iqr_variable_star = torch.tensor(np.nan, device=pred.device)
            if n_variables:
                iqr_variable_star = self.iqr_loss(y_d[variable])

            out.update({'val_IQR': iqr, 'val_IQR_var': iqr_variable,
                        'val_IQR_noise': iqr_noise, 'val_IQR_var_noise': iqr_variable_noise,
                        'val_IQR_star': iqr_star, 'val_IQR_var_star': iqr_variable_star,
                        })
            self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        results = dict()
        if self.task == 'ssl':
            for dataloader_idx in range(len(outputs)):
                # print(outputs[dataloader_idx])
                for name, val in outputs[dataloader_idx].items():
                    if name in results:
                        results[name] += val.item()
                    else:
                        results[name] = val.item()
            
                    # for x in outputs[dataloader_idx]:
                    #     print(x)
                    # score = torch.stack([x[name]
                    #                     for x in outputs[dataloader_idx]]).mean()
                    self.log(name, val, prog_bar=True)
            results = {k: v/len(outputs) for k, v in results.items()}
            # print(f"Epoch {self.current_epoch} validation: {results}")
        else:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
            self.log('val_loss', avg_loss)
            self.log('val_acc', avg_acc)
            print(f"Epoch {self.current_epoch} val loss: {avg_loss}, val acc: {avg_acc}")

        

    def test_step(self, batch, batch_index, dataloader_idx=None):
        d_out = self.validation_step(batch, batch_index, dataloader_idx)
        return {k.replace('val', 'test'): v for k, v in d_out.items()}

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)


class PosEmbedding(nn.Module):
    def __init__(self, d_model, learned=False, max_len=5000, dtype=torch.float32):
        super(PosEmbedding, self).__init__()
        if learned:
            self.pe = LearnedPosEmbedding(
                d_model, max_len=max_len, dtype=dtype)
        else:
            self.pe = FixedPosEmbedding(d_model, max_len=max_len, dtype=dtype)

    def forward(self, x):
        return self.pe(x)


class FixedPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.float32):
        super(FixedPosEmbedding, self).__init__()
        # Compute the positional embeddings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=dtype)
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LearnedPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000, dtype=torch.float32):
        super(LearnedPosEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model, dtype=dtype))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required). (B, L, D)
        Shape:
            output: tensor of shape (B, L, D)
        """
        return self.pe[:, :x.size(1), :]


class BatchNorm(nn.BatchNorm1d):
    """Overrides nn.BatchNorm1d to define shape structure identical
    to LayerNorm, i.e. (N, L, C) and not (N, C, L)"""

    def forward(self, input):
        return super().forward(input.transpose(1, 2)).transpose(1, 2)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Overrides nn.TransformerEncoderLayer class with
    - BatchNorm option as suggested by Zerveas et al https://arxiv.org/abs/2010.02803
    - PrboSparse attention from Zhou et al 
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 norm='layer', attention='full', seq_len=None,
                 device=None, dtype=None) -> None:
        # this combination of shapes hasn't been dealt with yet
        assert batch_first or norm == 'layer'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,
                                                      layer_norm_eps, batch_first, norm_first, device, dtype)

        if attention == 'full':
            pass
        else:
            raise NotImplementedError
        if norm == 'layer':
            pass
        elif norm == 'batch':
            self.norm1 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = BatchNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            raise NotImplementedError


class   TransformerEncoder(nn.TransformerEncoder):
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, x, x, attn_mask=mask)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


def inverse_standardise_batch(x, mu, sigma):
    return x * sigma + mu


def detrend(x, trend):
    return x / trend

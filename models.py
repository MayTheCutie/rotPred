import torch
import torch.nn as nn
import torch.nn.functional as F
from lightPred.transformer_models import TransformerEncoderDecoder, TransformerEncoder as TEncoder, TransformerDecoder as TDecoder
import math
from torch.nn.utils import weight_norm
from lightPred.utils import residual_by_period
from lightPred.period_analysis import analyze_lc_torch, analyze_lc
from lightPred.Informer2020.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, HwinEncoderLayer
from lightPred.Informer2020.models.decoder import Decoder, DecoderLayer
from lightPred.Informer2020.models.attn import FullAttention, ProbAttention, AttentionLayer, HwinAttentionLayer
from lightPred.Informer2020.models.embed import DataEmbedding
from lightPred.Autoformer.layers.Autoformer_EncDec import Encoder as AutoformerEncoder, EncoderLayer as AutoformerEncoderLayer, my_Layernorm as AutoformerLayerNorm, series_decomp
from lightPred.Autoformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from torchvision.models.swin_transformer import SwinTransformer
from lightPred.conformer.conformer.convolution import ConformerConvModule, Conv2dSubampling
from lightPred.conformer.conformer.encoder import ConformerBlock


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # print("projection_MLP: ", x.shape)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Adjust the dimensions using a 1x1 convolutional layer if needed
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, block, blocks,seq_len=1024, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        layers = [self.make_layer(block, 64*2**i, blocks[i], stride=min(2,i+1) ) for i in range(len(blocks))]

        self.layers = torch.nn.Sequential(*layers)
        self.out_shape = self._out_shape()
        self.output_dim = self.out_shape[-1]
        # self.avg_pool = nn.MaxPool1d(2)
        # self.fc = nn.Linear(64*2**(len(blocks)-1), num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        print(stride)
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.randn(1,1, self.seq_len)
            output = self.forward(dummy_input)
            # n_features = output.numel() // output.shape[0]  
            return output.shape 
        finally:
            torch.set_rng_state(rng_state)

    def forward(self, x):
        # print("stating backbone with: ", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layers(out)
    
        # out = self.avg_pool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # print("after backbone: ", out.shape)

        return out.view(out.shape[0], -1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        print("ConvBlock: ", in_channels, out_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.pool = nn.MaxPool1d(kernel_size=stride)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        skip = self.skip(x)
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(self.pool(x))
        x = x + skip
        return x

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, seq_len=1024, hidden_size=256, num_layers=5, num_classes=4,
                 in_channels=1, channels=256, dropout=0.2, kernel_size=4 ,stride=4, image=False):
        super(LSTMFeatureExtractor, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.image = image
        self.stride = stride
        self.t_features = self.seq_len//self.stride
        print("image: ", image)
        # self.conv = Conv2dSubampling(in_channels=in_channels, out_channels=channels)
        if not image:
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool1d(kernel_size=stride)
            self.skip = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout1d(p=dropout)
            self.batchnorm1 = nn.BatchNorm1d(channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool2d(kernel_size=stride)
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout2d(p=dropout)
            self.batchnorm1 = nn.BatchNorm2d(channels)        
        # self.conv_list = nn.Sequential(
        #                         ConvBlock(1,channels//4, kernel_size=3, stride=2, padding=1, dropout=dropout),
        #                         ConvBlock(channels//4,channels//2, kernel_size=3, stride=2, padding=1, dropout=dropout),
        #                         ConvBlock(channels//2,channels, kernel_size=3, stride=2, padding=1, dropout=dropout))
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1, stride=2)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=channels, kernel_size=kernel_size, padding=1, stride=4)
        
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
       
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
        # try:
        print("calculating out shape")
        if not self.image:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len)
        else:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len, self.seq_len)
        # dummy_input = torch.randn(2,self.seq_len, self.in_channels)
        input_length = torch.ones(2, dtype=torch.int64)*self.seq_len
        print("dummy_input: ", dummy_input.shape)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(dummy_input)))))
        # x = self.conv(dummy_input, input_length)
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2)
        x_f,(h_f,_) = self.lstm(x)
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        print("finished")
        return h_f.shape[1] 
        # finally:
        #     torch.set_rng_state(rng_state)

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(-1)
        # elif len(x.shape) == 3 and x.shape[1] == 1:
        #     x = x.transpose(-1,-2)
        # input_length = torch.ones(x.shape[0], dtype=torch.int64)*self.seq_len
        # x = self.conv(x, input_length)
        skip = self.skip(x)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(x)))))
        x = x + skip
        # x = self.conv(x)
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2)
        x_f,(h_f,c_f) = self.lstm(x)
        if return_cell:
            return x_f, h_f, c_f
        h_f = h_f.transpose(0,1).transpose(1,2)
        # h_f = h_f.max(dim=1).values
        h_f = h_f.reshape(h_f.shape[0], -1)
        return h_f
    
class LSTM(nn.Module):
    def __init__(self, seq_len=1024, hidden_size=64, num_layers=5, num_classes=4,
                 in_channels=1, predict_size=128, channels=256, dropout=0.35, kernel_size=4,stride=4, image=False):
        super(LSTM, self).__init__()
        self.activation = nn.GELU()
        self.feature_extractor = LSTMFeatureExtractor(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,in_channels=in_channels,
                                                      channels=channels, dropout=dropout, kernel_size=kernel_size,
                                                      image=image)
        self.num_classes = num_classes
        self.out_shape = self.feature_extractor.num_features
        self.predict_size = predict_size
        self.fc1 = nn.Linear(self.out_shape, predict_size)
        self.fc2 = nn.Linear(predict_size, num_classes)
        # self.fc3 = nn.Linear(predict_size, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        h_f = self.feature_extractor(x)
        out = self.fc2(self.activation(self.fc1(h_f)))
        # out2 = self.fc3(self.fc1(x_f))
        return out

class LSTM_ATTN(LSTM):
    def __init__(self, **kwargs):
        super(LSTM_ATTN, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        # self.fc3 = nn.Linear(self.feature_extractor.hidden_size*4, self.predict_size)
        # self.fc4 = nn.Linear(self.predict_size, self.num_classes)


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
        return linear_combination

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        x_f, h_f, c_f = self.feature_extractor(x, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        values = self.attention(c_f, x_f, x_f) 
        out = self.fc2(self.activation(self.fc1(values)))
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return out

class LSTM_ATTN2(LSTM):
    def __init__(self, attn='prob',n_heads=4, factor=5, dropout=0.35, output_attention=False, num_att_layers=4, **kwargs):
        super(LSTM_ATTN2, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        self.pool = nn.AdaptiveMaxPool1d(self.feature_extractor.hidden_size)
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=self.feature_extractor.hidden_size*2,
            num_attention_heads=n_heads,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
        ) for _ in range(num_att_layers)])
        # Attn = ProbAttention if attn=='prob' else FullAttention
        # self.attention = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model=self.feature_extractor.hidden_size*2, n_heads=n_heads, mix=False)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        outputs, h_f, c_f = self.feature_extractor(x, return_cell=True)
        # print("x_f: ", x_f.shape)
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = self.pool(outputs.permute(0,2,1)).reshape(outputs.shape[0], -1)
        # c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        # values, _ = self.attention(x_f, x_f, x_f, attn_mask=None) 
        outputs = self.fc2(self.activation(self.fc1(outputs)))
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return outputs

class LSTM_ATTN_QUANT(LSTM):
    def __init__(self, attn='prob',n_heads=8, factor=5, dropout=0.1, output_attention=False, n_q=3, **kwargs):
        super(LSTM_ATTN_QUANT, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.attention = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model=self.feature_extractor.hidden_size*2, n_heads=n_heads, mix=False)
        self.quantiles = n_q
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        x_f, h_f, c_f = self.feature_extractor(x, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        values, _ = self.attention(x_f, x_f, x_f, attn_mask=None) 
        values = values.unsqueeze(1).expand( values.shape[0],self.quantiles, values.shape[1], values.shape[2])
        out = self.fc2(self.activation(self.fc1(values.max(dim=2).values)))
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return out.transpose(1,2)

class LSTM_SWIN(LSTM_ATTN):
    def __init__(self, patch_size=[16,16], im_embed_dim=64, depths=[4,6,6,4], num_heads=[4,8,8,4],
     window_size=[8,8], im_dropout=0.3, swin_weight=1, **kwargs):
        super(LSTM_SWIN, self).__init__(**kwargs)
        self.swin = SwinTransformer(patch_size=patch_size, embed_dim=im_embed_dim, depths=depths, num_heads=num_heads,
                                     window_size=window_size, dropout=im_dropout)
        self.swin.head = nn.Identity()
        num_swin_features =  im_embed_dim * 2 ** (len(depths) - 1)
        num_lstm_features = self.feature_extractor.hidden_size*2
        self.swin_weight = swin_weight
        self.fc1 = nn.Linear(num_swin_features + num_lstm_features, self.predict_size)

        
    def forward(self, x_im, x_t):
        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)
        if x_im.shape[1] == 1:
            x_im = x_im.repeat(1,3,1,1) # RGB like
        x_f, h_f, c_f = self.feature_extractor(x_t, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)
        t_features = self.attention(c_f, x_f, x_f) 
        im_features = self.swin_weight*self.swin(x_im)
        values = torch.cat([t_features, im_features], dim=1)
        out = self.fc2(self.activation(self.fc1(values)))
        return out

class BERTEncoder(nn.Module):
    def __init__(self, ntoken=1024, vocab_size=1024, d_model=768, nhead=12, nlayers=12, in_channels=1, dropout=0.2, num_classes=4):
        super(BERTEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=3, padding=1, stride=4)
        self.skip = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1, padding=0, stride=4)
        
        self.embedding = torch.nn.Sequential(self.conv1, nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, nlayers)
        self.output_dim = d_model*ntoken//4

        self.encoder2hidden = nn.Linear(d_model*ntoken//4, vocab_size)
        self.hidden2out = nn.Linear(vocab_size, num_classes)


        # self.token_prediction_layer = nn.Linear(d_model, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=-1)  
        # self.classification_layer = nn.Linear(d_model, 2)


    def forward(self, input_tensor, attention_mask=None):
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(1) 
        # print("input_tensor: ", input_tensor.shape)
        skip = self.skip(input_tensor)
        embedded = self.embedding(input_tensor)
        embedded = embedded + skip
        embedded = self.layer_norm(embedded.transpose(1,2))
        # print("nans2: ", torch.isnan(embedded).any()) 

        # print("max embedded: ", embedded.max())
        encoded = self.encoder(embedded, attention_mask) # [batch_size, seq_len, d_model]
        return encoded.view(encoded.shape[0], -1)  
        # print("nans3: ", torch.isnan(encoded).any()) 

        # hidden = self.encoder2hidden(encoded.view(encoded.shape[0], -1)) # [batch_size, vocab_size]
        # out = self.hidden2out(hidden) # [batch_size, num_classes]
        # return out
        # token_predictions = self.token_prediction_layer(encoded)  # [batch_size, seq_len, vocab_size]
        # print("nans4: ", torch.isnan(token_predictions).any()) 
        # print("nans5: ", torch.isnan(self.softmax(token_predictions)).any())
  
        # first_word = encoded[:, 0, :] # [batch_size, d_model]  
        # return self.softmax(token_predictions) 
        
    
class EncoderDecoder(nn.Module):
    def __init__(self, dropout=0.2):
        super(EncoderDecoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN1DBackBone(nn.Module):
    def __init__(self, input_channels, output_channels=512):
        super(CNN1DBackBone, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels//8, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm1d(output_channels//8)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=output_channels//8, out_channels=output_channels//4, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(output_channels//4)
        
        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=output_channels//4, out_channels=output_channels//2, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(output_channels//2)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv1d(in_channels=output_channels//2, out_channels=output_channels, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm1d(output_channels)

        # 1x1 Convolutional layers for skip connections
        self.skip_conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels//8, kernel_size=1, stride=2)
        self.skip_conv2 = nn.Conv1d(in_channels=output_channels//8, out_channels=output_channels//4, kernel_size=1, stride=2)
        self.skip_conv3 = nn.Conv1d(in_channels=output_channels//4, out_channels=output_channels//2, kernel_size=1, stride=2)
        
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=0.1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        
    def forward(self, x):
        skip = self.skip_conv1(x)
        x = self.drop(self.activation(self.bn1(self.conv1(x))))
        x = x + skip
        skip = self.skip_conv2(x)
        x = self.drop(self.activation(self.bn2(self.conv2(x))))
        x = x + skip
        skip = self.skip_conv3(x)
        x = self.drop(self.activation(self.bn3(self.conv3(x))))
        x = x + skip
        x = self.drop(self.activation(self.bn4(self.conv4(x))))
        
        # x = x.view(x.size(0), -1)  # Flatten the output
        
        return x

class CNN(nn.Module):
    """
    1D CNN model architecture.
    
    Attributes
    ----------
    num_in : int
        Exposure in seconds.
        
    log : _io.TextIOWrapper
        Log file.
    
    kernel1, kernel2 : int
        Kernel width of first and second convolution, respectively.
    stride1, stride2 : int
        Stride of first and second convolution, respectively.
    
    padding1, padding2 : int
        Zero-padding of first and second convolution, respectively.
    dropout : float
        Dropout probability applied to fully-connected part of network.
    
    hidden1, hidden2, hidden3 : int
        Number of hidden units in the first, second, and third fully-connected
        layers, respectively.
    
    Methods
    -------
    forward(x)
        Forward pass through the model architecture.
    """
    def __init__(self, t_samples, kernel1=3, kernel2=3, stride1=1, stride2=1, \
                 padding1=1, padding2=1, dropout=0.2, hidden1=2048, hidden2=1024, \
                 hidden3=256, out_channels1=64, out_channels2=16, out_dim=1):
    
        super(CNN, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.num_in = t_samples

        self.out_channels_1 = out_channels1
        dilation1 = 1
        poolsize1 = 4
        
        self.out_channels_2 = out_channels2
        dilation2 = 1
        poolsize2 = 2

        # first convolution
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=self.out_channels_1,
                               kernel_size=kernel1,
                               dilation=dilation1,
                               stride=stride1,
                               padding=padding1)
        self.num_out = ((self.num_in+2*padding1-dilation1* \
                         (kernel1-1)-1)/stride1)+1
        assert str(self.num_out)[-1] == '0'

        self.bn1 = nn.BatchNorm1d(num_features=self.out_channels_1)
        self.pool1 = nn.AvgPool1d(kernel_size=poolsize1)
        self.num_out = (self.num_out/poolsize1)
        assert str(self.num_out)[-1] == '0'

        
        # hidden convolution
        self.conv_hidden = nn.Conv1d(in_channels=self.out_channels_1,
                               out_channels=self.out_channels_1,
                               kernel_size=kernel2,
                               stride=stride2,
                               padding=padding2)
        self.bn_hidden = nn.BatchNorm1d(num_features=self.out_channels_1)
        self.pool_hidden = nn.AvgPool1d(kernel_size=poolsize2)

        self.conv2 = nn.Conv1d(in_channels=self.out_channels_1,
                               out_channels=self.out_channels_2,
                               kernel_size=kernel2,
                               stride=stride2,
                               padding=padding2)
        self.num_out = ((self.num_out+2*padding2-dilation2* \
                         (kernel2-1)-1)/stride2)+1
        assert str(self.num_out)[-1] == '0'
        self.bn2 = nn.BatchNorm1d(num_features=self.out_channels_2)
        self.pool2 = nn.AvgPool1d(kernel_size=poolsize2)
        self.num_out = (self.num_out/(poolsize2**5))
        assert str(self.num_out)[-1] == '0'
        
        # fully-connected network
        self.num_out = self.out_channels_2*self.num_out
        assert str(self.num_out)[-1] == '0'
        self.num_out = int(self.num_out)
        self.linear1 = nn.Linear(2*self.num_out, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden3)
        
        # output prediction
        self.predict = nn.Linear(hidden3, out_dim)
    

    def forward(self, x):
        """
        Forward pass through the model architecture.
            
        Parameters
        ----------
        x : array_like
            Input time series data.
            
        s : array_like
            Standard deviation array.
            
        Returns
        ----------
        x : array_like
            Output prediction.
        """
        s = torch.ones((x.shape[0], self.num_out),device=x.device)*torch.std(x)
        # print("s.shape", s.shape,   "x.shape", x.shape)
        x = self.pool1(F.relu(self.bn1((self.dropout((self.conv1(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv_hidden(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv_hidden(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv_hidden(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv_hidden(x)))))))
        x = self.pool2(F.relu(self.bn2((self.dropout((self.conv2(x)))))))
       
        x = x.view(-1, self.num_out)
        x = torch.cat((x, s), 1)

        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = F.relu(self.linear3(x))
        x = self.predict(x)
        
        return x.float()
    

class FeedForward(nn.Module):
    def __init__(self, t_samples, input_size_parameters, hidden_size=256, predict_size=64, out_dim=1):
        super(FeedForward, self).__init__()

        # Branch for processing time series data
        # self.branch_time_series = nn.Sequential(
        #     nn.Linear(t_samples, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        # )

        self.branch_time_series = LSTM_ATTN(seq_len=t_samples, hidden_size=hidden_size, num_layers=5, num_classes=out_dim,
                 in_channels=1, predict_size=predict_size, channels=256, dropout=0.2, kernel_size=4,stride=4, image=False)

        # Branch for processing additional parameters
        self.branch_parameters = nn.Sequential(
            nn.Linear(input_size_parameters, input_size_parameters*2),
            nn.ReLU(),
            nn.Linear(input_size_parameters*2, input_size_parameters),
            nn.ReLU(),
        )

        # Fully connected layers after concatenating the outputs from both branches
        self.fc = nn.Sequential(
            nn.Linear(predict_size + input_size_parameters, predict_size),
            nn.ReLU(),
            nn.Linear(predict_size, out_dim),
        )

    def forward(self, x_time_series, x_parameters):
        # print('nans before all: ', torch.any(torch.isnan(x_time_series)).item(), torch.any(torch.isnan(x_parameters)).item())
        x_f, h_f, c_f = self.branch_time_series.feature_extractor(x_time_series, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        values = self.branch_time_series.attention(c_f, x_f, x_f) 
        out_time_series = self.branch_time_series.activation(self.branch_time_series.fc1(values))
        # out_time_series = self.branch_time_series(x_time_series)
        out_parameters = self.branch_parameters(x_parameters)
        # Concatenate the outputs from both branches

        out = torch.cat((out_time_series, out_parameters), dim=1)
        # print('nans after cat: ', torch.any(torch.isnan(out)).item())


        # Pass through fully connected layers
        out = self.fc(out)
        return out
       
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int,
                 nlayers: int, in_channels=1, stride=1, dropout: float = 0.5, num_classes=2):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.ntoken = ntoken
        self.in_channels = in_channels

        self.backbone = CNN1DBackBone(in_channels, d_model)
        # self.conv = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=3, padding=1, stride=stride)
        # self.skip = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1, padding=0, stride=stride)
        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=ntoken)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, activation='gelu')
        # decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.transformer_decoder = nn.TransformerDecoder(encoder_layers, nlayers)
        self.dropout = nn.Dropout1d(p=dropout)
        # self.batchnorm = nn.BatchNorm1d(d_model)
        self.activation = nn.GELU()
        self.linear1 = nn.Linear(d_model*ntoken//16, 256)
        # self.linear2 = nn.Linear(2048, 1024)
        # self.linear3 = nn.Linear(1024, 256)
        self.hidden2output = nn.Linear(256, num_classes)

        # self.out_shape = self._out_shape() # Number of features extracted by the conv layers.
        # print("out shape: ", self.out_shape)
        # self.fc1 = nn.Linear(self.out_shape[1], d_hid)
        # self.fc2 = nn.Linear(d_hid, 2)
        
        # self.embedding = nn.Embedding(ntoken, d_model)
        # self.linear = nn.Linear(d_model*ntoken, 2)

        # self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.fc1.weight.data.uniform_(-initrange, initrange)
    #     self.fc1.bias.data.zero_()
    #     self.fc2.bias.data.zero_()
    #     self.fc2.weight.data.uniform_(-initrange, initrange)

    # def _out_shape(self) -> int:
    #     """
    #     Calculates the number of extracted features going into the the classifier part.
    #     :return: Number of features.
    #     """
    #     # Make sure to not mess up the random state.
    #     rng_state = torch.get_rng_state()
    #     # ====== YOUR CODE: ======
        
    #     dummy_input = torch.randn(2,self.in_channels, self.ntoken)
    #     x = self.drop(self.activation(self.batchnorm(self.conv(dummy_input))))
    #     # x = self.drop(self.activation(self.batchnorm(self.conv2(x))))
    #     # x = self.drop(self.activation(self.batchnorm(self.conv3(x))))
    #     x = torch.swapaxes(x, 1,2)
    #     x = self.transformer_encoder(x) # (batch_size, seq_len, d_model)
    #     # h_f = h_f.transpose(0,1).transpose(1,2)
    #     x = x.reshape(x.shape[0], -1)

        
    #     # n_features = output.numel() // output.shape[0]  
    #     return x.shape 

    def forward(self, src, src_mask = None):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if len(src.shape) == 2:
            src = src.unsqueeze(1)
        elif len(src.shape) == 3 and src.shape[-1] == 1:
            src = src.transpose(-1,-2)
        # skip = self.skip(src)
        # x = self.drop(self.activation(self.batchnorm(self.conv(src))))
        # x = self.drop(self.activation(self.batchnorm(self.conv2(x))))
        # x = self.drop(self.activation(self.batchnorm(self.conv3(x))))
        # x = x + skip
        x = self.backbone(src)
        # print("after conv: ", x.shape)
        x = torch.swapaxes(x, 1,2)
        memory = self.transformer_encoder(x, src_mask)
        out = self.dropout(self.activation(self.linear1(memory.reshape(memory.shape[0], -1))))
        # out = self.dropout(self.activation(self.linear2(out)))
        # out = self.dropout(self.activation(self.linear3(out)))
        out = F.softplus(self.hidden2output(out))
        # output = self.transformer_decoder(tgt_in, memory)
        # print("encoded shape ", output.shape)
        # output = output.transpose(0,1).transpose(1,2)
        # output = output.reshape(output.shape[0], -1)
        # output = self.fc2(self.activation(self.fc1(output)))
        return out

class InformerEncoder(nn.Module):
    def __init__(self,enc_in, c_out, seq_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                dropout=0.1, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True, ssl=False  ):
        super(InformerEncoder, self).__init__()
        # Conv
        self.conv = nn.Sequential(ConvBlock(1,64, kernel_size=3, stride=2, padding=1, dropout=0.1),
                                ConvBlock(64,128, kernel_size=3, stride=2, padding=3, dropout=0.1),
                                ConvBlock(128,d_model, kernel_size=3, stride=2, padding=1, dropout=0.1))
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, max_len=seq_len)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.ffd = nn.Linear(d_model*seq_len, 1024, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.ssl = ssl
        self.output_dim = d_model

    def forward(self, x_enc, enc_self_mask=None):
        if len(x_enc.shape) == 2:
            x_enc = x_enc.unsqueeze(-1)
        if len(x_enc.shape) == 3 and x_enc.shape[1] == 1:
            x_enc = x_enc.transpose(-1,-2)
        enc_out = self.enc_embedding(x_enc, None)
        # enc_out = self.conv(enc_out.transpose(-1,-2))
        # enc_out = enc_out.transpose(-1,-2)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print("enc_out: ", enc_out.shape)
        # enc_out = self.projection(self.ffd(enc_out.reshape(enc_out.shape[0], -1)))
        if self.ssl:
            return enc_out.max(dim=1).values
        enc_out = self.projection(enc_out.max(dim=1).values)
        return enc_out

# class ConformerEncoder(nn.Module):
#     def __init__(self, channels ):
#         super(ConformerEncoder, self).__init__()
#         self.conv = Conv2dSubampling(in_channels=1, out_channels=channels)


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, max_len=seq_len)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout, max_len=seq_len)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
    
    def reshape_input(self, x_enc, x_dec):
        if len(x_enc.shape) == 2:
            x_enc = x_enc.unsqueeze(-1)
        if len(x_enc.shape) == 3 and x_enc.shape[1] == 1:
            x_enc = x_enc.transpose(-1,-2)
        if len(x_dec.shape) == 2:
            x_dec = x_dec.unsqueeze(-1)
        if len(x_dec.shape) == 3 and x_enc.shape[1] == 1:
            x_dec = x_dec.transpose(-1,-2)
        return x_enc, x_dec
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc, x_dec = self.reshape_input(x_enc, x_dec)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

class HwinEncoder(nn.Module):
    def __init__(self,enc_in, c_out, seq_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, window_size=6, n_windows=4,
                dropout=0.1, predict_size=256, attn='full', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True, ssl=False  ):
        super(HwinEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        shrink_factor =seq_len/(sum([seq_len/(window_size*2**i) for i in range(n_windows)]))

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, max_len=seq_len)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                HwinEncoderLayer(
                    HwinAttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model=d_model, n_heads=n_heads, window_size=window_size, n_windows=n_windows, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model, c_out=d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.ffd = nn.Linear(d_model*seq_len, 1024, bias=True)
        self.output_dim = self._out_shape()
        self.projection = nn.Linear(self.output_dim, predict_size, bias=True)
        self.prediction = nn.Linear(predict_size, c_out, bias=True)
        # self.prediction2 = nn.Linear(predict_size, c_out, bias=True)

        self.ssl = ssl

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.randn(2,self.seq_len, self.d_model)
            x,_ = self.encoder(dummy_input)
            print("x: ", x.shape)
            x = x.view(x.size(0), -1)  # Flatten the output
            return x.shape[1] 
        finally:
            torch.set_rng_state(rng_state)

    def forward(self, x_enc, enc_self_mask=None):
        if len(x_enc.shape) == 2:
            x_enc = x_enc.unsqueeze(-1)
        if len(x_enc.shape) == 3 and x_enc.shape[1] == 1:
            x_enc = x_enc.transpose(-1,-2)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print("enc_out: ", enc_out.shape)
        # enc_out = self.projection(self.ffd(enc_out.reshape(enc_out.shape[0], -1)))
        if self.ssl:
            return enc_out.view(enc_out.size(0, -1))
        enc_out = self.projection(enc_out.view(enc_out.size(0), -1))
        out = self.prediction(enc_out)
        # out2 = self.prediction2(enc_out)
        return out



class AutoEncoder(nn.Module):
    def __init__(self,enc_in, c_out, seq_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, moving_avg=25, 
                dropout=0.1, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True, ssl=False  ):
        super(AutoEncoder, self).__init__()
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout, max_len=seq_len)
        # Encoder
        self.encoder = AutoformerEncoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer = AutoformerLayerNorm(d_model)
        )

        # self.ffd = nn.Linear(d_model*seq_len, 1024, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.ssl = ssl

    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print("enc_out: ", enc_out.shape)
        # enc_out = self.projection(self.ffd(enc_out.reshape(enc_out.shape[0], -1)))
        if self.ssl:
            return enc_out.max(dim=1).values
        enc_out = self.projection(enc_out.max(dim=1).values)
        return enc_out


class DLInear(nn.Module):
    def __init__(self, seq_len, pred_len, c_out, moving_avg=25, dropout=0.2):
        super(DLInear, self).__init__()
        self.decompsition = series_decomp(moving_avg)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.pred = nn.Linear(self.pred_len, c_out)

    def forward(self, x):
        season, trend = self.decompsition(x)
        season = self.Linear_Seasonal(season.squeeze())
        trend = self.Linear_Trend(trend.squeeze())
        out = self.pred(self.dropout(self.activation(season + trend)))
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        # self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=[2], dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            k = kernel_size[i]
            layers += [TemporalBlock(in_channels, out_channels, k, stride=1, dilation=dilation_size,
                                     padding=(k-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



# class ResidualNet(nn.Module):
#     def __init__(self, backbone, predict_size):
#         super(ResidualNet, self).__init__()
#         self.network = backbone
#         self.num_features = self.network.num_features
#         self.hidden2output = nn.Linear(self.num_features, predict_size)
#         self.batchnorm = nn.BatchNorm1d(predict_size)
#         self.p_head = nn.Linear(predict_size, 1)
#         self.i_head = nn.Linear(predict_size, 1)
#         self.activation = nn.GELU()
#     def forward(self, x):
#         p = analyze_lc_torch(x[:,1,:]).to(x.device).unsqueeze(-1)
#         # with torch.no_grad():
#         #     out = self.activation(self.batchnorm(self.hidden2output(self.network(x))))
#         #     p  = self.p_head(out)
#         residuals = residual_by_period(x[:,0,:].clone(), p)
#         x[:,0,:] = residuals
#         out =  self.activation(self.batchnorm(self.hidden2output(self.network(x))))
#         i = self.i_head(out)

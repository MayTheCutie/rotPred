import torch
import torch.nn as nn


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
        
    
class ConvEncoderDecoder(nn.Module):
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

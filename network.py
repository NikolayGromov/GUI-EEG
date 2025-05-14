import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio


def NormalizeAndClip(signal):
    means = signal.mean(axis=1)[..., None]
    stds = signal.std(axis=1)[..., None]
    signal = np.clip((signal - means) / stds, a_min=-10, a_max=10)
    return signal

### TRANSFORM TO ONE FREQUENCY ###

def Transform(sample_rate, records, freqs, times): # only for 199
    transform = 0
    new_freq = 0
    if sample_rate == 199:
        transform500 = torchaudio.transforms.Resample(250625, 100000)
        transform1000 = torchaudio.transforms.Resample(50125, 10000)
        
        new_freq = 80000 / 401
    else:
        transform = torchaudio.transforms.Resample(100000, 250625) 
        new_freq = 500
    for i in range(len(records)):
        if int(freqs[i]) != sample_rate:
            if int(freqs[i]) == 500:
                new_sigbufs = []
                sigbufs = records[i]
                for sig in tqdm(sigbufs):
                    new_sigbufs.append(transform500(torch.FloatTensor(sig)))
                new_sigbufs = np.array(new_sigbufs)
                records[i] = new_sigbufs
                freqs[i] = new_freq
                times[i] = [1/new_freq * j for j in range(len(new_sigbufs[0]))]   
            elif int(freqs[i]) == 1000:
                new_sigbufs = []
                sigbufs = records[i]
                for sig in tqdm(sigbufs):
                    new_sigbufs.append(transform1000(torch.FloatTensor(sig)))
                new_sigbufs = np.array(new_sigbufs)
                records[i] = new_sigbufs
                freqs[i] = new_freq
                times[i] = [1/new_freq * j for j in range(len(new_sigbufs[0]))]
    for i in range(len(freqs)):
        freqs[i] = 80000 / 401
        times[i] = [1/freqs[i] * j for j in range(len(records[i][0]))]

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, 
                               padding = int(np.ceil(dilation * (kernel_size-1) / 2)), bias=True) # for stride=1, else need to calculate and change
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        inp_shape = int(np.ceil(x.shape[2] / self.stride))
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)[:, :, :inp_shape] 
        #print("conbr_out", out.shape)
        return out      

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)        
        x_out = torch.add(x, x_re)
        return x_out 
    
class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size, n_down_layers, depth, n_features=1): # n_features for additional features in some other exps
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.n_down_layers = n_down_layers
        self.depth = depth
        
        self.AvgPool1D = nn.ModuleList([nn.AvgPool1d(input_dim, stride=5**i, padding=8) for i in range(1, self.n_down_layers)])
        
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, depth)
        self.layer1_sneo = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, self.depth)
        self.layer1_mc = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, self.depth)
        
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, self.depth)
        
        self.down_layers = nn.ModuleList([self.down_layer(self.layer_n*(1+i)+n_features*self.input_dim, self.layer_n*(2+i), 
                                            self.kernel_size,5, self.depth) for i in range(1, self.n_down_layers)])


        self.cbr_up = nn.ModuleList([conbr_block(int(self.layer_n*(2*i+1)), int(self.layer_n*i), self.kernel_size, 1, 1) 
                       for i in range(self.n_down_layers, 0, -1)]) #input size is a sizes sum of outs of 2 down layers for current down depth
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest') 
        
        self.outcov = nn.Conv1d(self.layer_n, 2, kernel_size=self.kernel_size, stride=1,
                                padding = int(np.ceil(1 * (self.kernel_size-1) / 2)))
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
        
        
            
    def forward(self, x):
        inp_shape = x.shape[2]
        
        
        
        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        outs = [out_0, out_1]
        for i in range(self.n_down_layers-1):
            pool = self.AvgPool1D[i](x)
            x_down = torch.cat([outs[-1],pool],1)
            outs.append(self.down_layers[i](x_down))




        #############Decoder####################
        up = self.upsample(outs[-1])[:, :, :outs[-2].shape[2]]
        for i in range(self.n_down_layers):
                        
            up = torch.cat([up,outs[-2-i]],1)
            up = self.cbr_up[i](up)
            if i + 1 < self.n_down_layers:
                up = self.upsample(up)[:, :, :outs[-3-i].shape[2]]

        out = self.outcov(up)


        return out[:, :, :inp_shape] 


def MergeClose(predictions, threshold):
    i = 0
    in_event = False
    while i < len(predictions):
        while i < len(predictions) and predictions[i] == 1:
            in_event = True
            i += 1
        if  i < len(predictions) and in_event:
            if np.any(predictions[i:i+threshold]):
                while  i < len(predictions) and predictions[i] == 0:
                    predictions[i] = 1
                    i += 1
            else:
                in_event = False
        i += 1

def DeleteShortEvents(predictions, threshold):
    i = 0
    while i < len(predictions):
        event_len = 0
        event_idx_start = i
        while i < len(predictions) and predictions[i] == 1:
            i += 1
            event_len += 1
        if event_len < threshold:
            predictions[event_idx_start:i] = 0
        i += 1
def PostProcessing(predictions, threshold1, threshold2=None):
    MergeClose(predictions, threshold1)
    if threshold2 is None:
        DeleteShortEvents(predictions, threshold1)
    else:
        DeleteShortEvents(predictions, threshold2)

OVERLAP = 0
RECEPTIVE_FIELD = 4000

def CollectingPreds(model, test_data):
    model.eval()
    model.cpu()
    
    record_preds = []
    for idx in tqdm(range(OVERLAP, test_data.size()[1]- RECEPTIVE_FIELD - OVERLAP, RECEPTIVE_FIELD)):

        test_seq = test_data[:, idx-OVERLAP:idx+RECEPTIVE_FIELD+OVERLAP][None, ...]
        
        out = model(test_seq)
                
        m = nn.Softmax(dim=1)
        out = m(out)
        
        preds = np.argmax(out.detach().cpu().numpy(), axis=1)
        record_preds.append(preds)
    shapes = np.array(record_preds).shape
    record_preds = np.array(record_preds).reshape(shapes[0] * shapes[1] * shapes[2])
    return record_preds


def SendPredictsGUI(test_data, freq):
    test_data = NormalizeAndClip(test_data)
    Transform(199, [test_data], [freq], [None])
    test_data = torch.FloatTensor(test_data)
    model = UNET_1D(20,128,7,3,1)
    model.load_state_dict(torch.load("Unet1d", map_location=torch.device('cpu')))
    test_predicts = CollectingPreds(model, test_data)
    PostProcessing(test_predicts, 20)
    return test_predicts
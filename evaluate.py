import torch
import numpy as np
from vgg import *
import random
from tools import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


np.random.seed(0)

conv_model = joblib.load('trainedModel/forest_reg_conv.pkl')
fc_model = joblib.load('trainedModel/forest_reg_fc.pkl')


layers = []

def unfoldLayer(model):
    layer_list = list(model.named_children())
    for item in layer_list:
        module = item[1]
        sublayer = list(module.named_children())
        sublayer_num = len(sublayer)

        if sublayer_num == 0:
            layers.append(module)
        elif isinstance(module, torch.nn.Module):
            unfoldLayer(module)

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
for _ in range(1,6):
    for i in range(0,17):
        if cfg[i]=='M':
            continue
        else:
            cfg[i]=random.randint(32,512)
    print("cfg=",cfg)
    model = vgg(cfg).cuda()
    latency = measure_latency(model)
    print("latency=",latency,"ms")

    layers = []
    unfoldLayer(model)
    predLatency = 0.0
    data = torch.randn(128,3,32,32).cuda()
    index = 0
    for l in layers:    
        if index==30:
            data = nn.AvgPool2d(2)(data)
            data = data.view(data.size(0), -1)

        index += 1

        if isinstance(l,nn.Conv2d):
            hin = data.size()[2]
            win = data.size()[3]
            cin = data.size()[1]
            hout = (hin-l.kernel_size[0]+2*l.padding[0])/l.stride[0] + 1
            wout = (win-l.kernel_size[0]+2*l.padding[0])/l.stride[0] + 1
            flops = l.out_channels * hout * wout *(cin*l.kernel_size[0]*l.kernel_size[1]+1)
            data = l(data)
            param_size = l.kernel_size[0]*l.kernel_size[0]*l.in_channels*l.out_channels+1 
            vector = np.array([[128,win,l.in_channels,l.out_channels,l.kernel_size[0],l.stride[0],flops,param_size]])
            print("conv vector:",vector)
            print("conv pred:",conv_model.predict(vector))
        elif isinstance(l,nn.Linear):
            data = l(data)
            param_size = l.in_features * l.out_features + l.out_features
            flops = 2 * l.in_features * l.out_features
            vector = np.array([[128,l.in_features,l.out_features,flops,param_size]])
            print("fc vector:",vector)
            print("fc pred:",fc_model.predict(vector))
        else:
            data = l(data)



import torch
import torchvision
import torchprof
import torch.nn as nn
from tools import *
from thop import profile
from models import *
import torch.nn.functional as F
import numpy as np
from fvcore.nn import ActivationCountAnalysis

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

def FLOPs_and_params(model):
    input = torch.randn(1, 3, 32, 32).cuda()
    flops, params = profile(model, inputs=(input, ))
    return flops, params

def getLatency(model):
    # gpu warm up
    #data = torch.randn(128,3,32,32).cuda()
    data = torch.randn(64,3,32,32).cuda()
    for _ in range(20):
        _ = model(data)
    latency = measure_latency(model,dataset="cifar")
    return latency

def getActs(model):
    inputs = torch.randn((1,3,32,32)).cuda()
    acts = ActivationCountAnalysis(model,inputs)
    return acts.total()

def extractFeature(model):
    flops, params = FLOPs_and_params(model)
    latency = getLatency(model)
    acts = getActs(model)
    unfoldLayer(model)
    num_conv = 0
    num_fc = 0
    num_bn = 0
    conv_param = 0
    fc_param = 0
    weighted_neuron_counts = 0
    for l in layers:
        if isinstance(l,nn.Conv2d):
            num_conv += 1
            conv_param += l.kernel_size[0]*l.kernel_size[0]*l.in_channels*l.out_channels+1
            weighted_neuron_counts += l.kernel_size[0]*l.kernel_size[0]*l.in_channels*l.out_channels
        elif isinstance(l,nn.Linear):
            num_fc += 1
            fc_param += l.in_features * l.out_features + l.out_features
            weighted_neuron_counts += l.out_features
        elif isinstance(l,nn.BatchNorm2d):
            num_bn += 1
    return np.array([len(layers),flops,params,acts,weighted_neuron_counts,num_conv,num_fc,num_bn,latency])

data = np.zeros(9)
for rep in range(1000):
    if rep%100==0:
        print("process bar: ",rep)
    honey = np.random.randint(1,11,36)
    #model = BeeVGG('vgg16',honey).cuda()
    #model = resnet('resnet56', honey).cuda()
    #model = googlenet(honey=honey).cuda()
    model = densenet(honey=honey).cuda()
    
    layers = []
    feats = extractFeature(model)
    data = np.vstack((data,feats))

print(data)
np.savetxt('densenetArch.txt',data)

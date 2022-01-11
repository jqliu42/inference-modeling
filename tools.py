import torch
import torch.nn as nn
from thop import profile
import numpy as np

feat_vec = ["one-hot encoding","FLOPs","MACs","param size","in channel","out channel","activation counts","kernel size","stride","padding"]

def convFeat(layer,x):
    hin = x.shape[2]
    win = x.shape[3]
    in_channel = layer.in_channels
    out_channel = layer.out_channels
    kernel_size = layer.kernel_size[0]
    stride_size = layer.stride[0]
    padding = layer.padding[0]
    hout = (hin-kernel_size+2*padding)/stride + 1
    wout = (win-kernel_size+2*padding)/stride + 1
    FLOPs = out_channel * hout * wout * (in_channel*kernel_size*kernel_size+1)
    param_size = kernel_size * kernel_size * in_channel * out_channel + 1
    MACs = x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3] + param_size + x.shape[0]*out_channel*hout*wout 
    activation_counts = 0
    return [1,0,0,0,FLOPs,MACs,param_size,in_channel, out_channel,activation_counts, kernel_size, stride_size, padding]

def fcFeat(layer,x):
    in_channel = layer.in_features
    out_channel = layer.out_features
    FLOPs = 2*in_channel*out_channel
    param_size = in_channel * out_channel + out_channel
    MACs = x.shape[0]*x.shape[1] + param_size + x.shape[0]*out_channel
    activation_counts = out_channel
    return [0,1,0,0,FLOPs,MACs,param_size,in_channel,out_channel,activation_counts,0,0,0]

def BNFeat(layer):
    
    return [0,0,1,0,]

def poolingFeat(layer):
    
    return [0,0,0,1,]

def parseLatency(prof):
    prof = str(prof)
    index = prof.rfind(':')
    latency = float(prof[index+1:-3])
    if prof[-3]=='m':
        latency *= 1000
    return latency


def measure_latency(model,dataset='cifar'):
   if dataset=="cifar":
       dummy_input = torch.randn(128,3,32,32,dtype=torch.float).cuda()
   else:
       dummy_input = torch.randn(128,3,224,224,dtype=torch.float).cuda()

   
   #print(model)

   starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
   repetitions = 300
   timings = np.zeros((repetitions,1))

   # GPU warm up

   for _ in range(10):
       _ = model(dummy_input)

   # mesure performance
   with torch.no_grad():
       for rep in range(repetitions):
           starter.record()
           _ = model(dummy_input)
           ender.record()
           # wait for gpu sync
           torch.cuda.synchronize()
           curr_time = starter.elapsed_time(ender)
           timings[rep] = curr_time

   mean_syn = np.sum(timings) / repetitions
   std_syn = np.std(timings)

   return mean_syn


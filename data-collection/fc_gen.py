import torch
import torchvision
import torchprof
import torch.nn as nn
from tools import *
import random

# 随机生成全连接层配置，并评测inference latency

for _ in range(1000):
    in_channel = random.randint(32,4096) 
    out_channel = random.randint(32,4096)
    batch_size = random.randint(1,16)*32
    test_input = torch.randn(batch_size,in_channel).cuda()
    conv_layer = nn.Linear(in_channel,out_channel).cuda()


    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions,1))


    # mesure performance
    with torch.no_grad():
       for rep in range(repetitions):
           starter.record()
           _ = conv_layer(test_input)
           ender.record()
           # wait for gpu sync
           torch.cuda.synchronize()
           curr_time = starter.elapsed_time(ender)
           timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    FLOPs = in_channel * out_channel * 2 
    param_size = in_channel*out_channel+ out_channel

    print("out model time: ",mean_syn," ms")
    f = open("fc_data.txt",'a+')
    f.write(str(batch_size)+'\t'+str(in_channel)+'\t'+str(out_channel)+'\t'+str(FLOPs)+'\t'+str(param_size)+'\t'+str(mean_syn)+'\n')

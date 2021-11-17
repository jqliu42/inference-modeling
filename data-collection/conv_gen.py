import torch
import torchvision
import torchprof
import torch.nn as nn
from tools import *
import random
import time

# 随机生成卷积层配置，并评测inference latency

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
for _ in range(1000):
    in_channel = random.randint(1,16)*32 
    out_channel = random.randint(1,16)*32
    kernel_size = random.randint(2,5)
    padding = 1
    stride = random.randint(1,2) 
    in_width = random.randint(1,2)*32
    in_height = in_width
    batch_size = random.randint(1,8)*32
    test_input = torch.randn(batch_size,in_channel,in_width,in_height).cuda()
    conv_layer = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,padding=padding,stride=stride,bias=False).cuda()


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

    hin = in_height
    win = in_width
    cin = in_channel
    hout = (hin-kernel_size+2*padding)/stride + 1
    wout = (win-kernel_size+2*padding)/stride + 1
    FLOPs = out_channel * hout * wout *(cin*kernel_size*kernel_size+1)
    param_size = kernel_size*kernel_size*in_channel*out_channel+1 

    print("out model time: ",mean_syn," ms")
    f = open("conv_data.txt",'a+')
    f.write(str(batch_size)+'\t'+str(in_width)+'\t'+str(in_channel)+'\t'+str(out_channel)+'\t'+str(kernel_size)+'\t'+str(stride)+'\t'+str(FLOPs)+'\t'+str(param_size)+'\t'+str(mean_syn)+'\n')


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

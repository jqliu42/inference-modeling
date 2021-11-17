import torch
import torchvision
import torchprof
import torch.nn as nn
from tools import *
from thop import profile
from models import *
import torch.nn.functional as F

# 验证model latency是否等于sum of layer latency

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

modelName = 'ghostnet'
#model = VGG().cuda()
#model = ResNet18().cuda()
#model = ShuffleNetG2().cuda()
model = ghost_net().cuda()
unfoldLayer(model)

print(model)
# gpu warm up
data = torch.randn(128,3,32,32).cuda()
for _ in range(20):
    _ = model(data)

latency = measure_latency(model,dataset="cifar")
print("target latency: ",latency, " ms")
total = 0.0

index = 0
for l in layers:
    print(index)
    if modelName=='vgg' and index==44: 
        data = data.view(data.size(0), -1)
        index += 1
        continue

    if modelName=='resnet' and index in [10,17,24,29]:
        if index==29:
            data = F.avg_pool2d(data, 4)
            data = data.view(data.size(0), -1)
        index += 1
        continue

    index += 1
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions,1))

    if isinstance(l,nn.Conv2d):
        for rep in range(repetitions):
            starter.record()
            _ = l(data)
            ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        hin = data.size()[2]
        win = data.size()[3]
        cin = data.size()[1]
        hout = (hin-l.kernel_size[0]+2*l.padding[0])/l.stride[0] + 1
        wout = (win-l.kernel_size[0]+2*l.padding[0])/l.stride[0] + 1
        flops = l.out_channels * hout * wout *(cin*l.kernel_size[0]*l.kernel_size[1]+1)
        data = l(data)
        total += mean_syn
        if index in [13,24,35]:
            total += mean_syn
        param_size = l.kernel_size[0]*l.kernel_size[0]*l.in_channels*l.out_channels+1 
        print("conv: in=",l.in_channels,", out=",l.out_channels,", kernel_size=",l.kernel_size[0],", stride_size=",l.stride[0],", padding=",l.padding[0],"FLOPS=",flops," ,param_size=",param_size,", latency=", mean_syn," ms")
    elif isinstance(l,nn.Linear):
        #print(data.size())
        for rep in range(repetitions):
            starter.record()
            _ = l(data)
            ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        data = l(data)
        total += mean_syn
        if index in [3,13,24,35]:
            total += mean_syn
        param_size = l.in_features * l.out_features + l.out_features
        flops = 2 * l.in_features * l.out_features
        print("fc: in=",l.in_features,", out=",l.out_features,", param_size=",param_size,", FLOPs=",flops,", latency=",mean_syn," ms")
    else:
        data = l(data)

print("target latency: ",latency, " ms")
print("sum of layer latency: ",total," ms")

import numpy as np
import torch

'''
方案1：FLOPs, parameter size, layer 数量, 卷积层数量，全连接层数量，input size, ... ,latency
方案2：将每一层的feature拼接起来（如何归一化长度？）(每一个layer是句子，这样就相当于序列数据，可以用lstm试试)
'''
def extractFeature(model):
    pass



def convFeat(module):
    pass

def fcFeat(module):
    pass

def poolingFeat(module):
    pass



#latency
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

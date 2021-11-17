import torch
import torch.nn as nn
from thop import profile
import numpy as np


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


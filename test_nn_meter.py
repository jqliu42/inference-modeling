import os
from nn_meter.dataset import bench_dataset
import numpy as np

datasets = bench_dataset()
for data in datasets:
    print(f"Model group: {os.path.basename(data)}")

import jsonlines
test_data = datasets[0]

with jsonlines.open(test_data) as data_reader:
    True_lat = []
    Pred_lat = []
    for i, item in enumerate(data_reader):
        print("dict key:",list(item.keys()))
        print("model id",item['id'])
        print('cpu latency: ',item['cortexA76cpu_tflite21'])
        print('adreno640gpu latency: ',item['adreno640gpu_tflite21'])
        print('adreno630gpu latency: ',item['adreno630gpu_tflite21'])
        print('intelvpu latency: ',item['myriadvpu_openvino2019r2'])
        print("model graph is stored in nn-meter IR (shows only one node here): ",item['graph']['conv1.conv/Conv2D'])
        break



import nn_meter

predictor_name = 'adreno640gpu_tflite21'

predictor = nn_meter.load_latency_predictor(predictor_name)
test_data = datasets[0]
with jsonlines.open(test_data) as data_reader:
    True_lat = []
    Pred_lat = []
    for i,item in enumerate(data_reader):
        if i>20:
            break
        graph = item['graph']
        pred_lat = predictor.predict(graph,model_type='nnmeter-ir')
        real_lat = item[predictor_name]
        print(f'[RESULT] {os.path.basename(test_data)}[{i}]: predict: {pred_lat}, real: {real_lat}')

        if real_lat != None:
            True_lat.append(real_lat)
            Pred_lat.append(pred_lat)
if len(True_lat) > 0:
    rmse, rmspe, error, acc5, acc10, _ = nn_meter.latency_metrics(Pred_lat, True_lat)
    print(f'[SUMMARY] The first 20 cases from {os.path.basename(test_data)} on {predictor_name} : rmse: {rmse},5%accuracy: {acc5}, 10%accuracy: {acc10}')


import os
from nn_meter.dataset import gnn_dataloader

target_device = "cortexA76cpu_tflite21"

print("Processing Training Set")
train_set = gnn_dataloader.GNNDataset(train=True, device=target_device)
print("Processing Testing Set")
test_set = gnn_dataloader.GNNDataset(train=False, device=target_device)

train_loader = gnn_dataloader.GNNDataloader(train_set,batchsize=1,shuffle=True)
test_loader = gnn_dataloader.GNNDataloader(test_set,batchsize=1,shuffle=True)
print('Train Dataset Size: ', len(train_set))
print('Test Dataset Size: ', len(test_set))
print('Attribute tensor shape:',next(train_loader)[1].ndata['h'].size(1))
ATTR_COUNT = next(train_loader)[1].ndata['h'].size(1)

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import dgl.nn as dglnn
from dgl.nn.pytorch.glob import MaxPooling

class GNN(Module):
    def __init__(self,num_features=0,num_layers=2,num_hidden=32,dropout_ratio=0):
        super(GNN,self).__init__()
        self.nfeat = num_features
        self.nlayer = num_layers
        self.nhid = num_hidden
        self.dropout_ratio = dropout_ratio
        self.gc = nn.ModuleList([dglnn.SAGEConv(self.nfeat if i==0 else self.nhid, self.nhid,'pool') for i in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.LayerNorm(self.nhid) for i in range(self.nlayer)])
        self.relu = nn.ModuleList([nn.ReLU() for i in range(self.nlayer)])
        self.pooling = MaxPooling()
        self.fc = nn.Linear(self.nhid,1)
        self.fc1 = nn.Linear(self.nhid,self.nhid)
        self.dropout = nn.ModuleList([nn.Dropout(self.dropout_ratio) for i in range(self.nlayer)])

    def forward_single_model(self,g,features):
        x = self.relu[0](self.bn[0](self.gc[0](g,features)))
        x = self.dropout[0](x)
        for i in range(1,self.nlayer):
            x = self.relu[i](self.bn[i](self.gc[i](g,x)))
            x = self.dropout[i](x)
        return x

    def forward(self, g, features):
        x = self.forward_single_model(g, features)
        with g.local_scope():
            g.ndata['h'] = x
            x = self.pooling(g,x)
            x = self.fc1(x)
            return self.fc(x)

from torch.optim.lr_scheduler import CosineAnnealingLR

if torch.cuda.is_available():
    print("Using CUDA")

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = GNN(ATTR_COUNT, 3,400,0.1).to(device)
opt = torch.optim.AdamW(model.parameters(),lr=4e-4)
EPOCHS=100
loss_func = nn.L1Loss()

real = []
pred = []

lr_scheduler = CosineAnnealingLR(opt, T_max=EPOCHS)
loss_sum = 0
for epoch in range(EPOCHS):
    train_length = len(train_set)
    train_acc_ten = 0
    loss_sum = 0

    for batched_l, batched_g in train_loader:
        opt.zero_grad()
        batched_l = batched_l.to(device).float()
        batched_g = batched_g.to(device)
        batched_f = batched_g.ndata['h'].float()
        logits = model(batched_g, batched_f)
        for i in range(len(batched_l)):
            pred_latency = logits[i].item()
            prec_latency = batched_l[i].item()
            if (pred_latency >= 0.9 * prec_latency) and (pred_latency <= 1.1 * prec_latency):
                train_acc_ten += 1

        batched_l = torch.reshape(batched_l,(-1,1))
        loss = loss_func(logits, batched_l)
        loss_sum += loss
        loss.backward()
        opt.step()
    lr_scheduler.step()
    print("[Epoch ", epoch,"]: ","Training accuracy within 10%: ", train_acc_ten / train_length * 100," %.")



test_length = len(test_set)
test_acc_ten = 0

for batched_l, batched_g in test_loader:
    batched_l = batched_l.to(device).float()
    batched_g = batched_g.to(device)
    batched_f = batched_g.ndata['h'].float()
    logits = model(batched_g, batched_f)
    for i in range(len(batched_l)):
        pred_latency = logits[i].item()
        prec_latency = batched_l[i].item()
        real.append(prec_latency)
        pred.append(pred_latency)
        if (pred_latency >= 0.9 * prec_latency) and (pred_latency <= 1.1 * prec_latency):
            test_acc_ten += 1

np.save('pred.npy',np.array(pred))
np.save('real.npy',np.array(real))

print("Test accuracy within 10%: ", test_acc_ten / test_length * 100," %.")






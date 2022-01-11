import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class BeeLeNet(nn.Module):
    def __init__(self,honey):
        super(BeeLeNet,self).__init__()
        self.honey = honey
        self.conv1 = nn.Conv2d(3,int(6*honey[0]/10),5)
        self.conv2 = nn.Conv2d(int(6*honey[0]/10),int(16*honey[1]/10),5)
        self.fc1 = nn.Linear(int(16*5*5*honey[1]/10),int(120*honey[2]/10))
        self.fc2 = nn.Linear(int(120*honey[2]/10),int(84*honey[3]/10))
        self.fc3 = nn.Linear(int(84*honey[3]/10),10)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out,2)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


def test():
    honey = [8,8,8,8]
    model = BeeLeNet(honey)
    print(model)

test()

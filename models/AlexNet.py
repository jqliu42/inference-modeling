import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classed=10):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*1*1,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classed),)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),256*1*1)
        x = self.classifier(x)
        return x

class BeeAlexNet(nn.Module):
    def __init__(self,honey,num_classes=10):
        super(BeeAlexNet,self).__init__()
        self.honey = honey
        
        self.features = nn.Sequential(
            nn.Conv2d(3,int(64*honey[0]/10),kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(int(64*honey[0]/10),int(192*honey[1]/10),kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(int(192*honey[1]/10),int(384*honey[2]/10),kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(384*honey[2]/10),int(256*honey[3]/10),kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(256*honey[3]/10),int(256*honey[4]/10),kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(int(256*honey[4]/10)*1*1,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes),)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),int(256*honey[4]/10)*1*1)
        x = self.classifier(x)
        return x

def test():
    honey = [8,8,8,8,8]
    model = BeeAlexNet(honey)
    print(model)

test()


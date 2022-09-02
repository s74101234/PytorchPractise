import torch
from torch import nn
class CNNModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 7)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(128 * 25 * 25), out_features=512),
            nn.Dropout(p=0.2, inplace=True), 
            nn.ReLU(), 
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.2, inplace=True), 
            nn.ReLU(), 
            nn.Linear(in_features=256, out_features=n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # flatten
        outputs = self.fc(x)

        return outputs

def basemodel(ClassesNum):
    net = CNNModel(n_classes = ClassesNum)
    print(net)
    return net
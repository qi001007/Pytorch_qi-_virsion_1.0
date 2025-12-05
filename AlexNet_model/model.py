import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as ft


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 凯明初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s3(x)

        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = ft.dropout(x, p=0.5)
        x = self.ReLU(self.f2(x))
        x = ft.dropout(x, p=0.5)
        x = self.f3(x)
        y = x

        return y
    

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    AlexNet = AlexNet().to(device)
    print(summary(AlexNet, input_size=(1, 227, 227)))

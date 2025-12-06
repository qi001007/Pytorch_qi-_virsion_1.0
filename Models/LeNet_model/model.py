import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmod = nn.Sigmoid()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.s1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(in_features=400, out_features=120)
        self.l2 = nn.Linear(in_features=120, out_features=84)
        self.l3 = nn.Linear(in_features=84, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 凯明初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

            if isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.s1(self.sigmod(self.c1(x)))
        x = self.s2(self.sigmod(self.c2(x)))
        x = self.flatten(x)
        x = self.l3(self.sigmod(self.l2(self.sigmod(self.l1(x)))))
        y = x
        return y


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    LeNet = LeNet().to(device)
    print(summary(LeNet, input_size=(1, 28, 28)))








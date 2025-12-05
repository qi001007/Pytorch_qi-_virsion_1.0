import torch
from torch import nn
from torchsummary import summary


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,  stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.block5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        # 多分类(原版)
        # self.block6 = nn.Sequential(nn.Flatten(),
        #                             nn.Linear(in_features=7*7*512, out_features=4096),
        #                             nn.ReLU(),
        #                             nn.Dropout(p=0.5),
        #                             nn.Linear(in_features=4096, out_features=4096),
        #                             nn.ReLU(),
        #                             nn.Dropout(p=0.5),
        #                             nn.Linear(in_features=4096, out_features=10))
        # 简化参数版
        self.block6 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features=7*7*512, out_features=1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=1024, out_features=512),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(in_features=512, out_features=10))

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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        y = x
        return y


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    VGG16 = VGG16().to(device)
    print(summary(VGG16, input_size=(1, 224, 224)))


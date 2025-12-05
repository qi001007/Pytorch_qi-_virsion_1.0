import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, c1_c3_stride=1, ues_1_conv=False):
        super(Residual, self).__init__()
        self.ReLu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                            kernel_size=3, stride=c1_c3_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.c2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels,
                            kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        if ues_1_conv:
            self.c3 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                kernel_size=1, stride=c1_c3_stride, padding=0)
        else:
            self.c3 = None

    def forward(self, x):
        y = self.ReLu(self.bn1(self.c1(x)))
        y = self.bn2(self.c2(y))
        if self.c3:
            x = self.c3(x)
        y = self.ReLu(y + x)
        return y


class ResNet18(nn.Module):
    def __init__(self, residual):
        super(ResNet18, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            residual(input_channels=64, output_channels=64, c1_c3_stride=1, ues_1_conv=False),
            residual(input_channels=64, output_channels=64, c1_c3_stride=1, ues_1_conv=False),
        )
        self.block3 = nn.Sequential(
            residual(input_channels=64, output_channels=128, c1_c3_stride=2, ues_1_conv=True),
            residual(input_channels=128, output_channels=128, c1_c3_stride=1, ues_1_conv=False),
        )
        self.block4 = nn.Sequential(
            residual(input_channels=128, output_channels=256, c1_c3_stride=2, ues_1_conv=True),
            residual(input_channels=256, output_channels=256, c1_c3_stride=1, ues_1_conv=False),
        )
        self.block5 = nn.Sequential(
            residual(input_channels=256, output_channels=512, c1_c3_stride=2, ues_1_conv=True),
            residual(input_channels=512, output_channels=512, c1_c3_stride=1, ues_1_conv=False),
        )
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )

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

    ResNet18 = ResNet18(Residual).to(device)
    print(summary(ResNet18, input_size=(1, 224, 224)))

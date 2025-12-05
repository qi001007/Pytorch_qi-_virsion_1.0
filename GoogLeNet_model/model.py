import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    # 输入参数：传入通道数in_channels, 路线1传出通道数c1_out_channels, 路线1传出通道数c2_out_channels(二维元组),
    # 路线1传出通道数c3_out_channels(二维元组), 路线1传出通道数c4_out_channels
    def __init__(self, in_channels, c1_out_channels, c2_out_channels, c3_out_channels, c4_out_channels):
        super(Inception, self).__init__()
        self.Relu = nn.ReLU()
        # 路线1,单1x1卷积层
        self.c1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1_out_channels,
                              kernel_size=1, stride=1, padding=0)
        # 路线2,1x1+3x3卷积层
        self.c2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2_out_channels[0],
                              kernel_size=1, stride=1, padding=0)
        self.c2_2 = nn.Conv2d(in_channels=c2_out_channels[0], out_channels=c2_out_channels[1],
                              kernel_size=3, stride=1, padding=1)
        # 路线3,1x1+5x5卷积层
        self.c3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3_out_channels[0],
                              kernel_size=1, stride=1, padding=0)
        self.c3_2 = nn.Conv2d(in_channels=c3_out_channels[0], out_channels=c3_out_channels[1],
                              kernel_size=5, stride=1, padding=2)
        # 路线4,3x3最大池化层+1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.c4_1 = nn.Conv2d(in_channels=in_channels, out_channels=c4_out_channels,
                              kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Relu(self.c1_1(x))
        x2 = self.Relu(self.c2_2(self.Relu(self.c2_1(x))))
        x3 = self.Relu(self.c3_2(self.Relu(self.c3_1(x))))
        x4 = self.Relu(self.c4_1(self.p4_1(x)))
        # 输出结构是（batch_size, channel, H, W)，所以dim是1
        y = torch.concat(tensors=(x1, x2, x3, x4), dim=1)
        return y


class GoogLeNet(nn.Module):
    def __init__(self, inception):
        super(GoogLeNet, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_block1 = nn.Sequential(inception(in_channels=192, c1_out_channels=64, c2_out_channels=(96, 128),
                                                        c3_out_channels=(16, 32), c4_out_channels=32),
                                              inception(in_channels=256, c1_out_channels=128, c2_out_channels=(128, 192),
                                                        c3_out_channels=(32, 96), c4_out_channels=64),
                                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_block2 = nn.Sequential(inception(in_channels=480, c1_out_channels=192, c2_out_channels=(96, 208),
                                                        c3_out_channels=(16, 48), c4_out_channels=64),
                                              inception(in_channels=512, c1_out_channels=160, c2_out_channels=(112, 224),
                                                        c3_out_channels=(24, 64), c4_out_channels=64),
                                              inception(in_channels=512, c1_out_channels=128, c2_out_channels=(128, 256),
                                                        c3_out_channels=(24, 64), c4_out_channels=64),
                                              inception(in_channels=512, c1_out_channels=112, c2_out_channels=(128, 288),
                                                        c3_out_channels=(32, 64), c4_out_channels=64),
                                              inception(in_channels=528, c1_out_channels=256, c2_out_channels=(160, 320),
                                                        c3_out_channels=(32, 128), c4_out_channels=128),
                                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.inception_block3 = nn.Sequential(inception(in_channels=832, c1_out_channels=256, c2_out_channels=(160, 320),
                                                        c3_out_channels=(32, 128), c4_out_channels=128),
                                              inception(in_channels=832, c1_out_channels=384, c2_out_channels=(192, 384),
                                                        c3_out_channels=(48, 128), c4_out_channels=128),
                                              nn.AdaptiveAvgPool2d((1, 1)))
        self.Linear_block1 = nn.Sequential(nn.Flatten(),
                                           nn.Linear(in_features=1024, out_features=10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 凯明初始化(fan_in 保证输入方差为 1,fan_out 保证反向传播时梯度方差为 1,梯度消失/爆炸时，把 variance 留给反向会更稳)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.inception_block3(x)
        x = self.Linear_block1(x)
        y = x
        return y


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    GoogLeNet = GoogLeNet(Inception).to(device)
    print(summary(GoogLeNet, input_size=(1, 224, 224)))

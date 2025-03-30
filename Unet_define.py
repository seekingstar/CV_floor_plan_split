# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 定义UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # 定义卷积块：Conv2d + BatchNorm2d + ReLU 的组合
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # 添加卷积层
            layers += [nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride, 
                               padding=padding,
                               bias=bias)]
            # 添加批归一化层
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # 添加激活函数
            layers += [nn.ReLU()]
            return nn.Sequential(*layers)  # 将层组合成序列

        ### 编码器（下采样）
        
        # 第1个编码阶段（输入分辨率最高）
        self.enc1_1 = CBR2d(in_channels, 64)   # 输入通道→64通道
        self.enc1_2 = CBR2d(64, 64)            # 保持64通道
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 下采样（高宽减半）

        self.enc2_1 = CBR2d(64, 128)         
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(128, 256)       
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(256, 512)         
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # 第5个编码阶段（最底层，分辨率最小）
        self.enc5_1 = CBR2d(512, 1024)       
        self.enc5_2 = CBR2d(1024, 1024)


        ### 解码器（上采样）
        # 第5个解码阶段（最底层开始上采样）
        self.dec5_1 = CBR2d(1024, 512)         # 1024→512通道
        self.dec5_2 = CBR2d(512, 512)
        # 转置卷积实现上采样（分辨率加倍）
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.dec4_2 = CBR2d(1024, 256)        
        self.dec4_1 = CBR2d(256, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.dec3_2 = CBR2d(512, 128)       
        self.dec3_1 = CBR2d(128, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.dec2_2 = CBR2d(256, 64)         
        self.dec2_1 = CBR2d(64, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # 第1个解码阶段（输出层）
        self.dec1_2 = CBR2d(128, 64)           # 处理64上采样 + 64编码特征 → 128通道
        self.dec1_1 = CBR2d(64, out_channels)  # 最终输出层（无激活函数，通常后续接Sigmoid或Softmax）

    def forward(self, x):
        ### 编码过程
        # 第1阶段
        enc1_1 = self.enc1_1(x)          # 初始卷积
        enc1_2 = self.enc1_2(enc1_1)     # 二次卷积
        pool1 = self.pool1(enc1_2)       # 下采样

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        # 第5阶段（最底层）
        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)     # 编码器最终输出


        ###  解码过程
        # 第5阶段解码
        dec5_1 = self.dec5_1(enc5_2)
        dec5_2 = self.dec5_2(dec5_1)
        unpool4 = self.unpool4(dec5_2)   # 上采样

        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # 拼接上采样结果和编码器对应层特征
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool3(dec4_1)

        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1)

        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1)

        # 第1阶段解码（输出层）
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)    # 最终输出

        x = dec1_1
        return x

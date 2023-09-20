import torch.nn as nn
import torch
from TPT import C_Att

class CFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFB, self).__init__()
        middle_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.convr1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.convr2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.convr3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.relu(out)

        y = self.relu(self.convr1(x))
        y = self.convr2(y)
        x = self.convr3(x)
        x2 = self.relu(x + y)

        x1 = torch.relu((x1 + x2) / 2)
        x1 = torch.sigmoid(x1)
        x2 = torch.relu(x2)
        M1 = x1
        M2 = x2
        YT = torch.mean(M2)
        Trp = torch.max(M1) / torch.min(M1)
        volumes = torch.multiply(Trp, M1)
        spectral = torch.where(M1 > YT, volumes, M2)
        reward = 2.5 * M1
        punishment = torch.zeros_like(spectral)
        M1 = torch.where(M1 > 0.2, reward, punishment)
        A = torch.multiply(M1, M2)
        return A

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = CFB(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.CAtt1 = C_Att(64)

        self.conv2 = CFB(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.CAtt2 = C_Att(128)

        self.conv3 = CFB(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.CAtt3 = C_Att(256)

        self.conv4 = CFB(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.CAtt4 = C_Att(512)

        self.conv5 = CFB(512, 1024)

        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = CFB(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = CFB(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = CFB(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = CFB(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)



    def forward(self, x):
        c1 = self.conv1(x)
        c1 = self.CAtt1(c1)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        c2 = self.CAtt2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        c3 = self.CAtt3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        c4 = self.CAtt4(c4)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)

        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        #out = nn.Sigmoid()(c10)  # 化成(0~1)区间 change
        out = c10
        return out


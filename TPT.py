import numpy
import torch
import torch.nn as nn
import numpy as np





class C_Att(nn.Module):

    def __init__(self,in_dim, **kwargs):  #
        super().__init__()
        in_dim = in_dim
        self.Horizontal_Convd = nn.Conv2d(in_channels = 2*in_dim, out_channels=in_dim,kernel_size=3,padding=1)
        self.vertical_Convd = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=3,padding=1)
        #self.Horizontal_Convu = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=2,stride=2)
        #self.vertical_Convu = nn.Conv2d(in_channels=2*in_dim,out_channels=in_dim,kernel_size=2,stride=2)
        # self.Sakura_Mint = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.x_dim = nn.sigmoid(dim=1)
        # self.H_dim = nn.sigmoid(dim=2)
        # self.V_dim = nn.sigmoid(dim=3)
        self.convr1 = nn.Conv2d(in_dim, 2*in_dim, kernel_size=3, padding=1)  # 先做卷积
        self.convr2 = nn.Conv2d(2*in_dim, in_dim, kernel_size=3, padding=1)
        self.convr3 = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )




    def forward(self, x):
        # x is a list (Feature matrix, Laplacian (Adjcacency) Matrix).
        #assert isinstance(x, list)
        #x = torch.tensor(x)
        _,C_dim1,H_dim2,W_dim3 = x.shape
        #print(x.size())
        C_Vertical = x.permute(0, 1, 3, 2).contiguous()
        WT_Vertical = x.permute(0, 3, 2, 1).contiguous()
        HD_Vertical = C_Vertical.permute(0, 2, 1, 3).contiguous()

        x1 = self.sigmoid(x)
        C_Vertical = self.relu(C_Vertical)
        WT_Vertical = self.sigmoid(WT_Vertical)
        HD_Vertical = self.relu(HD_Vertical)

        H1D_Vertical = HD_Vertical.permute(0, 1, 3, 2).contiguous()
        W1T_Vertical = WT_Vertical.permute(0, 1, 3, 2).contiguous()

        WT_Vertical = self.sigmoid((H1D_Vertical + WT_Vertical)/2)
        HD_Vertical = self.relu((W1T_Vertical + HD_Vertical)/2)

        WT_Vertical = WT_Vertical.permute(0, 3, 2, 1).contiguous()
        HD_Vertical = HD_Vertical.permute(0, 2, 1, 3).contiguous()

        WT_Vertical = self.sigmoid(self.conv(WT_Vertical))
        HD_Vertical = self.relu(self.conv(HD_Vertical))

        C1_Vertical = torch.cat([WT_Vertical, x1], 1)
        C2_Vertical = torch.cat([HD_Vertical, C_Vertical], 1)

        C1_Vertical = self.sigmoid(self.Horizontal_Convd(C1_Vertical))
        C2_Vertical = self.relu(self.Horizontal_Convd(C2_Vertical))

        C2_Vertical = C2_Vertical.permute(0,1,3,2).contiguous()

        M1 = self.sigmoid(self.vertical_Convd(torch.cat([C1_Vertical, C2_Vertical], 1)))


        # y = self.relu(self.convr1(x))
        # y = self.convr2(y)
        # x = self.convr3(x)
        # M2 = self.relu(x + y)
        #M2 = self.sigmoid(x)
        #
        punishment = torch.zeros_like(M1)  # tf.zeros_like(M1)
        # #Loss
        M1 = torch.where(M1 > 0.9, M1 * 0.1, punishment)  # tf.where(M1 > 0.2, x=reward, y=punishment)

        #A = (M1 + M2)/2#torch.multiply(M1, M2)#add
        M1 = x + M1
        return M1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = C_Att(3)
    x = torch.randn(1, 3, 6, 6)
    print(x)
    print(x.shape)
    for i in range(2):
        out = model(x)
    print(out.shape)
    print(out)

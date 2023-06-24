import torch
import torch.nn as nn

### create CNN block
class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding_mode="reflect")
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x=self.conv(x)
        x=self.bnorm(x)
        x=self.lrelu(x)
        return x

### create discriminator class
class Discriminator(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, 64,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.l1 = CNNBlock(in_channels=64,out_channels=128)
        self.l2= CNNBlock(in_channels=128,out_channels=256)
        self.l3 = CNNBlock(in_channels=256,out_channels=512)
        self.l4 = CNNBlock(in_channels=512,out_channels=1)

    def forward(self,x,y):
        x=torch.cat([x,y],dim=1)
        x=self.initial(x)
        x=self.l1(x)
        x=self.l2(x)
        x=self.l3(x)
        x=self.l4(x)
        return x

### block used in DownConv
class GenBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,padding_mode="reflect")
        self.BN = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x=self.conv(x)
        x=self.BN(x)
        x=self.lrelu(x)
        return x

# class DownConv(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super().__init__()
#         self.initial_layer = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
#             nn.LeakyReLU(0.2)
#         )
#         self.down1= GenBlock(out_channels,out_channels*2)
#         self.down2= GenBlock(out_channels*2,out_channels*4)
#         self.down3= GenBlock(out_channels*4,out_channels*8)
#         self.down4= GenBlock(out_channels*8,out_channels*8)
#         self.down5= GenBlock(out_channels*8,out_channels*8)
#         self.down6= GenBlock(out_channels*8,out_channels*8)
        
#     def forward(self,x):
#         d1=self.initial_layer(x)
#         d2=self.down1(d1)
#         d3=self.down2(d2)
#         d4=self.down3(d3)
#         d5=self.down4(d4)
#         d6=self.down5(d5)
#         d7=self.down6(d6)
#         return d7

### block used in Upconv
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.BN(x)
        x=self.relu(x)
        return x


# class UpConv(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super().__init__()
#         self.initial_layer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1),
#             nn.ReLU()
#         )
#         self.up1 = UpBlock(out_channels*8,out_channels*8)
#         self.up2 = UpBlock(out_channels*8*2,out_channels*8)
#         self.up3 = UpBlock(out_channels*8*2,out_channels*8)
#         self.up4 = UpBlock(out_channels*8*2,out_channels*8)
#         self.up5 = UpBlock(out_channels*8*2,out_channels*4)
#         self.up6 = UpBlock(out_channels*4*2,out_channels*2)
#         self.up7 = UpBlock(out_channels*2*2,out_channels)
#         self.dropout = nn.Dropout(0.4)

#     def forward(self,x):
#         # x=self.initial_layer(x)
#         up1=self.up1(x)
#         up1=self.dropout(up1)
#         up2=self.up2(torch.cat([up1,DownConv.d7],1))
#         up2=self.dropout(up2)
#         up3=self.up3(torch.cat([up2,d6],1))
#         up3=self.dropout(up3)
#         up4=self.up4(torch.cat([up3,d5],1))
#         up5=self.up5(torch.cat([up4,d4],1))
#         up6=self.up6(torch.cat([up5,d3],1))
#         up7=self.up7(torch.cat([up6,d2],1))
#         return x

class Generator(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down1= GenBlock(out_channels,out_channels*2)
        self.down2= GenBlock(out_channels*2,out_channels*4)
        self.down3= GenBlock(out_channels*4,out_channels*8)
        self.down4= GenBlock(out_channels*8,out_channels*8)
        self.down5= GenBlock(out_channels*8,out_channels*8)
        self.down6= GenBlock(out_channels*8,out_channels*8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels*8,out_channels*8,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.ReLU()
        )

        self.up1 = UpBlock(out_channels*8,out_channels*8)
        self.up2 = UpBlock(out_channels*8*2,out_channels*8)
        self.up3 = UpBlock(out_channels*8*2,out_channels*8)
        self.up4 = UpBlock(out_channels*8*2,out_channels*8)
        self.up5 = UpBlock(out_channels*8*2,out_channels*4)
        self.up6 = UpBlock(out_channels*4*2,out_channels*2)
        self.up7 = UpBlock(out_channels*2*2,out_channels)
        self.dropout = nn.Dropout(0.4)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels*2,out_channels=in_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        d1=self.initial_layer(x)
        d2=self.down1(d1)
        d3=self.down2(d2)
        d4=self.down3(d3)
        d5=self.down4(d4)
        d6=self.down5(d5)
        d7=self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1=self.up1(bottleneck)
        up1=self.dropout(up1)
        up2=self.up2(torch.cat([up1,d7],1))
        up2=self.dropout(up2)
        up3=self.up3(torch.cat([up2,d6],1))
        up3=self.dropout(up3)
        up4=self.up4(torch.cat([up3,d5],1))
        up5=self.up5(torch.cat([up4,d4],1))
        up6=self.up6(torch.cat([up5,d3],1))
        up7=self.up7(torch.cat([up6,d2],1))
        result= self.final_layer(torch.cat([up7,d1],1))
        return result


# x=torch.randn((1,3,256,256))
# model = Generator(3,64)
# pred = model(x)
# print(pred.shape)
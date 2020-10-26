from torch import nn
import torch.nn.functional as F
from pdb import set_trace as bp

# Support: ['LightNet']

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"

        krnl_sz=3
        strd = 1
                    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu1_1 = nn.PReLU()
        self.prelu1_2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=64, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=krnl_sz, stride=strd, padding=1)
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=krnl_sz, stride=strd, padding=1)
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()

        
        if input_size[0] == 112:
            self.out = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(3136 * 4 * 4, 512),
            )
        else:
            self.out = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(3136 * 8 * 8, 512),
            )

    def forward(self, x):
        mp_ks=2
        mp_strd=2

        x = self.prelu1_1(self.conv1(x))
        x = self.prelu1_2(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = self.prelu2_1(self.conv3(x))
        x = self.prelu2_2(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = self.prelu3_1(self.conv5(x))
        x = self.prelu3_2(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = self.prelu4_1(self.conv7(x))
        x = self.prelu4_2(self.conv8(x))
        x = F.max_pool2d(x, kernel_size=mp_ks, stride=mp_strd)

        x = x.view(x.size(0), -1) # Flatten           
        x = self.out(x)

        return x




# class Net(nn.Module):
#     def __init__(self, input_size):
#         super(Net, self).__init__()
#         assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
#         self.input_size = input_size
        
#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True)
#             )

#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#                 nn.BatchNorm2d(inp),
#                 nn.ReLU(inplace=True),
    
#                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.ReLU(inplace=True),
#             )

#         self.model = nn.Sequential(
#             conv_bn(  3,  32, 2), 
#             conv_dw( 32,  64, 1),
#             conv_dw( 64, 128, 2),
#             conv_dw(128, 128, 1),
#             conv_dw(128, 256, 2),
#             conv_dw(256, 256, 1),
#             conv_dw(256, 512, 2),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 512, 1),
#             conv_dw(512, 1024, 2),
#             conv_dw(1024, 1024, 1),
#             nn.AvgPool2d(7),
#         )
#         self.fc = nn.Linear(1024, 512)

#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1024)
#         x = self.fc(x)
        
#         return x


def LightNet(input_size, **kwargs):
    model = Net(input_size, **kwargs)
    return model


import torch
from torch_geometric.nn import *
import torch.nn.functional as F


from MYgcn import GCNconv

import os

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNconv(in_channels = 18+7 + 19 +20, out_channels=50, edge_length=112)  # 160
        self.BN1 = torch.nn.BatchNorm1d(50)
        self.conv2 = GCNconv(50,50)
        self.BN2 = torch.nn.BatchNorm1d(50)
        self.conv3 = GCNconv(50,50)
        self.BN3 = torch.nn.BatchNorm1d(50)
        self.conv4 = GCNconv(50,50)
        self.BN4 = torch.nn.BatchNorm1d(50)
        self.conv5 = GCNconv(50,50)
        self.BN5 = torch.nn.BatchNorm1d(50)
        self.conv6 = GCNconv(50,50)
        self.BN6 = torch.nn.BatchNorm1d(50)
        self.conv7 = GCNconv(50,50)
        self.BN7 = torch.nn.BatchNorm1d(50)
        self.conv8 = GCNconv(50,50)
        self.BN8 = torch.nn.BatchNorm1d(50)
        self.conv9 = GCNconv(50,50)
        self.BN9 = torch.nn.BatchNorm1d(50)
        self.conv10 = GCNconv(50,50)
        self.BN10 = torch.nn.BatchNorm1d(50)
        #self.act1 = torch.nn.ReLU()
        #self.act2 = torch.nn.ReLU()
        self.act = torch.nn.Softmax()

        self.conv11 = GCNconv(50,50)
        self.BN11 = torch.nn.BatchNorm1d(50)
        self.conv12 = GCNconv(50,50)
        self.BN12 = torch.nn.BatchNorm1d(50)
        self.conv13 = GCNconv(50,50)
        self.BN13 = torch.nn.BatchNorm1d(50)
        self.conv14 = GCNconv(50,50)
        self.BN14 = torch.nn.BatchNorm1d(50)
        self.conv15 = GCNconv(50,50)
        self.BN15 = torch.nn.BatchNorm1d(50)
        self.conv16 = GCNconv(50,50)
        self.BN16 = torch.nn.BatchNorm1d(50)
        self.conv17 = GCNconv(50,50)
        self.BN17 = torch.nn.BatchNorm1d(50)
        self.conv18 = GCNconv(50,50)
        self.BN18 = torch.nn.BatchNorm1d(50)
        self.conv19 = GCNconv(50,50)
        self.BN19 = torch.nn.BatchNorm1d(50)
        self.conv20 = GCNconv(50,50)
        self.BN20 = torch.nn.BatchNorm1d(50)

        self.lin = torch.nn.Linear(50+ 8  ,49*4) #380  +8  +20 prior
        self.BN = torch.nn.BatchNorm1d(49*4)
        self.drop = torch.nn.Dropout(0.3)

        self.Conv2D1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 1 +4 + 2*2, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D5 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D6 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D7 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D8 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D9 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D10 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(8),
                torch.nn.ELU(),
                )
        self.Conv2D_L1 = torch.nn.Linear(8,10)
        self.Conv2D_L2 = torch.nn.Linear(8,10)

        self.conditionLin1 = torch.nn.Linear(49,2)
        self.conditionBN1 = torch.nn.BatchNorm1d(2)
        self.conditionLin2 = torch.nn.Linear(2,4000)
        self.conditionBN2 = torch.nn.BatchNorm1d(4000)

        self.reducelin = torch.nn.Linear(50,2)
        self.reduceBN = torch.nn.BatchNorm1d(2)

        self.endConv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8 , out_channels = 38, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(38),
                torch.nn.ELU(),
                )

        self.endConv2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels = 8 , out_channels = 48, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(48),
                torch.nn.ELU(),
                )


    def forward(self,x,edge_index,edge_attr,distance,):#condition):  # distanc
        #x = torch.cat((x,prior),1)
        x,edge_attr = self.conv1(x,edge_index,edge_attr)
        x1 = self.BN1(x)
        x1 = F.elu(x1)
        x2,edge_attr2 = self.conv2(x1,edge_index,edge_attr)
        x2 = self.BN2(x2)
        x2 = F.elu(x2)
        x2_C = x1+x2   # torch.cat((x1,x2),1)
        #edge_attr2_C = edge_attr + edge_attr2
        x3, edge_attr3 = self.conv3(x2_C,edge_index,edge_attr2)
        x3 = F.elu( self.BN3(x3) ) # + x1 # self.lin(x1)
        x3_C = x1+x2+x3
        #edge_attr3_C = edge_attr + edge_attr2 + edge_attr3
        x4, edge_attr4 = self.conv4(x3_C,edge_index,edge_attr3)
        x4 = self.BN4(x4)
        x4 = F.elu(x4)
        x4_C = x1+x2+x3+x4
        #edge_attr4_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4
        x5, edge_attr5 = self.conv5(x4_C,edge_index,edge_attr4)
        x5 = self.BN5(x5)
        x5 = F.elu(x5)
        x5_C = x1+x2+x3+x4+x5
        #edge_attr5_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5
        x6, edge_attr6 = self.conv6(x5_C,edge_index,edge_attr5)
        x6 = self.BN6(x6)
        x6 = F.elu(x6)
        x6_C = x1+x2+x3+x4+x5+x6
        #edge_attr6_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6
        x7, edge_attr7 = self.conv7(x6_C,edge_index,edge_attr6)
        x7 = self.BN7(x7)
        x7 = F.elu(x7)
        x7_C = x1+x2+x3+x4+x5+x6+x7
        #edge_attr7_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7
        x8, edge_attr8 = self.conv8(x7_C,edge_index,edge_attr7)
        x8 = self.BN8(x8)
        x8 = F.elu(x8)
        x8_C = x1+x2+x3+x4+x5+x6+x7+x8
        #edge_attr8_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8
        x9, edge_attr9 = self.conv9(x8_C,edge_index,edge_attr8)
        x9 = self.BN9(x9)
        x9 = F.elu(x9)
        x9_C = x1+x2+x3+x4+x5+x6+x7+x8+x9
        #edge_attr9_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9
        x10, edge_attr10 = self.conv10(x9_C,edge_index,edge_attr9)
        x10 = self.BN10(x10)
        x10 = F.elu(x10)
        x10_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10
        #edge_attr10_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10
        
        ''' 
        x11, edge_attr11 = self.conv11(x10_C,edge_index,edge_attr10) 
        x11 = self.BN11(x11)
        x11 = F.elu(x11)
        x11_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11
        #edge_attr11_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11
        x12, edge_attr12 = self.conv12(x11_C,edge_index,edge_attr11)
        x12 = self.BN12(x12)
        x12 = F.elu(x12)
        x12_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12
        #edge_attr12_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12
        x13, edge_attr13 = self.conv13(x12_C,edge_index,edge_attr12)
        x13 = self.BN13(x13)
        x13 = F.elu(x13)
        x13_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13
        #edge_attr13_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13
        x14, edge_attr14 = self.conv14(x13_C,edge_index,edge_attr13)
        x14 = self.BN14(x14)
        x14 = F.elu(x14)
        x14_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14
        #edge_attr14_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13 + edge_attr14
        x15, edge_attr15 = self.conv15(x14_C,edge_index,edge_attr14)
        x15 = self.BN15(x15)
        x15 = F.elu(x15)
        x15_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15
        #edge_attr15_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13 + edge_attr14 + edge_attr15
        x16, edge_attr16 = self.conv16(x15_C,edge_index,edge_attr15)
        x16 = self.BN16(x16)
        x16 = F.elu(x16)
        x16_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16
        #edge_attr16_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13 + edge_attr14 + edge_attr15 + edge_attr16
        x17, edge_attr17 = self.conv17(x16_C,edge_index,edge_attr16)
        x17 = self.BN17(x17)
        x17 = F.elu(x17)
        x17_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17
        #edge_attr17_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13 + edge_attr14 + edge_attr15 + edge_attr16 + edge_attr17
        x18, edge_attr18 = self.conv18(x17_C,edge_index,edge_attr17)
        x18 = self.BN18(x18)
        x18 = F.elu(x18)
        x18_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18
        #edge_attr18_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13 + edge_attr14 + edge_attr15 + edge_attr16 + edge_attr17 + edge_attr18
        x19, edge_attr19 = self.conv19(x18_C,edge_index,edge_attr18)
        x19 = self.BN19(x19)
        x19 = F.elu(x19)
        x19_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19
        #edge_attr19_C = edge_attr + edge_attr2 + edge_attr3 + edge_attr4 + edge_attr5 + edge_attr6 + edge_attr7 + edge_attr8 +edge_attr9 + edge_attr10 + edge_attr11 + edge_attr12 + edge_attr13 + edge_attr14 + edge_attr15 + edge_attr16 + edge_attr17 + edge_attr18 + edge_attr19
        x20, edge_attr20 = self.conv20(x19_C,edge_index,edge_attr19)
        x20 = self.BN20(x20)
        x20 = F.elu(x20)
        x20_C = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20
        '''

        '''
        for num_layer in range(5):
            x = F.relu(GCNconv(20,20)(x,edge_index,edge_attr))
        '''
        
        #distance = distance.unsqueeze(0).unsqueeze(0)  ## exchange L*L to 1*1*L*L
        distance = distance.permute(2,0,1 ).unsqueeze(0)
        
        x10_C_m = self.reducelin(x10_C)
        x10_C_m = self.reduceBN(x10_C_m)
        x10_C_c = x10_C_m.repeat( len(x10_C),1)
        x10_C_r = x10_C_m.repeat(1, len(x10_C) ).view(-1,2)
        x10_C_rc = torch.cat(( x10_C_r,x10_C_c),1 ).view( len(x10_C),len(x10_C),2*2)
        x10_C_rc = x10_C_rc.unsqueeze(0)
        
        distance = torch.cat((distance, x10_C_rc.permute(0,3,1,2) ) , 1) 
        
        d1 = self.Conv2D1(distance)
        d2 = self.Conv2D2(d1)
        d2_C = d1 + d2
        d3 = self.Conv2D3(d2_C)
        d3_C = d1 + d2 + d3
        d4 = self.Conv2D4(d3_C)
        d4_C = d1 + d2 + d3 + d4
        d5 = self.Conv2D5(d4_C)
        d5_C = d1 + d2 + d3 + d4 + d5
        d6 = self.Conv2D6(d5_C)
        d6_C = d1 + d2 + d3 + d4 + d5 + d6
        d7 = self.Conv2D7(d6_C)
        d7_C = d1 + d2 + d3 + d4 + d5 + d6 + d7
        d8 = self.Conv2D8(d7_C)
        d8_C = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8
        d9 = self.Conv2D9(d8_C)
        d9_C = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9
        d10 = self.Conv2D10(d9_C)
        d10_C = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10
        #print(d10_C.size())        
        d10_end = self.endConv(d10_C)
        d10_end2 = self.endConv2(d10_C)

        d10_C = d10_C.permute(0,2,3,1)
        #d10_C = torch.cat((d10_end,d10_end2),1).permute(0,2,3,1)
        Conv2D_L1 = self.Conv2D_L1(d10_C)
        Conv2D_L2 = self.Conv2D_L2(d10_C)
        Conv2D_L1 = F.softmax(Conv2D_L1,-1).permute(0,1,3,2)
        Conv2D_L2 = F.softmax(Conv2D_L2,-1).permute(0,1,3,2)
        o = torch.matmul(Conv2D_L1,d10_C) + torch.matmul(Conv2D_L2,d10_C)
        #o = F.relu(o).sum(2).squeeze(0)
        o = F.elu(o).mean(2).squeeze(0)
        '''
        #condition = condition.view(-1)
        #condition = torch.scatter( torch.zeros(condition.size(0),48), 1, condition.unsqueeze(1),1 ).float()
        condition1 = self.conditionLin1(condition)
        condition1_norm = self.conditionBN1(condition1)
        condition1_act = F.elu(condition1_norm)
        condition2 = self.conditionLin2(condition1_act)
        condition2_norm = self.conditionBN2(condition2)
        #condition2_act = F.elu(condition2_norm)
        condition2_norm = condition2_norm.t()[:condition.size(0)]
        condition2_act = F.softmax(condition2_norm,-1)
        #condition2_act = condition2_act - torch.diag_embed( torch.diag(condition2_act) )
        condition2_act = torch.tril(condition2_act,diagonal=-1)
        condition_out = torch.matmul(  condition2_act, condition )
        '''
        mid = torch.cat((x10_C,o),1)
        #mid = mid.repeat(1,4).view(-1,mid.size(1))
        #o = torch.cat((mid , condition_out),1)  #prior
        
        
        o =self.drop(mid)  # x20_C x10_C

        o = self.lin(o) # x10_C
        o = self.BN(o)
        #print(d10_end.size())
        return o.view(-1,49,4), d10_end, d10_end2








from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torch import nn


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SelfAttn2(nn.Module):
    def __init__(self, in_dim, stride=4, group_num=2):
        super(SelfAttn2, self).__init__()
        self.in_dim    = in_dim
        self.stride    = (stride, stride) if isinstance(stride, int) else stride
        self.group_num = group_num
        self.qkv_conv  = nn.Conv2d(in_dim, in_dim // 2 + in_dim, kernel_size=1)
        self.out_conv  = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma     = nn.Parameter(torch.zeros(1))
        print("##########################")
        print(self.stride, self.group_num)

    def forward(self, x, mode=0):
        b, c, h, w = x.size()
        g, s = self.group_num, self.stride
        b1  = b * g
        c1  = c // 4 // g
        c2  = c // g
        h1  = h // s[0]
        w1  = w // s[1]

        qkv = self.qkv_conv(x)                                               # (b, c, H, W)
        Q, K, V = torch.split(qkv, [c//4, c//4, c], dim=1)                   # (b, c, H, W)

        q   = F.avg_pool2d(Q, kernel_size=s, stride=s, padding=0)            # (b, c, h, w)
        k   = F.avg_pool2d(K, kernel_size=s, stride=s, padding=0)            # (b, c, h, w)

        q   = q.reshape(b1, c1, -1)                                          # (b*g, c, q)
        k   = k.reshape(b1, c1, -1)                                          # (b*g, c, k)
        Q   = Q.reshape(b1, c1, -1)                                          # (b*g, c, Q)
        K   = K.reshape(b1, c1, -1)                                          # (b*g, c, K)
        V   = V.reshape(b1, c2, -1)                                          # (b*g, c, K)

        A_c = torch.bmm(k.permute(0, 2, 1), Q)                               # (b*g, k, Q)
        A_m = torch.bmm(k.permute(0, 2, 1), q)                               # (b*g, k, q)
        A_s = torch.bmm(K.permute(0, 2, 1), q)                               # (b*g, K, q)

        A_c = F.softmax(A_c, dim=1)                                          # (b*g, k, Q) (b*g, h1*w1, h1*s*w1*s)
        A_m = F.softmax(A_m, dim=1)                                          # (b*g, k, q) (b*g, h1*w1, h1*w1)
        A_s = F.softmax(A_s, dim=1)                                          # (b*g, K, q) (b*g, h1*s*w1*s, h1*w1)

        e   = torch.eye(h1*w1, dtype=x.dtype, device=x.device)               # (h1*w1, h1*w1)
        E   = e.view(h1*w1, h1, w1)[:, :, None, :, None].expand(-1, -1, s[0], -1, s[1]).reshape(h1*w1, -1)  # (h1*w1, h1*s*w1*s)

        A_c = A_c + E                                                        # (b*g, k, Q)
        A_m = A_m + e                                                        # (b*g, k, q)
        A_m = torch.inverse(A_m)                                             # (b*g, q, k)
        A_c = torch.bmm(A_m, A_c)                                            # (b*g, q, Q)

        O   = torch.bmm(V, A_s)                                              # (b*g, c, q)
        O   = torch.bmm(O, A_c)                                              # (b*g, c, Q)
        out = O.view(b, -1, h, w)                                            # (b, c, h, w)
        out = self.out_conv(out)                                             # (b, c, h, w)
        if mode == 0:
            out = self.gamma * out + x                                       # (b, c, h, w)
        return out


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        for mo in self.base.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        self.num_features = num_features
        self.num_classes = num_classes  # 751 #num_classes to be changed according to dataset
        self.dropout = dropout
        out_planes0 = self.base.layer1[0].conv1.in_channels
        out_planes1 = self.base.layer2[0].conv1.in_channels
        out_planes2 = self.base.layer3[0].conv1.in_channels
        out_planes3 = self.base.layer4[0].conv1.in_channels
        out_planes4 = self.base.fc.in_features

        self.local_conv = nn.Conv2d(out_planes4, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer1 = nn.Conv2d(out_planes1, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer2 = nn.Conv2d(out_planes2, self.num_features, kernel_size=1, padding=0, bias=False)
        self.local_conv_layer3 = nn.Conv2d(out_planes3, self.num_features, kernel_size=1, padding=0, bias=False)

        init.kaiming_normal_(self.local_conv.weight, mode='fan_out')
        init.kaiming_normal_(self.local_conv_layer1.weight, mode='fan_out')
        init.kaiming_normal_(self.local_conv_layer2.weight, mode='fan_out')
        init.kaiming_normal_(self.local_conv_layer3.weight, mode='fan_out')

        #       init.constant_(self.local_conv.bias,0)
        self.feat_bn2d = nn.BatchNorm2d(self.num_features)  # may not be used, not working on caffe
        init.constant_(self.feat_bn2d.weight, 1)  # initialize BN, may not be used
        init.constant_(self.feat_bn2d.bias, 0)  # iniitialize BN, may not be used

        # self.offset = ConvOffset2D(256)

        self.SA0 = SelfAttn2(out_planes1, stride=4)
        self.SA1 = SelfAttn2(out_planes1, stride=4)
        self.SA2 = SelfAttn2(out_planes2, stride=4)
        self.SA3 = SelfAttn2(out_planes3, stride=2)
        self.SA4 = SelfAttn2(out_planes4, stride=2)

        ##---------------------------stripe1----------------------------------------------#
        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance0.weight, std=0.001)
        init.constant_(self.instance0.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance1.weight, std=0.001)
        init.constant_(self.instance1.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance2.weight, std=0.001)
        init.constant_(self.instance2.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance3.weight, std=0.001)
        init.constant_(self.instance3.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance4.weight, std=0.001)
        init.constant_(self.instance4.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance5 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance5.weight, std=0.001)
        init.constant_(self.instance5.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance6 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance6.weight, std=0.001)
        init.constant_(self.instance6.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance_layer1 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance_layer1.weight, std=0.001)
        init.constant_(self.instance_layer1.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance_layer2 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance_layer2.weight, std=0.001)
        init.constant_(self.instance_layer2.bias, 0)
        ##---------------------------stripe1----------------------------------------------#
        ##---------------------------stripe1----------------------------------------------#
        self.instance_layer3 = nn.Linear(self.num_features, self.num_classes)
        init.normal_(self.instance_layer3.weight, std=0.001)
        init.constant_(self.instance_layer3.bias, 0)

        self.fusion_conv = nn.Conv1d(4, 1, kernel_size=1, bias=False)

        self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            if name == 'layer1':
                x = module[0](x)
                x = module[1](x)
                x = self.SA0(x, 0)
                x = module[2](x)
            else:
                x = module(x)
            if name == 'layer1':
                x_layer1 = self.SA1(x, 1)
            if name == 'layer2':
                x_layer2 = self.SA2(x, 1)
            if name == 'layer3':
                x_layer3 = self.SA3(x, 1)
        
        # Deep Supervision
        x_layer1 = F.avg_pool2d(x_layer1, kernel_size=(96, 32), stride=(1, 1))
        x_layer1 = self.local_conv_layer1(x_layer1)
        x_layer1 = x_layer1.contiguous().view(x_layer1.size(0), -1)
        x_layer1 = self.instance_layer1(x_layer1)

        x_layer2 = F.avg_pool2d(x_layer2, kernel_size=(48, 16), stride=(1, 1))
        x_layer2 = self.local_conv_layer2(x_layer2)
        x_layer2 = x_layer2.contiguous().view(x_layer2.size(0), -1)
        x_layer2 = self.instance_layer2(x_layer2)

        x_layer3 = F.avg_pool2d(x_layer3, kernel_size=(24, 8), stride=(1, 1))
        x_layer3 = self.local_conv_layer3(x_layer3)
        x_layer3 = x_layer3.contiguous().view(x_layer3.size(0), -1)
        x_layer3 = self.instance_layer3(x_layer3)

        # Part-Level Feature
        sx = x.size(2) // 6
        kx = x.size(2) - sx * 5
        x  = F.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))  # H4 W8

        out0 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)  # use this feature vector to do distance measure
        # out0 = torch.cat([f3,out0],dim=1)
        x  = self.drop(x)
        x  = self.local_conv(x)
        x  = self.feat_bn2d(x)
        x  = F.relu(x)  # relu for local_conv feature
        x6 = F.avg_pool2d(x, kernel_size=(6, 1), stride=(1, 1))
        x6 = x6.contiguous().view(x6.size(0), -1)

        c6 = self.instance6(x6)

        x  = x.chunk(6, 2)
        x0 = x[0].contiguous().view(x[0].size(0), -1)
        x1 = x[1].contiguous().view(x[1].size(0), -1)
        x2 = x[2].contiguous().view(x[2].size(0), -1)
        x3 = x[3].contiguous().view(x[3].size(0), -1)
        x4 = x[4].contiguous().view(x[4].size(0), -1)
        x5 = x[5].contiguous().view(x[5].size(0), -1)
        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        c4 = self.instance4(x4)
        c5 = self.instance5(x5)
        return out0, (c0, c1, c2, c3, c4, c5, c6, x_layer1, x_layer2, x_layer3)

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


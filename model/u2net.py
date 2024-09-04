import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        #self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.conv_s1 = GhostConv(in_ch, out_ch, k=3, s=1, g=1, act=True)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


################################################################################
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
#########################################################################################


############################################################################################
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape: b, num_channels, h, w  -->  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channelshuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class shuffleNet_unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(shuffleNet_unit, self).__init__()

        mid_channels = out_channels // 4
        self.stride = stride
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups
        self.GConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups,
                      bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.GConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 添加卷积层以匹配shortcut的通道数
        self.shortcut_match = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=self.groups,
                                        bias=False)

        if self.stride == 2:
            self.shortcut_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut_pool = nn.Identity()

    def forward(self, x):
        out = self.GConv1(x)
        out = channel_shuffle(out, groups=self.groups)
        out = self.DWConv(out)
        out = self.GConv2(out)

        # 使用shortcut_match调整shortcut的通道数
        short = self.shortcut_match(x)
        short = self.shortcut_pool(short)

        if self.stride == 2:
            out = F.relu(out + short)
        else:
            out = F.relu(out + short)

        return out


############################################################################################
## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
# class RSU7(nn.Module):#UNet07DRES(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU7,self).__init__()
#
#         self.rebnconvin = GhostConv(in_ch,out_ch,3)
#
#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
#
#         self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)
#
#         self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
#
#     def forward(self,x):
#
#         hx = x
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)
#
#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)
#
#         hx5 = self.rebnconv5(hx)
#         hx = self.pool5(hx5)
#
#         hx6 = self.rebnconv6(hx)
#
#         hx7 = self.rebnconv7(hx6)
#
#         hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
#         hx6dup = _upsample_like(hx6d,hx5)
#
#         hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)
#
#         hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
#
#         return hx1d + hxin
#
# ### RSU-6 ###
# class RSU6(nn.Module):#UNet06DRES(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU6,self).__init__()
#
#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
#
#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
#
#         self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)
#
#         self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
#
#     def forward(self,x):
#
#         hx = x
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)
#
#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)
#
#         hx5 = self.rebnconv5(hx)
#
#         hx6 = self.rebnconv6(hx5)
#
#
#         hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)
#
#         hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
#
#         return hx1d + hxin
#
# ### RSU-5 ###
# class RSU5(nn.Module):#UNet05DRES(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU5,self).__init__()
#
#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
#
#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
#
#         self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)
#
#         self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
#
#     def forward(self,x):
#
#         hx = x
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)
#
#         hx4 = self.rebnconv4(hx)
#
#         hx5 = self.rebnconv5(hx4)
#
#         hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
#
#         return hx1d + hxin
#
# ### RSU-4 ###
# class RSU4(nn.Module):#UNet04DRES(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU4,self).__init__()
#
#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
#
#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
#         self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
#
#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)
#
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
#
#     def forward(self,x):
#
#         hx = x
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#
#         hx4 = self.rebnconv4(hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
#
#         return hx1d + hxin
#
# ### RSU-4F ###
# class RSU4F(nn.Module):#UNet04FRES(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
#         super(RSU4F,self).__init__()
#
#         self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
#
#         self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
#         self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
#         self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)
#
#         self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)
#
#         self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
#         self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
#         self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
#
#     def forward(self,x):
#
#         hx = x
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx2 = self.rebnconv2(hx1)
#         hx3 = self.rebnconv3(hx2)
#
#         hx4 = self.rebnconv4(hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
#         hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
#         hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))
#
#         return hx1d + hxin


class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = shuffleNet_unit(in_ch, out_ch, stride=1, groups=1)

        self.rebnconv1 = shuffleNet_unit(out_ch, mid_ch, stride=1, groups=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.rebnconv7 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv6d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv5d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv4d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv3d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv2d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv1d = shuffleNet_unit(mid_ch * 2, out_ch, stride=1, groups=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = shuffleNet_unit(in_ch, out_ch, stride=1, groups=1)

        self.rebnconv1 = shuffleNet_unit(out_ch, mid_ch, stride=1, groups=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv6 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv5d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv4d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv3d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv2d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv1d = shuffleNet_unit(mid_ch * 2, out_ch, stride=1, groups=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = shuffleNet_unit(in_ch, out_ch, stride=1, groups=1)

        self.rebnconv1 = shuffleNet_unit(out_ch, mid_ch, stride=1, groups=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv5 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv4d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv3d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv2d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv1d = shuffleNet_unit(mid_ch * 2, out_ch, stride=1, groups=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = shuffleNet_unit(in_ch, out_ch, stride=1, groups=1)

        self.rebnconv1 = shuffleNet_unit(out_ch, mid_ch, stride=1, groups=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv4 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv3d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv2d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv1d = shuffleNet_unit(mid_ch * 2, out_ch, stride=1, groups=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = shuffleNet_unit(in_ch, out_ch, stride=1, groups=1)

        self.rebnconv1 = shuffleNet_unit(out_ch, mid_ch, stride=1, groups=1)
        self.rebnconv2 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.rebnconv3 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)
        self.rebnconv4 = shuffleNet_unit(mid_ch, mid_ch, stride=1, groups=1)

        self.rebnconv3d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv2d = shuffleNet_unit(mid_ch * 2, mid_ch, stride=1, groups=1)
        self.rebnconv1d = shuffleNet_unit(mid_ch * 2, out_ch, stride=1, groups=1)

    def forward(self, x):
            hx = x

            hxin = self.rebnconvin(hx)

            hx1 = self.rebnconv1(hxin)
            hx2 = self.rebnconv2(hx1)
            hx3 = self.rebnconv3(hx2)

            hx4 = self.rebnconv4(hx3)

            hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
            hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
            hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

            return hx1d + hxin


############################################################SPPM模块

class PPContextModule(nn.Module):
    """
    简单的上下文模块（Context module）。

    Args:
        in_channels (int): 金字塔池化模块的输入通道数。
        inter_channels (int): 金字塔池化模块中间卷积层的通道数。
        out_channels (int): 金字塔池化模块输出通道数。
        bin_sizes (tuple, optional): 池化特征图的输出尺寸。默认为（1，3）。
        align_corners (bool): F.interpolate函数的一个参数，当特征图的输出尺寸为偶数时，应设置为False；否则设置为True。
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes=(1, 3, 5),
                 align_corners=False):
        super(PPContextModule, self).__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = torch.nn.functional.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


##############################################################
##############################################################CBAM注意力机制

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
############################################################



##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()
        # 对于每一个RSU来说，本质其实就是一个Unet，多个下采样多个上采样

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = PPContextModule(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)



        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        #CBAM
        self.attention1 = CBAM(channel=64)####################
        self.attention2 = CBAM(channel=128)###################
        self.attention3 = CBAM(channel=256)###################
        self.attention4 = CBAM(channel=512)###################

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        # 通过一个个Unet得到相应的mask
        hx1 = self.attention1(hx1) ##############
        hx = self.pool12(hx1)


        #stage 2
        hx2 = self.stage2(hx)
        hx2 = self.attention2(hx2) ##############
        hx = self.pool23(hx2)


        #stage 3
        hx3 = self.stage3(hx)
        hx3 = self.attention3(hx3) ##############
        hx = self.pool34(hx3)


        #stage 4
        hx4 = self.stage4(hx)
        hx4 = self.attention4(hx4) ##############
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5 = self.attention4(hx5)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6 = self.attention4(hx6)
        hx6up = _upsample_like(hx6, hx5)



        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        # 这里类似FPN。每个block的输出结果和上一个（下一个block）结果做融合（cat），然后输出。
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)
        # 这里本质就是把每一个block输出结果，转换成WxHx1的mask最后过一个sigmod就可以得到每个block输出的概率图。

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)


        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)



        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)



        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)



        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)


        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6), 1))
        # 6个block concat一起之后做特征融合，然后再做输出，结果就是d0的结果，其他的输出都是为了计算loss
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

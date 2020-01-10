from model import common
import torch.nn as nn
import torch
import torch.nn.functional as F

def make_model(args, parent=False):
    return RDID_RES(args)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ASPP(nn.Module):
    def __init__(self, channels_out, channels_in, reduction):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(channels_in, channels_out, kernel_size=1)

        self.ca_1x1_1 = CALayer(channels_in,reduction)
        # self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.bn_conv_3x3_1 = nn.BatchNorm2d(256)
        self.ca_3x3_1 = CALayer(channels_in, reduction)

        self.conv_3x3_2 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.bn_conv_3x3_2 = nn.BatchNorm2d(256)
        self.ca_3x3_2 = CALayer(channels_in, reduction)

        self.conv_3x3_3 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.bn_conv_3x3_3 = nn.BatchNorm2d(256)
        self.ca_3x3_3 = CALayer(channels_in, reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        # self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(channels_in*5, channels_out, kernel_size=1)  # (1280 = 5*256)
        # self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, channels_out, kernel_size=1)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out_1x1 =self.ca_1x1_1(F.relu(self.conv_1x1_1(feature_map)))# (shape: (batch_size, 256, h/16, w/16))
        out_1x1_ca = out_1x1 + self.ca_1x1_1(out_1x1)

        out_3x3_1 = F.relu(self.conv_3x3_1(feature_map))  # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1_ca= out_3x3_1+self.ca_3x3_1(out_3x3_1)

        out_3x3_2 = F.relu(self.conv_3x3_2(feature_map))# (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2_ca = out_3x3_2 + self.ca_3x3_1(out_3x3_2)

        out_3x3_3 = F.relu(self.conv_3x3_3(feature_map)) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3_ca = out_3x3_3 + self.ca_3x3_1(out_3x3_3)

        out_img = self.avg_pool(feature_map)  # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.conv_1x1_2(out_img))  # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w),
                             mode="bilinear")  # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1_ca, out_3x3_1_ca, out_3x3_2_ca, out_3x3_3_ca, out_img],
                        1)
        out = F.relu(self.conv_1x1_3(out))  # (shape: (batch_size, 256, h/16, w/16))
        # out = self.conv_1x1_4(out)  # (shape: (batch_size, num_classes, h/16, w/16))
        # residual short cut
        # out = out+feature_map

        return out

class DIDB(nn.Module):
    def __init__(self, conv, in_channels, n_feat, kernel_size, reduction,n_resblocks,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(DIDB, self).__init__()
        self.conv = conv(in_channels, n_feat, kernel_size, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RCAB(nn.Module):
    def __init__(self, conv, in_channels, n_feat, kernel_size, reduction,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []

        for i in range(2):
            modules_body.append(conv(in_channels, n_feat, kernel_size, bias=bias))
            in_channels = n_feat

            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            modules_body.append(act)
        # modules_body.append(ASPP(n_feat, n_feat, reduction))
        # modules_body.append(nn.Conv2d(n_feat*2,n_feat,kernel_size=1))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res += x
        return torch.cat((x, res), 1)


class RIDB(nn.Module):
    def __init__(self, conv, in_channels, n_feat, kernel_size, reduction,n_resblocks,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RIDB, self).__init__()

        modules_head=[]
        modules_head.append(conv(in_channels, n_feat, kernel_size, bias=bias))
        # modules_head.append(act)
        modules_body = []
        for i in range(n_resblocks):
            modules_body.append(RCAB(
                conv, n_feat * i + n_feat, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                res_scale=1))
        # modules_body = [
        #
        #     RCAB(
        #         conv, n_feat * i + n_feat, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
        #         res_scale=1) \
        #     for i in range(n_resblocks)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        self.DLFF = nn.Sequential(*[nn.Conv2d(n_feat * n_resblocks + n_feat, n_feat, kernel_size=1), act])
        # self.aspp = nn.Sequential(*[ASPP(n_feat, n_feat, reduction)])
        # modules_body.append(nn.Conv2d(n_feat * n_resblocks + n_feat, n_feat, kernel_size=1))

    def forward(self, x):
        head = self.head(x)
        res = self.body(head)
        res =res+x
        res=self.DLFF(res)
        # res_2 =self.aspp(res)
        res = res+head
        return torch.cat((x, res), 1)




## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RIDB(
                conv, n_feat*i+n_feat, n_feat, kernel_size, reduction, i,bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for i in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat*n_resblocks+n_feat,n_feat,kernel_size=1))
        # modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*[ASPP(n_feat, n_feat, reduction)])
    def forward(self, x):
        res = self.body(x)
        res =res+x
        res_2 =self.tail(res)
        res_2 = res_2+res
        return res_2


#MS-RHDN
class RDID_RES(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RDID_RES, self).__init__()

        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        self.modules_body = nn.ModuleList()
        for i in range(self.n_resgroups):
            self.modules_body.append(
                ResidualGroup(
                    conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
            )
        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]

        self.modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.n_resgroups * n_feats, n_feats, 1, padding=0, stride=1),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        ])

        # define tail module
        # modules_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]
        # Up-sampling net
        if scale == 2 or scale == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats*scale*scale, kernel_size, padding=(kernel_size-1)//2, stride=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size-1)//2, stride=1)
            ])
        elif scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size-1)//2, stride=1)
            ])
        elif scale == 8:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size, padding=(kernel_size - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*self.modules_body)
        # self.tail = nn.Sequential(*modules_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        head = x

        RDBs_out = []

        for i in range(self.n_resgroups):
            x = self.modules_body[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        # x += f__1
        # res = self.body(x)
        x = head+x

        x = self.UPNet(x)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

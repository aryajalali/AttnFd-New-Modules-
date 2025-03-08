import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math
from spectral import *

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()


        # height and width
        spatial_dims = [33, 33, 129]

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        # self.atten_modules = [(s_channels[i], model = 'student').cuda() 
                    #   for i in range(self.start_layer, len(s_channels))]

        self.atten_modules = [MultiSpectralAttentionLayer(s, spatial_dims[idx], spatial_dims[idx]) for idx, s in
                              enumerate(s_channels[3:])]
        
        self.t_atten_modules = [MultiSpectralAttentionLayer(t, spatial_dims[idx], spatial_dims[idx]) for idx, t in
                                enumerate(t_channels[3:])]
        
        self.atten_modules = nn.ModuleList(self.atten_modules)
        self.t_atten_modules = nn.ModuleList(self.t_atten_modules)


        self.t_net = t_net
        self.s_net = s_net
        self.args = args

    def forward(self, x):

        t_feats, _, t_out = self.t_net.extract_feature(x)
        s_feats, _, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)
        
        loss_attnfd = 0
        for i in range(3, feat_num):
            b,c,h,w = t_attens[i-3].shape

            s_attens = self.Connectors[i](self.atten_modules[i - 3](s_feats, is_student = True))
            t_attens = self.Connectors[i](self.t_atten_modules[i - 3](t_feats))
            
            # s_attens[i-3] = self.Connectors[i](s_attens[i-3])
            # loss_attnfd += (s_attens[i-3] / torch.norm(s_attens[i-3], p = 2) - t_attens[i-3] / torch.norm(t_attens[i-3], p = 2)).pow(2).sum() / (b)

            loss_attnfd += (s_attens / torch.norm(s_attens, p = 2) - t_attens / torch.norm(t_attens, p = 2)).pow(2).sum() / (b)
        
        loss_attnfd = loss_attnfd * self.args.attn_lambda

        return s_out, loss_attnfd

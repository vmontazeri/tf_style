import torch
import torch.nn as nn
import numpy as np
from includes.Utils import color_print

class M1(nn.Module):
    def __init__(self, input_size, L_depth, kernel_size, padding, stride):
        super().__init__()

        self.input_size = input_size # batch x Cin x H x W
        self.L_size = L_depth
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = 0

        self.net = nn.Sequential()
        i = 0
        h_ = self.input_size[2]
        w_ = self.input_size[3]
        print('Layer 1')
        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l1_1 = nn.Conv2d(
            in_channels=self.input_size[1],out_channels=self.L_size[i],kernel_size=self.kernel_size[i],
            padding=(w_pad,h_pad),stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=self.input_size[2], s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=self.input_size[3], s=self.stride[i], k=self.kernel_size[i])
        l1_2 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=h_, s=2, k=2, w_out=int(h_/2))
        w_pad = self.get_pad_size(w_in=w_, s=2, k=2, w_out=int(w_/2))
        l1_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(h_pad,w_pad))

        print('Layer 2')
        h_ = int(h_/2)
        w_ = int(w_/2)
        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l2_1 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l2_2 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=h_, s=2, k=2, w_out=int(h_ / 2))
        w_pad = self.get_pad_size(w_in=w_, s=2, k=2, w_out=int(w_ / 2))
        l2_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(h_pad, w_pad))

        print('Layer 3')
        h_ = int(h_ / 2)
        w_ = int(w_ / 2)
        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l3_1 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l3_2 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l3_3 = nn.Conv2d(
            in_channels=self.L_size[i - 1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l3_4 = nn.Conv2d(
            in_channels=self.L_size[i - 1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=2, k=2, w_out=int(h_ / 2))
        w_pad = self.get_pad_size(w_in=w_, s=2, k=2, w_out=int(w_ / 2))
        l3_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(h_pad, w_pad))

        print('Layer 4')
        h_ = int(h_ / 2)
        w_ = int(w_ / 2)
        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l4_1 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l4_2 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l4_3 = nn.Conv2d(
            in_channels=self.L_size[i - 1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l4_4 = nn.Conv2d(
            in_channels=self.L_size[i - 1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )

        h_pad = self.get_pad_size(w_in=h_, s=2, k=2, w_out=int(h_ / 2))
        w_pad = self.get_pad_size(w_in=w_, s=2, k=2, w_out=int(w_ / 2))
        l4_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(h_pad, w_pad))

        print('Layer 5')
        h_ = int(h_ / 2)
        w_ = int(w_ / 2)
        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l5_1 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i+=1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l5_2 = nn.Conv2d(
            in_channels=self.L_size[i-1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l5_3 = nn.Conv2d(
            in_channels=self.L_size[i - 1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )
        i += 1

        h_pad = self.get_pad_size(w_in=h_, s=self.stride[i], k=self.kernel_size[i])
        w_pad = self.get_pad_size(w_in=w_, s=self.stride[i], k=self.kernel_size[i])
        l5_4 = nn.Conv2d(
            in_channels=self.L_size[i - 1], out_channels=self.L_size[i], kernel_size=self.kernel_size[i],
            padding=(w_pad, h_pad), stride=self.stride[i]
        )

        h_pad = self.get_pad_size(w_in=h_, s=2, k=2, w_out=int(h_ / 2))
        w_pad = self.get_pad_size(w_in=w_, s=2, k=2, w_out=int(w_ / 2))
        l5_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=(h_pad, w_pad))

        self.net.add_module('l1_1', l1_1)
        self.net.add_module('l1_2', l1_2)
        self.net.add_module('l1_pool', l1_pool)
        self.net.add_module('l2_1', l2_1)
        self.net.add_module('l2_2', l2_2)
        self.net.add_module('l2_pool', l2_pool)
        self.net.add_module('l3_1', l3_1)
        self.net.add_module('l3_2', l3_2)
        self.net.add_module('l3_3', l3_3)
        self.net.add_module('l3_4', l3_4)
        self.net.add_module('l3_pool', l3_pool)
        self.net.add_module('l4_1', l4_1)
        self.net.add_module('l4_2', l4_2)
        self.net.add_module('l4_3', l4_3)
        self.net.add_module('l4_4', l4_4)
        self.net.add_module('l4_pool', l4_pool)
        self.net.add_module('l5_1', l5_1)
        self.net.add_module('l5_2', l5_2)
        self.net.add_module('l5_3', l5_3)
        self.net.add_module('l5_4', l5_4)
        self.net.add_module('l5_pool', l5_pool)

        color_print('Initiated the M1 model', type_='success')

    def get_pad_size(self, w_in, s, k, w_out=None):
        if w_out == None:
            w_out = w_in
        return int((w_out*s-1-w_in+k)/2)

    def forward(self, x):
        l1_out = self.net(x)
        return l1_out

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size




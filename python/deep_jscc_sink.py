#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-deep_jscc_test author.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import numbers
from collections import Counter
from itertools import combinations
import math
import cv2
import onnx
import onnx_tensorrt.backend as backend
from gnuradio import gr
import pmt

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()
        self.padding = kernel_size // 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32)
                for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


def ss_warp(x, flo):
    """
    warp an scaled space volume (x) back to im1, according to scale space flow
    x: [B, C, D, H, W]
    flo: [B, 3, 1, H, W] ss flow
    """
    B, C, D, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    zz = torch.zeros_like(xx)
    grid = torch.cat((xx, yy, zz), 1).float()
    grid = grid.unsqueeze(2)

    if x.is_cuda:
        grid = grid.to(x.device)
    # vgrid = Variable(grid) + flo
    grid.requires_grad = True
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0
    vgrid[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :].clone() - 1.0

    vgrid = vgrid.permute(0, 2, 3, 4, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output.squeeze(2)


def generate_ss_volume(x, sigma, kernel_size, M):
    B, C, H, W = x.size()
    out = [x]
    for i in range(M):
        kernel = GaussianSmoothing(C, kernel_size, (2**i) * sigma).to(x.device)
        out.append(kernel(x))
    out = torch.stack(out, dim=2)
    return out

class deep_jscc_sink(gr.sync_block):
    """
    docstring for block deep_jscc_sink
    """
    def __init__(self, model_dir, packet_len):
        gr.sync_block.__init__(self,
            name="deep_jscc_sink",
            in_sig=[np.complex64, np.uint8],
            out_sig=None)
        
        self.packet_len = packet_len
        models = self.get_model(model_dir)                           # load models
        self.key_decoder, self.interp_decoder, self.ssf_net = self.get_engine(models) 

        self.gop_size = 5
        self.ss_sigma = 0.01
        self.ss_levels = 5
        #self.bw_size = 20
        self.n_gops = (self.n_frames - 1) // (self.gop_size - 1)      # number of gops, = number of frame-1 / 4
        self.first = True

        self.gop_idx = 0                                        # gop index
        self.pair_idx = 0                                       # pair index
        self.running_idx = 0                                    # 
        self.curr_codeword = 0                                  # 
        self.curr_codes = None                                  # 
        self.bw_per_gop = 240 * 15 * 20 // 2                    # divided by 2 because real part + imaginary part of 
        self.total_bw = self.bw_per_gop * (self.n_gops + 1)
        self.payload = []

    def get_engine(self, nets):
        key_decoder, interp_decoder, ssf_net = nets
        key_decoder_engine = backend.prepare(key_decoder)
        interp_decoder_engine = backend.prepare(interp_decoder)
        ssf_engine = backend.prepare(ssf_net)
        engines = (key_decoder_engine, interp_decoder_engine, ssf_engine)
        return engines

    def get_model(self, model_dir):                                  # load model
        key_decoder = onnx.load(model_dir + '/key_decoder_simp.onnx')
        interp_decoder = onnx.load(model_dir + '/interp_decoder_simp.onnx')
        ssf_net = onnx.load(model_dir + '/ssf_net_simp.onnx')
        nets = (key_decoder, interp_decoder, ssf_net)
        return nets

    def receive(self, codes, frame_lengths):
        code_gop = [None] * (len(frame_lengths)+1)
        code_gop[0] = first_code # Todo

        for idx in len(frame_lengths):
            if idx == 0:
                code_gop[idx + 1] = codes[0 : frame_lengths[idx]].reshape(1, frame_lengths[idx]/300 , 15, 20)
            else:
                code_gop[idx + 1] = codes[frame_lengths[idx - 1] :frame_lengths[idx]].reshape(1, frame_lengths[idx]/300 , 15, 20)

        last_code = self.key_decoder.run((code_gop[-1]))[0]

        for pred_idx in [2 ,1 ,3]:
            if pred_idx == 2:
                dist = 2
            else:
                dist = 1  
            #flow1, r, flow2, mask = self.interp_decoder.run((code_gop[code_idx]))[0]
            #w1 = ss_warp(code_gop[0], torch.from_numpy(flow1).unsqueeze(2)).detach().numpy()
            #w2 = ss_warp(code_gop[-1], torch.from_numpy(flow2).unsqueeze(2)).detach().numpy()
        
            # frame = mask * w1 + mask * w2 + mask * r
        return True
        
    def work(self, input_items, output_items):
        reconstruct = False
        payload_in = input_items[0]
        byte_in = input_items[1]
	payload_in1 = []
	payload_in2 = []
        # payload convert from img to real number
        for number in len(payload_in):
            payload_in1[number] = payload_in[number].real
            payload_in2[number] = payload_in[number].imag
	payload_in = [payload_in1, payload_in2]

        # self.get_tags_in_window(which_input, rel_start, rel_end)
        # Catch the tags and convert to python
        tags = self.get_tags_in_window(0, 0, len(byte_in))
        tags = pmt.to_python(tags)

        self.packet_len = tags['packet_len']
        new_gop = tags['new_gop']
        frame_lengths = tags['frame_lengths']

        if new_gop == True:
            reconstruct = self.receive(self.payload, frame_lengths)
            self.payload = []
        self.payload.append(payload_in)

        print('Payload size is : {}\n' .format(payload_in.shape))
        print('Packet length is : {}, whether is a new gop: {}, frame length is : {}\n' .format(self.packet_len, new_gop, frame_lengths))
        print('Reconstruction result is {}' .format(reconstruct))
        return len(input_items[0])

if __name__ == '__main__':
    x = np.zeros(1)
    source = deep_jscc_sink('/home/xaviernx/onnx_output', 96)
    source.work(None, [[None]*100, [None]*100])
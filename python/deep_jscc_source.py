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

# Generates permutations without repetition, generate all the ways to allocate bandwidth in get_bw_set
def perms_without_reps(s):
    partitions = list(Counter(s).items())

    def _helper(idxset, i):
        if len(idxset) == 0:
            yield ()
            return
        for pos in combinations(idxset, partitions[i][1]):
            for res in _helper(idxset - set(pos), i+1):
                yield (pos,) + res

    n = len(s)
    for poses in _helper(set(range(n)), 0):
        out = [None] * n
        for i, pos in enumerate(poses):
            for idx in pos:
                out[idx] = partitions[i][0]
        yield out

# Bw split
def split_list_by_val(x, s):
    size = len(x)
    idx_list = [idx + 1 for idx, val in enumerate(x) if val == s]
    res = [x[i:j] for i, j in zip([0] + idx_list, idx_list + [size])]
    return res


class deep_jscc_source(gr.sync_block):
    """
    docstring for block deep_jscc_source
    """
    def __init__(self, video_file, model_dir, snr, packet_len):
        gr.sync_block.__init__(self,
            name="deep_jscc_source",
            in_sig=None,
            out_sig=[np.complex64, np.uint8])

        snr = np.float32(snr)
        self.snr = np.array([[snr]])             		     # channel snr, should be estimated
        self.packet_len = packet_len                                 # packet length = 48

        self.video_file = cv2.VideoCapture(video_file)               # load video file
        models = self.get_model(model_dir)                           # load models
        self.key_encoder, self.interp_encoder, \
            self.bw_allocator, self.ssf_net = self.get_engine(models) 

        self.gop_size = 5
        self.ss_sigma = 0.01
        self.ss_levels = 5
        self.bw_size = 20
        self.n_frames = self.video_file.get(cv2.CAP_PROP_FRAME_COUNT) # number of frames of video
        self.n_gops = (self.n_frames - 1) // (self.gop_size - 1)      # number of gops, = number of frame-1 / 4
        self.first = True
        self.get_bw_set()

        self.gop_idx = 0                                        # gop index
        self.pair_idx = 0                                       # pair index
        self.running_idx = 0                                    # 
        self.curr_codeword = 0                                  # 
        self.curr_codes = None                                  # 
        self.bw_per_gop = 240 * 15 * 20 // 2                    # divided by 2 because real part + imaginary part of 
        self.total_bw = self.bw_per_gop * (self.n_gops + 1)     # total bandwidth = individual bandwidth * number of gops

    def get_bw_set(self):
        bw_set = [1] * self.bw_size + [0] * (self.gop_size - 2)             # 
        bw_set = perms_without_reps(bw_set)                                 # 
        bw_set = [split_list_by_val(action, 0) for action in bw_set]
        self.bw_set = [[sum(bw) for bw in action] for action in bw_set]

    def get_engine(self, nets):
        key_encoder, interp_encoder, bw_allocator, ssf_net = nets
        key_encoder_engine = backend.prepare(key_encoder)
        interp_encoder_engine = backend.prepare(interp_encoder)
        bw_allocator_engine = backend.prepare(bw_allocator)
        ssf_engine = backend.prepare(ssf_net)
        engines = (key_encoder_engine, interp_encoder_engine,
                    bw_allocator_engine, ssf_engine)
        return engines

    def get_model(self, model_dir):                                  # load model
        key_encoder = onnx.load(model_dir + '/key_encoder_simp.onnx')
        interp_encoder = onnx.load(model_dir + '/interp_encoder_simp.onnx')
        bw_allocator = onnx.load(model_dir + '/bw_allocator_simp.onnx')
        ssf_net = onnx.load(model_dir + '/ssf_net_simp.onnx')
        nets = (key_encoder, interp_encoder, bw_allocator, ssf_net)
        return nets

    def forward(self, gop):
        codes = [None] * self.gop_size
        code_lengths = [None] * self.gop_size
        # gop = np.array(gop, dtype = gop.dtype, order = 'C')		# Convert the frame to row-major order
        if self.first:
            self.first = False
            init_frame = gop[0]
            # print(init_frame.dtype())		#dtype = float32
            init_code = self.key_encoder.run((init_frame, self.snr))[0]
            codes[0] = init_code.reshape(-1, 2)                         # Because real+img?
            code_lengths[0] = init_code.size // 2
        else:
            codes[0] = self.prev_last.reshape(-1, 2)                    # what is prev_last
            code_lengths[0] = self.prev_last.size // 2

        interp_inputs = [None] * self.gop_size
        interp_inputs[0] = gop[0]
        interp_inputs[-1] = gop[-1]

        for pred_idx in [2, 1, 3]:
            if pred_idx == 2:                   # if the middle(no. 3)
                dist = 2                        # the distance between midlle and key is 2
            else:                               # if the no.1 / 2
                dist = 1                        # the distance is 1

            vol1 = generate_ss_volume(torch.from_numpy(gop[pred_idx - dist]),
                                        self.ss_sigma, 3, self.ss_levels)
            vol2 = generate_ss_volume(torch.from_numpy(gop[pred_idx + dist]),
                                        self.ss_sigma, 3, self.ss_levels)
            flow1 = self.ssf_net.run(np.concatenate((gop[pred_idx], gop[pred_idx - dist]), axis=1))[0]		# flow1 = [predict, 1st]
            flow2 = self.ssf_net.run(np.concatenate((gop[pred_idx], gop[pred_idx + dist]), axis=1))[0]		# flow2 = [predict, 5th]

            w1 = ss_warp(vol1, torch.from_numpy(flow1).unsqueeze(2)).detach().numpy()
            w2 = ss_warp(vol2, torch.from_numpy(flow2).unsqueeze(2)).detach().numpy()
            r1 = gop[pred_idx] - w1
            r2 = gop[pred_idx] - w2
            interp_input = np.concatenate([gop[pred_idx], w1, w2, r1, r2, flow1, flow2], axis=1)
            interp_inputs[pred_idx] = interp_input # interp_inputs = [gop0, information1, information2, information3,]

        bw_state = np.concatenate(interp_inputs, axis=1) # shape = (1, 3+21+21+21+3, 240, 320)
        bw_policy = self.bw_allocator.run((bw_state, self.snr))[0]
        # bw_alloc = np.argmax(bw_policy) # , axis=0)			# allocate the bw with highest ..?
        # bw_alloc = self.bw_set[bw_alloc[0, 0]] * self.bw_size
        bw_alloc = self.bw_set[bw_policy[0]] * self.bw_size

        last = interp_inputs[-1]				# =gop[-1]
        last_code = self.key_encoder.run((last, self.snr))[0]
        last_code = last_code[:, :bw_alloc[0]]
        codes[-1] = last_code.reshape(-1, 2)
        code_lengths[-1] = last_code.size // 2
        self.prev_last = last_code

        for pred_idx in [2, 1, 3]:
            if pred_idx == 2:
                dist = 2
            else:
                dist = 1

            interp_input = interp_inputs[pred_idx]
            interp_code = self.interp_encoder.run((interp_input, self.snr))[0] # (1, 240, 15, 20)
            interp_code = interp_code[:, :bw_alloc[pred_idx]]			# (1, 240-bw_alloc , 15 ,20)
            codes[pred_idx] = interp_code.reshape(-1, 2)
            code_lengths[pred_idx] = interp_code.size // 2

        return codes, code_lengths

    def get_gop(self, gop_idx):
        start_frame = int(gop_idx * 4)
	# CAP_PROP_POS_FRAMES: 0-based index of the frame to be decoded/captured next.
        self.video_file.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # frames = np.array([])
        frames = []
        for _ in range(5):
            flag, frame = self.video_file.read()		# frame.shape = (hight, width, 3) = (240, 320, 3) 
            assert flag
            frame = np.array(frame, dtype=np.float32, order='C') # conver the frame to row-major order, https://github.com/onnx/onnx-tensorrt/issues/237
            frame = np.swapaxes(frame, 0, 2)			# [3, width, height] = (3, 320, 240)
            frame = np.swapaxes(frame, 1, 2)			# [3, height, width] = (3, 240, 320)
            frame = np.expand_dims(frame, axis=0) / 255.0	# normalisation to [0,1]
            frame = np.array(frame, dtype=np.float32, order='C') # conver the frame to row-major order
            frame.flags							# verify the contiguous
            # frame32 = np.float32(frame)				# float 64 to numpy float 32
            # frames = np.append(frames, frame32)
            frames.append(frame)
            # frames = np.array(frames, dtype=np.float32, order='C') # conver the frame to row-major order
        return frames

    def work(self, input_items, output_items):
        payload_out = output_items[0]
        byte_out = output_items[1]
        frame_lengths = pmt.make_s32vector(4, 0)
        curr_code_lengths = []

        for payload_idx in range(len(payload_out)):
            if self.curr_codes is None:
                curr_gop = self.get_gop(self.gop_idx)
                self.curr_codes, self.curr_code_lengths = self.forward(curr_gop) # gather a new gop

            if self.pair_idx == (self.curr_code_lengths[self.curr_codeword] - 1):
                self.curr_codeword += 1
                new_codeword = True
                self.pair_idx = 0
            else:
                new_codeword = False

            if self.curr_codeword == (len(self.curr_codes) - 1):
                self.gop_idx += 1
                new_gop = True

                curr_gop = self.get_gop(self.gop_idx)
                self.curr_codes, self.curr_code_lengths = self.forward(curr_gop)  # gather a new gop
                self.curr_codeword = 0
            else:
                new_gop = False

	    # proportion of frames k1:k2:k3:k4 = lengths[1]:lengths[2]:length[3]:length[4]
            for frame_index in range(self.gop_size - 1):	                                        # -1 because the first frame is not transmitted
                pmt.s32vector_set(frame_lengths, frame_index, self.curr_code_lengths[frame_index + 1])	# +1 to map from [0:3] to [1:4]

	
            if self.running_idx % self.packet_len == 0:
		# add_item_tag(which_output, abs_offset, key, value)
                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('packet_len'), pmt.from_long(self.packet_len))
                self.add_item_tag(1, payload_idx + self.nitems_written(1), pmt.intern('packet_len'), pmt.from_long(self.packet_len))

                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('new_gop'), pmt.from_bool(new_gop)) 
                self.add_item_tag(1, payload_idx + self.nitems_written(1), pmt.intern('new_gop'), pmt.from_bool(new_gop))
            
                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('frame_lengths'), frame_lengths)
                self.add_item_tag(1, payload_idx + self.nitems_written(0), pmt.intern('frame_lengths'), frame_lengths)

                # self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('new_codeword'), pmt.from_bool(new_codeword)) 
                # self.add_item_tag(1, payload_idx + self.nitems_written(1), pmt.intern('new_codeword'), pmt.from_bool(new_codeword))

	        # codes.size = [;,2]?
            symbol = self.curr_codes[self.curr_codeword][self.pair_idx, 0] + self.curr_codes[self.curr_codeword][self.pair_idx, 1]*1j
            payload_out[payload_idx] = symbol
            byte_out[payload_idx] = np.uint8(7)

            self.pair_idx += 1
            self.running_idx += 1

        
        return len(output_items[0])

if __name__ == '__main__':
    x = np.zeros(1)
    source = deep_jscc_source('/home/xaviernx/Downloads/UCF-101/HulaHoop/v_HulaHoop_g13_c03.avi',
                            '/home/xaviernx/onnx_output', 0.35 , 96)
    source.work(None, [[None]*100, [None]*100])
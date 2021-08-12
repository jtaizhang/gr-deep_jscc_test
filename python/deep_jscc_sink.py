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
import ffmpeg
from collections import Counter
from itertools import combinations
import math
import cv2
import onnx
import onnx_tensorrt.backend as backend
from gnuradio import gr
import pmt
import subprocess
#import os
import gi 
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

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

class deep_jscc_sink(gr.sync_block):
    """
    docstring for block deep_jscc_sink
    """
    def __init__(self, video_file, model_dir, snr, packet_len):
        gr.sync_block.__init__(self,
            name="deep_jscc_sink",
            in_sig=None,
            out_sig=None)

        snr = np.float32(snr)
        self.snr = np.array([[snr]])             		     # channel snr, should be estimated
        self.packet_len = packet_len
        self.video_file = cv2.VideoCapture(video_file)               # load video file

        in_port_name = 'pdu_in'
        self.message_port_register_in(pmt.intern(in_port_name))
        self.set_msg_handler(pmt.intern(in_port_name), self.msg_handler)
        
        self.packet_len = packet_len
        models = self.get_model(model_dir)                           # load models
        self.key_decoder, self.interp_decoder = self.get_engine(models) 

        self.gop_size = 5
        self.ss_sigma = 0.01
        self.ss_levels = 5
        self.bw_size = 20
        self.bw_block_size = 12
        self.n_frames = self.video_file.get(cv2.CAP_PROP_FRAME_COUNT) # number of frames of video
        self.n_gops = (self.n_frames - 1) // (self.gop_size - 1)      # number of gops, = number of frame-1 / 4
        self.first_received = False
        self.get_bw_set()

        self.gop_idx = 0                                        # gop index
        self.pair_idx = 0                                       # pair index
        self.running_idx = 0                                    # 
        self.curr_codeword = 0                                  # 
        self.curr_codes = None                                  # 
        self.bw_per_gop = 240 * 15 * 20 // 2  
        self.max_tensor_channels = 240                  # divided by 2 because real part + imaginary part of 
        self.total_bw = self.bw_per_gop * (self.n_gops + 1)     # total bandwidth = individual bandwidth * number of gops
        self.gop_IQ = []
        self.gop_bw_policy = []  

        # ffmpeq error: export DBUS_FATAL_WARNINGS=0 https://bugs.launchpad.net/ubuntu/+source/libsdl2/+bug/1775067
        # unset XMODIFIERS
        self.output_pipe = (ffmpeg
                            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(320, 240))
                            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                            .overwrite_output()
                            .run_async(pipe_stdin=True, pipe_stdout=True)
                            )

        self.output_play = subprocess.Popen(
            [
                'ffplay',
                '-f', 'rawvideo',
                '-pixel_format', 'rgb24',
                '-video_size', '{}x{}'.format(320, 240),
                '-i', 'pipe:',
            ],
            stdin = self.output_pipe.stdout,
        )
	
        # Not working because it only accept files
        #self.output_play = subprocess.Popen(
        #    [
        #        'gst-launch-1.0', 
        #        'filesrc location=video.raw', '!',
        #        'queue', '!',
        #        'rawvideoparse use-sink-caps=false', '\ ',
        #        'width=320 height=240 format=RGB ! autovideosink',
        #    ],
        #    #stdin = self.output_pipe.stdout
        #)

    def get_bw_set(self):
        bw_set = [1] * self.bw_size + [0] * (self.gop_size - 2)             # 
        bw_set = perms_without_reps(bw_set)                                 # 
        bw_set = [split_list_by_val(action, 0) for action in bw_set]
        self.bw_set = [[sum(bw) for bw in action] for action in bw_set]

    def get_engine(self, nets):
        key_decoder, interp_decoder = nets
        key_decoder_engine = backend.prepare(key_decoder)
        interp_decoder_engine = backend.prepare(interp_decoder)
        engines = (key_decoder_engine, interp_decoder_engine)
        return engines

    def get_model(self, model_dir):                                  # load model
        key_decoder = onnx.load(model_dir + '/key_decoder_simp.onnx')
        interp_decoder = onnx.load(model_dir + '/interp_decoder_simp.onnx')
        nets = (key_decoder, interp_decoder)
        return nets

    def forward(self, codes, first):
        gop_frames = [None] * self.gop_size
        if first:
            init_frame = self.key_decoder.run((codes[0], self.snr))[0]
            gop_frames[0] = init_frame
        else:
            gop_frames[0] = self.last_key

        last_frame = self.key_decoder.run((codes[-1], self.snr))[0]
        gop_frames[-1] = last_frame
        self.last_key = last_frame

        for pred_idx in [2, 1, 3]:
            if pred_idx == 2:
                dist = 2
            else:
                dist = 1

            interp_decoder_out = self.interp_decoder.run((codes[pred_idx], self.snr))[0]
            interp_decoder_out = torch.from_numpy(interp_decoder_out)

            f1, f2, a, r = torch.split(interp_decoder_out,
                                        split_size_or_sections=[3, 3, 3, 3],
                                        dim=1)

            a = F.softmax(a, dim=1)
            a1, a2, a3 = torch.chunk(a, chunks=3, dim=1)
            r = torch.sigmoid(r)

            a1 = a1.repeat_interleave(3, dim=1)
            a2 = a2.repeat_interleave(3, dim=1)
            a3 = a3.repeat_interleave(3, dim=1)

            pred_vol1 = generate_ss_volume(torch.from_numpy(gop_frames[pred_idx - dist]),
                                            self.ss_sigma, 3, self.ss_levels)
            pred_vol2 = generate_ss_volume(torch.from_numpy(gop_frames[pred_idx + dist]),
                                            self.ss_sigma, 3, self.ss_levels)
            pred_1 = ss_warp(pred_vol1, f1.unsqueeze(2))
            pred_2 = ss_warp(pred_vol2, f2.unsqueeze(2))

            pred = a1 * pred_1 + a2 * pred_2 + a3 * r
            gop_frames[pred_idx] = pred.numpy()

        return gop_frames

    def append_zeros(self, codes, bw_policy, is_first):
        padded_codes = []

        bw_allocation = self.bw_set[bw_policy]
        bw_allocation = [bw * self.bw_block_size for bw in bw_allocation]

        if is_first:
            bw_allocation = bw_allocation.insert(0, 240)

        symbol_len = [bw * 15 * 20 for bw in bw_allocation]
        assert sum(symbol_len) == codes.shape[0]
        symbol_len = np.roll(symbol_len, -1)
        symbol_len = np.cumsum(symbol_len)

        code_segments = np.split(codes, symbol_len, axis=0)

        for i, codeword in enumerate(code_segments):
            codeword = codeword.reshape(1, -1, 15, 20)
            zero_pad = self.max_tensor_channels - codeword.shape[1]
            zero_pad = np.zeros((1, zero_pad, 15, 20))
            padded_codeword = np.concatenate((codeword, zero_pad), axis=1)
            padded_codes.append(padded_codeword)

        return padded_codes

    def get_gop(self, gop_idx):
        start_frame = int(gop_idx * 4)
	    # CAP_PROP_POS_FRAMES: 0-based index of the frame to be decoded/captured next.
        self.video_file.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # frames = np.array([])
        frames = []
        for _ in range(30):
            flag, frame = self.video_file.read()		# frame.shape = (hight, width, 3) = (240, 320, 3) 
            assert flag
            frame = np.array(frame, dtype=np.float32, order='C') # conver the frame to row-major order, https://github.com/onnx/onnx-tensorrt/issues/237
            frame = np.swapaxes(frame, 0, 2)			# [3, width, height] = (3, 320, 240)
            frame = np.swapaxes(frame, 1, 2)			# [3, height, width] = (3, 240, 320)
            frame = np.expand_dims(frame, axis=0) / 255.0	# normalisation to [0,1]
            frame = np.array(frame, dtype=np.float32, order='C') # conver the frame to row-major order
            #frame.flags							# verify the contiguous
            # frame32 = np.float32(frame)				# float 64 to numpy float 32
            # frames = np.append(frames, frame32)
            frames.append(frame)
            # frames = np.array(frames, dtype=np.float32, order='C') # conver the frame to row-major order
        return frames

    def write_to_pipe(self, frames):
        for frame in frames:
            # swapping colour axis necessary
            r = np.copy(frame[:, 0, :, :])
            b = np.copy(frame[:, 2, :, :])
            frame[:, 0, :, :] = b
            frame[:, 2, :, :] = r

            frame = np.squeeze(frame, axis=0)
            frame = np.swapaxes(frame, 0, 2)			# [3, width, height] = (3, 320, 240)
            frame = np.swapaxes(frame, 0, 1)			# [3, height, width] = (3, 240, 320)
            frame = np.round(frame * 255.0)
            # os.environ["DBUS_FATAL_WARNINGS"] = 0
            self.output_pipe.stdin.write(frame.astype(np.uint8).tobytes())
	    # The following stores the video and read it in gstreamer
	    #fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
	    #out = cv2.VideoWriter('output.avi', fourcc, fps, (320,2 40))
	    #out.write(frame)
            #Gst.init(None)
            #pipeline = Gst.parse_launch("filesrc location='output.avi' ! rawvideoparse use-sink-caps=false \
            #                            width=320 height=240 format=RGB ! autovideosink")


    def msg_handler(self, msg_pmt):
        tags = pmt.to_python(pmt.car(msg_pmt)) # should be a dictionary of tags
        payload_in = pmt.to_python(pmt.cdr(msg_pmt)) # should be the packet data

        # Verify if it is the first frame
        if tags['first'] and not self.first_received:
            self.first_received = True
            is_first = True
        else:
            is_first = False

        # When the first frame is received
        if self.first_received:
            new_gop = tags['new_gop']
            if new_gop:
                self.gop_IQ = []
                self.gop_bw_policy = []       

            for pair in payload_in:
                self.gop_IQ.append([pair.real, pair.imag])
                self.running_idx += 1

            self.gop_bw_policy.append(tags['bw_policy'])

            if (len(self.gop_IQ) == self.bw_per_gop and not is_first) or (len(self.gop_IQ) == (self.bw_per_gop+240) and is_first):
                bw_policy = max(set(self.gop_bw_policy), key=self.gop_bw_policy.count)
                received_IQ = np.array(self.gop_IQ)
                padded_codes = self.append_zeros(received_IQ, bw_policy, is_first)
                pred_frames = self.forward(padded_codes, is_first)
                self.write_to_pipe(pred_frames)

            if self.running_idx == self.total_bw:
                self.first_received = False
                self.running_idx = 0
        
    def work(self, input_items, output_items):
        #reconstruct = False
        #payload_in = input_items[0]
        #byte_in = input_items[1]
        #payload_in1 = []
        #payload_in2 = []
        # payload convert from img to real number
        #for number in len(payload_in):
        #    payload_in1[number] = payload_in[number].real
        #    payload_in2[number] = payload_in[number].imag
        #payload_in = [payload_in1, payload_in2]

        # self.get_tags_in_window(which_input, rel_start, rel_end)
        # Catch the tags and convert to python
        #tags = self.get_tags_in_window(0, 0, len(byte_in))
        #tags = pmt.to_python(tags)

        #self.packet_len = tags['packet_len']
        #new_gop = tags['new_gop']
        #frame_lengths = tags['frame_lengths']

        #if new_gop == True:
        #    reconstruct = self.receive(self.payload, frame_lengths)
        #    self.payload = []
        #self.payload.append(payload_in)

        #print('Payload size is : {}\n' .format(payload_in.shape))
        #print('Packet length is : {}, whether is a new gop: {}, frame length is : {}\n' .format(self.packet_len, new_gop, frame_lengths))
        #print('Reconstruction result is {}' .format(reconstruct))
        #return len(input_items[0])
        return 0

if __name__ == '__main__':
    x = np.zeros(1)
    source = deep_jscc_sink('/home/xaviernx/Downloads/UCF-101/HulaHoop/v_HulaHoop_g13_c03.avi',
                            '/home/xaviernx/onnx_output', 0.35 , 96)
    #source.work(None, [[None]*100, [None]*100])
    frames = source.get_gop(3)
    source.write_to_pipe(frames)
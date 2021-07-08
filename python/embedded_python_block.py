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


import numpy as np
from gnuradio import gr
import pmt
 
class embedded_python_block(gr.sync_block):
    def __init__(self, probability):
        gr.sync_block.__init__(
            self,
            name='Embedded Python Block',
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.probability = probability
 
    def work(self, input_items, output_items):
        for indx, sample in enumerate(input_items[0]): # Enumerate through the input samples in port in0
            if np.random.rand() > self.probability: # 5% chance this sample is chosen
                key = pmt.intern("example_key")
                value = pmt.intern("example_value")
                self.add_item_tag(0, # Write to output port 0
                        self.nitems_written(0) + indx, # Index of the tag in absolute terms
                        key, # Key of the tag
                        value # Value of the tag
                )
                # note: (self.nitems_written(0) + indx) is our current sample, in absolute time
        output_items[0][:] = input_items[0] # copy input to output
        return len(output_items[0])


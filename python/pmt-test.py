import numpy as np
import numbers
from collections import Counter
from itertools import combinations
import math
from gnuradio import gr
import pmt

gop_size = 5
frame_lengths = pmt.make_s32vector(4, 0)
payload_idx = 0
curr_code_lengths = [1, 10000, 1000, 100, 10]
for frame_index in range(gop_size - 1):	# -1 because the first frame is not transmitted					
    pmt.s32vector_set(frame_lengths, frame_index, curr_code_lengths[frame_index + 1])	# +1 to map from [0:3] to [1:4]
print(pmt.s32vector_elements(frame_lengths))
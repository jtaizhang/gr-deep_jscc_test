#!/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir="/home/xaviernx/gr-deep_jscc_test/lib"
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/xaviernx/gr-deep_jscc_test/build/lib":$PATH
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH
deep_jscc_test_qa_packet_header_jscc.cc 

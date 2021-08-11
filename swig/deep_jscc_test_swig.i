/* -*- c++ -*- */

#define DEEP_JSCC_TEST_API
#define DIGITAL_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "deep_jscc_test_swig_doc.i"

%{
#include "deep_jscc_test/ofdm_jscc_equalizer_single_tap.h"
#include "deep_jscc_test/packet_header_jscc.h"
%}

%include "gnuradio/digital/ofdm_equalizer_base.h"
%template(ofdm_equalizer_base_sptr) boost::shared_ptr<gr::digital::ofdm_equalizer_base>;
%template(ofdm_equalizer_1d_pilots_sptr) boost::shared_ptr<gr::digital::ofdm_equalizer_1d_pilots>;
%pythoncode %{
    ofdm_equalizer_1d_pilots_sptr.__repr__ = lambda self: "<OFDM equalizer 1D base class>"
    %}

using namespace gr::digital;
%include "deep_jscc_test/ofdm_jscc_equalizer_single_tap.h"
%template(ofdm_jscc_equalizer_single_tap_sptr) boost::shared_ptr<gr::deep_jscc_test::ofdm_jscc_equalizer_single_tap>;
%pythoncode %{
    ofdm_jscc_equalizer_single_tap_sptr.__repr__ = lambda self: "OFDM_JSCC_equalizer_single_tap"
    ofdm_jscc_equalizer_single_tap = ofdm_jscc_equalizer_single_tap .make;
    %}

%include "gnuradio/digital/packet_header_default.h"
%template(packet_header_default_sptr) boost::shared_ptr<gr::digital::packet_header_default>;
%include "deep_jscc_test/packet_header_jscc.h"
%template(packet_header_jscc_sptr) boost::shared_ptr<gr::deep_jscc_test::packet_header_jscc>;
%pythoncode %{
packet_header_jscc_sptr.__repr__ = lambda self: "<packet_header_jscc>"
packet_header_jscc = packet_header_jscc .make;
%}



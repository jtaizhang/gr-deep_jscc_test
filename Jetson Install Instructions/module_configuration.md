### To install the module in GRC, some external packages are required to be included in cmakelist amd yml files.



### 1. The *CMakeLists.txt* in base directory

`find_package(Gnuradio "3.8" REQUIRED digital filter fft blocks analog)`



### 2. The *CMakeLists.txt* in lib directory

`target_link_libraries(gnuradio-your_module_name gnuradio::gnuradio-runtime gnuradio::gnuradio-blocks gnuradio::gnuradio-fft gnuradio::gnuradio-digital gnuradio::gnuradio-filter gnuradio::gnuradio-analog Volk::volk)`



### 3. The *yml* file in grc directory

change the make method

change all parameters and inputs/outputs as desired



### 4. *swig* file

```idl
/* -*- c++ -*- */

#define YOUR_MODULE_NAME_API
#define DIGITAL_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "your_module_name_swig_doc.i"

%{
#include "your_module_name/ofdm_jscc_equalizer_single_tap.h"
#include "your_module_name/packet_header_jscc.h"
%}

%include "gnuradio/digital/ofdm_equalizer_base.h"
%template(ofdm_equalizer_base_sptr) boost::shared_ptr<gr::digital::ofdm_equalizer_base>;
%template(ofdm_equalizer_1d_pilots_sptr) boost::shared_ptr<gr::digital::ofdm_equalizer_1d_pilots>;
%pythoncode %{
    ofdm_equalizer_1d_pilots_sptr.__repr__ = lambda self: "<OFDM equalizer 1D base class>"
    %}

using namespace gr::digital;
%include "your_module_name/ofdm_jscc_equalizer_single_tap.h"
%template(ofdm_jscc_equalizer_single_tap_sptr) boost::shared_ptr<gr::your_module_name::ofdm_jscc_equalizer_single_tap>;
%pythoncode %{
    ofdm_jscc_equalizer_single_tap_sptr.__repr__ = lambda self: "OFDM_JSCC_equalizer_single_tap"
    ofdm_jscc_equalizer_single_tap = ofdm_jscc_equalizer_single_tap .make;
    %}

%include "gnuradio/digital/packet_header_default.h"
%template(packet_header_default_sptr) boost::shared_ptr<gr::digital::packet_header_default>;
%include "your_module_name/packet_header_jscc.h"
%template(packet_header_jscc_sptr) boost::shared_ptr<gr::your_module_name::packet_header_jscc>;
%pythoncode %{
packet_header_jscc_sptr.__repr__ = lambda self: "<packet_header_jscc>"
packet_header_jscc = packet_header_jscc .make;
%}
```



### 5. in the file "/etc/ld.so.conf"

include two lines



 `/usr/local/lib`

 and

 `/usr/lib`

and then run 

`sudo ldconfig`

 ### 6. cmake command

```cmake -DCMAKE_INSTALL_PREFIX=/usr/local ../```



### 7. At receiver

`pip install ffmpeg-python`

for mmpeg:

``` bash
sudo nano /etc/.bash
export DBUS_FATAL_WARNINGS=0
unset XMODIFIERS
```

Intall jtop:

```bash
sudo -H pip3 install jetson-stats
sudo jtop
```


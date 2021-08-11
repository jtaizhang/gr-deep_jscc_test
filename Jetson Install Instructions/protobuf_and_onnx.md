L4T version: R32(release), Revesion: 5.1

### 1. cmake update

[Download the latest release of make][cmake] 

```bash
# current version==3.10.2
# download the source distribution of latest release
tar -zxvf cmake-3.20.5.tar.gz 
cd cmake-3.20.5
sudo apt-get install libssl-dev # install openssl dependency first
sudo cmake .
sudo make
sudo make install
# reopen the terminal
cmake --version # check the version
```



### 2. Install protobuf

[Github repository][protobuf]

```bash
# Follow the README instruction
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure
sudo make
sudo make check
sudo make install
sudo ldconfig # refresh shared library cache.
```



### 3. Install onnx_tensorrt 7.1

[Github repository][onnx]

```bash
# The following will install an old version -----------------------------------------
git clone https://github.com/onnx/onnx-tensorrt.git
#------------------------------------------------------------------------------------
# download at https://github.com/protocolbuffers/protobuf/releases
cd onnx-tensorrt
git checkout 7.1
git submodule update --init --recursive
mkdir build && cd build
cmake -DTENSORRT_ROOT=/usr/lib/aarch64-linux-gnu ..
make -j
sudo make install
sudo ldconfig
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
# About onnx installation --------------------------------------------------------------------------------------
# this needs a try
sudo apt-get install protobuf-compiler libprotoc-dev
pip install --upgrade pip
python3 -m pip install onnx==1.6.0

# by doing the following, the latest version of onnx is going to be installed
sudo apt-get install python-pip protobuf-compiler libprotoc-dev
# The following may cause WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.
pip3 install Cython --user
pip3 install onnx --user --verbose

sudo pip3 install -U pybind11
pip3 install pytest

# about the setup.py
#---------------------------------------------------------------------------------------------------------------
# https://github.com/onnx/onnx-tensorrt/issues/350#issuecomment-569975247
# Hi all, I found a way to get it to work (not sure if it's a hack or not). 
# Add #define TENSORRTAPI at the top of NvOnnxParser.h, and re-run python setup.py install. I believe the error comes from swig seeing a syntax error with TENSORRTAPI.
#---------------------------------------------------------------------------------------------------------------
# there are other errors, but I can't remember
# Another - https://github.com/onnx/onnx-tensorrt/issues/162
# But I try edit setup.py and delete line 37 （ '-m64'，）
#---------------------------------------------------------------------------------------------------------------
sudo nano ${HOME}/.bashrc
# add the following to your ${HOME}/.bashrc
export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib
#---------------------------------------------------------------------------------------------------------------
sudo pip3 install pycuda --user
sudo python3 setup.py install

pip3 install numpy==1.19.4
#---------------------------------------------------------------------------------------------------------------
# https://github.com/onnx/onnx/issues/3268#issuecomment-777629332
# numpy==1.19.4
#---------------------------------------------------------------------------------------------------------------

# module test
python3
import onnx_tensorrt
python onnx_backend_test.py OnnxBackendRealModelTest
python onnx_backend_test.py
```



[cmake]: <https://cmake.org/download>

[protobuf]:  https://github.com/protocolbuffers/protobuf/tree/master/src
[onnx]: https://github.com/onnx/onnx-tensorrt/tree/7.1


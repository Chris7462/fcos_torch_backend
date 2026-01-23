# FCOS Object Detection Torch Backend

## Preparation

### Clone PyTorch and TorchVision from GitHub
```bash
cd ~/thirdparty/
git clone git@github.com:pytorch/pytorch.git --recurse-submodules
git clone git@github.com:pytorch/vision.git --recurse-submodules
```

### Install NCCL from NVIDIA repos
```bash
# If not already installed
sudo apt install libnccl2 libnccl-dev
```

### Build LibTorch
```bash
cd ~/thirdparty/pytorch
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_CUDA=ON \
         -DUSE_CUDNN=ON \
         -DUSE_CUDSS=ON \
         -DUSE_CUFILE=ON \
         -DUSE_CUSPARSELT=ON \
         -DUSE_SYSTEM_NCCL=ON \
         -DCMAKE_INSTALL_PREFIX=/opt/libtorch
cmake --build . -j8
cmake --install .
```
If you encountered issue, try to add the following in your `~/.bashrc` before build the libtorch
```bash
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda/include:$CPATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

## Point to your custom libtorch installation
Add the following to your `~/.bashrc` file
```bash
export Torch_DIR="/opt/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="/opt/libtorch/lib:$LD_LIBRARY_PATH"
```

### Build LibTorchVision
```bash
cd ~/thirdparty/vision
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DWITH_CUDA=ON \
         -DCMAKE_PREFIX_PATH=/opt/libtorch \
         -DCMAKE_INSTALL_PREFIX=/opt/libtorchvision
cmake --build . -j8
cmake --install .
```

## Point to your custom libtorchvision installation
Add the following to your `~/.bashrc` file
```bash
export TorchVision_DIR="/opt/libtorchvision/share/cmake/TorchVision"
export LD_LIBRARY_PATH="/opt/libtorchvision/lib:$LD_LIBRARY_PATH"
```

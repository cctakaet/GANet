mkdir result
mkdir result/Synthetic
mkdir result/Real
mkdir checkpoint
wget --no-check-certificate "https://onedrive.live.com/download?cid=F80778D80179676E&resid=F80778D80179676E%211349&authkey=AJOnXyDTUNZTeXI" -O checkpoint/kitti2015_final.pth
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev

#export LD_LIBRARY_PATH="/home/feihu/anaconda3/lib:$LD_LIBRARY_PATH"
#export LD_INCLUDE_PATH="/home/feihu/anaconda3/include:$LD_INCLUDE_PATH"
#export CUDA_HOME="/usr/local/cuda-10.0"
#export PATH="/home/feihu/anaconda3/bin:/usr/local/cuda-10.0/bin:$PATH"
#export CPATH="/usr/local/cuda-10.0/include"
#export CUDNN_INCLUDE_DIR="/usr/local/cuda-10.0/include"
#export CUDNN_LIB_DIR="/usr/local/cuda-10.0/lib64"

#export LD_LIBRARY_PATH="/home/zhangfeihu/anaconda3/lib:$LD_LIBRARY_PATH"
#export LD_INCLUDE_PATH="/home/zhangfeihu/anaconda3/include:$LD_INCLUDE_PATH"
#export CUDA_HOME="/home/work/cuda-9.2"
#export PATH="/home/zhangfeihu/anaconda3/bin:/home/work/cuda-9.2/bin:$PATH"
#export CPATH="/home/work/cuda-9.2/include"
#export CUDNN_INCLUDE_DIR="/home/work/cudnn/cudnn_v7/include"
#export CUDNN_LIB_DIR="/home/work/cudnn/cudnn_v7/lib64"
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))") || TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
cd libs/GANet
python setup.py clean || python3 setup.py clean
rm -rf build
python setup.py build || python3 setup.py build
cp -r build/lib* build/lib

cd ../sync_bn
python setup.py clean || python3 setup.py clean
rm -rf build
python setup.py build || python3 setup.py build
cp -r build/lib* build/lib

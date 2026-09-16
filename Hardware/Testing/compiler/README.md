#/usr/local/cuda-13.0/bin/nvcc cuda_check.cu -o cuda_check
#CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 ./cuda_check
#readelf -d /usr/bin/nvidia-smi
#ldd /usr/bin/nvidia-smi
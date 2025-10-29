#README
#instruction
#lab.sh
#GPU_burn

#compiler
gpu_sm.cu
--- GPU 7 -------------------------------------
Name                : NVIDIA B200
Architecture        : Blackwell (B100 / B200 / GB200)  sm_100
SM Count            : 148
CUDA Cores / SM     : 64
Total CUDA Cores    : 9472
Tensor Cores / SM   : 8
Memory (VRAM)       : 182631 MiB (178 GiB)
L2 Cache            : 129536 KiB
Clock (base)        : N/A (CUDA version)
NVLink capable      : NO
P2P with next GPU   : YES


gpu_precision.cu
--- GPU 7 -------------------------------------
Name                : NVIDIA B200
Architecture        : Blackwell (B100 / B200 / GB200)  sm_100
SM Count            : 148
CUDA Cores / SM     : 64
Total CUDA Cores    : 9472
Tensor Cores / SM   : 8
Memory (VRAM)       : 182631 MiB (178 GiB)
L2 Cache            : 129536 KiB
Clock (base)        : 1965 MHz
NVLink capable      : NO
P2P with next GPU   : YES
  Precision Support:
    FP32           : YES (native)
    FP8 (E4M3/E5M2): YES (Tensor Cores)
    FP4 (E2M1)     : YES (native, NVFP4 w/ microscaling)
    FP6            : YES (native)
    MXINT (INT8/INT4): YES (Tensor Cores + GPTQ/AWQ)
    FP8 Intrinsics : N/A
    FP4 Intrinsics : N/A

  MIG Mode          : Not supported

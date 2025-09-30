#https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/NVIDIA/code/llama3_1-405b/tensorrt
#https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/cpp
#https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/NVIDIA/code/mixtral-8x7b/tensorrt
#https://huggingface.co/models



+Dev: docker run --rm -it --gpus all nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04 /bin/bash
+benchmark: docker run --rm -it --gpus all nvcr.io/nvidia/hpc-benchmarks:25.04  /bin/bash

cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/examples/OpenACC/samples
make all 
cd /nbody; make ; ./nbody.out
/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/profilers/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
docker run --rm --gpus all \
    -v /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/examples/OpenACC/samples/nbody:/workspace \
    nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04 \
    /workspace/nbody.out

docker run --rm -it --gpus all nvcr.io/nvidia/hpc-benchmarks:25.04  /bin/bash
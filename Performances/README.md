#https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/NVIDIA/code/llama3_1-405b/tensorrt
#https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/cpp
#https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/NVIDIA/code/mixtral-8x7b/tensorrt
#https://huggingface.co/models
https://github.com/ollama/ollama



#ollama
curl -fsSL https://ollama.com/install.sh | sh or curl https://ollama.com/install.sh -o install.sh; chmod +x install.sh; ./install.sh
ollama run mistral-nemo

https://ollama.com/install.sh
https://github.com/ollama/ollama

>>> What is the capital of France?
>>> """
Write me a Python function that
prints hello world.
"""

>>> /show
>>> /load mistral
>>> /load llama2
>>> /save mysession
>>> /load mysession
>>> /bye
ollama run mistral-nemo "New about GPUs"


#nvbandwidth - Install the nvbandwidth for quick check
apt install libboost-program-options-dev -y; git clone https://github.com/NVIDIA/nvbandwidth; cd $(pwd)/nvbandwidth; sudo ./debian_install.sh; cmake .; make
./nvbandwidth

#nccl
sudo apt install -y libnccl-dev libnccl2 -y; git clone https://github.com/NVIDIA/nccl-tests.git; cd $(pwd)/nccl-tests; make #make MPI=1 to run with mpi
./build/all_reduce_perf -b 8 -e 32G -f 2 -t 8
./build/alltoall_perf -b 4G -e 32G -f 2 -t 8



#gburn
git clone https://github.com/wilicc/gpu-burn; cd $(pwd)/gpu-burn ; make #make COMPUTE=100 for B200
docker build -t gpu_burn .
docker run --rm --gpus all gpu_burn

+gpu_burn-drv.cpp (line 113 is the issue)
        #checkError(cuCtxCreate(&d_ctx, 0, d_dev)); #remove completely
+add: 
        CUctxCreateParams params = {0}; // Initialize params to zero
        checkError(cuCtxCreate(&d_ctx, &params, 0, d_dev));
+Modify the MAKEFILE.txt
CUDAPATH ?= /usr/local/cuda-13.0
CUDA_VERSION ?= 13.0.0
COMPUTE ?= 100


# Build and run TRT LLM container
git clone https://github.com/NVIDIA/TensorRT-LLM.git -b v0.17.0
cd TensorRT-LLM
git submodule update --init --recursive
apt-get install git-lfs
git lfs install && git lfs pull
make -C docker release_build #make -C docker -j180 release_build (j180: 180 cores)
docker run -it --gpus all --shm-size 64g docker.io/tensorrt_llm/release:latest /bin/bash
-v /opt/testing/app/TensorRT-LLM:/app python3
docker run -it docker.io/tensorrt_llm/release:latest /bin/bash
# Number of requests for various ISL, OSL can be found in this link: https://github.com/NVIDIA/TensorRT-LLM/blob/deepseek/docs/source/performance/perf-overview.md
export NUM_REQUESTS=<number of requests>
export ISL=<input sequence length>
export OSL=<output sequence length>
export HF_TOKEN=<your huggingface token>
export TP=<number of GPUs: 1,2,4,8>
export PP=<for this benchmark, PP is set to 1>
export QUANTIZATION=<NVFP4 for FP4 and NVFP8 for FP8>

#download model inside the container
docker run -it --gpus all --shm-size 128g -v  /opt/testing/perf/:/app/tensorrt_llm --entrypoint /bin/bash tensorrt_llm/release:latest

# Download checkpoints from Huggingface
huggingface-cli login --token hf auth login --token ${HF_TOKEN} --add-to-git-credential
huggingface-cli login --token ${HF_TOKEN}
export HF_HOME=/tmp/HF
huggingface-cli download meta-llama/Llama-3.1-70B
git clone https://huggingface.co/meta-llama/Llama-3.1-70B
git clone https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct


# Prepare dataset with NUM_REQUESTS, ISL and OSL
# For example, to run ISL:OSL = 128:128, choose num_requests = 30000 from the above link
python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-70B token-norm-dist --num-requests=30000 --input-mean=128 --output-mean=128 --input-stdev=0 --output-stdev=0 > /tmp/dataset.txt

+python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-70B token-norm-dist --num-requests=${NUM_REQUESTS} --input-mean=${ISL} --output-mean=${OSL} --input-stdev=0 --output-stdev=0 > /tmp/dataset.txt
python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=. token-norm-dist --num-requests=20000 --input-mean=2048 --output-mean=2048 --input-stdev=0 --output-stdev=0 > /app/tensorrt_llm/dataset_20000_2048.txt



# Build and run engine with TP and PP
+trtllm-bench --workspace /tmp/Llama-3.1-70B --model meta-llama/Llama-3.1-70B build --tp_size ${TP} --pp_size ${PP} --dataset /tmp/dataset.txt --quantization ${QUANTIZATION} #not FB4 NVFB4 for B200
trtllm-bench --workspace /tmp/Llama-3.1-70B --model /app/tensorrt_llm build --tp_size 8 --pp_size 1 --dataset /app/tensorrt_llm/dataset.txt --quantization FP8
trtllm-bench --workspace /tmp/Llama-3.1-70B --model /app/tensorrt_llm build --tp_size 4 --pp_size 1 --dataset /app/tensorrt_llm/dataset.txt --quantization FP8

+trtllm-bench --model meta-llama/Llama-3.1-70B throughput --dataset /tmp/dataset.txt --engine_dir /tmp/Llama-3.1-70B/meta-llama/Llama-3.1-70B/tp_${TP}_pp_${PP} --kv_cache_free_gpu_mem_fraction 0.95
trtllm-bench --model dummy  --model_path /app/tensorrt_llm   --workspace /tmp/Llama-3.1-70B   throughput   --dataset /app/tensorrt_llm/dataset.txt   --engine_dir /app/tensorrt_llm/tp_8_pp_1   --kv_cache_free_gpu_mem_fraction 0.95
trtllm-bench --model dummy  --model_path /app/tensorrt_llm   --workspace /tmp/Llama-3.1-70B   throughput   --dataset /app/tensorrt_llm/dataset_20000_2048.txt   --engine_dir /app/tensorrt_llm/tp_8_pp_1   --kv_cache_free_gpu_mem_fraction 0.95


+Dev: docker run --rm -it --gpus all nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04 /bin/bash
+benchmark: docker run --rm -it --gpus all nvcr.io/nvidia/hpc-benchmarks:25.04  /bin/bash

cd /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/examples/OpenACC/samples
make all 
cd /nbody; make ; ./nbody.out
/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/extras/qd/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/nvshmem/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/nccl/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/profilers/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
docker run --rm --gpus all -v /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/examples/OpenACC/samples/nbody:/workspace \
    nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04 /workspace/nbody.out

docker run --rm -it --gpus all nvcr.io/nvidia/hpc-benchmarks:25.04  /bin/bash


An IOMMU (Input-Output Memory Management Unit) is a hardware component that translates virtual addresses used by I/O devices into physical addresses, similar to how a CPU's MMU handles CPU addresses. It provides memory protection for devices using Direct Memory Access (DMA), 
enhances security by protecting against malicious devices, and is crucial for device passthrough in virtualization to allow virtual machines to use hardware like GPUs directly

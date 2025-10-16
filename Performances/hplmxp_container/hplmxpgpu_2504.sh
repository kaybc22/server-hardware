#!/bin/bash
#https://docs.nvidia.com/nvidia-hpc-benchmarks/HPL_MxP_benchmark.html
#$(date +%s%3N)
#CONTAINER_TAG= "nvcr.io/nvidia/hpc-benchmarks:25.09"
CONTAINER_TAG=25.04

OUTFILE8=hplmxpgpu-fp8
OUTFILE16=hplmxpgpu-fp16

TMPDIR8=/tmp/hplmxpgpu-fp8
TMPDIR16=/tmp/hplmxpgpu-fp16
TMPOUT8=$TMPDIR8/$OUTFILE8$(date +%s%3N)
TMPOUT16=$TMPDIR16/$OUTFILE16$(date +%s%3N)
rm -rf $TMPDIR8
rm -rf $TMPDIR16
mkdir -p $TMPDIR8
mkdir -p $TMPDIR16

# Print out the GPUs. The GPU type will be used in the outcome script to determine which threshold values to use
nvidia-smi --list-gpus > $TMPOUT8
nvidia-smi --list-gpus > $TMPOUT16

LASTCPU=$(grep ^processor /proc/cpuinfo | tail -1 | awk '{print $3}')
#echo "=====FP16====="  &>> $TMPOUT16
#docker run --gpus all --privileged --rm -v $TMPOUT16:$TMPOUT16 --shm-size 68g nvcr.io/nvidia/hpc-benchmarks:${CONTAINER_TAG} mpirun -np 8  /workspace/hpl-mxp.sh --gpu-affinity 0:1:2:3:4:5:6:7 --mem-affinity 0:0:0:0:1:1:1:1 --cpu-affinity 0-13:14-27:28-41:42-55:56-69:70-83:84-97:98-111 --n 90000 --nb 2048 --nprow 4 --npcol 2 --nporder row --preset-gemm-kernel 0 --u-panel-chunk-nbs 8   --use-mpi-panel-broadcast 50 --sloppy-type 2 --call-dgemv-with-multiple-threads 0 --Anq-device 0 --mpi-use-mpi 1 --prioritize-trsm 0 --prioritize-factorization 1 &>> $TMPOUT16
echo "=====FP8====="  &>> $TMPOUT8
docker run --gpus all --privileged --rm -v $TMPOUT8:$TMPOUT8 --shm-size 68g nvcr.io/nvidia/hpc-benchmarks:${CONTAINER_TAG} mpirun -np 8  /workspace/hpl-mxp.sh --gpu-affinity 0:1:2:3:4:5:6:7 --mem-affinity 0:0:0:0:1:1:1:1 --cpu-affinity 0-13:14-27:28-41:42-55:56-69:70-83:84-97:98-111 --n 190000 --nb 4096 --nprow 4 --npcol 2 --nporder row --preset-gemm-kernel 0 --u-panel-chunk-nbs 8 --use-mpi-panel-broadcast 50 --sloppy-type 1 --call-dgemv-with-multiple-threads 0 --Anq-device 0 --mpi-use-mpi 1 --prioritize-trsm 0 --prioritize-factorization 1 &>> $TMPOUT8




#--n 160000 :size of N-by-N matrix 

# ****** HPL MxP Result    ******
#docker run --gpus all --privileged --rm -v $TMPOUT8:$TMPOUT8 --shm-size 68g nvcr.io/nvidia/hpc-benchmarks:${CONTAINER_TAG} mpirun -np 8  /workspace/hpl-mxp.sh --gpu-affinity 0:1:2:3:4:5:6:7 --mem-affinity 0:0:0:0:1:1:1:1 --cpu-affinity 0-13:14-27:28-41:42-55:56-69:70-83:84-97:98-111 --n 120000 --nb 4096 --nprow 4 --npcol 2 --nporder row --preset-gemm-kernel 0 --u-panel-chunk-nbs 8 --use-mpi-panel-broadcast 50 --sloppy-type 1 --call-dgemv-with-multiple-threads 0 --Anq-device 0 --mpi-use-mpi 1 --prioritize-trsm 0 --prioritize-factorization 1 &>> $TMPOUT8
#    N = 120000, NB = 4096, NPROW = 4, NPCOL = 2, SLOPPY-TYPE = 1
#       GFLOPS = 1.8936e+05, per GPU =   23669.68 ------ The HPL-MxP performance to report
#    LU GFLOPS = 7.4471e+05, per GPU =   93089.10 ------ The performance excluding the iterative solver part

#docker run --gpus all --privileged --rm -v $TMPOUT8:$TMPOUT8 --shm-size 68g nvcr.io/nvidia/hpc-benchmarks:${CONTAINER_TAG} mpirun -np 8  /workspace/hpl-mxp.sh --gpu-affinity 0:1:2:3:4:5:6:7 --mem-affinity 0:0:0:0:1:1:1:1 --cpu-affinity 0-13:14-27:28-41:42-55:56-69:70-83:84-97:98-111 --n 160000 --nb 4096 --nprow 4 --npcol 2 --nporder row --preset-gemm-kernel 0 --u-panel-chunk-nbs 8 --use-mpi-panel-broadcast 50 --sloppy-type 1 --call-dgemv-with-multiple-threads 0 --Anq-device 0 --mpi-use-mpi 1 --prioritize-trsm 0 --prioritize-factorization 1 &>> $TMPOUT8
#    N = 120000, NB = 4096, NPROW = 4, NPCOL = 2, SLOPPY-TYPE = 1
#       GFLOPS = 1.9939e+05, per GPU =   24923.21 ------ The HPL-MxP performance to report
#    LU GFLOPS = 8.8688e+05, per GPU =  110860.20 ------ The performance excluding the iterative solver part

#docker run --gpus all --privileged --rm -v $TMPOUT8:$TMPOUT8 --shm-size 68g nvcr.io/nvidia/hpc-benchmarks:${CONTAINER_TAG} mpirun -np 8  /workspace/hpl-mxp.sh --gpu-affinity 0:1:2:3:4:5:6:7 --mem-affinity 0:0:0:0:1:1:1:1 --cpu-affinity 0-13:14-27:28-41:42-55:56-69:70-83:84-97:98-111 --n 240000 --nb 4096 --nprow 4 --npcol 2 --nporder row --preset-gemm-kernel 0 --u-panel-chunk-nbs 8 --use-mpi-panel-broadcast 50 --sloppy-type 1 --call-dgemv-with-multiple-threads 0 --Anq-device 0 --mpi-use-mpi 1 --prioritize-trsm 0 --prioritize-factorization 1 &>> $TMPOUT8
#    N = 240000, NB = 4096, NPROW = 4, NPCOL = 2, SLOPPY-TYPE = 1
#       GFLOPS = 4.3040e+05, per GPU =   53799.84 ------ The HPL-MxP performance to report
#    LU GFLOPS = 3.5410e+06, per GPU =  442625.49 ------ The performance excluding the iterative solver part


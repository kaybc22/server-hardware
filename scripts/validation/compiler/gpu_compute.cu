#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error: Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }
#nvcc -o gpu_test gpu_test.cu -gencode arch=compute_100,code=sm_100
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    printf("Found %d CUDA-capable device(s):\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    }

    return 0;
}

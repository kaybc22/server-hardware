#include <iostream>
#include <cuda_runtime.h>

int main() {
    int gpuA, gpuB;
    size_t dataMB;

    std::cout << "Enter source GPU ID (1–8): ";
    std::cin >> gpuA;
    std::cout << "Enter destination GPU ID (1–8): ";
    std::cin >> gpuB;
    std::cout << "Enter data size in MB (e.g., 10, 100000): ";
    std::cin >> dataMB;

    gpuA -= 1;
    gpuB -= 1;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (gpuA < 0 || gpuB < 0 || gpuA >= deviceCount || gpuB >= deviceCount) {
        std::cerr << "Invalid GPU IDs. Available GPUs: 1 to " << deviceCount << std::endl;
        return 1;
    }

    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, gpuB, gpuA);
    if (!canAccessPeer) {
        std::cerr << "GPU " << gpuB + 1 << " cannot access GPU " << gpuA + 1 << " via P2P." << std::endl;
        return 1;
    }

    cudaSetDevice(gpuB);
    cudaDeviceEnablePeerAccess(gpuA, 0);

    size_t dataSize = dataMB * 1024 * 1024; // Convert MB to bytes

    cudaSetDevice(gpuA);
    float* src;
    cudaMalloc(&src, dataSize);
    cudaMemset(src, 1, dataSize);

    cudaSetDevice(gpuB);
    float* dst;
    cudaMalloc(&dst, dataSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpyPeer(dst, gpuB, src, gpuA, dataSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double gb = static_cast<double>(dataSize) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = gb / (ms / 1000.0);

    std::cout << "\n? Data transferred from GPU " << gpuA + 1 << " to GPU " << gpuB + 1 << std::endl;
    std::cout << "?? Transfer size: " << gb << " GB" << std::endl;
    std::cout << "?? Transfer time: " << ms << " ms" << std::endl;
    std::cout << "?? Bandwidth: " << bandwidth << " GB/s\n" << std::endl;

    cudaFree(src);
    cudaFree(dst);
    cudaDeviceDisablePeerAccess(gpuA);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
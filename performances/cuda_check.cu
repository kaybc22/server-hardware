#include <iostream>
#include <cuda_runtime.h>
int main() {
   int deviceCount = 0;
   cudaError_t err = cudaGetDeviceCount(&deviceCount);
   if (err != cudaSuccess) {
       std::cerr << "cudaGetDeviceCount failed: "
<< cudaGetErrorString(err) << std::endl;
       return 1;
   }
   std::cout << "Detected " << deviceCount << " CUDA device(s)\n";
   for (int i = 0; i < deviceCount; i++) {
       std::cout << "\nChecking GPU " << i << "...\n";
       // Try to set the device
       err = cudaSetDevice(i);
       if (err != cudaSuccess) {
           std::cerr << "  Failed to set device " << i << ": "
<< cudaGetErrorString(err) << std::endl;
           continue;
       }
       // Query device properties
       cudaDeviceProp prop;
       err = cudaGetDeviceProperties(&prop, i);
       if (err != cudaSuccess) {
           std::cerr << "  Failed to get properties: "
<< cudaGetErrorString(err) << std::endl;
           continue;
       }
       std::cout << "  Name: " << prop.name
<< " | Global Mem: " << (prop.totalGlobalMem >> 20) << " MB"
<< " | SMs: " << prop.multiProcessorCount << std::endl;
       // Test a small allocation
       void* d_ptr = nullptr;
       err = cudaMalloc(&d_ptr, 1024);
       if (err != cudaSuccess) {
           std::cerr << "  cudaMalloc failed: "
<< cudaGetErrorString(err) << std::endl;
       } else {
           std::cout << "  cudaMalloc test OK\n";
           cudaFree(d_ptr);
       }
   }
   return 0;
}

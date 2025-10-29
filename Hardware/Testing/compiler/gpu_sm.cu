// gpu_sm.cu
// ---------------------------------------------------------------
//  Detect Hopper vs Blackwell + SMs, memory, cores, clock, NVLink
//  Works on CUDA 11.8 → 12.5+ 
//  Compile: nvcc -O2 -arch=sm_90 gpu_sm.cu -o gpu_sm
// ---------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %d: %s at %s:%d\n",               \
                    err, cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

/* ------------------------------------------------------------------
   Helper: CUDA cores per SM
   ------------------------------------------------------------------ */
static inline int coresPerSM(int major, int minor) {
    switch ((major << 4) + minor) {
        case 0x90: return 64;  // Hopper
        case 0xA0: return 64;  // Blackwell
        default:   return 0;
    }
}

/* ------------------------------------------------------------------ */
__global__ void dummy() { }

int main() {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));

    if (devCount == 0) {
        printf("No CUDA-capable GPUs found.\n");
        return 0;
    }

    printf("=== Found %d GPU(s) ===\n\n", devCount);

    for (int dev = 0; dev < devCount; ++dev) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // ------------------- Architecture -------------------
        const char *archName = "Unknown";
        const char *genName  = "Unknown";
        int tensorCoresPerSM = 0;

        if (prop.major == 9 && prop.minor == 0) {
            archName = "Hopper";
            genName  = "H100 / H200";
            tensorCoresPerSM = 4;
        } else if (prop.major == 10 && prop.minor == 0) {
            archName = "Blackwell";
            genName  = "B100 / B200 / GB200";
            tensorCoresPerSM = 8;
        }

        // ------------------- Memory -------------------
        size_t memMiB = prop.totalGlobalMem / (1024ULL * 1024ULL);
        size_t memGiB = memMiB / 1024ULL;

        // ------------------- Clock (kHz → MHz) -------------------
        int baseClockMHz = 0;
#ifdef cudaDevPropClockRate
        baseClockMHz = prop.clockRate / 1000;  // kHz → MHz
#else
        // Fallback: use memory clock or skip
        baseClockMHz = 0;
        printf("Warning: clockRate not available in this CUDA version\n");
#endif

        // ------------------- NVLink / P2P -------------------
        int p2pAccess = 0;
        if (devCount > 1) {
            int access = 0;
            if (cudaDeviceCanAccessPeer(&access, dev, (dev + 1) % devCount) == cudaSuccess && access)
                p2pAccess = 1;
        }
        int nvlinkCapable = prop.tccDriver ? 1 : 0;

        // ------------------- Print -------------------
        printf("--- GPU %d -------------------------------------\n", dev);
        printf("Name                : %s\n", prop.name);
        printf("Architecture        : %s (%s)  sm_%d%d\n",
               archName, genName, prop.major, prop.minor);
        printf("SM Count            : %d\n", prop.multiProcessorCount);
        printf("CUDA Cores / SM     : %d\n", coresPerSM(prop.major, prop.minor));
        printf("Total CUDA Cores    : %d\n",
               prop.multiProcessorCount * coresPerSM(prop.major, prop.minor));
        printf("Tensor Cores / SM   : %d\n", tensorCoresPerSM);
        printf("Memory (VRAM)       : %zu MiB (%zu GiB)\n", memMiB, memGiB);
        printf("L2 Cache            : %d KiB\n", prop.l2CacheSize / 1024);
        if (baseClockMHz > 0)
            printf("Clock (base)        : %d MHz\n", baseClockMHz);
        else
            printf("Clock (base)        : N/A (CUDA version)\n");
        printf("NVLink capable      : %s\n", nvlinkCapable ? "YES" : "NO");
        printf("P2P with next GPU   : %s\n", p2pAccess ? "YES" : "NO");
        printf("\n");
    }

    dummy<<<1,1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Tip: Run `nvidia-smi nvlink -s` for detailed NVLink status.\n");
    return 0;
}

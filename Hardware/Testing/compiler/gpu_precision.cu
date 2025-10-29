// gpu_precision.cu
// ---------------------------------------------------------------
//  Detect Hopper/Blackwell + FP4/FP8/FP32/MXINT, SMs, MIG, Clock
//  Works on CUDA 12.4 → 13.0+ (no clockRate)
//  Compile: nvcc -O2 -arch=sm_90 gpu_precision.cu -o gpu_precision
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

/* ------------------------------------------------------------------
   Get GPU base clock via attribute (replacement for prop.clockRate)
   ------------------------------------------------------------------ */
int getBaseClockMHz(int dev) {
    int clockMHz = 0;
    cudaError_t err = cudaDeviceGetAttribute(&clockMHz, cudaDevAttrClockRate, dev);
    if (err == cudaSuccess && clockMHz > 0) {
        return clockMHz / 1000;  // kHz → MHz
    }
    return 0;
}

/* ------------------------------------------------------------------
   Print precision support
   ------------------------------------------------------------------ */
void printPrecisionSupport(int major, int minor) {
    printf("  Precision Support:\n");
    printf("    FP32           : YES (native)\n");

    if (major >= 9) {
        printf("    FP8 (E4M3/E5M2): YES (Tensor Cores)\n");
    } else {
        printf("    FP8            : Emulated (slow)\n");
    }

    if (major == 10) {
        printf("    FP4 (E2M1)     : YES (native, NVFP4 w/ microscaling)\n");
        printf("    FP6            : YES (native)\n");
    } else {
        printf("    FP4/FP6        : Emulated (via FP8 fallback)\n");
    }

    if (major >= 8) {
        printf("    MXINT (INT8/INT4): YES (Tensor Cores + GPTQ/AWQ)\n");
    } else {
        printf("    MXINT          : Limited\n");
    }

#ifdef __CUDA_FP8__
    printf("    FP8 Intrinsics : Available\n");
#else
    printf("    FP8 Intrinsics : N/A\n");
#endif

#ifdef __CUDA_FP4__
    printf("    FP4 Intrinsics : Available\n");
#else
    printf("    FP4 Intrinsics : N/A\n");
#endif

    printf("\n");
}

/* ------------------------------------------------------------------
   MIG info (guarded)
   ------------------------------------------------------------------ */
#ifdef cudaDevAttrMigMode
void printMigInfo(int dev) {
    int migMode = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&migMode, cudaDevAttrMigMode, dev));

    if (migMode == 0) {
        printf("  MIG Mode          : Disabled\n");
        return;
    }

    printf("  MIG Mode          : Enabled\n");

    int instanceCount = 0;
    cudaError_t err = cudaDeviceGetMigInstanceCount(&instanceCount, dev);
    if (err != cudaSuccess) {
        printf("  MIG Instances     : <error>\n");
        return;
    }

    if (instanceCount == 0) {
        printf("  MIG Instances     : 0\n");
        return;
    }

    printf("  MIG Instances     : %d\n", instanceCount);

    for (int i = 0; i < instanceCount; ++i) {
        cudaMigDeviceInstance_t migInst;
        err = cudaDeviceGetMigInstance(&migInst, dev, i);
        if (err != cudaSuccess) continue;

        int profileId = 0, placementStart = 0, placementSize = 0, memorySizeMiB = 0;
        err = cudaMigInstanceGetInfo(migInst, &profileId, &placementStart, &placementSize, &memorySizeMiB);
        if (err != cudaSuccess) continue;

        const char* profileName = "Unknown";
        switch (profileId) {
            case 19: profileName = "1g.10gb"; break;
            case 22: profileName = "4g.40gb"; break;
            case 30: profileName = "1g.24gb"; break;
        }

        printf("    Instance %d:\n", i);
        printf("      Profile         : %s (ID=%d)\n", profileName, profileId);
        printf("      Memory          : %d MiB\n", memorySizeMiB);
        printf("      Approx SMs      : %d\n", placementSize);
        printf("\n");
    }
}
#else
void printMigInfo(int dev) {
    printf("  MIG Mode          : Not supported\n");
}
#endif

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

        // ------------------- Clock (via attribute) -------------------
        int baseClockMHz = getBaseClockMHz(dev);

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
            printf("Clock (base)        : N/A\n");
        printf("NVLink capable      : %s\n", nvlinkCapable ? "YES" : "NO");
        printf("P2P with next GPU   : %s\n", p2pAccess ? "YES" : "NO");

        // ------------------- Precision -------------------
        printPrecisionSupport(prop.major, prop.minor);

        // ------------------- MIG -------------------
        printMigInfo(dev);

        printf("\n");
    }

    dummy<<<1,1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Tip: Use Transformer Engine for FP8/FP4 autocast.\n");
    return 0;
}

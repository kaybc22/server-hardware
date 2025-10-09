#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#numactl --membind=1 ./my_inference_app # 1=CXl 0=CPU DRAM
#define SIZE (1024 * 1024 * 1024) // 1 GB elements
#gcc -O3 test_mem.c -o test_mem

int main() {
    double *a, *b, *c;
    a = (double*) malloc(SIZE * sizeof(double));
    b = (double*) malloc(SIZE * sizeof(double));
    c = (double*) malloc(SIZE * sizeof(double));

    if (!a || !b || !c) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize
    for (size_t i = 0; i < SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    // Time the computation
    clock_t start = clock();
    for (size_t i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }
    clock_t end = clock();

    double secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Completed vector addition of %zu elements in %.3f seconds\n", SIZE, secs);

    free(a); free(b); free(c);
    return 0;
}
// mpi_alltoall.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    const int N = 4; // each process sends N ints to every other process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[size * N];
    int recvbuf[size * N];

    // Fill sendbuf with something unique per rank
    for (int i = 0; i < size * N; i++) {
        sendbuf[i] = rank * 100 + i;
    }

    // Perform all-to-all exchange
    MPI_Alltoall(sendbuf, N, MPI_INT,
                 recvbuf, N, MPI_INT, MPI_COMM_WORLD);

    // Print what each rank received
    printf("Rank %d received:", rank);
    for (int i = 0; i < size * N; i++) {
        printf(" %d", recvbuf[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}

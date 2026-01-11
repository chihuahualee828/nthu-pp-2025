#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]), 
		           k = atoll(argv[2]),
			   pixels = 0;
	// for (unsigned long long x = 0; x < r; x++)
	// {
	// 	unsigned long long y = ceil(sqrtl(r * r - x * x));
	// 	pixels += y;
	// 	pixels %= k;
	// }
	
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	unsigned long long chunk = r / size;
    unsigned long long start = rank * chunk;
    unsigned long long end   = (rank == size - 1) ? r : start + chunk;
	// printf("Rank %d: start = %llu, end = %llu\n", rank, start, end);

	double t0 = MPI_Wtime();


	unsigned long long local_pixels = 0;
    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl((long double)r * r - (long double)x * x));
        local_pixels += y;
        local_pixels %= k;
    }
	// printf("Rank %d: local_pixels = %llu\n", rank, local_pixels);

	double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

	
	unsigned long long total_pixels = 0;
	MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // add up all local_pixels to total_pixels at rank 0


	// printf("Rank %d took %.6f seconds\n", rank, elapsed); 

    if (rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k); // mod total_pixels = mod of each rank local_pixels and add up then mod
    }

	MPI_Finalize();


	// printf("%llu\n", (4 * pixels) % k);
	return 0;
}


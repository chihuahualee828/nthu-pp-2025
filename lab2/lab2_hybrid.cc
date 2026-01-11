#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define NUM_THREADS 8

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]), k = atoll(argv[2]), pixels = 0;

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int rem = r % size;
	unsigned long long base_chunk = r / size, start = rank * base_chunk+(rank < rem ? rank : rem), end = start + base_chunk + (rank < rem ? 1 : 0);


	// char *env = getenv("SLURM_CPUS_PER_TASK"); 
	// int num_threads = 8;
	// if (env) { 
	// 	// printf("Using %d threads\n", atoi(env));
	// 	num_threads = atoi(env); // 
	// }

	unsigned long long r_square = r * r, thread_chunk = (base_chunk + (rank < rem ? 1 : 0))/NUM_THREADS, thread_rem = (base_chunk + (rank < rem ? 1 : 0))%NUM_THREADS;

	#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:pixels) schedule(static)
	// for (unsigned long long x = 0; x < r; x++) {
	// 	unsigned long long y = ceil(sqrtl(r_square - x*x));
	// 	pixels += y;
	// 	pixels %= k;
	// }
	for (int i = 0; i < NUM_THREADS; i++) {
		
		// chunk each mpi rank with # threads
		unsigned long long thread_start = start+thread_chunk*i+(i < (int)thread_rem ? i : thread_rem),
			thread_end = thread_start + thread_chunk + (i < (int)thread_rem ? 1 : 0);

		unsigned long long first = ceil(sqrtl(r_square-start*start));
		// for (unsigned long long x = start; x < end; x++) {
		// 	pixels += ceil(sqrtl(r_square - x*x));
		// 	pixels %= k;
		// }
		for (unsigned long long x = thread_start; x < thread_end; x++) {
			while(r_square-x*x <= (first-1)*(first-1)) {
				first -= 1;
			}
			while(r_square-x*x > first*first) {
				first++;
			}
			pixels += first;
			
		}
		pixels %= k;
	}
	// printf("Rank %d: start = %llu, end = %llu, pixels = %llu\n", rank, start, end, pixels);
	unsigned long long total_pixels = 0;
	MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // add up all local_pixels to total_pixels at rank 0

	MPI_Finalize();

	if (rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k); // mod total_pixels = mod of each rank local_pixels and add up then mod
    }


    return 0; 
}
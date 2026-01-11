#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define NUM_THREADS 8

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]), k = atoll(argv[2]), pixels = 0;

	char *env = getenv("SLURM_CPUS_PER_TASK"); 
	
	// int num_threads = 8;
	// if (env) { 
	// 	// printf("Using %d threads\n", atoi(env));
	// 	num_threads = atoi(env); // 
	// }


	unsigned long long base = r / NUM_THREADS, rem  = r % NUM_THREADS, r_square = r * r;

	#pragma omp parallel for num_threads(NUM_THREADS) reduction(+:pixels) schedule(static)
	// for (unsigned long long x = 0; x < r; x++) {
	// 	unsigned long long y = ceil(sqrtl(r_square - x*x));
	// 	pixels += y;
	// 	pixels %= k;
	// }
	for (int i = 0; i < NUM_THREADS; i++) {
		unsigned long long len = base + (i < (int)rem ? 1 : 0);
		unsigned long long start = i*base + (i < (int)rem ? i : rem);
		unsigned long long end   = start + len;

		unsigned long long first = ceil(sqrtl(r_square - start*start));
		// for (unsigned long long x = start; x < end; x++) {
		// 	pixels += ceil(sqrtl(r_square - x*x));
		// 	pixels %= k;
		// }

		for (unsigned long long x = start; x < end; x++) {
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
	printf("%llu\n", (4 * pixels) % k);

    return 0;    
}
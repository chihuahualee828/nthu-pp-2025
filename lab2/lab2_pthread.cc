#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#define NUM_THREADS 8

struct ThreadArgs {
	unsigned long long tid, r, k, start, end, pixels;
};



void *compute(void *arg) {
    struct ThreadArgs *data = (struct ThreadArgs *)arg; // cast back

    unsigned long long tid = data->tid, r = data->r, k = data->k,
		start = data->start, end = data->end, pixels = data->pixels, local_pixels = 0;
    // unsigned long long y = safe_y(r, tid);

	unsigned long long r_square = r * r;
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
		local_pixels += first;
		
	}

	data->pixels = local_pixels % k;

	// printf("Thread %lu finished (tid=%llu)\n",
    //        (unsigned long)pthread_self(), tid);
	// printf("Thread %llu: start = %llu, end = %llu, pixels = %llu\n", tid, start, end, pixels);

	pthread_exit(NULL);

}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]), k = atoll(argv[2]);


	// char *env = getenv("SLURM_CPUS_PER_TASK"); 
	
	// int num_threads = 1;
	// if (env) { 
	// 	// printf("Using %d threads\n", atoi(env));
	// 	num_threads = atoi(env); // 
	// }
	pthread_t threads[NUM_THREADS];
	struct ThreadArgs args[NUM_THREADS];
	// int ids[num_threads];

	unsigned long long base = r / NUM_THREADS, rem  = r % NUM_THREADS, cur = 0;
	
	for (int i = 0; i < NUM_THREADS; i++) {
		unsigned long long len = base + (i < (int)rem ? 1 : 0);
        args[i].start = cur;
        args[i].end   = cur + len;
        cur += len;
		
		args[i].tid = i;
		args[i].r = r; 
		args[i].k = k;

		pthread_create(&threads[i], NULL, compute, (void *)&args[i]);

	}
	unsigned long long res = 0;
	for (int i = 0; i < NUM_THREADS; i++) {
		// pthread_join(threads[i], NULL);
		// unsigned long long *res;
		// pthread_join(threads[i], (void **)&res);
		// // printf("Thread result = %d\n", *res);
		// pixels += *res;
		// pixels %= k;
		// free(res);
		pthread_join(threads[i], NULL);
		res += args[i].pixels;
		res %= k;
	}
	printf("%llu\n", (4 * res) % k);
	pthread_exit(NULL);
    return 0;
}
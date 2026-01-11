#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
// #include <xmmintrin.h>
#include <emmintrin.h> 
// #define NUM_THREADS 32


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_set_compression_strategy(png_ptr, Z_RLE); 
    png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    // png_set_compression_level(png_ptr, 1);
    png_set_compression_level(png_ptr, 0); //  faster compression
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


double start, end, left, right, lower, upper;
int width, height, iters, num_threads;
int* image;
int chunk_counter = 0;
int base_chunk, blocks;

pthread_mutex_t mutex;
int get_chunk(){
    pthread_mutex_lock(&mutex);
    int chunk = chunk_counter++;
    pthread_mutex_unlock(&mutex);
    return chunk;
}

struct ThreadArgs {
	int tid;
};



void *compute(void *arg) {
    struct ThreadArgs *data = (struct ThreadArgs *)arg; // cast back

    int tid = data->tid;

    const double dx = (right - left) / width;
    const double dy = (upper - lower) / height;
    
    // printf("Thread %lu: start = %lu, end = %lu\n", (unsigned long)pthread_self(), (unsigned long)start, (unsigned long)end);
    const __m128d two   = _mm_set1_pd(2.0);
    const __m128d four  = _mm_set1_pd(4.0);
    const __m128d one = _mm_set1_pd(1.0);

    while (true) {
        int chunk_id = get_chunk();
        int start = chunk_id * base_chunk;
        int end = (start + base_chunk) < (width * height) ? (start + base_chunk) : (width * height);
        
        if (chunk_id >= blocks) break;

        // int idx = tid;
        int idx = start;
        for (; idx < end; idx += 2) {
        // for (; idx+1 < width * height; idx += num_threads*2) {

            int idx1 = idx;
            int y1 = idx1 / width;
            int x1 = idx1 % width;
            double x0_1 = left + x1 * dx;
            double y0_1 = lower + y1 * dy;

            int idx2 = idx + 1; // The next pixel for this thread
            if (idx2 >= end) idx2 = 0; // Check bounds
            int y2 = idx2 / width;
            int x2 = idx2 % width;
            double x0_2 = left + x2 * dx;
            double y0_2 = lower + y2 * dy;

            __m128d y0v = _mm_set_pd(y0_2, y0_1); // reversed because first one is high 64 bits
            __m128d x0v = _mm_set_pd(x0_2, x0_1);

            __m128d x = _mm_setzero_pd();
            __m128d y = _mm_setzero_pd();
            __m128d repeats_pd = _mm_setzero_pd();  // doubles; convert back to int at the end
            __m128d active_mask_pd = _mm_set1_pd(1.0);
            for (int k = 0; k < iters; ++k) { // iterate up to iters since multiple lanes
                // x^2, y^2, xy
                __m128d xx = _mm_mul_pd(x, x);
                __m128d yy = _mm_mul_pd(y, y);
                __m128d xy = _mm_mul_pd(x, y);

                x = _mm_add_pd(_mm_sub_pd(xx, yy), x0v); // x^2 - y^2 + x0
                y = _mm_add_pd(_mm_add_pd(xy, xy), y0v);  // 2*xy + y0

                __m128d len2   = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
                active_mask_pd = _mm_cmplt_pd(len2, four); // check if length_squared < 4
                
                // If no lanes active, break
                if (_mm_movemask_pd(active_mask_pd) == 0) {
                    break;
                }

                // repeats += active ? 1 : 0
                // __m128d inc = _mm_and_pd(active_mask_pd, one);
                repeats_pd = _mm_add_pd(repeats_pd, _mm_and_pd(active_mask_pd, one)); // results of two pixels

            }

            repeats_pd = _mm_add_pd(repeats_pd, _mm_andnot_pd(active_mask_pd, _mm_set1_pd(1.0))); // invert because for lanes become inactive before last iteration, we didn't count the last iteration
            __m128i repeats_i = _mm_cvtpd_epi32(repeats_pd); // convert back to int

            // Store two pixels: note _mm_cvtpd_epi32 packs to low 2 ints of __m128i
            int tmp[2];
            _mm_storel_epi64((__m128i*)tmp, repeats_i);

            // data->image[idx1] = tmp[0];
            // data->image[idx2] = tmp[1];
            image[idx1] = tmp[0];
            if (idx+1 < end) image[idx2] = tmp[1];
        }
    }


	pthread_exit(NULL);

}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_threads = CPU_COUNT(&cpu_set)*1;
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    // int* image = (int*)malloc(width * height * sizeof(int));
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[num_threads];
	struct ThreadArgs args[num_threads];

	// int ids[num_threads];
    pthread_mutex_init(&mutex, NULL);
    
    base_chunk = (width * height) / (num_threads*500) < 10 ? 10 : (width * height) / (num_threads*500);
    base_chunk = base_chunk % 2 == 0 ? base_chunk : base_chunk+1; // make even
    blocks = (width * height + base_chunk - 1) / base_chunk;
    // printf("num_threads: %d, base_chunk: %d, blocks: %d\n", num_threads, base_chunk, blocks);
    // start = rank * base_chunk+(rank < rem ? rank : rem), end = start + base_chunk + (rank < rem ? 1 : 0), chunk_size = end - start;
    
	for (int i = 0; i < num_threads; i++) {
		args[i].tid = i;
		pthread_create(&threads[i], NULL, compute, (void *)&args[i]);

	}

    // unsigned long long res = 0;
	for (int i = 0; i < num_threads; i++) {
		pthread_join(threads[i], NULL);
	}
    
    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
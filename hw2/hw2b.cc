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
#include <mpi.h>
#include <omp.h>
#include <emmintrin.h>

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
    png_set_compression_level(png_ptr, 0);
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




int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    int NUM_THREADS = CPU_COUNT(&cpu_set)*1;

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int total_pixels = width * height;



    int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	int rem = total_pixels % size;

	int base_chunk = total_pixels / size, start = rank * base_chunk+(rank < rem ? rank : rem), end = start + base_chunk + (rank < rem ? 1 : 0), mpi_chunk_size = end - start;

    int* rank_image = (int*)malloc(mpi_chunk_size * sizeof(int));
    assert(rank_image);

    int* indices = (int*)malloc(mpi_chunk_size * sizeof(int));
    // indices with stride: r, r+size, r+2*size, ...
    for (int k = 0, g = rank; k < mpi_chunk_size; ++k, g += size)
        indices[k] = g;
    assert(indices);
    // printf("Rank %d: first index = %d, last index = %d, chunk_size: %d\n", rank, indices[0], indices[mpi_chunk_size-1], mpi_chunk_size);
    // printf("total_pixels: %d, start: %d, end: %d\n", total_pixels, start, end-1);

    

    const double dx = (right - left) / width;
    const double dy = (upper - lower) / height;

    const __m128d two   = _mm_set1_pd(2.0);
    const __m128d four  = _mm_set1_pd(4.0);
    const __m128d one = _mm_set1_pd(1.0);
    
    
    int chunk_size = (mpi_chunk_size / NUM_THREADS)/500;
    chunk_size = chunk_size < 10 ? 10 : chunk_size;
    // #pragma omp parallel for schedule(static, chunk_size)
    #pragma omp parallel for schedule(dynamic, chunk_size)
    // #pragma omp parallel for schedule(guided)
    // #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < mpi_chunk_size; i += 2) {
            int idx1 = indices[i];
            int y1 = idx1 / width;
            int x1 = idx1 % width;
            double x0_1 = left + x1 * dx; 
            double y0_1 = lower + y1 * dy;

            // Pixel 2: (x2, y2)
            int idx2 = (i+1 >= mpi_chunk_size) ? 0 : indices[i+1]; // Check bounds
            int y2 = idx2 / width;
            int x2 = idx2 % width;
            double x0_2 = left + x2 * dx;
            double y0_2 = lower + y2 * dy;


            __m128d y0v = _mm_set_pd(y0_2, y0_1); // reversed because first one is high 64 bits
            __m128d x0v = _mm_set_pd(x0_2, x0_1);

            __m128d x = _mm_setzero_pd();
            __m128d y = _mm_setzero_pd();
            __m128d repeats_pd = _mm_setzero_pd();  // doubles; convert back to int at the end
            // __m128i repeats_i = _mm_setzero_si128();
            __m128d active_mask_pd = _mm_set1_pd(1.0);

            for (int k = 0; k < iters; ++k) { // iterate up to iters since multiple lanes
                // x^2, y^2, xy
                __m128d xx = _mm_mul_pd(x, x);
                __m128d yy = _mm_mul_pd(y, y);
                __m128d xy = _mm_mul_pd(x, y);
                
                // __m128d x_tmp = _mm_add_pd(_mm_sub_pd(xx, yy), x0v); // x^2 - y^2 + x0
                x = _mm_add_pd(_mm_sub_pd(xx, yy), x0v); // x^2 - y^2 + x0
                y = _mm_add_pd(_mm_add_pd(xy, xy), y0v);  // 2*xy + y0

                // __m128d len2   = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
                active_mask_pd = _mm_cmplt_pd(_mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), four); // check if length_squared < 4
                
                // If no lanes active, break
                if (_mm_movemask_pd(active_mask_pd) == 0) {
                    break;
                }

                // repeats += active ? 1 : 0
                // __m128d inc = _mm_and_pd(active_mask_pd, one);
                repeats_pd = _mm_add_pd(repeats_pd, _mm_and_pd(active_mask_pd, one)); // results of two pixels

            }

            repeats_pd = _mm_add_pd(repeats_pd, _mm_andnot_pd(active_mask_pd, one)); // invert because for lanes become inactive before last iteration, we didn't count the last iteration
            __m128i repeats_i = _mm_cvtpd_epi32(repeats_pd); // convert back to int

            // Store two pixels: note _mm_cvtpd_epi32 packs to low 2 ints of __m128i
            int tmp[2];
            _mm_storel_epi64((__m128i*)tmp, repeats_i);

            rank_image[i] = tmp[0];

            if (i+1 < mpi_chunk_size){
                rank_image[i+1] = tmp[1];
            }
        // }
        
    }

    int *image = NULL;

    if (rank == 0) {
        image = (int*)malloc(width * height * sizeof(int));
        MPI_Request *reqs = (MPI_Request*) malloc(size * sizeof(*reqs));
        // receive each rank's data
        for (int r = 0; r < size; ++r) {
            int counts = base_chunk + (r < rem ? 1 : 0);

            MPI_Datatype vec;
            MPI_Type_vector(counts, 1, size, MPI_INT, &vec); // stride=# of ranks
            MPI_Type_commit(&vec);

            // Place pixels to corresponding indices in image
            MPI_Irecv(&image[r], 1, vec, r, 0, MPI_COMM_WORLD, &reqs[r]);
            // MPI_Recv(&image[r], 1, vec, r, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            MPI_Type_free(&vec);
        }

        if (mpi_chunk_size > 0)
            MPI_Send(rank_image, (int)mpi_chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD);

        MPI_Waitall(size, reqs, MPI_STATUSES_IGNORE);
        free(reqs);

    } else {
        if (mpi_chunk_size > 0)
            MPI_Send(rank_image, (int)mpi_chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    free(rank_image);
    
	MPI_Finalize();
        
	if (rank == 0) {
        // free(counts); free(displs);
        /* draw and cleanup */
        write_png(filename, iters, width, height, image);
        free(image);
    }
    
}
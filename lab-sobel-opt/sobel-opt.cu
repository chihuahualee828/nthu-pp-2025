#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

// extern __shared__ unsigned char S[];
__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    // int tid = blockIdx.x * blockDim.x + threadIdx.x; // row-major, each thread process one row
    int tid = threadIdx.y * blockDim.y + threadIdx.x; // thread index within a block , row-major
    // int tid = threadIdx.x; // row-major
    // int idx = blockIdx.x * (blockDim.x * blockDim.y) + tid; // global index

    // Global coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = gy * width + gx;
    
    float val[Z][3]; // register cannot store array, it's put onto local memory(off SM) by compiler
    if (gx >= width || gy >= height) return;
    // if (idx >= height) return;

    // S[tid* channels+0]=s[idx* channels+0];
    // S[tid* channels+1]=s[idx* channels+1];
    // S[tid* channels+2]=s[idx* channels+2];

    __shared__ unsigned char S[3][(32 + 2 * xBound)][(32 + 2 * yBound)];
    // Each thread loads one or more pixels until all 1296 are loaded
    for (int s_idx = tid; s_idx < (blockDim.x + 2 * xBound)*(blockDim.y + 2 * yBound); s_idx += blockDim.x * blockDim.y) {

        int s_y = s_idx / (blockDim.x + 2 * xBound);
        int s_x = s_idx % (blockDim.x + 2 * xBound);
        
        // Calculate corresponding global memory coordinate
        int g_x = (blockIdx.x * blockDim.x - xBound) + s_x;
        int g_y = (blockIdx.y * blockDim.y - yBound) + s_y;

        if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height) {
            
            int g_idx_1d = g_y * width + g_x;
            
            S[0][s_y][s_x] = s[channels * g_idx_1d + 0]; 
            S[1][s_y][s_x] = s[channels * g_idx_1d + 1];
            S[2][s_y][s_x] = s[channels * g_idx_1d + 2];

        } else {
            // Out-of-bounds: pad with 0
            S[0][s_y][s_x] = 0;
            S[1][s_y][s_x] = 0;
            S[2][s_y][s_x] = 0;
        }
    }
    __syncthreads();


    // int y = tid;
    // for (int x = 0; x < width; ++x) {
    // for (int j = tid; j < blockDim.x * blockDim.y; j+= blockDim.x * blockDim.y) {
        // printf("Processing pixel (%d, %d) (%d, %d) in block (%d, %d)\n", threadIdx.x, threadIdx.y, gx, gy, blockIdx.x, blockIdx.y);
        /* Z axis of mask */
    // for (int i = 0; i < Z; ++i) {

        // val[i][2] = 0.;
        // val[i][1] = 0.;
        // val[i][0] = 0.;

        // from local mem to register
        val[0][2] = 0.; 
        val[0][1] = 0.;
        val[0][0] = 0.;
        val[1][2] = 0.;
        val[1][1] = 0.;
        val[1][0] = 0.;

        /* Y and X axis of mask */
        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                // if (bound_check(gx + u, 0, width) && bound_check(gy + v, 0, height)) {
                    // const unsigned char R = S[channels * ((blockDim.y) * (threadIdx.y + v ) + (threadIdx.x + u )) + 2];
                    // const unsigned char G = S[channels * ((blockDim.y) * (threadIdx.y + v ) + (threadIdx.x + u )) + 1];
                    // const unsigned char B = S[channels * ((blockDim.y) * (threadIdx.y + v ) + (threadIdx.x + u )) + 0];
                    // const unsigned char R = S[channels * ((blockDim.y+2*yBound) * (threadIdx.y + v + yBound) + (threadIdx.x + u + xBound)) + 2];
                    // const unsigned char G = S[channels * ((blockDim.y+2*yBound) * (threadIdx.y + v + yBound) + (threadIdx.x + u + xBound)) + 1];
                    // const unsigned char B = S[channels * ((blockDim.y+2*yBound) * (threadIdx.y + v + yBound) + (threadIdx.x + u + xBound)) + 0];
                    const unsigned char R = S[2][threadIdx.y + v + yBound][threadIdx.x + u + xBound];
                    const unsigned char G = S[1][threadIdx.y + v + yBound][threadIdx.x + u + xBound];
                    const unsigned char B = S[0][threadIdx.y + v + yBound][threadIdx.x + u + xBound];
                    // const unsigned char R = s[channels * (width * (gy + v) + (gx + u)) + 2];
                    // const unsigned char G = s[channels * (width * (gy + v) + (gx + u)) + 1];
                    // const unsigned char B = s[channels * (width * (gy + v) + (gx + u)) + 0];

                    val[0][2] += R * mask[0][u + xBound][v + yBound];
                    val[0][1] += G * mask[0][u + xBound][v + yBound];
                    val[0][0] += B * mask[0][u + xBound][v + yBound];

                    val[1][2] += R * mask[1][u + xBound][v + yBound];
                    val[1][1] += G * mask[1][u + xBound][v + yBound];
                    val[1][0] += B * mask[1][u + xBound][v + yBound];
                // }
            }
        }
    // }
    float totalR = 0.;
    float totalG = 0.;
    float totalB = 0.;
    for (int i = 0; i < Z; ++i) {
        totalR += val[i][2] * val[i][2];
        totalG += val[i][1] * val[i][1];
        totalB += val[i][0] * val[i][0];
    }
    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
    const unsigned char cR = (totalR > 255.) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.) ? 255 : totalB;
    t[channels * idx + 2] = cR;
    t[channels * idx + 1] = cG;
    t[channels * idx + 0] = cB;
    // }
    // printf("Thread %d in block %d finish %d\n", threadIdx.x, blockIdx.x, pixel);
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }
    
    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // decide to use how many blocks and threads
    const int size = 32;
    dim3 threadsPerBlock(size, size);
    dim3 numBlocks((width  + size-1) / size,
               (height + size-1) / size);
    // launch cuda kernel
    // sobel<<<numBlocks, threadsPerBlock, (size+2*xBound) * (size+2*yBound) * channels * sizeof(unsigned char)>>>(dsrc, ddst, height, width, channels);
    sobel<<<numBlocks, threadsPerBlock>>>(dsrc, ddst, height, width, channels);
    // sobel<<<numBlocks, threadsPerBlock, size * size * channels * sizeof(unsigned char)>>>(dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}

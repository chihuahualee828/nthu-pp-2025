#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* =========== utility function =========== */
#define DEV_NO 0
cudaDeviceProp prop;

#define BLOCK_FACTOR 78 // max shared mem is 49152
#define THREAD_PER_BLOCK_X 39 
#define THREAD_PER_BLOCK_Y 13 
#define MAX2(a,b) ((a) > (b) ? (a) : (b))

const int B = BLOCK_FACTOR;
const int INF = ((1 << 30) - 1);

int n, m;
FILE* file;

void input(char* infile) {
    file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n=%d, m=%d\n", n, m);

    // return file;

}

void output(const char* outFileName, int* h_Dist) {
    FILE* outfile = fopen(outFileName, "wb");
    size_t total = (size_t)n * n;
    fwrite(h_Dist, sizeof(int), total, outfile);
    fclose(outfile);
}


__global__ void init_dist(int *s, int n, int start_row, int end_row) {

    int b_i = blockIdx.y;   // tile row index
    int b_j = blockIdx.x;   // tile column index

    if (b_i < start_row || b_i >= end_row )
        return;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // if (gy >= n_padded || gx >= n_padded) return;

    int block_internal_start_x = b_i * BLOCK_FACTOR;
    int block_internal_start_y = b_j * BLOCK_FACTOR;

    int gx = block_internal_start_y + tx;
    int gy = block_internal_start_x + ty;
    
    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
            s[(i * THREAD_PER_BLOCK_Y + gy) * n + (j * THREAD_PER_BLOCK_X + gx)] =  ((i * THREAD_PER_BLOCK_Y + gy) == (j * THREAD_PER_BLOCK_X + gx)) ? 0 : INF;
        }
    }

}

__global__ void dist_fill(int* s, int *pairs, int n, int m) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int u = pairs[3 * idx];
        int v = pairs[3 * idx + 1];
        int w = pairs[3 * idx + 2];
        s[u * n + v] = w;
    }

}


__global__ void phase1_cal(int *s, int n, int B, int r) {

    // index within a tile(B*B)
    // int gx = blockIdx.x * blockDim.x + threadIdx.x; 
    // int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (tx >= BLOCK_FACTOR || ty >= BLOCK_FACTOR) return;


    int b_i = r;   // tile row index
    int b_j = r;   // tile column index


    // __shared__ int S[32*32]; // shared memory for a block(a tile can be processed by multiple blocks)
    __shared__ int S[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile

    int stride = blockDim.y;
    
    int block_internal_start_y = b_i * B;
    int block_internal_start_x = b_j * B;

    int gx = block_internal_start_x + tx;
    int gy = block_internal_start_y + ty;

    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {

            S[i*THREAD_PER_BLOCK_Y+ty][j*THREAD_PER_BLOCK_X+tx] = s[(i * THREAD_PER_BLOCK_Y + gy) * n + (j * THREAD_PER_BLOCK_X + gx)];
        }
    }
    __syncthreads();
    

    for (int k = 0; k < BLOCK_FACTOR; ++k) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
            #pragma unroll
            for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
                S[i*THREAD_PER_BLOCK_Y+ty][j*THREAD_PER_BLOCK_X+tx] = min(S[i*THREAD_PER_BLOCK_Y+ty][k] + S[k][j*THREAD_PER_BLOCK_X+tx], S[i*THREAD_PER_BLOCK_Y+ty][j*THREAD_PER_BLOCK_X+tx]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
            s[(gy + i* THREAD_PER_BLOCK_Y)*n + (gx + j* THREAD_PER_BLOCK_X)] = S[i*THREAD_PER_BLOCK_Y+ty][j*THREAD_PER_BLOCK_X+tx];
        }
    }


}

__global__ void phase2_cal(int *s, int n, int rounds, int r,int tid, int start_row, int end_row) {
    
    int b_i = blockIdx.y;   // tile row index
    int b_j = blockIdx.x;   // tile column index

    bool is_pivot_row = (b_i == r && b_j != r);
    bool is_pivot_col = (b_j == r && b_i != r);

    
    // This block does no work if it's not in the cross
    if (!is_pivot_row && !is_pivot_col)
        return;

    // if (b_i < start_row || b_i >= end_row)
    //     return;

    // // tid(GPU id): allocate blocks based on GPU 
    // if (is_pivot_row && tid % 2 != 0) // GPU 0 handles pivot row tiles
    //     return;

    // if (is_pivot_col && tid % 2 != 1) // GPU 1 handles pivot column tiles
    //     return;

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    
    if (tx >= BLOCK_FACTOR || ty >= BLOCK_FACTOR) return;

    // if (tid == 1 && r == 0 && ty ==0 && tx ==0) {
    //     printf("GPU%d process (%d, %d) \n", tid, b_i, b_j);
    // }

    __shared__ int S[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile
    __shared__ int S_pivot[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile's pivot block
    
    
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;

    int gx = block_internal_start_y + tx;
    int gy = block_internal_start_x + ty;
    

    int stride = blockDim.y;
    // if (gridDim.z > 1) {
    //     stride = B;
    // }

   #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) { 

                S[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx] = s[(i * THREAD_PER_BLOCK_Y + gy) * n + (j * THREAD_PER_BLOCK_X + gx)];
                S_pivot[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx] = s[(r * B + i * THREAD_PER_BLOCK_Y + ty) * n + (r * B + j * THREAD_PER_BLOCK_X + tx)];
            
        }
    }
    __syncthreads();
    
    

    #pragma unroll
    for (int k = 0; k < BLOCK_FACTOR; ++k) {
        #pragma unroll
        for (int i = 0; i <  MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
            #pragma unroll
            for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
                if (is_pivot_col) {
                    // S[i][j] = min(S[i][j], S[i][k] + S_pivot[k][j]);
                    S[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx] = min(S[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx], S[i * THREAD_PER_BLOCK_Y + ty][k] + S_pivot[k][j * THREAD_PER_BLOCK_X + tx]);
                }else{
                    S[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx] = min(S[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx], S_pivot[i * THREAD_PER_BLOCK_Y + ty][k] + S[k][j * THREAD_PER_BLOCK_X + tx]);
                }
            }
        }

    }


    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
            s[(i * THREAD_PER_BLOCK_Y + gy) * n + (j * THREAD_PER_BLOCK_X + gx)] = S[i * THREAD_PER_BLOCK_Y + ty][j * THREAD_PER_BLOCK_X + tx];
        }
    }
    

  
}

__global__ void phase3_cal(int *s, int n, int rounds, int r, int tid, int start_row, int end_row) {

    int b_i = blockIdx.y;   // tile row index
    int b_j = blockIdx.x;   // tile column index


    if (b_i == r || b_j == r)
        return;

    if (b_i < start_row || b_i >= end_row )
        return;
    
    // // tid(GPU id): allocate blocks based on GPU 
    // if (b_i%2 != 0 && tid % 2 != 0) // GPU 0 handles even rows
    //     return;

    // if (b_i%2 == 0 && tid % 2 != 1) // GPU 1 handles odd rows
    //     return;


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (tx >= B || ty >= B) return;

    
    int block_internal_start_x = b_i * B;
    int block_internal_end_x = min((b_i + 1) * B, n);
    int block_internal_start_y = b_j * B;
    int block_internal_end_y = min((b_j + 1) * B, n);

    int gx = block_internal_start_y + tx;
    int gy = block_internal_start_x + ty;
    
    
    // // __shared__ int S[32*32]; // shared memory for a block(a tile can be processed by multiple blocks)
    __shared__ int S_pivot_row[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile's pivot row tile
    __shared__ int S_pivot_col[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile's pivot col tile
    // __shared__ int S[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile
    
    

    int stride = blockDim.y;
    // Check if z dimension is used (z > 1 means we're using multiple blocks per tile)
    // if (gridDim.z > 1) {
    //     stride = B;
    // }

    // --- pivot row tile (r, bj) ---
    int pivot_row_start_x = r * B;                 // rows
    int pivot_row_end_x   = min((r + 1) * B, n);
    int pivot_row_start_y = b_j * B;                // cols
    int pivot_row_end_y   = min((b_j + 1) * B, n);

    // // --- pivot column tile (bi, r) ---
    int pivot_col_start_x = b_i * B;                // rows
    int pivot_col_end_x   = min((b_i + 1) * B, n);
    int pivot_col_start_y = r * B;                 // cols
    int pivot_col_end_y   = min((r + 1) * B, n);


    int local_vals[MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1)][MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1)];

    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); i += 1) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); j += 1) {
            // if ((pivot_row_start_x + i) < n && (j+ pivot_row_start_y) < n)
                S_pivot_row[i*THREAD_PER_BLOCK_Y+ty][j*THREAD_PER_BLOCK_X+tx] = s[(pivot_row_start_x + i*THREAD_PER_BLOCK_Y + ty) * n + (j*THREAD_PER_BLOCK_X+tx+ pivot_row_start_y)];
            // if ((pivot_col_start_x + i) < n && (j+ pivot_col_start_y) < n)
                S_pivot_col[i*THREAD_PER_BLOCK_Y+ty][j*THREAD_PER_BLOCK_X+tx] = s[(pivot_col_start_x + i*THREAD_PER_BLOCK_Y + ty) * n + (j*THREAD_PER_BLOCK_X+tx+ pivot_col_start_y)];
                local_vals[i][j] = s[(i*THREAD_PER_BLOCK_Y + gy) * n + (j*THREAD_PER_BLOCK_X+ gx)];
        }
    }

    __syncthreads();
    

    #pragma unroll
    for (int k = 0; k < BLOCK_FACTOR; ++k) {
        #pragma unroll
        for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
            #pragma unroll
            for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {

                int sum = S_pivot_col[i*THREAD_PER_BLOCK_Y+ty][k] + S_pivot_row[k][j*THREAD_PER_BLOCK_X+tx];
                local_vals[i][j] = min(local_vals[i][j], sum);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
            s[(i * THREAD_PER_BLOCK_Y + gy) * n + (j * THREAD_PER_BLOCK_X + gx)] = local_vals[i][j];
        }
    }

        
}



int main(int argc, char* argv[]) {


    // int *h_Dist, *h_Dist_padded;
    
    // h_Dist = input(argv[1]);
    // h_Dist_padded = input(argv[1]);
    input(argv[1]);
    int rounds = (n + B - 1) / B;
    // pad the matrix size to multiple of B to avoid boundary checks
    int n_padded = rounds * B;
    
    int *h_Dist_padded;
    h_Dist_padded = (int*)malloc(n_padded * n_padded * sizeof(int));
    // pinned memory for faster transfer
    // cudaMallocHost(&h_Dist_padded, n_padded * n_padded * sizeof(int));


    
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    int *h_pair, *s_pair[numDevices];
    // h_pair = (int*)malloc(3 * m * sizeof(int));
    cudaMallocHost(&h_pair, 3 * m * sizeof(int));
    fread(h_pair, sizeof(int), 3 * m, file);



    
    // cudaMemcpy(d_Dist, h_Dist_padded, n_padded * n_padded * sizeof(int), cudaMemcpyHostToDevice);
    

    int *d_Dist[numDevices];
    // omp_set_num_threads(numDevices);
    #pragma omp parallel num_threads(numDevices)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid); // Bind CPU thread to specific GPU

        cudaGetDeviceProperties(&prop, tid);
        printf("GPU%d maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", tid, prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

        // Calculate Data Slice
        int rows_per_gpu = n_padded / numDevices;
        int start_row_global = tid * rows_per_gpu;     // Global index where this GPU starts
        int end_row_global   = start_row_global + rows_per_gpu;

        int start_row = (rounds/numDevices)*tid;
        int end_row = (rounds * (tid + 1)) / numDevices;


        // cudaMalloc(&d_Dist, n_padded * n_padded * sizeof(int));
        cudaSetDevice(tid);
        // if (tid == 0) {
        //     cudaMalloc(&d_Dist[tid], 2*n_padded*n_padded*sizeof(int));
        // } else {
        //     cudaMalloc(&d_Dist[tid], n_padded*n_padded*sizeof(int));
        // }
        cudaMalloc(&d_Dist[tid], n_padded*n_padded*sizeof(int));
        // cudaMemcpy(d_Dist[tid], h_Dist_padded, n_padded*n_padded*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&s_pair[tid], 3 * m * sizeof(int));
        cudaMemcpy(s_pair[tid], h_pair, 3 * m * sizeof(int), cudaMemcpyHostToDevice);
        

        // cut data in half by rows
        #pragma omp barrier
        // cudaMemcpy(d_Dist[tid]+(start_row*BLOCK_FACTOR*n_padded), h_Dist_padded+(start_row*BLOCK_FACTOR*n_padded), (end_row-start_row)*BLOCK_FACTOR*n_padded*sizeof(int), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y);
        init_dist<<<dim3(rounds, rounds), threadsPerBlock>>>(d_Dist[tid], n_padded, start_row, end_row);
        dist_fill<<<(m+1024-1)/1024, 1024>>>(d_Dist[tid], s_pair[tid], n_padded, m);

        cudaFree(s_pair[tid]);
        
        printf("GPU %d process start %d to %d \n", tid, start_row, end_row);
        for (int r = 0; r < rounds; ++r) {
            // copy specific row to other GPU, r*BLOCK_FACTOR*n_padded: row start index
            cudaMemcpyPeer(d_Dist[!tid]+(r*BLOCK_FACTOR*n_padded), !tid, d_Dist[tid]+(r*BLOCK_FACTOR*n_padded), tid, (r>=start_row && r<end_row)*BLOCK_FACTOR*n_padded*sizeof(int));
            #pragma omp barrier

            phase1_cal<<<1, threadsPerBlock>>>(d_Dist[tid], n_padded, B, r); 

            phase2_cal<<<dim3(rounds, rounds), threadsPerBlock>>>(d_Dist[tid], n_padded, rounds, r, tid, start_row ,end_row); 

            phase3_cal<<<dim3(rounds, rounds), threadsPerBlock>>>(d_Dist[tid], n_padded, rounds, r, tid, start_row ,end_row);

        }

        // cudaFree(d_Dist);
        // cudaMemcpy(h_Dist_padded+(start_row*BLOCK_FACTOR*n_padded), d_Dist[tid]+(start_row*BLOCK_FACTOR*n_padded), (end_row-start_row)*BLOCK_FACTOR*n_padded*sizeof(int), cudaMemcpyDeviceToHost);

        int rows_to_copy = (end_row - start_row) * BLOCK_FACTOR;

        if (end_row * BLOCK_FACTOR > n) {
            rows_to_copy = max(0, n - start_row * BLOCK_FACTOR);
        }

        cudaMemcpy2D(
            h_Dist_padded + start_row * BLOCK_FACTOR * n,        // correct place in global host matrix
            n * sizeof(int),                             // unpadded pitch
            d_Dist[tid] + start_row * BLOCK_FACTOR * n_padded,   // correct place in this GPU's padded matrix
            n_padded * sizeof(int),                      // padded pitch
            n * sizeof(int),                             // only n columns (unpad)
            rows_to_copy,                                // only this GPU's rows
            cudaMemcpyDeviceToHost
        );

        cudaFree(d_Dist[tid]);
        // cudaFree(d_RowBuffer);
    }

    cudaFreeHost(h_pair);

    // for(int i=0; i<n; ++i) {
    //     for(int j=0; j<n; ++j) {
    //         h_Dist_padded[i * n + j] = h_Dist_padded[i * n_padded + j];
    //     }
    // }

    output(argv[2], h_Dist_padded);
    // cudaFree(d_Dist);
    // cudaFreeHost(h_Dist_padded);
    free(h_Dist_padded);
    
    return 0;
}


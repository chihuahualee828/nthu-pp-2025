#include <stdio.h>
#include <stdlib.h>

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
    // printf("n=%d, m=%d\n", n, m);

    // return file;

}

void output(const char* outFileName, int* h_Dist) {
    FILE* outfile = fopen(outFileName, "wb");
    size_t total = (size_t)n * n;
    fwrite(h_Dist, sizeof(int), total, outfile);
    fclose(outfile);
}


__global__ void init_dist(int *s, int n) {

    int b_i = blockIdx.y;   // tile row index
    int b_j = blockIdx.x;   // tile column index

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int tid = ty * blockDim.x + tx;

    if (tx >= BLOCK_FACTOR || ty >= BLOCK_FACTOR) return;

    int block_internal_start_y = b_i * BLOCK_FACTOR;
    int block_internal_start_x = b_j * BLOCK_FACTOR;

    int gx = block_internal_start_x + tx;
    int gy = block_internal_start_y + ty;

    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
            int gi = i * THREAD_PER_BLOCK_Y + gy;
            int gj = j * THREAD_PER_BLOCK_X + gx;
            s[gi * n + gj] =  (gi == gj) ? 0 : INF;
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
    int tid = ty * blockDim.x + tx;
    if (tx >= BLOCK_FACTOR || ty >= BLOCK_FACTOR) return;


    int b_i = r;   // tile row index
    int b_j = r;   // tile column index


    // __shared__ int S[32*32]; // shared memory for a block(a tile can be processed by multiple blocks)
    __shared__ int S[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile
    // __shared__ int S[BLOCK_FACTOR*BLOCK_FACTOR]; // shared memory store current tile

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
                int gi = i * THREAD_PER_BLOCK_Y + ty;
                int gj = j * THREAD_PER_BLOCK_X + tx;
                S[gi][gj] = min(S[gi][k] + S[k][gj], S[gi][gj]);
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


__global__ void phase2_cal(int *s, int n, int rounds, int r) {
    

    int b_i = blockIdx.y;   // tile row index
    int b_j = blockIdx.x;   // tile column index


    bool is_pivot_row = (b_i == r && b_j != r);
    bool is_pivot_col = (b_j == r && b_i != r);

    if (!is_pivot_row && !is_pivot_col)
        return;

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    
    int tid = ty * blockDim.x + tx;

    if (tx >= BLOCK_FACTOR || ty >= BLOCK_FACTOR) return;

    __shared__ int S[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile
    __shared__ int S_pivot[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile's pivot block
    
    
    int block_internal_start_y = b_i * B;
    int block_internal_start_x = b_j * B;

    int gx = block_internal_start_x + tx;
    int gy = block_internal_start_y + ty;
    

    int stride = blockDim.y;
    // if (gridDim.z > 1) {
    //     stride = B;
    // }
   #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) { 
            int gi = i * THREAD_PER_BLOCK_Y + ty;
            int gj = j * THREAD_PER_BLOCK_X + tx;

                S[gi][gj] = s[(i * THREAD_PER_BLOCK_Y + gy) * n + (j * THREAD_PER_BLOCK_X + gx)];
                S_pivot[gi][gj] = s[(r * B + i * THREAD_PER_BLOCK_Y + ty) * n + (r * B + j * THREAD_PER_BLOCK_X + tx)];
            
        }
    }

    __syncthreads();
    
    
    
    #pragma unroll
    for (int k = 0; k < BLOCK_FACTOR; ++k) {
        #pragma unroll
        for (int i = 0; i <  MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
            #pragma unroll
            for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
                int gi = i * THREAD_PER_BLOCK_Y + ty;
                int gj = j * THREAD_PER_BLOCK_X + tx;
                if (is_pivot_col) {
                    // S[i][j] = min(S[i][j], S[i][k] + S_pivot[k][j]);
                    S[gi][gj] = min(S[gi][gj], S[gi][k] + S_pivot[k][gj]);
                }else{
                    S[gi][gj] = min(S[gi][gj], S_pivot[gi][k] + S[k][gj]);
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

__global__ void phase3_cal(int *s, int n, int rounds, int r) {

    int b_i = blockIdx.y;   // tile row index
    int b_j = blockIdx.x;   // tile column index

    if (b_i == r || b_j == r)
        return;


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int tid = ty * blockDim.x + tx;

    if (tx >= BLOCK_FACTOR || ty >= BLOCK_FACTOR) return;
    
    int block_internal_start_y = b_i * B;
    int block_internal_start_x = b_j * B;

    int gx = block_internal_start_x + tx;
    int gy = block_internal_start_y + ty;
    

    __shared__ int S_pivot_row[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile's pivot row tile
    __shared__ int S_pivot_col[BLOCK_FACTOR][BLOCK_FACTOR]; // shared memory store current tile's pivot col tile
    

    // --- pivot row tile (r, bj) ---
    int pivot_row_start_x = r * B;                 // rows
    int pivot_row_start_y = b_j * B;                // cols

    // // --- pivot column tile (bi, r) ---
    int pivot_col_start_x = b_i * B;                // rows
    int pivot_col_start_y = r * B;                 // cols

    int local_vals[MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1)][MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1)];

    #pragma unroll
    for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); i += 1) {
        #pragma unroll
        for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); j += 1) {
            // if ((pivot_row_start_x + i) < n && (j+ pivot_row_start_y) < n)
            int gi = i * THREAD_PER_BLOCK_Y + ty;
            int gj = j * THREAD_PER_BLOCK_X + tx;
            S_pivot_row[gi][gj] = s[(pivot_row_start_x + gi) * n + (gj + pivot_row_start_y)];
        // if ((pivot_col_start_x + i) < n && (j+ pivot_col_start_y) < n)
            S_pivot_col[gi][gj] = s[(pivot_col_start_x + gi) * n + (gj + pivot_col_start_y)];
            local_vals[i][j] = s[(i*THREAD_PER_BLOCK_Y + gy) * n + (j*THREAD_PER_BLOCK_X + gx)];
        }
    }

    
    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < BLOCK_FACTOR; ++k) {
        #pragma unroll
        for (int i = 0; i < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_Y, 1); ++i) {
            #pragma unroll
            for (int j = 0; j < MAX2(BLOCK_FACTOR/THREAD_PER_BLOCK_X, 1); ++j) {
                int gi = i * THREAD_PER_BLOCK_Y + ty;
                int gj = j * THREAD_PER_BLOCK_X + tx;
                int sum = S_pivot_col[gi][k] + S_pivot_row[k][gj];
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

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // // start timing
    // cudaEventRecord(start);
    // float ms = 0;
    
    
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

    int *h_pair, *s_pair;
    // h_pair = (int*)malloc(3 * m * sizeof(int));
    cudaMallocHost(&h_pair, 3 * m * sizeof(int));
    fread(h_pair, sizeof(int), 3 * m, file);


    int *d_Dist;
    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&d_Dist, n_padded * n_padded * sizeof(int));
    // cudaMemcpy(d_Dist, h_Dist_padded, n_padded * n_padded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&s_pair, 3 * m * sizeof(int));
    cudaMemcpy(s_pair, h_pair, 3 * m * sizeof(int), cudaMemcpyHostToDevice);

    cudaFreeHost(h_pair);
    // free(h_pair);



    dim3 threadsPerBlock(THREAD_PER_BLOCK_X, THREAD_PER_BLOCK_Y);
    // dim3 blocksPerGrid((B + threadsPerBlock.x - 1) / threadsPerBlock.x, (B + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    init_dist<<<dim3(rounds, rounds), threadsPerBlock>>>(d_Dist, n_padded);
    dist_fill<<<(m+1024-1)/1024, 1024>>>(d_Dist, s_pair, n_padded, m);
    cudaFree(s_pair);

    for (int r = 0; r < rounds; ++r) {

        phase1_cal<<<1, threadsPerBlock>>>(d_Dist, n_padded, B, r); 

        phase2_cal<<<dim3(rounds, rounds), threadsPerBlock>>>(d_Dist, n_padded, rounds, r); 

        phase3_cal<<<dim3(rounds, rounds), threadsPerBlock>>>(d_Dist, n_padded, rounds, r);
    }

    cudaMemcpy2D(h_Dist_padded, n*sizeof(int), d_Dist, n_padded*sizeof(int), n*sizeof(int), n, cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);
    
    output(argv[2], h_Dist_padded);
    free(h_Dist_padded);
    // cudaFreeHost(h_Dist_padded);
    
    return 0;
}


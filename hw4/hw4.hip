#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>


#define Bc 16 
#define Br 16 


void input(char *input_filename);
void output(char *output_filename);
__global__ void FusedKernel(float *q, float *k, float *v, float *o, int N, int d, float scale);

__device__ float _max(float a, float b) { return a > b ? a : b; }
__device__ float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;
float scale;

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);
    
    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    #pragma unroll
    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}



int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);
    
    scale = 1.0 / sqrtf((float)d);

    // double start, end;
    // start = getTimeStamp();

    // entire batches
    float *d_K, *d_V, *d_Q, *d_O;
    
    hipMalloc(&d_K, B * N * d * sizeof(float));
    hipMalloc(&d_V, B * N * d * sizeof(float));
    hipMalloc(&d_Q, B * N * d * sizeof(float));
    hipMalloc(&d_O, B * N * d * sizeof(float));
    
    hipMemcpy(d_K, K, B * N * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V, B * N * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_Q, Q, B * N * d * sizeof(float), hipMemcpyHostToDevice);
    // hipMemcpy(d_O, O, B * N * d * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_O, 0, B * N * d * sizeof(float));

    dim3 gridDim(N/Br, B); // N/Br query tiles per batch, B batches, each block each tile in one batch
    int shared_mem_size = (Br *d + Bc *(d+1) + Bc * d) * sizeof(float); // q,o,k,v tiles size
    // FusedKernel<<<gridDim, Bc, shared_mem_size>>>(d_Q, d_K, d_V, d_O, N, d, scale); // each thread each KV tile
    FusedKernel<<<gridDim, dim3(Bc, Br), shared_mem_size>>>(d_Q, d_K, d_V, d_O, N, d, scale); 

    hipDeviceSynchronize();

    // write back to host
    hipMemcpy(O, d_O, B * N * d * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);


    // end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    // printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}


__global__ void FusedKernel(float *q, float *k, float *v, float *o, int N, int dim, float scale) {
    
    extern __shared__ float smem[]; // (Br*d + Bc*d + Bc*d)*sizeof(float) for q, k, v tiles

    float *q_i = smem; // Br * d
    float *k_j = &smem[Br * dim]; // Bc * d
    float *v_j = &smem[Br * dim + Bc * (dim+1)]; // Bc * d


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int batch_idx = blockIdx.y;
    int tile_idx = blockIdx.x; // Q tile index: N/Br
    
    int global_tile_start = batch_idx * N + tile_idx * Br;

    int tid = ty * blockDim.x + tx; // coalesced access
    int total_threads = blockDim.x * blockDim.y;

    // 2D block
    for (int idx = tid; idx < Br * dim; idx += total_threads) {
        q_i[idx] = q[(global_tile_start) * dim + idx]; 
    }
    
    float m_val = -FLT_MAX;
    float l_val = 0.0f;
    
    __syncthreads();

    // loop over each k,v tile
    for (int kv_tile_idx=0; kv_tile_idx < N/Bc; kv_tile_idx++) {
        // each thread load a k,v tile in a batch into shared mem
        int kv_offset = batch_idx * N + kv_tile_idx * Bc;

        for (int r = ty; r < Bc; r += blockDim.y) {
            for (int c = tx; c < dim; c += blockDim.x) {
                k_j[r * (dim + 1) + c] = k[(kv_offset + r) * dim + c];
                v_j[r * dim + c]       = v[(kv_offset + r) * dim + c];
            }
        }
        
        __syncthreads();

        float pij = 0.0f;

        
        // 2. Compute QK (sij)
        float sij = 0.0f;
        for (int d = 0; d < dim; d++) {
            sij += q_i[ty * dim + d] * k_j[tx * (dim+1) + d];
        }
        sij *= scale; 
        
        // 3. Row Max 
        float mij = sij;
        for (int offset = Bc/2; offset > 0; offset /= 2)
            mij = fmaxf(mij, __shfl_xor(mij, offset, Bc)); // butterfly reduction within warp, find max within a warp
        
        // 4. MinusMaxAndExp
        pij = expf(sij - mij);

        // 5. RowSum
        float lij = pij;
        for (int offset = Bc/2; offset > 0; offset /= 2)
            lij += __shfl_xor(lij, offset, Bc);

        // 6. UpdateMiLiOi
        float m_prev = m_val;
        float l_prev = l_val;
        
        float m_new = fmaxf(m_prev, mij);
        float l_new = expf(m_prev - m_new) * l_prev + expf(mij - m_new) * lij;
        
        m_val = m_new;
        l_val = l_new;

        float scale_O = (l_prev * expf(m_prev - m_new)); 
        float scale_PV = (expf(mij - m_new));
        
        // 7. Compute Output (O=PV) and Write to Global Memory
        for (int col = tx; col < dim; col += blockDim.x) {
            float pv_val = 0.0f;
            // Oi = PijVj, iterate over each of Bc
            for (int c = 0; c < Bc; c++) {
                float p_c = __shfl(pij, c, Bc);  // Pij of each tile
                pv_val += p_c * v_j[c * dim + col]; 
            }

            int o_idx = (global_tile_start + ty) * dim + col;
            float o_old = o[o_idx];
            o[o_idx] = (scale_O * o_old + scale_PV * pv_val) / l_new;
        }

        __syncthreads(); 
    }

}


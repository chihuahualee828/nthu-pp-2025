// odd_even_sort_seq.cpp
#include <utility>  // for std::swap
#include <algorithm>  // for std::max

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mpi.h>
using namespace std;



void sort_nlogn_mergesort(float* arr, size_t n) {
    if (n < 2) return;
    std::vector<float> buf(n);
    for (size_t width = 1; width < n; width <<= 1) {
        for (size_t i = 0; i < n; i += (width << 1)) {
            size_t mid = std::min(i + width, n);
            size_t hi  = std::min(i + (width << 1), n);

            // merge [i, mid) and [mid, hi) into buf
            size_t p = i, q = mid, k = i;
            while (p < mid && q < hi) {
                if (arr[p] <= arr[q]) buf[k++] = arr[p++];
                else                  buf[k++] = arr[q++];
            }
            while (p < mid) buf[k++] = arr[p++];
            while (q < hi)  buf[k++] = arr[q++];

            // copy back
            std::copy(buf.begin() + i, buf.begin() + hi, arr + i);
        }
    }
}

void radix_sort_floats(float* arr, size_t n) {
    if (n <= 1) return;

    // reinterpret floats as uint32_t for bitwise tricks
    std::vector<uint32_t> keys(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        std::memcpy(&bits, &arr[i], sizeof(bits));
        // flip bits so that float ordering == uint32 ordering
        if (bits & 0x80000000u) {      // negative
            bits = ~bits;
        } else {                       // positive
            bits ^= 0x80000000u;
        }
        keys[i] = bits;
    }

    std::vector<uint32_t> tmp(n);
    const int BITS = 32;
    const int RADIX = 8;       // process 8 bits per pass
    const int BUCKETS = 1 << RADIX;

    for (int shift = 0; shift < BITS; shift += RADIX) {
        size_t count[BUCKETS] = {0};

        // histogram
        for (size_t i = 0; i < n; i++) {
            count[(keys[i] >> shift) & (BUCKETS - 1)]++;
        }

        // prefix sum
        size_t sum = 0;
        for (int b = 0; b < BUCKETS; b++) {
            size_t c = count[b];
            count[b] = sum;
            sum += c;
        }

        // scatter into tmp
        for (size_t i = 0; i < n; i++) {
            int bucket = (keys[i] >> shift) & (BUCKETS - 1);
            tmp[count[bucket]++] = keys[i];
        }

        keys.swap(tmp);
    }

    // undo the transform back to floats
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = keys[i];
        if (bits & 0x80000000u) {      // originally positive
            bits ^= 0x80000000u;
        } else {                       // originally negative
            bits = ~bits;
        }
        std::memcpy(&arr[i], &bits, sizeof(bits));
    }
}



int main(int argc, char **argv)
{
    // if (argc != 4) {
    //     cerr << "Usage: " << argv[0] << " N INF_FILE OUT_FILE\n";
    //     return 1;
    // }

    size_t n = static_cast<size_t>(atoll(argv[1]));
    const char *const input_filename = argv[2],
               *const output_filename = argv[3];  
    
    int i, rank, size, namelen;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status stat;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Get_processor_name(name, &namelen);
    
    double tik, tok;

    tik = MPI_Wtime();

    MPI_File input_file, output_file;
    
    long long chunk = n / size;    // base chunk size
    int r = (int)(n % size);       // leftover: how many ranks get +1
    // printf("n = %zu, size = %d, chunk = %lld, r = %d\n", n, size, chunk, r);
    if (rank < r) {
        chunk += 1;                 // first r ranks get +1
    }

    // printf("rank = %d\n chunk = %lld\n", rank, chunk);
    int last_rank = size-1;
    if (n <= size) {
        last_rank = n-1;
    }

    // if (rank > last_rank) {
    //     MPI_Finalize();
    //     return 0;
    // }

    vector<float> data(chunk);

    
    MPI_Offset offset = (MPI_Offset)rank * (MPI_Offset)chunk * (MPI_Offset)sizeof(float); // byte offset, rank0: 0~chunk*4bytes
    if (rank >= r) {
        offset += (MPI_Offset)r * (MPI_Offset)sizeof(float); // first r ranks has +1 chunk
    }


    // printf("rank %d: offset = %lld\n", rank, (long long)offset);
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    // MPI_File_read_at(input_file, offset, data.data(), chunk, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_read_at_all(input_file, offset, data.data(), chunk, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_read_at(input_file, offset, data.data(), chunk, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // MPI_Info info;
    // MPI_Info_create(&info);
    // // Provide hints to the I/O system
    // MPI_Info_set(info, "access_style", "read_once");
    // MPI_Info_set(info, "collective_buffering", "true");
    // MPI_Info_set(info, "romio_ds_read", "disable");  // Lustre-specific
    // MPI_Info_set(info, "romio_lustre_co_ratio", "1");  // Lustre-specific
    // MPI_Info_set(info, "striping_factor", "8");  // Increase for large files
    // MPI_Info_set(info, "striping_unit", "8388608");  // 8MB stripes for Lustre

    // // Use the info object when opening the file
    // MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, info, &input_file);
    // MPI_File_read_at_all(input_file, offset, data.data(), chunk, MPI_FLOAT, MPI_STATUS_IGNORE);

    // // ... use the file ...
    // MPI_Info_free(&info);


    // // printf("rank %d got float: %f\n", rank, data[0]);
    bool global_swapped = true;
    // setvbuf(stdout, NULL, _IONBF, 0);
    int phase = 0;
    thread_local std::vector<float> recvBuf;
    static thread_local std::vector<float> tmp;
    if (chunk > 0) {
        tmp.resize(chunk);
        recvBuf.resize(chunk);
    }

    // // size_t n = arr.size();
    // tik = MPI_Wtime();
    while (global_swapped) { 
        bool local_swapped = false;
        // sort each rank array locally after swaps

        if (chunk > 1 && phase == 0){
            radix_sort_floats(data.data(), chunk); // n
        }
        if (chunk>0) {
            int peerChunk = chunk;
            // Even phase: (0,1), (2,3), ...; Odd phase: (1,2), (3,4), ... 
            if (rank < last_rank &&
                ((phase % 2 != 0 && rank % 2 != 0) ||
                (phase % 2 == 0 && rank % 2 == 0))) { //current rank is the former of the pair
                    if (rank < r && rank + 1 >= r){
                        peerChunk -= 1; // next rank has 1 less chunk
                        recvBuf.resize(peerChunk);
                    }
                    // printf("rank %d sending %d floats to rank %d which has %d\n", rank, chunk, rank + 1, peerChunk);
                    
                    // blocking, non-blocking: isend irecv
                    MPI_Sendrecv(data.data(), chunk, MPI_FLOAT, rank + 1, 0,
                            recvBuf.data(), peerChunk, MPI_FLOAT, rank + 1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // MPI_Sendrecv(&data[chunk - 1],   1, MPI_FLOAT, rank + 1, 0,
                    //             &recvBuf[0],1, MPI_FLOAT, rank + 1, 0,
                    //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    local_swapped = (data[chunk - 1] > recvBuf[0]);

                    // printf("rank %d local_swapped = %d\n", rank, local_swapped);
                    if (local_swapped) {
                            
                        // // std::merge(recvBuf.begin(), recvBuf.end(), data.begin(), data.end(), tmp.begin()); // merge sort

                        // // // keep first half (smaller elements), replace rank's data
                        // // std::copy(tmp.begin(), tmp.begin() + chunk, data.begin());

                        int i = 0, j = 0, k = 0;
                        while (k < chunk) {
                            if (j >= peerChunk || (i < chunk && data[i] <= recvBuf[j])){
                                tmp[k++] = data[i++];
                            } else {
                                tmp[k++] = recvBuf[j++];
                            }
                        }
                        std::memcpy(data.data(), tmp.data(), size_t(chunk) * sizeof(float)); // copy from tmp to current rank data

                    }
            } else if (rank > 0 &&
                ((phase % 2 != 0 && rank % 2 == 0) ||
                (phase % 2 == 0 && rank % 2 != 0))) { // current rank is the latter of the pair
                    if (rank >= r && rank - 1 < r){
                        peerChunk += 1; // previous rank has 1 more chunk
                        recvBuf.resize(peerChunk);
                    }
                    // printf("rank %d sending %d floats to rank %d which has %d\n", rank, chunk, rank - 1, peerChunk);

                    // tmp.resize(chunk + peerChunk);
                    // tmp.resize(chunk);

                    MPI_Sendrecv(data.data(), chunk, MPI_FLOAT, rank - 1, 0,
                            recvBuf.data(), peerChunk, MPI_FLOAT, rank -1, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // MPI_Sendrecv(&data[0], 1, MPI_FLOAT, rank - 1, 0,
                    //             &recvBuf[peerChunk-1],1, MPI_FLOAT, rank - 1, 0,
                    //             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                    local_swapped = (data[0] < recvBuf[peerChunk-1]);
                    // printf("rank %d local_swapped = %d\n", rank, local_swapped);
                    if (local_swapped) {
                        // // std::merge(data.begin(), data.end(), recvBuf.begin(), recvBuf.end(), tmp.begin()); // merge sort

                        // std::copy(tmp.end() - chunk, tmp.end(), data.begin());
                        int i = chunk - 1, j = peerChunk - 1, k = chunk - 1;
                        while (k >= 0) {
                            if (j < 0 || (i >= 0 && data[i] >= recvBuf[j])) {
                                tmp[k--] = data[i--];
                            }else{
                                tmp[k--] = recvBuf[j--];
                            }
                        }
                        std::memcpy(data.data(), tmp.data(), size_t(chunk) * sizeof(float)); 
                        
                    }
            }

        }

        if (phase > 0){
            // if no swaps happened for all ranks, stop
            int local = local_swapped ? 1 : 0;
            int any = 0;
            MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            global_swapped = (any != 0);
        }

        // if (phase == 1){
        //     tok = MPI_Wtime();
        //     printf("That took %f seconds\n", tok - tik);
        // }
        phase++;
    }

    // tok = MPI_Wtime();
    // printf("Sorting took %f seconds\n", tok - tik);
    // printf("rank %d finished after %d phases\n", rank, phase);
    
    
    // tik = MPI_Wtime();
    // MPI_Info winfo;
    // MPI_Info_create(&winfo);
    // MPI_Info_set(winfo, "access_style", "write_once");
    // MPI_Info_set(winfo, "collective_buffering", "true");
    // MPI_Info_set(winfo, "striping_factor", "4");
    // // On Lustre file systems, set stripe size for better performance
    // MPI_Info_set(winfo, "striping_unit", "4194304"); 


    // --- open output file ---
    MPI_File_open(MPI_COMM_WORLD, output_filename,
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    // MPI_File_open(MPI_COMM_WORLD, output_filename,
    //             MPI_MODE_CREATE | MPI_MODE_WRONLY, winfo, &output_file);

    // // --- collective write ---
    // MPI_File_write_at_all(output_file, offset,
    //                     data.data(), (int)chunk, MPI_FLOAT, MPI_STATUS_IGNORE);
    // MPI_File_write_at(output_file, sizeof(float) * (int)chunk * rank, data.data(), (int)chunk, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_write_at(output_file, offset, data.data(), (int)chunk, MPI_FLOAT, MPI_STATUS_IGNORE);
    // printf("rank %d wrote %d floats, write from %d\n", rank, chunk, sizeof(float) * (int)chunk * rank);
    // --- close file ---
    MPI_File_close(&output_file);
    // MPI_Info_free(&winfo);
    
    // tok = MPI_Wtime();
    // printf("Writing file took %f seconds\n", tok - tik);

    MPI_Finalize();
    return 0;
}

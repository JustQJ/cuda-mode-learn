#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda_runtime.h>
#include<cuda.h>


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

u_int64_t get_us_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000000 + tv.tv_usec;
}

#define BLOCK_SIZE 512

/*
problem of this file:
we have a sequence string, and we want to count the number of each hisgram in the string
for example: aabbbccxxxyhhssddfflgg
the histogram is: a-d , e-h, i-l, m-p, q-t, u-x, y-z, we want to count the number of each histogram in the string

cpu version:

void histogram_cpu_sequential(char *data, int length, int *histogram){
    for(int i=0; i<length; i++){
        int index = (data[i] - 'a'); // get the index of the histogram
        if(index < 0 || index >= 26){
            continue;
        }
        histogram[index/4]++;
    }
}

*/


void histogram_cpu_sequential(char *data, int length, int *histogram){
    for(int i=0; i<length; i++){
        int index = (data[i] - 'a'); // get the index of the histogram
        if(index < 0 || index >= 26){
            continue;
        }
        histogram[index/4]++;
    }
}



//use gpu to calculate the histogram
//base version: each thread deal one char, and use atomicAdd to update the histogram

__global__ void histogram_kernel(char *data, int length, int *histogram){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < length){
        int pos = (data[index] - 'a'); 
        if(pos < 0 || pos >= 26){
            return;
        }
        atomicAdd(&histogram[pos/4], 1);
    }
}


//each block has a copy of the histogram, which can reduce the race condition
//use shared memory to store the histogram
//add the result of each block to the final histogram
__global__ void histogram_shared_mem_kernel(char *data, int length, int *histogram){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int histo_size = 26/4+1;
    __shared__ int temp_histo[histo_size];
    if(threadIdx.x < histo_size){
        temp_histo[threadIdx.x] = 0;
    }

    __syncthreads();

    if(index < length){
        int pos = (data[index] - 'a'); 
        if(pos >= 0 && pos < 26){
            atomicAdd(&temp_histo[pos/4], 1);
        }
        
    }

    __syncthreads();

    //write back

    if(threadIdx.x < histo_size){
        int value = temp_histo[threadIdx.x];
        if(value > 0){
            atomicAdd(&histogram[threadIdx.x], value);
        }
    }

}

//one thread deal with multi chars, therefore we has less blocks, which can reduce the overhead of resource allocation and reduce the times of atomicAdd operation to global memroy
//two version: 
// contiguous (one thread deal with contiguous chars) 
// interleaved (one thread deal with interleaved chars, which means the swap can deal with contiguous chars, can improve the read performance)


#define CHAR_NUM_PER_THREAD 4
__global__ void histogram_shared_mem_multi_contiguous_kernel(char *data, int length, int *histogram){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int histo_size = 26/4+1;
    __shared__ int temp_histo[histo_size];
    if(threadIdx.x < histo_size){
        temp_histo[threadIdx.x] = 0;
    }

    __syncthreads();

    for(int idx = CHAR_NUM_PER_THREAD*index; idx<min(CHAR_NUM_PER_THREAD*(index+1), length); idx++){
        int pos = (data[idx] - 'a'); 
        if(pos >= 0 && pos < 26){
            atomicAdd(&temp_histo[pos/4], 1);
        }
        
    }

    __syncthreads();

    //write back

    if(threadIdx.x < histo_size){
        int value = temp_histo[threadIdx.x];
        if(value > 0){
            atomicAdd(&histogram[threadIdx.x], value);
        }
    }

}


__global__ void histogram_shared_mem_multi_interleaved_kernel(char *data, int length, int *histogram){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int histo_size = 26/4+1;
    __shared__ int temp_histo[histo_size];
    if(threadIdx.x < histo_size){
        temp_histo[threadIdx.x] = 0;
    }

    __syncthreads();

    for(int idx = index; idx<length; idx+=blockDim.x*gridDim.x){
        int pos = (data[idx] - 'a'); 
        if(pos >= 0 && pos < 26){
            atomicAdd(&temp_histo[pos/4], 1);
        }
        
    }

    __syncthreads();

    //write back

    if(threadIdx.x < histo_size){
        int value = temp_histo[threadIdx.x];
        if(value > 0){
            atomicAdd(&histogram[threadIdx.x], value);
        }
    }

}


//some char may occurs frequently, so we can reduce the atomicAdd operation by recording the number 
__global__ void histogram_shared_mem_multi_interleaved_counter_kernel(char *data, int length, int *histogram){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    const int histo_size = 26/4+1;
    __shared__ int temp_histo[histo_size];
    if(threadIdx.x < histo_size){
        temp_histo[threadIdx.x] = 0;
    }

    __syncthreads();

    int temp_accumulator = 0;
    int pred_bin_pos = -1;
    for(int idx = index; idx<length; idx+=blockDim.x*gridDim.x){
        int pos = (data[idx] - 'a'); 
        if(pos >= 0 && pos < 26){
            // atomicAdd(&temp_histo[pos/4], 1);
            int bin_pos = pos/4;
            if(bin_pos != pred_bin_pos){
                if(temp_accumulator > 0){
                    atomicAdd(&temp_histo[pred_bin_pos], temp_accumulator);
                }
                temp_accumulator = 1;
                pred_bin_pos = bin_pos;
            }else{
                temp_accumulator++;
            }
        }
        
    }

    if(temp_accumulator > 0){
        atomicAdd(&temp_histo[pred_bin_pos], temp_accumulator);
    }

    __syncthreads();

    //write back

    if(threadIdx.x < histo_size){
        int value = temp_histo[threadIdx.x];
        if(value > 0){
            atomicAdd(&histogram[threadIdx.x], value);
        }
    }

}


int main(int argc, char **argv){

    int char_num = 1024*1024*1024;
    char *data = (char *)malloc(char_num*sizeof(char));
    for(int i=0; i<char_num; i++){
        data[i] = 'a' + rand()%26;
    }
    int histo_size = 26/4+1;
    int *cpu_histogram = (int *)malloc(histo_size*sizeof(int));
    int *histogram = (int *)malloc(histo_size*sizeof(int));
    memset(cpu_histogram, 0, histo_size*sizeof(int));

    int num = 10;
    





    char *d_data;
    int *d_histogram;


    CHECK_CUDA_ERROR(cudaMalloc(&d_data, char_num*sizeof(char)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, histo_size*sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, data, char_num*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, histo_size*sizeof(int)));

    dim3 block_(BLOCK_SIZE);
    dim3 grid_((char_num+BLOCK_SIZE-1)/BLOCK_SIZE);


    //verify the result

    histogram_cpu_sequential(data, char_num, cpu_histogram);

    histogram_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    CHECK_CUDA_ERROR(cudaMemcpy(histogram, d_histogram, histo_size*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i=0; i<histo_size; i++){
        if(cpu_histogram[i] != histogram[i]){
            printf("histogram_kernel error: %d %d %d\n", i, cpu_histogram[i], histogram[i]);
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, histo_size*sizeof(int)));
    histogram_shared_mem_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    CHECK_CUDA_ERROR(cudaMemcpy(histogram, d_histogram, histo_size*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i=0; i<histo_size; i++){
        if(cpu_histogram[i] != histogram[i]){
            printf("histogram_shared_mem_kernel error: %d %d %d\n", i, cpu_histogram[i], histogram[i]);
            break;
        }
    }


    //multi char per thread
    grid_ = dim3((char_num+BLOCK_SIZE*CHAR_NUM_PER_THREAD-1)/(BLOCK_SIZE*CHAR_NUM_PER_THREAD));

    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, histo_size*sizeof(int)));
    histogram_shared_mem_multi_contiguous_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    CHECK_CUDA_ERROR(cudaMemcpy(histogram, d_histogram, histo_size*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i=0; i<histo_size; i++){
        if(cpu_histogram[i] != histogram[i]){
            printf("histogram_shared_mem_multi_contiguous_kernel error: %d %d %d\n", i, cpu_histogram[i], histogram[i]);
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, histo_size*sizeof(int)));
    histogram_shared_mem_multi_interleaved_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    CHECK_CUDA_ERROR(cudaMemcpy(histogram, d_histogram, histo_size*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i=0; i<histo_size; i++){
        if(cpu_histogram[i] != histogram[i]){
            printf("histogram_shared_mem_multi_interleaved_kernel error: %d %d %d\n", i, cpu_histogram[i], histogram[i]);
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, histo_size*sizeof(int)));
    histogram_shared_mem_multi_interleaved_counter_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    CHECK_CUDA_ERROR(cudaMemcpy(histogram, d_histogram, histo_size*sizeof(int), cudaMemcpyDeviceToHost));
    for(int i=0; i<histo_size; i++){
        if(cpu_histogram[i] != histogram[i]){
            printf("histogram_shared_mem_multi_interleaved_counter_kernel error: %d %d %d\n", i, cpu_histogram[i], histogram[i]);
            break;
        }
    }

















    //cpu time
    u_int64_t start_time_cpu = get_us_time();
    for(int i=0; i<num; i++)
        histogram_cpu_sequential(data, char_num, histogram);
    u_int64_t end_time_cpu = get_us_time();
    printf("cpu time: %d us\n", (end_time_cpu-start_time_cpu)/num);



    //grid
    grid_ = dim3((char_num+BLOCK_SIZE-1)/BLOCK_SIZE);


    //warm up
    for(int i=0; i<num; i++){
        histogram_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    



    u_int64_t start_time = get_us_time();
    for(int i=0; i<num; i++){
        histogram_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t end_time = get_us_time();
    printf("histogram_kernel time: %d us, speedup: %.2f\n", (end_time-start_time)/num, (end_time_cpu-start_time_cpu)*1.0/(end_time-start_time));

    

    start_time = get_us_time();
    for(int i=0; i<num; i++){
        histogram_shared_mem_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();
    printf("histogram_shared_mem_kernel time: %d us, speedup: %.2f\n", (end_time-start_time)/num, (end_time_cpu-start_time_cpu)*1.0/(end_time-start_time));



    //multi char per thread
    grid_ = dim3((char_num+BLOCK_SIZE*CHAR_NUM_PER_THREAD-1)/(BLOCK_SIZE*CHAR_NUM_PER_THREAD));
    start_time = get_us_time();
    for(int i=0; i<num; i++){
        histogram_shared_mem_multi_contiguous_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();
    printf("histogram_shared_mem_multi_contiguous_kernel time: %d us, speedup: %.2f\n", (end_time-start_time)/num, (end_time_cpu-start_time_cpu)*1.0/(end_time-start_time));


    start_time = get_us_time();
    for(int i=0; i<num; i++){
        histogram_shared_mem_multi_interleaved_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();

    printf("histogram_shared_mem_multi_interleaved_kernel time: %d us, speedup: %.2f\n", (end_time-start_time)/num, (end_time_cpu-start_time_cpu)*1.0/(end_time-start_time));


    start_time = get_us_time();
    for(int i=0; i<num; i++){
        histogram_shared_mem_multi_interleaved_counter_kernel<<<grid_, block_>>>(d_data, char_num, d_histogram);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    end_time = get_us_time();
    printf("histogram_shared_mem_multi_interleaved_counter_kernel time: %d us, speedup: %.2f\n", (end_time-start_time)/num, (end_time_cpu-start_time_cpu)*1.0/(end_time-start_time));


    //free
    free(data);
    free(histogram);
    free(cpu_histogram);
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_histogram));

    return 0;



}
/*

histogram_kernel time: 25466 us
histogram_shared_mem_kernel time: 345 us
histogram_shared_mem_multi_contiguous_kernel time: 145 us
histogram_shared_mem_multi_interleaved_kernel time: 245 us
histogram_shared_mem_multi_interleaved_counter_kernel time: 274 us
*/
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda.h>
#include<cuda_runtime.h>



#define BLOCK_DIM 1024

u_int64_t get_us_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000000 + tv.tv_usec;
}


/*
program to demonstrate the use of reduction in CUDA
reduction is a technique to combine a set of values into a single value throgugh a binary operation (sum, max, min, product, etc)

for example, sum of all elements in an array
cpu version
float sum = 0.0;
for(int i=0; i<n; i++){
    sum += a[i];
}


cuda reduction tree


*/

//cpu version
//Kahan Summation to reduce the error in floating point addition
float cpu_sum(float *a, u_int64_t n){
    float sum = 0.0f;
    float c = 0.0f;  

    for (uint64_t i = 0; i < n; ++i) {
        float y = a[i] - c;   
        float t = sum + y;     
        c = (t - sum) - y;     
        sum = t;              
    }

    return sum;
}




/*
first we will implement a simple reduction kernel that reduces an array of floats to a single float
only use one block and each thread will reduce two elements, so this kernel will only process at most 2048 elements because of the 1024 threads per block limit

*/
__global__ void one_block_reduction_kernel(float *input, float *output){

    int i = 2*threadIdx.x; //location of the thread 
    for(int stride=1; stride<=blockDim.x; stride*=2){ //the size of the array is 2*blockDim.x
        if(threadIdx.x % (stride) == 0){
            input[i] += input[i + stride];
        }
        __syncthreads(); //wait for all threads to finish, go to the next iteration
    }

    if(threadIdx.x == 0){
        *output = input[0];
    }
}


/*
reduce the thread divergence by decreasing stride by half in each iteration, rather than doubling it
*/

__global__ void one_block_reduction_decreasing_stride_kernel(float *input, float *output){

    int i = threadIdx.x; //location of the thread 
    for(int stride=blockDim.x; stride>=1; stride/=2){ //the size of the array is 2*blockDim.x
        if(threadIdx.x < stride){
            input[i] += input[i + stride];
        }
        __syncthreads(); //wait for all threads to finish, go to the next iteration
    }

    if(threadIdx.x == 0){
        *output = input[0];
    }
}


/*
using shared memory to reduce the number of global memory accesses
*/

__global__ void one_block_reduction_decreasing_stride_shared_mem_kernel(float *input, float *output){

    __shared__ float temp_sum[BLOCK_DIM]; //shared memory to store the partial sums
    int i = threadIdx.x; //location of the thread 
    temp_sum[i] = input[i] + input[i + blockDim.x]; //fisrt reduce result stored in shared memory
    //do the reduction in shared memory
    for(int stride=blockDim.x/2; stride>=1; stride/=2){ //the size of the array is 2*blockDim.x
        __syncthreads(); //wait for all threads to finish, go to the next iteration
        if(threadIdx.x < stride){
            temp_sum[i] += temp_sum[i + stride];
        }
        
    }

    if(threadIdx.x == 0){
        *output = temp_sum[0];
    }
}


/*
multiple blocks reduction kernel to deal with abritrary number of elements
*/

__global__ void multiple_blocks_reduction_kernel(float *input, float *output, u_int64_t n){

    __shared__ float temp_sum[BLOCK_DIM];
    u_int64_t offset = 2*blockDim.x*blockIdx.x; //offset to the start of the block, each block will reduce 2*blockDim.x elements
    u_int64_t i = threadIdx.x; //location of the thread
    if(offset + i + blockDim.x < n){
        temp_sum[i] = input[offset + i] + input[offset + i + blockDim.x];
    }else if(offset +i < n){
        temp_sum[i] = input[offset + i]; 
    }else{
        temp_sum[i] = 0.0;
    }

    for(int stride=blockDim.x/2; stride>=1; stride/=2){
        __syncthreads();
        if(i<stride){
            temp_sum[i] += temp_sum[i + stride];
        }
    }

    if(i == 0){
       atomicAdd(output, temp_sum[0]); //atomic add results to the final output for multiple blocks
    }


}


/*
one thread dealing more than two elements in the array, thread coarsening
*/
#define COARSEN_FACTOR 2 //each thread will reduce COARSEN_FACTOR*2 elements
__global__ void multiple_blocks_reduction_thread_coarsening_kernel(float *input, float *output, u_int64_t n){

    __shared__ float temp_sum[BLOCK_DIM];
    u_int64_t offset = COARSEN_FACTOR*2*blockDim.x*blockIdx.x; //offset to the start of the block, each block will reduce 2*COARSEN_FACTOR*blockDim.x elements
    u_int64_t i = threadIdx.x; //location of the thread

    float partial_sum = 0.0; //first reduce the COARSEN_FACTOR*2 elements
    if(offset + i < n){ //check if the thread is within the array
        partial_sum = input[offset + i];
    }
    for(int stride=1; stride<COARSEN_FACTOR*2; stride++){
        if(offset + i + stride*blockDim.x < n){ //check the location is within the array
            partial_sum = partial_sum + input[offset + i + stride*blockDim.x];
        }
    }
    temp_sum[i] = partial_sum;
    

    

    for(int stride=blockDim.x/2; stride>=1; stride/=2){
        __syncthreads();
        if(i<stride){
            temp_sum[i] += temp_sum[i + stride];
        }
    }

    if(i == 0){
       atomicAdd(output, temp_sum[0]); //atomic add results to the final output for multiple blocks
    }


}




void check_kernels(){

    u_int64_t n = 1024*1024*512;
    float *a = (float *)malloc(n*sizeof(float));
    for(u_int64_t i=0; i<n; i++){
        a[i] = 1.0;
    }

    int block_size = BLOCK_DIM;

    float cpu_result = cpu_sum(a, n);
    printf("cpu result for %ld elements: %f\n", n ,cpu_result);

    cpu_result = cpu_sum(a, block_size*2);
    printf("cpu result for %d elements: %f\n", block_size*2, cpu_result);

    float *d_a, *d_result;
    cudaMalloc(&d_a, n*sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    float *h_result = (float *)malloc(sizeof(float));

    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);


    //one block reduction kernel
    cudaMemset(d_result, 0, sizeof(float));
    dim3 block(block_size, 1,1);
    dim3 grid(1,1,1);
    one_block_reduction_kernel<<<grid, block>>>(d_a, d_result);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("one block reduction kernel result for %d elements: %f\n", block_size*2,*h_result);

    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);

    //one block reduction decreasing stride kernel
    cudaMemset(d_result, 0, sizeof(float));
    one_block_reduction_decreasing_stride_kernel<<<grid, block>>>(d_a, d_result);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("one_block_reduction_decreasing_stride_kernel result for %d elements: %f\n", block_size*2, *h_result);

    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
    //one block reduction decreasing stride shared memory kernel
    cudaMemset(d_result, 0, sizeof(float));
    one_block_reduction_decreasing_stride_shared_mem_kernel<<<grid, block>>>(d_a, d_result);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("one_block_reduction_decreasing_stride_shared_mem_kernel result for %d elements: %f\n", block_size*2, *h_result);

    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
    //multiple blocks reduction kernel
    cudaMemset(d_result, 0, sizeof(float));
    grid.x = (n+2*block.x-1)/(2*block.x);
    multiple_blocks_reduction_kernel<<<grid, block>>>(d_a, d_result, n);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("multiple blocks reduction kernel result for %ld elements: %f\n", n,*h_result);

    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);
    //multiple blocks reduction thread coarsening kernel
    cudaMemset(d_result, 0, sizeof(float));
    grid.x = (n+2*COARSEN_FACTOR*block.x-1)/(2*COARSEN_FACTOR*block.x);
    multiple_blocks_reduction_thread_coarsening_kernel<<<grid, block>>>(d_a, d_result, n);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("multiple blocks reduction thread coarsening kernel result for %ld elements: %f\n", n, *h_result);

    //free all the memory
    free(a);
    free(h_result);
    cudaFree(d_a);
    cudaFree(d_result);


}


void peformance_evaluation(){
     u_int64_t n = 1024*1024*512;
    float *a = (float *)malloc(n*sizeof(float));
    for(u_int64_t i=0; i<n; i++){
        a[i] = rand()/(float)RAND_MAX;
    }

    int eval_num = 10; 

    int block_size = BLOCK_DIM;


    u_int64_t start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        float cpu_result = cpu_sum(a, n);
    u_int64_t end_time = get_us_time();
    u_int64_t cpu_time1 = end_time - start_time;
    printf("cpu compute %ld elements cost time: %f us\n", n , (end_time - start_time)/(float)eval_num);

     start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        float cpu_result = cpu_sum(a, block_size*2);
     end_time = get_us_time();
     u_int64_t cpu_time2 = end_time - start_time;
    printf("cpu compute %ld elements cost time: %f us\n", block_size*2 , (end_time - start_time)/(float)eval_num);

    float *d_a, *d_result;
    cudaMalloc(&d_a, n*sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    float *h_result = (float *)malloc(sizeof(float));

    cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice);


    //one block reduction kernel
    
    dim3 block(block_size, 1,1);
    dim3 grid(1,1,1);

    cudaMemset(d_result, 0, sizeof(float));

    cudaDeviceSynchronize();
    start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        one_block_reduction_kernel<<<grid, block>>>(d_a, d_result);

     cudaDeviceSynchronize();
    end_time = get_us_time();
    printf("one block reduction kernel compute %ld elements cost time: %f us, speedup %.2f\n", block_size*2 , (end_time - start_time)/(float)eval_num, cpu_time2/(float)(end_time - start_time));
 

    
    cudaMemset(d_result, 0, sizeof(float));

    //one block reduction decreasing stride kernel
    cudaDeviceSynchronize();
    start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        one_block_reduction_decreasing_stride_kernel<<<grid, block>>>(d_a, d_result);

     cudaDeviceSynchronize();
    end_time = get_us_time();
    printf("one block reduction decreasing stride kernel compute %ld elements cost time: %f us, speedup %.2f\n", block_size*2 , (end_time - start_time)/(float)eval_num, cpu_time2/(float)(end_time - start_time));
   

    cudaMemset(d_result, 0, sizeof(float));
    //one block reduction decreasing stride shared memory kernel
    cudaDeviceSynchronize();
    start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        one_block_reduction_decreasing_stride_shared_mem_kernel<<<grid, block>>>(d_a, d_result);

     cudaDeviceSynchronize();
    end_time = get_us_time();
    printf("one block reduction decreasing stride shared memory kernel compute %ld elements cost time: %f us, speedup %.2f\n", block_size*2 , (end_time - start_time)/(float)eval_num, cpu_time2/(float)(end_time - start_time));
    
   

    
    //multiple blocks reduction kernel
    cudaMemset(d_result, 0, sizeof(float));
    grid.x = (n+2*block.x-1)/(2*block.x);
    cudaDeviceSynchronize();
    start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        multiple_blocks_reduction_kernel<<<grid, block>>>(d_a, d_result, n);

     cudaDeviceSynchronize();
    end_time = get_us_time();
    printf("multiple blocks reduction kernel compute %ld elements cost time: %f us, speedup %.2f\n", n , (end_time - start_time)/(float)eval_num, cpu_time1/(float)(end_time - start_time));
  


    //multiple blocks reduction thread coarsening kernel
    cudaMemset(d_result, 0, sizeof(float));
    grid.x = (n+2*COARSEN_FACTOR*block.x-1)/(2*COARSEN_FACTOR*block.x);
    cudaDeviceSynchronize();
    start_time = get_us_time();
    for(int i=0; i<eval_num; i++)
        multiple_blocks_reduction_thread_coarsening_kernel<<<grid, block>>>(d_a, d_result, n);

     cudaDeviceSynchronize();
    end_time = get_us_time();
    printf("multiple blocks reduction thread coarsening kernel compute %ld elements cost time: %f us, speedup %.2f\n", n , (end_time - start_time)/(float)eval_num, cpu_time1/(float)(end_time - start_time));


    //free all the memory
    free(a);
    free(h_result);
    cudaFree(d_a);
    cudaFree(d_result);
}
int main(){
    printf("check result correct\n");
    check_kernels();
    printf("performance evaluation\n");
    peformance_evaluation();
    return 0;
}
/*

check result correct
cpu result for 536870912 elements: 536870912.000000
cpu result for 2048 elements: 2048.000000
one block reduction kernel result for 2048 elements: 2048.000000
one_block_reduction_decreasing_stride_kernel result for 2048 elements: 2048.000000
one_block_reduction_decreasing_stride_shared_mem_kernel result for 2048 elements: 2048.000000
multiple blocks reduction kernel result for 536870912 elements: 536870912.000000
multiple blocks reduction thread coarsening kernel result for 536870912 elements: 536870912.000000
performance evaluation
cpu compute 536870912 elements cost time: 4928176.000000 us
cpu compute 2048 elements cost time: 18.200001 us
one block reduction kernel compute 2048 elements cost time: 6.400000 us, speedup 2.84
one block reduction decreasing stride kernel compute 2048 elements cost time: 3.100000 us, speedup 5.87
one block reduction decreasing stride shared memory kernel compute 2048 elements cost time: 3.100000 us, speedup 5.87
multiple blocks reduction kernel compute 536870912 elements cost time: 3401.399902 us, speedup 1448.87
multiple blocks reduction thread coarsening kernel compute 536870912 elements cost time: 2240.300049 us, speedup 2199.78
*/
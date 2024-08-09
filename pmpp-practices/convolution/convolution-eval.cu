#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>



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

#define BENCHMARK(call) { \
    u_int64_t start = get_us_time(); \
    call; \
    u_int64_t end = get_us_time(); \
    printf("Time cost by %s: %lu us\n", #call ,end-start); \
}

#define FILTER_RADIUS 4

//define INPUT TILE DIM
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM-2*FILTER_RADIUS)


//define the constant memory for the filter tensor because it is small and read-only, can be cached in constant memory
__constant__ float dc_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];


//basic 2d convolution, each thread compute the each element of the output tensor
__global__ void convolution_2d_basic_kernel(float *N, float *F, float *P, int r, int width, int height){
    /*
    N: input tensor
    F: filter tensor
    P: output tensor
    r: filter radius (2r+1)x(2r+1)
    width: width of input tensor
    height: height of input tensor
    N and P has the same size
    */

    int col = blockIdx.x*blockDim.x + threadIdx.x; 
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    float p_value = 0.0;
    for(int f_row = 0; f_row < 2*r+1; f_row++){
        for(int f_col = 0; f_col < 2*r+1; f_col++){
            int n_row = row - r + f_row;
            int n_col = col - r + f_col;
            if(n_row >= 0 && n_row < height && n_col >= 0 && n_col < width){
                p_value += N[n_row*width + n_col] * F[f_row*(2*r+1) + f_col];
            }
        }
    }
    P[row*width+col] = p_value;

}



//constatn memory version of 2d convolution, each thread compute the each element of the output tensor
__global__ void convolution_2d_constant_mem_kernel(float *N, float *P, int r, int width, int height){
    /*
    N: input tensor
    P: output tensor
    r: filter radius (2r+1)x(2r+1)
    width: width of input tensor
    height: height of input tensor
    N and P has the same size
    use constant memory for the filter tensor F
    */

    int col = blockIdx.x*blockDim.x + threadIdx.x; 
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    float p_value = 0.0;
    for(int f_row = 0; f_row < 2*r+1; f_row++){
        for(int f_col = 0; f_col < 2*r+1; f_col++){
            int n_row = row - r + f_row;
            int n_col = col - r + f_col;
            if(n_row >= 0 && n_row < height && n_col >= 0 && n_col < width){
                p_value += N[n_row*width + n_col] * dc_F[f_row*(2*r+1) + f_col];
            }
        }
    }
    P[row*width+col] = p_value;
}


//tiled version of 2d convolution, a block of threads compute a tile of the output tensor, use the shared memory to cache the input tensor
//the thread block map to the input tensor tile, which means some threads may not do the computation for output tensor, just used to load the input tensor into shared memory
__global__ void convolution_2d_tiled_const_mem_kernel(float *N, float *P, int width, int height){
    /*
    N: input tensor
    P: output tensor
    width: width of input tensor
    height: height of input tensor
    N and P has the same size
    use constant memory for the filter tensor F
    */

    //shared memory for the tile of input tensor
    __shared__ float s_N[IN_TILE_DIM][IN_TILE_DIM];

    //input tensor row and column index, to load data
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    if(col >= 0 && col < width && row >= 0 && row < height){
        s_N[threadIdx.y][threadIdx.x] = N[row*width + col];
    }else{
        s_N[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();


    //only the threads in the output tensor tile do the computation
    int tile_col = threadIdx.x - FILTER_RADIUS;
    int tile_row = threadIdx.y - FILTER_RADIUS;

    //compute the output tensor
    if(col >= 0 && col < width && row >= 0 && row < height){
        if(tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0 && tile_row < OUT_TILE_DIM){
            float p_value = 0.0;
            for(int f_row = 0; f_row < 2*FILTER_RADIUS+1; f_row++){
                for(int f_col = 0; f_col < 2*FILTER_RADIUS+1; f_col++){
                    p_value += s_N[tile_row+f_row][tile_col+f_col] * dc_F[f_row*(2*FILTER_RADIUS+1) + f_col];
                }
            }
            P[row*width+col] = p_value;
        }

    }
    
}



#define TILE_DIM 32
//tiled version of 2d convolution, a block of threads compute a tile of the output tensor, use the shared memory to cache the input tensor
//only load the inner input tensor into shared memory, the boundary input tensor is not loaded
__global__ void convolution_2d_cached_tiled_const_mem_kernel(float *N, float *P, int width, int height){
    /*
    N: input tensor
    P: output tensor
    width: width of input tensor
    height: height of input tensor
    N and P has the same size
    use constant memory for the filter tensor F
    */

    //shared memory for the tile of input tensor
    __shared__ float s_N[TILE_DIM][TILE_DIM];

    //input tensor row and column index, to load data
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;
    if(col >= 0 && col < width && row >= 0 && row < height){
        s_N[threadIdx.y][threadIdx.x] = N[row*width + col];
    }else{
        s_N[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    //turning off the boundary threads
    if(col<width && row<height){
        float p_value = 0.0;

        for(int f_row = 0; f_row < 2*FILTER_RADIUS+1; f_row++){
            for(int f_col = 0; f_col < 2*FILTER_RADIUS+1; f_col++){
                if(threadIdx.x - FILTER_RADIUS + f_col >= 0 && threadIdx.x - FILTER_RADIUS + f_col < TILE_DIM && threadIdx.y - FILTER_RADIUS + f_row >= 0 && threadIdx.y - FILTER_RADIUS + f_row < TILE_DIM){
                    p_value += s_N[threadIdx.y - FILTER_RADIUS + f_row][threadIdx.x - FILTER_RADIUS + f_col] * dc_F[f_row*(2*FILTER_RADIUS+1) + f_col];
                }else{ //load from global memory
                    if(row-FILTER_RADIUS+f_row >= 0 && row-FILTER_RADIUS+f_row < height && col-FILTER_RADIUS+f_col >= 0 && col-FILTER_RADIUS+f_col < width){
                        p_value += N[(row-FILTER_RADIUS+f_row)*width + col-FILTER_RADIUS+f_col] * dc_F[f_row*(2*FILTER_RADIUS+1) + f_col];
                    }

                }
            }
        }
        P[row*width+col] = p_value;
    }

    
    
}



void call_basic_kernel(float *N, float *F, float *P, int width, int height){

    //device memory pointers
    float *d_N, *d_F, *d_P;

    //allocate device memory for input tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_N, width*height*sizeof(float)));
    //allocate device memory for filter tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float)));
    //allocate device memory for output tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, width*height*sizeof(float)));

    //copy input tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, N, width*height*sizeof(float), cudaMemcpyHostToDevice));
    //copy filter tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float), cudaMemcpyHostToDevice));

    //block size
    dim3 block_size(32, 32);
    //grid size
    dim3 grid_size((width+block_size.x-1)/block_size.x, (height+block_size.y-1)/block_size.y);

    //call the kernel
    
    convolution_2d_basic_kernel<<<grid_size, block_size>>>(d_N, d_F, d_P, FILTER_RADIUS, width, height);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //benchmark the kernel call
    u_int64_t start = get_us_time();
    for(int i=0; i<10; i++){
        convolution_2d_basic_kernel<<<grid_size, block_size>>>(d_N, d_F, d_P, FILTER_RADIUS, width, height);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t end = get_us_time();
    printf("Time cost by convolution_2d_basic_kernel: %lu us\n", (end-start)/10);



    //copy output tensor to host memory
    CHECK_CUDA_ERROR(cudaMemcpy(P, d_P, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    //print output tensor
    for(int i = 0; i < 10; i++){
        printf("%f, ", P[i]);
    }
    printf("\n");

    CHECK_CUDA_ERROR(cudaFree(d_N));
    CHECK_CUDA_ERROR(cudaFree(d_F));
    CHECK_CUDA_ERROR(cudaFree(d_P));
   
}


//call the kernel with constant memory



void call_constant_mem_kernel(float *N, float *F, float *P, int width, int height){

    //device memory pointers
    float *d_N, *d_P;

    //allocate device memory for input tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_N, width*height*sizeof(float)));
    //allocate device memory for output tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, width*height*sizeof(float)));

    //copy input tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, N, width*height*sizeof(float), cudaMemcpyHostToDevice));
    //copy filter tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dc_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float)));

    //block size
    dim3 block_size(32, 32);
    //grid size
    dim3 grid_size((width+block_size.x-1)/block_size.x, (height+block_size.y-1)/block_size.y);

    //call the kernel
    
    convolution_2d_constant_mem_kernel<<<grid_size, block_size>>>(d_N, d_P, FILTER_RADIUS, width, height);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //benchmark the kernel call
    u_int64_t start = get_us_time();
    for(int i=0; i<10; i++){
        convolution_2d_constant_mem_kernel<<<grid_size, block_size>>>(d_N, d_P, FILTER_RADIUS, width, height);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t end = get_us_time();
    printf("Time cost by convolution_2d_constant_mem_kernel: %lu us\n", (end-start)/10);



    //copy output tensor to host memory
    CHECK_CUDA_ERROR(cudaMemcpy(P, d_P, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    //print output tensor
    for(int i = 0; i < 10; i++){
        printf("%f, ", P[i]);
    }
    printf("\n");

    CHECK_CUDA_ERROR(cudaFree(d_N));
    CHECK_CUDA_ERROR(cudaFree(d_P));
   
}



void call_2d_tiled_constant_mem_kernel(float *N, float *F, float *P, int width, int height){

    //device memory pointers
    float *d_N, *d_P;

    //allocate device memory for input tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_N, width*height*sizeof(float)));
    //allocate device memory for output tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, width*height*sizeof(float)));

    //copy input tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, N, width*height*sizeof(float), cudaMemcpyHostToDevice));
    //copy filter tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dc_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float)));

    //block size
    dim3 block_size(IN_TILE_DIM, IN_TILE_DIM);
    //grid size
    dim3 grid_size((width+OUT_TILE_DIM-1)/OUT_TILE_DIM, (height+OUT_TILE_DIM-1)/OUT_TILE_DIM);

    //call the kernel
    
    convolution_2d_tiled_const_mem_kernel<<<grid_size, block_size>>>(d_N, d_P, width, height);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //benchmark the kernel call
    u_int64_t start = get_us_time();
    for(int i=0; i<10; i++){
        convolution_2d_tiled_const_mem_kernel<<<grid_size, block_size>>>(d_N, d_P, width, height);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t end = get_us_time();
    printf("Time cost by convolution_2d_constant_mem_kernel: %lu us\n", (end-start)/10);



    //copy output tensor to host memory
    CHECK_CUDA_ERROR(cudaMemcpy(P, d_P, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    //print output tensor
    for(int i = 0; i < 10; i++){
        printf("%f, ", P[i]);
    }
    printf("\n");

    CHECK_CUDA_ERROR(cudaFree(d_N));
    CHECK_CUDA_ERROR(cudaFree(d_P));
   
}

void call_2d_cached_tiled_constant_mem_kernel(float *N, float *F, float *P, int width, int height){

    //device memory pointers
    float *d_N, *d_P;

    //allocate device memory for input tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_N, width*height*sizeof(float)));
    //allocate device memory for output tensor
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, width*height*sizeof(float)));

    //copy input tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, N, width*height*sizeof(float), cudaMemcpyHostToDevice));
    //copy filter tensor to device memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(dc_F, F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float)));

    //block size
    dim3 block_size(TILE_DIM, TILE_DIM);
    //grid size
    dim3 grid_size((width+TILE_DIM-1)/TILE_DIM, (height+TILE_DIM-1)/TILE_DIM);

    //call the kernel
    
    convolution_2d_cached_tiled_const_mem_kernel<<<grid_size, block_size>>>(d_N, d_P, width, height);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //benchmark the kernel call
    u_int64_t start = get_us_time();
    for(int i=0; i<10; i++){
        convolution_2d_cached_tiled_const_mem_kernel<<<grid_size, block_size>>>(d_N, d_P, width, height);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t end = get_us_time();
    printf("Time cost by convolution_2d_cached_tiled_const_mem_kernel: %lu us\n", (end-start)/10);



    //copy output tensor to host memory
    CHECK_CUDA_ERROR(cudaMemcpy(P, d_P, width*height*sizeof(float), cudaMemcpyDeviceToHost));

    //print output tensor
    for(int i = 0; i < 10; i++){
        printf("%f, ", P[i]);
    }
    printf("\n");

    CHECK_CUDA_ERROR(cudaFree(d_N));
    CHECK_CUDA_ERROR(cudaFree(d_P));
   
}




int main(int argc, char **argv){

    //input tensor
    float *N;
    //filter tensor
    float *F;
    //output tensor
    float *P;

    //size of input tensor
    int width = 4096*4;
    int height = 4096*4;
    //size of filter tensor
    

    //allocate memory for input tensor
    N = (float *)malloc(width*height*sizeof(float));
    //allocate memory for filter tensor
    F = (float *)malloc((2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    //allocate memory for output tensor
    P = (float *)malloc(width*height*sizeof(float));

    //initialize input tensor
    for(int i = 0; i < width*height; i++){
        N[i] = i*0.1;
    }

    //initialize filter tensor
    for(int i = 0; i < (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1); i++){
        F[i] = i*0.05;
    }

    
    

    //call the basic kernel
    call_basic_kernel(N, F, P, width, height);

    //call the kernel with constant memory
    call_constant_mem_kernel(N, F, P, width, height);

    //call the tiled kernel
    call_2d_tiled_constant_mem_kernel(N, F, P, width, height);

    //call the cached tiled kernel
    call_2d_cached_tiled_constant_mem_kernel(N, F, P, width, height);

    //free memory
    free(N);
    free(F);
    free(P);
    

    return 0;

}



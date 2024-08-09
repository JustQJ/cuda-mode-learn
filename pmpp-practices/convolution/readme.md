# 二维卷积
有一个二维的矩阵N，一个二维的卷积核矩阵F，计算得到一个输出矩阵P。               
基本计算就是F在N上移动，每次做一个加权和计算，得到P中的一个值。

## 计算优化

1. 基本卷积核                 
    每个线程每次从N和F中读取相应的元素，然后计算得到P中的一个值。
    ```cpp
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
    ```
    上面的计算中，两次浮点计算（一次加法和一次乘法）对应从global memory中载入8个字节的数据，**计算密度0.25 OP/B**，很低。

2. 使用constant memory
    GPU的constant memory有64KB，是一块只读的全局内存。由于只读，其有自己专门的一个cache，所以当一个swap或者block中的线程读取constant memory中的相同的地址时，就会有cache hit，提高读取速度。根据这篇文章（https://leimao.github.io/blog/CUDA-Constant-Memory/），constant memory只有在同一个block或者swap访问同一个地址时才有优势，如果一个swap中的线程访问不同的地址，还不如global memory访问的速度。                              
    由于F是只读数组，而且一般很小，所以可以放在constant memory中。而且可以看到，F被访问的特点是同一个swap的线程会访问相同的地址。                  
    ```cpp
    __constant__ float dc_F[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];
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
    ```
    constant memory的访问由于cache的存在可以忽略，所以计算密度提升为**0.5 OP/B**。                    
   

3. 使用shared memory
    将P划分成不同的block，那么该block需要的数据也是N对应的block向外扩展F的半径的block。那么有两种方案安排cuda线程block：（1）线程block和N的block大小相同，每个线程load一个数据到shared memory，那么在计算时有部分线程不需要参与计算；（2）线程block和P的block大小相同，那么每个线程负责计算P中的一个数据，而在load数据时，一个线程可能需要负责多个数据的载入。这里选择第一个方案实现。
    ```cpp
    #define IN_TILE_DIM 32
    #define OUT_TILE_DIM (IN_TILE_DIM-2*FILTER_RADIUS)
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
    ```
    上面计算中，一个block处理了$OUT\_TILE\_DIM^2$个元素，每个元素需要$2*(2*FILTER\_RADIUS+1)^2$次计算，所以总的计算次数是$OUT\_TILE\_DIM^2*2*(2*FILTER\_RADIUS+1)^2$。同时，一个block载入了 $IN\_TILE\_DIM^2*4 = (OUT\_TILE\_DIM+2*FILTER\_RADIUS)^2*4$ 字节。所以最后的计算密度是
    $$
        \frac{OUT\_TILE\_DIM^2*2*(2*FILTER\_RADIUS+1)^2}{(OUT\_TILE\_DIM+2*FILTER\_RADIUS)^2*4}
    $$
    从上面的公式，可以得到大的$FILTER\_RADIUS$和$IN\_TILE\_DIM$有助于提高计算密度。

4. 忽略输入block比输出block大的部分
    由于输入block会比输出block多卷积核半径的宽度，为了方便，可以直接忽略这部分，只将中间部分载入shared memory，而外围部分直接从global memory中访问。由于外围部分会被多个block访问，所以有比较好的cache hit。
    ```cpp
    #define TILE_DIM 32
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
    ```


## 测试
使用 $9*9$ 的F和$16384*16384$的N在 RTX4090 GPU上测试，得到结果
|基本卷积核|使用constant memory|使用shared memory|忽略输入block比输出block大的部分|
|---|---|---|---|
|11798 us|10413 us|7572 us|13940 us|

结果显示，使用constant memory和不使用区别不大。使用shared memory是最好的，但是一部分不使用shared memory的结果比基本方法都差，没有弄明白为什么。


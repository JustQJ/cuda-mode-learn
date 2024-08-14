# 规约
对一个集合的数据进行规约操作（add, max, min, product），得到最后的结果。             
以下面的cpu代码为基础，构建相应的cuda并行代码。         

```cpp
/*
对数组进行求和
*/
float cpu_sum(float *a, u_int64_t n){
    float sum = 0.0f;
    for (uint64_t i = 0; i < n; ++i) {
        sum += a[i];           
    }

    return sum;
}
```


## 计算优化
1. 基本算子
    由于规约需要线程的同步，而cuda无法在整个grid进行同步，只能在block进行同步，因此先只使用一个block进行规约。
    ```cpp
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
    ```
    一个thread处理两个元素，所以一个block处理的线程数量是`2*blockDim.x`。一个block最多只有1024个thread，最多处理2048个元素的规约。


2. 减少control divergence 和 memory divergence
    `one_block_reduction_kernel()`中，每个thread首先处理相邻的元素，然后`stride`变大，并且激活的thread减少。但是激活的thread分布在不同的swap中，造成同一个swap中只激活部分的thread，导致control divergence。并且同一个swap中的激活的thread的访存并不连续，所以无法合并访存。            
    将`stride` 增加变成减小，即所有线程开始的位置是连续位于数组的开始位置。 
    ```cpp
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
    
    ```
    上面的实现保证了激活的thread始终是从开始位置连续的thread，尽可能出现在同一个swap。同时连续的thread的访存是连续的。
    

3. 使用shared memory
    ```cpp
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
    ```


4. 多个block处理任意大小数组        
    每个block处理 `2*blockDim.x` 大小的部分，最后所有的block的第一个thread将自己的结果使用`atomicAdd()` 加到最后的结果中。
    ```cpp
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
    
    ```

5. 每个线程处理多个元素
    上面每个线程只处理两个元素，但是当数组很大时，会造成过多的block，抢占硬件资源。定义一个`COARSEN_FACTOR` ，每个线程处理`COARSEN_FACTOR*2` 个元素，每个block处理`COARSEN_FACTOR * 2 * blockDim.x` 个元素。
    ```
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
    
    ```
    开始的一个thread处理多个元素的规约是线程之间独立的，不需要同步。





**注：优化2是重点，简单的计算模式的修改以充分利用GPU的硬件特性**


## 测试

使用数据大小为$1024*1024*512$ 的数组在RTX4090 GPU上测试，得到结果
```
cpu compute 536870912 elements cost time: 4928176.000000 us
cpu compute 2048 elements cost time: 18.200001 us
one block reduction kernel compute 2048 elements cost time: 6.400000 us, speedup 2.84
one block reduction decreasing stride kernel compute 2048 elements cost time: 3.100000 us, speedup 5.87
one block reduction decreasing stride shared memory kernel compute 2048 elements cost time: 3.100000 us, speedup 5.87
multiple blocks reduction kernel compute 536870912 elements cost time: 3401.399902 us, speedup 1448.87
multiple blocks reduction thread coarsening kernel compute 536870912 elements cost time: 2240.300049 us, speedup 2199.78
```

|kernel|size|speedup|
|---|---|---|
|cpu|2048|1|
|one_block_reduction_kernel|2048|2.84|
|one_block_reduction_decreasing_stride_kernel|2048|5.87|
|one_block_reduction_decreasing_stride_shared_mem_kernel|2048|5.87|
|cpu|1024x1024x512|1|
|multiple_blocks_reduction_kernel|1024x1024x512|1448.87|
|multiple_blocks_reduction_thread_coarsening_kernel|1024x1024x512|2199.78|

结果基本符合预期，但是`one_block_reduction_decreasing_stride_shared_mem_kernel` 相对于 `one_block_reduction_decreasing_stride_kernel` 基本没有提升，应该是只启动了一个block，差距太小没体现出来。
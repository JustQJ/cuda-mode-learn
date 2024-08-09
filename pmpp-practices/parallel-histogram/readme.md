# 数据统计
对数据进行统计，方便绘制直方图。     
下面均以以下cpu代码为基础，构建相应的cuda并行代码。          
```cpp
/*
统计一个字符串中的各个字符区间中的字符出现的次数
字符区间： a-d , e-h, i-l, m-p, q-t, u-x, y-z
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

```

要点：该代码的cuda实现主要关注的是多个线程写同一个地址，如何减少竞争。              
```cpp
//主要算子
atomicAdd()
```


## 计算优化

1. 基本算子
    每个线程处理一个元素，并将结果写回全局内存的数组中。所有的线程同时竞争一个数组，等待时间长。
    ```cpp
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

    ```
    

2. 每个block拥有一份拷贝
    每个block在共享内存中创建一个结果数组，然后block内的线程只将结果写入到私有的数组。竞争只在一个block的线程内发生。

    ```cpp
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
    ```

3. 一个线程处理多个字符
    当数据量很大时，一个线程处理一个字符会导致产生大量的block。一是会导致硬件资源的不够用，因为每个block都会需要一个共享内存数组；二是最后写回全局内存结果数组时竞争会随着block的数量增加而增加。                         
    一共有两种方法，第一种是一个线程处理连续的几个字符（`contiguous`），第二种是间隔处理（`interleaved`）。在cpu上第一种方法能提高cache，但在gpu上，由于数据的读取是一个swap的所有线程同时进行的，所以第一种方法会导致一个swap中的线程读取的数据地址不连续，无法将读取操作合并。而第二种方法可以。
    ```cpp
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

    
    ```


4. 使用计算器减少写入的次数
    对于一个线程，其每次都会读取一个字符，然后写回结果数组。但是如果数据的重复比例较高，那么可能连续访问的几个字符都是相同的，可以记录完再一次性写回，减少写回次数。但这样会增加判断，造成线程分化，不一定能提高性能。
    ```cpp
    
    __global__ void histogram_shared_mem_multi_interleaved_counter_kernel(char *data,int length, int *histogram){
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

    ```

## 测试
使用数据大小为$1024*1024*1024$ 的字符串在RTX4090 GPU上测试，得到结果
```
cpu time: 2953368 us
histogram_kernel time: 203713 us, speedup: 14.50
histogram_shared_mem_kernel time: 2746 us, speedup: 1075.28
histogram_shared_mem_multi_contiguous_kernel time: 1135 us, speedup: 2601.86
histogram_shared_mem_multi_interleaved_kernel time: 1935 us, speedup: 1525.97
histogram_shared_mem_multi_interleaved_counter_kernel time: 2152 us, speedup: 1372.13
```
`histogram_shared_mem_multi_interleaved_kernel`比`histogram_shared_mem_multi_contiguous_kernel`差，有些奇怪。

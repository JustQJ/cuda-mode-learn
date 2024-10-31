
#include<stdio.h>
#include<stdlib.h>
#include<unordered_set>
#include<random>
#include<vector>
#include<assert.h>
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


#define BLOCK_SIZE 256


/*
focus on the matrix-vector multiplication of sparse matrix: y = A*x
A is a sparse matrix, x is a vector, y is the result vector
formats:
    - coo
    - csr
    - ell
    - jds

*/


//original matrix-vector multiplication 
//y = A*x
__global__ void spmv_kernel(float *A, float *x, float *y, int m, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<m){
        float sum = 0.0;
        for(int j=0; j<n; j++){
            sum += A[i*n+j]*x[j];
        }
        y[i] = sum;
    }
}

//coo format

__global__ void spmv_coo_kernel(int *rows, int *cols, float *vals, float *x, float *y, int nnz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<nnz){
        int col_idx = cols[i];
        int row_idx = rows[i];

        atomicAdd(&y[row_idx], vals[i]*x[col_idx]);
    }
}


// csr format
//avoid the atomic operation
__global__ void spmv_csr_kernel(int *row_ptr, int *cols, float *vals, float *x, float *y, int m, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<m){
        float sum = 0.0;
        for(int j=row_ptr[i]; j<row_ptr[i+1]; j++){ // row_ptr[i] is the start index of row i, row_ptr[i+1] is the start index of row i+1, its size is m+1 for the last element
            sum += vals[j]*x[cols[j]]; //sum the row i of all non-zero elements
        }
        y[i] = sum;
    }
}


// ell format
// coalesced memory access
// padding all rows to the same max length
//good for uniform  distribution of non-zero elements
__global__ void spmv_ell_kernel(int *cols, float *vals, float *x, float *y, int m, int n, int max_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<m){
        float sum = 0.0;
        for(int j=0; j<max_length; j++){
            int idx = j*m+i;
            int col_idx = cols[idx]; // 0 padding, no impact on the result
            
            sum += vals[idx]*x[col_idx];
            
        }
        y[i] = sum;
    }
}


// jds format
// sort rows by the number of non-zero elements
// good for non-uniform distribution of non-zero elements
__global__ void spmv_jds_kernel(int *cols, float *vals, int *row_ptr, int *jds_ptr, float *x, float *y, int m, int n, int max_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<m){
        float sum = 0.0;
        int row_idx = row_ptr[i];
        for(int j=0; j<max_length; j++){
            int idx = jds_ptr[j]+i;
            if(idx<jds_ptr[j+1]){
                int col_idx = cols[idx];
                sum += vals[idx]*x[col_idx];
            }
        }
        
        y[row_idx] = sum;
    }
}





std::vector<int> generate_unique_random_numbers(int n, int k) {
    std::unordered_set<int> unique_numbers; // use set to avoid duplicates
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n-1);

    while (unique_numbers.size() < k) { // generate k unique random numbers
        unique_numbers.insert(dis(gen));
    }

    return std::vector<int>(unique_numbers.begin(), unique_numbers.end());
}


int main(){

    int m = 10000;
    int n = 10000;
    int nnz = 1000000; // number of non-zero elements 0.1% of the matrix

    //construct a sparse matrix
    float *A = (float *)malloc(m*n*sizeof(float));
    float *x = (float *)malloc(n*sizeof(float));
    float *y = (float *)malloc(m*sizeof(float));

    //random generate nzz index from 0 to m*n, no duplicate
    // int *indexs = (int *)malloc(nnz*sizeof(int));
    std::vector<int> unique_indexs = generate_unique_random_numbers(m*n, nnz);
    for(int i=0; i<nnz; i++){
        int idx = unique_indexs[i];
        int row = idx/n;
        int col = idx%n;
        A[row*n+col] = (float)rand()/RAND_MAX;
    }
    


    //copy to device
    float *d_A, *d_x, *d_y;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, m*n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, m*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, y, m*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, m*sizeof(float)));

    //measure the time
    int number = 10;
    //warm up
    spmv_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_x, d_y, m, n);
    spmv_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_x, d_y, m, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t start_time = get_us_time();
    for(int i=0; i<number; i++){
        spmv_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_x, d_y, m, n);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    u_int64_t end_time = get_us_time();
    printf("original matrix-vector multiplication time: %f us\n", (end_time-start_time)*1.0/number);
    float base_time = (end_time-start_time)*1.0/number;



    //coo format
    int *rows = (int *)malloc(nnz*sizeof(int));
    int *cols = (int *)malloc(nnz*sizeof(int));
    float *vals = (float *)malloc(nnz*sizeof(float));

    for(int i=0; i<nnz; i++){
        int idx = unique_indexs[i];
        rows[i] = idx/n;
        cols[i] = idx%n;
        vals[i] = A[idx];
    }

    int *d_rows, *d_cols;
    float *d_vals;
    CHECK_CUDA_ERROR(cudaMalloc(&d_rows, nnz*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cols, nnz*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_vals, nnz*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_rows, rows, nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cols, cols, nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vals, vals, nnz*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, m*sizeof(float)));

    //measure the time
    //warm up
    spmv_coo_kernel<<<(nnz+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_rows, d_cols, d_vals, d_x, d_y, nnz);
    spmv_coo_kernel<<<(nnz+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_rows, d_cols, d_vals, d_x, d_y, nnz);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    start_time = get_us_time();
    for(int i=0; i<number; i++){
        spmv_coo_kernel<<<(nnz+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_rows, d_cols, d_vals, d_x, d_y, nnz);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();
    printf("coo matrix-vector multiplication time: %f us, speedup: %f\n", (end_time-start_time)*1.0/number, base_time*1.0/((end_time-start_time)*1.0/number));

    //csr format
    int *row_ptr = (int *)malloc((m+1)*sizeof(int));
    //cols and values are the same as coo format
    row_ptr[0] = 0;
    for(int i=0; i<m; i++){
        row_ptr[i+1] = row_ptr[i];
        for(int j=0; j<n; j++){
            if(A[i*n+j]!=0){
                row_ptr[i+1]++;
            }
        }
    }
    // row_ptr[m] should be nnz
    assert (row_ptr[m] == nnz);

    int *d_row_ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_row_ptr, (m+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_row_ptr, row_ptr, (m+1)*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, m*sizeof(float)));

    //measure the time
    //warm up
    spmv_csr_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_row_ptr, d_cols, d_vals, d_x, d_y, m, n);
    spmv_csr_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_row_ptr, d_cols, d_vals, d_x, d_y, m, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    start_time = get_us_time();
    for(int i=0; i<number; i++){
        spmv_csr_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_row_ptr, d_cols, d_vals, d_x, d_y, m, n);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();
    printf("csr matrix-vector multiplication time: %f us, speedup: %f\n", (end_time-start_time)*1.0/number, base_time*1.0/((end_time-start_time)*1.0/number));



    //ell format

    int max_length = 0;
    for(int i=0; i<m; i++){
        max_length = std::max(max_length, row_ptr[i+1]-row_ptr[i]);
    }
    int *ell_cols = (int *)malloc(m*max_length*sizeof(int));
    float *ell_vals = (float *)malloc(m*max_length*sizeof(float));

    for(int i=0; i<m; i++){
        for(int j=0; j<max_length; j++){
            if(j<row_ptr[i+1]-row_ptr[i]){
                ell_cols[j*m+i] = cols[row_ptr[i]+j];
                ell_vals[j*m+i] = vals[row_ptr[i]+j];
            }else{
                ell_cols[j*m+i] = 0;
                ell_vals[j*m+i] = 0;
            }
        }
    }

    int *d_ell_cols;
    float *d_ell_vals;
    CHECK_CUDA_ERROR(cudaMalloc(&d_ell_cols, m*max_length*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ell_vals, m*max_length*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_ell_cols, ell_cols, m*max_length*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ell_vals, ell_vals, m*max_length*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, m*sizeof(float)));

    //measure the time
    //warm up
    spmv_ell_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_ell_cols, d_ell_vals, d_x, d_y, m, n, max_length);
    spmv_ell_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_ell_cols, d_ell_vals, d_x, d_y, m, n, max_length);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    start_time = get_us_time();
    for(int i=0; i<number; i++){
        spmv_ell_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_ell_cols, d_ell_vals, d_x, d_y, m, n, max_length);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();
    printf("ell matrix-vector multiplication time: %f us, speedup: %f\n", (end_time-start_time)*1.0/number, base_time*1.0/((end_time-start_time)*1.0/number));




    //jds format
    int *jds_rows = (int *)malloc(m*sizeof(int));   
    int *rows_length = (int *)malloc(m*sizeof(int));
    int *jds_cols = (int *)malloc(nnz*sizeof(int));
    float *jds_vals = (float *)malloc(nnz*sizeof(float));
    int *jds_ptr = (int *)malloc((max_length+1)*sizeof(int));

    for(int i=0; i<m; i++){
        jds_rows[i] = i;
        rows_length[i] = row_ptr[i+1]-row_ptr[i];
    }
    //sort jds_rows by rows_length in  descending order
    for(int i=0; i<m; i++){
        for(int j=i+1; j<m; j++){
            if(rows_length[j]>rows_length[i]){
                std::swap(jds_rows[i], jds_rows[j]);
                std::swap(rows_length[i], rows_length[j]);
            }
        }
    }

    jds_ptr[0] = 0;
    for(int j=0; j<max_length; j++){
        jds_ptr[j+1] = jds_ptr[j];
        for(int i=0; i<m; i++){
            if(j<rows_length[i]){
                jds_cols[jds_ptr[j]+i] = cols[row_ptr[jds_rows[i]]+j];
                jds_vals[jds_ptr[j]+i] = vals[row_ptr[jds_rows[i]]+j];
                jds_ptr[j+1]++;
            }
        }
    }

    int *d_jds_rows, *d_jds_cols, *d_jds_ptr;
    float *d_jds_vals;
    CHECK_CUDA_ERROR(cudaMalloc(&d_jds_rows, m*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_jds_cols, nnz*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_jds_vals, nnz*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_jds_ptr, (max_length+1)*sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_jds_rows, jds_rows, m*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_jds_cols, jds_cols, nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_jds_vals, jds_vals, nnz*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_jds_ptr, jds_ptr, (max_length+1)*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemset(d_y, 0, m*sizeof(float)));

    //measure the time
    //warm up
    spmv_jds_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_jds_cols, d_jds_vals, d_row_ptr, d_jds_ptr, d_x, d_y, m, n, max_length);
    spmv_jds_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_jds_cols, d_jds_vals, d_row_ptr, d_jds_ptr, d_x, d_y, m, n, max_length);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    start_time = get_us_time();
    for(int i=0; i<number; i++){
        spmv_jds_kernel<<<(m+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_jds_cols, d_jds_vals, d_row_ptr, d_jds_ptr, d_x, d_y, m, n, max_length);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end_time = get_us_time();
    printf("jds matrix-vector multiplication time: %f us, speedup: %f\n", (end_time-start_time)*1.0/number, base_time*1.0/((end_time-start_time)*1.0/number));


    





















    //free memory
    free(A);
    free(x);
    free(y);
    free(rows);
    free(cols);
    free(vals);
    free(row_ptr);
    free(ell_cols);
    free(ell_vals);
    free(jds_rows);
    free(rows_length);
    free(jds_cols);
    free(jds_vals);
    free(jds_ptr);

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_rows));
    CHECK_CUDA_ERROR(cudaFree(d_cols));
    CHECK_CUDA_ERROR(cudaFree(d_vals));
    CHECK_CUDA_ERROR(cudaFree(d_row_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_ell_cols));
    CHECK_CUDA_ERROR(cudaFree(d_ell_vals));
    CHECK_CUDA_ERROR(cudaFree(d_jds_rows));
    CHECK_CUDA_ERROR(cudaFree(d_jds_cols));
    CHECK_CUDA_ERROR(cudaFree(d_jds_vals));
    CHECK_CUDA_ERROR(cudaFree(d_jds_ptr));
    

    return 0;







}
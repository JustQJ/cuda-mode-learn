
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<mpi.h>



void random_data(float *data, uint num_points){
    // for(uint i=0; i<num_points; i++){
    //     data[i] = (float)rand()/(float)RAND_MAX;
    // }
}

void store_output(float *data, int dimx, int dimy, int dimz){
    //store the output data
}


void server_process(int dimx, int dimy, int dimz){
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np); // get the number of processes

    uint num_comp_nodes = np-1, first_node=0, last_node=np-2; //compute nodes, first and last compute node id
    uint num_points = dimx*dimy*dimz; //number of data points
    uint num_bytes = num_points*sizeof(float); //number of bytes to store the data

    float *input=0, *output=0;
    input = (float*)malloc(num_bytes);
    output = (float*)malloc(num_bytes);
    if(input==NULL || output==NULL){
        printf("Error: unable to allocate memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
       
    }

    //initialize input data
    random_data(input, num_points);

    // compute the number of points to be processed by each compute node, first and last node may have different number of points 
    int edge_num_points = dimx*dimy*(dimz/num_comp_nodes + 4); //4 is needed for the halo from the previous or next node, first and last node
    int int_num_points = dimx*dimy*(dimz/num_comp_nodes + 8); //8 is needed for the halo from the previous and next node, internal nodes

    float *send_address = input;

    MPI_Send(send_address, edge_num_points, MPI_FLOAT, first_node, 0, MPI_COMM_WORLD); //send data to the first node
    send_address += dimx*dimy*(dimz/num_comp_nodes - 4); //move the pointer to the next node, because we need to send the halo, so -4
    for(int process=1; process<last_node; process++){
        MPI_Send(send_address, int_num_points, MPI_FLOAT, process, 0, MPI_COMM_WORLD); //send data to the internal nodes
        send_address += dimx*dimy*(dimz/num_comp_nodes); //move the pointer to the next node
    }
    MPI_Send(send_address, edge_num_points, MPI_FLOAT, last_node, 0, MPI_COMM_WORLD); //send data to the last node


    //receive the results

    MPI_Barrier(MPI_COMM_WORLD); //wait for all the processes to finish
    MPI_Status status;
    for(int process=0; process<num_comp_nodes; process++){
        MPI_Recv(output+process*num_points/num_comp_nodes, num_points/num_comp_nodes, MPI_FLOAT, process, 0, MPI_COMM_WORLD, &status);
        //process the results
    }
    store_output(output, dimx, dimy, dimz);
    free(input);
    free(output);




}


__global__ void stencil_kernel(float *output, float *input, int dimx, int dimy, int dimz){
    // idx for three dimensions
    int x = threadIdx.x + blockIdx.x*blockDim.x;    
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;

    if(x<dimx && y<dimy && z<dimz){
        int idx = x + y*dimx + z*dimx*dimy + 4*dimx*dimy; //4 is needed for the halo from the previous and next node
        output[idx] = 0.25f*(input[idx]+input[idx]+input[idx]+input[idx]+input[idx]);  // is not right, just a case  + 0.125f*(input[idx+1]+ input[idx-1] + input[idx+dimx] + input[idx-dimx] + input[idx+dimx*dimy] + input[idx-dimx*dimy]);
    }

}


void call_stencil_kernel(float *output, float *input, int dimx, int dimy, int dimz, cudaStream_t stream){
    dim3 block(8, 8, 8);
    dim3 grid((dimx+block.x-1)/block.x, (dimy+block.y-1)/block.y, (dimz-8+block.z-1)/block.z);  //remove the halo, 4 left, 4 right
    
    stencil_kernel<<<grid, block, 0, stream>>>(output, input, dimx, dimy, dimz-8);
    
}

void compute_process(int dimx, int dimy, int dimz, int nreps){
    int np, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &np); // get the number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &pid); // get the process id

    int server_process = np-1; //server process id
    uint num_points = dimx*dimy*(dimz+8); //number of data points, 8 is needed for the halo from the previous and next node
    uint num_bytes = num_points*sizeof(float); //number of bytes to store the data
    uint num_halo_points = dimx*dimy*4; //number of halo points
    uint num_halo_bytes = num_halo_points*sizeof(float); //number of bytes to store the halo data

    //allocate memory for the data
    float *h_input = (float*)malloc(num_bytes);
    //set zero to the input data
    memset(h_input, 0, num_bytes);
    float *d_input = NULL;
    cudaMalloc(&d_input, num_bytes);
    float *rcv_address = h_input + ((pid==0)?num_halo_points:0); //move the pointer to the correct position, first process has no previous halo
    MPI_Recv(rcv_address, num_points, MPI_FLOAT, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receive the data
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice); //copy the data to the device

    //allocation  output
    float *h_output = (float*)malloc(num_bytes);
    float *d_output = NULL;
    cudaMalloc(&d_output, num_bytes);

    float *h_left_boundary = NULL, *h_right_boundary = NULL;  //for send data to the previous and next node
    float *h_left_halo = NULL, *h_right_halo = NULL; //for receive data from the previous and next node

    //pinned memory for the asynchronous data transfer, if not the pinned memory, cudaMemcpuAsync will be not turlly asynchronous
    cudaHostAlloc(&h_left_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_left_halo, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_halo, num_halo_bytes, cudaHostAllocDefault);

    //create streams, for the computation and data transfer
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);


    int left_neighbor = (pid==0)?MPI_PROC_NULL:pid-1; //left neighbor
    int right_neighbor = (pid==np-2)?MPI_PROC_NULL:pid+1; //right neighbor

    //do computing iterations

    int left_halo_offset = 0, right_halo_offset = dimx*dimy*(dimz+4); //offset for the halo data
    int left_stage1_offset = 0;
    int right_stage1_offset = dimx*dimy*(dimz-4);
    int stage2_offset = num_halo_points;

    MPI_Barrier(MPI_COMM_WORLD); //wait for all the processes to start
    double start_time = MPI_Wtime();
    for(int i=0; i<nreps; i++){
        //compute the left halo and right halo first, need to send them to the previous and next node
        call_stencil_kernel(d_output+left_stage1_offset, d_input+left_stage1_offset, dimx, dimy, 12, stream0);
        call_stencil_kernel(d_output+right_stage1_offset, d_input+right_stage1_offset, dimx, dimy, 12, stream0);

        //compute the internal points
        call_stencil_kernel(d_output+stage2_offset, d_input+stage2_offset, dimx, dimy, dimz, stream1);

        //copy the halo data
        cudaMemcpyAsync(h_left_boundary, d_output+num_halo_points, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_right_boundary, d_output+right_stage1_offset+num_halo_points, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream0);
        // cudaMemcpy(h_left_boundary, d_output+num_halo_points, num_halo_bytes, cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_right_boundary, d_output+right_stage1_offset+num_halo_points, num_halo_bytes, cudaMemcpyDeviceToHost);


        //send data to the previous and next node and receive the halo data

        MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT, left_neighbor, i, h_right_halo, num_halo_points, MPI_FLOAT, right_neighbor, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_right_boundary, num_halo_points, MPI_FLOAT, right_neighbor, i, h_left_halo, num_halo_points, MPI_FLOAT, left_neighbor, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //copy the halo data to the device
        cudaMemcpyAsync(d_output+left_halo_offset, h_left_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_output+right_halo_offset, h_right_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);

        cudaDeviceSynchronize(); //wait for all tasks to finish

        //swap the input and output
        float *temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    double end_time = MPI_Wtime();
    printf("Process %d, elapsed time: %f seconds\n", pid, end_time-start_time);

    //wait all process
    
    //undo the swap in last iteration
    float *temp = d_input;
    d_input = d_output;
    d_output = temp;

    //send data to the server, skip all halo data
    cudaMemcpy(h_output, d_output, num_bytes, cudaMemcpyDeviceToHost);
    float *send_address = h_output + ((pid==0)?num_halo_points:0);
    MPI_Send(send_address, dimx*dimy*dimz, MPI_FLOAT, server_process, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    //free

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_left_boundary);
    cudaFreeHost(h_right_boundary);
    cudaFreeHost(h_left_halo);
    cudaFreeHost(h_right_halo);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);



}


/*
nvcc -o cluster cluster.cu  -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcudart
mpirun -np 4 ./cluster
*/



// int main(){

//     int num_bytes = 32*32*32*sizeof(float);

//     float *d_input = NULL;
//     cudaMalloc(&d_input, num_bytes);

//     //allocation  output
//     float *d_output = NULL;
//     cudaMalloc(&d_output, num_bytes);
//     cudaStream_t stream0, stream1;
//     cudaStreamCreate(&stream0);
//     cudaStreamCreate(&stream1);
//     call_stencil_kernel(d_output, d_input, 32, 32, 32, stream0);

//     return 0;

// }


int main(int argc, char* argv[]){
    int pad=0, dimx=2*480+pad, dimy=2*480, dimz=2*400, nreps=100;

    int pid=-1, np=-1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid); // get the process id
    MPI_Comm_size(MPI_COMM_WORLD, &np); // get the number of processes

    if(np<3){
        printf("This program requires at least 3 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        return 1;
    }
    
    //record the start time
    double start_time = MPI_Wtime();


    if(pid<np-1){
        compute_process(dimx, dimy, dimz/(np-1), nreps); //compute 
    }else{
        server_process(dimx, dimy, dimz); //generate data, assign work and collect results
    }

    //record the end time
    double end_time = MPI_Wtime();
    if(pid==np-1){
        printf("Elapsed time: %f seconds\n", end_time-start_time);
    }

    MPI_Finalize();
    return 0;
}
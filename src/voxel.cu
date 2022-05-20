#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void voxel_kernel(float * data, int * indexes, int n, int x_size, int y_size, int z_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n)
    {
        int x_idx = indexes[3 * index];
        int y_idx = indexes[3 * index + 1];
        int z_idx = indexes[3 * index + 2];

        *(data + z_idx * y_size * x_size + y_idx * x_size + x_idx) = 1;
    }
}

void voxel_gpu(int * indexes, float * data, int N, int x_size, int y_size, int z_size)
{
    

    int * d_indexes;
    cudaMalloc((void**)&d_indexes, N * sizeof(int));
    cudaMemcpy(d_indexes, indexes, N * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = ((N / 3) + block_size - 1) / block_size;
    voxel_kernel<<<num_blocks, block_size>>>(data, d_indexes, N / 3, x_size, y_size, z_size);
    cudaDeviceSynchronize();
    cudaFree(d_indexes);
}

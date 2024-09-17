#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>
#include <iostream>
#include <bitset>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <error.h>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

typedef unsigned __int128 uint128_t;
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)
#define CEIL(a, b) (((a)+(b)-1)/(b))

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}


__global__ void index_kernel(float *res, long *indices, float *src, int upper_bound, int dim)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < upper_bound){
        for(int i=threadIdx.x; i<dim; i+=blockDim.x){
            res[idx * dim + i] = src[indices[idx] * dim + i];
        }
    }
}

torch::Tensor zero_copyH2D(torch::Tensor emb, torch::Tensor indices, int dev_id) {

    cudaSetDevice(dev_id);
    dim3 block(32, 32);
    dim3 grids = (CEIL(indices.size(0), block.y));
    dim3 grids_vec = (CEIL(indices.size(0), block.y*block.x));
    torch::Device dev = indices.device();
    
    long * idx;
    CHECK(cudaMalloc(&idx, sizeof(long) * indices.size(0)));
    CHECK(cudaMemcpy(idx, indices.data_ptr<long>(), sizeof(long) * indices.size(0), cudaMemcpyHostToDevice));
    torch::Tensor res = torch::empty({indices.size(0), emb.size(1)}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA, dev_id));
    index_kernel<<< grids, block, 0 >>>(res.data_ptr<float>(), idx, emb.data_ptr<float>(), indices.size(0), emb.size(1));
    CHECK(cudaFree(idx));
    cudaDeviceSynchronize();return res;
        
}

__global__ void write_kernel(float *emb, long *indices, float *res, int upper_bound, int dim)
{
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < upper_bound){
        for(int i=threadIdx.x; i<dim; i+=blockDim.x){
            emb[indices[idx] * dim + i] += res[idx * dim + i];
        }
    }
}

void zero_writeD2H(torch::Tensor emb, torch::Tensor res, torch::Tensor indices, int dev_id){
    cudaSetDevice(dev_id);
    dim3 block(32, 32);
    dim3 grids = (CEIL(indices.size(0), block.y));

    torch::Device dev = indices.device();
    long * idx;
    CHECK(cudaMalloc(&idx, sizeof(long) * indices.size(0)));
    CHECK(cudaMemcpy(idx, indices.data_ptr<long>(), sizeof(long) * indices.size(0), cudaMemcpyHostToDevice));
    
    write_kernel<<< grids, block, 0 >>>(emb.data_ptr<float>(), idx, res.data_ptr<float>(), indices.size(0), emb.size(1));
    cudaDeviceSynchronize();
    
    CHECK(cudaFree(idx));
}



void pin_mem(torch::Tensor emb){
    CHECK(cudaHostRegister(emb.data_ptr<float>(), sizeof(float) * emb.size(0)*emb.size(1), cudaHostRegisterPortable| cudaHostAllocMapped));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("zero_copy_call", &zero_copyH2D, "zero copy data read from cpu to gpu");
    m.def("zero_write", &zero_writeD2H, "zero copy data read from cpu to gpu");
    m.def("pin_mem", &pin_mem, "pin memory on CPU");
}
import torch.multiprocessing as mp
import torch as th
import time
import sys
from torch.utils.cpp_extension import load
import numpy as np
import os 

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
zerocopy_cpp = load(name='testcase', sources=['zp_test.cu'], extra_cflags=['-I/usr/local/cuda/include'], extra_cuda_cflags=['-I/usr/local/cuda/include'], extra_ldflags=['-lcuda', '-ldl'])


class ZeroCopy(th.autograd.Function):
    @staticmethod
    def forward(ctx, emb, indices, device):
        output = zerocopy_cpp.zero_copy_call(emb, indices, device)
        return output

    @staticmethod
    def backward(ctx):
        pass

class ZeroWrite(th.autograd.Function):
    @staticmethod
    def forward(ctx, emb, res, indices, device):
        zerocopy_cpp.zero_write(emb, res, indices, device)

    @staticmethod
    def backward(ctx):
        pass

class Pin_Mem(th.autograd.Function):
    @staticmethod
    def forward(ctx, emb):
        zerocopy_cpp.pin_mem(emb)
        return emb

    @staticmethod
    def backward(ctx):
        pass

zero_copy = ZeroCopy.apply
zero_write = ZeroWrite.apply
pin_mem = Pin_Mem.apply


def train_mp(emb, rank):
    pin_mem(emb)
    copy_time=0
    write_time=0
    iter_num = 100
    warm_up  = 10
    numElements = emb.size(0)
    fetchNums  = 5000000
    
    for _ in range(warm_up):
        indices = th.randint(0, numElements, (fetchNums,))
        data = zero_copy(emb, indices, rank)
        grad = data*0.1
        zero_write(emb, grad, indices, rank)
       
    for _ in range(iter_num):
         
        #indices = th.randint(0, numElements, (fetchNums,))
        indices = th.arange(0,numElements).repeat(fetchNums//numElements+1)[:fetchNums]
        start = time.time()
 
        data = zero_copy(emb, indices, rank)
        th.cuda.synchronize()
        copy_time+=time.time() - start
        grad = data*0.1
        start = time.time()
        zero_write(emb, grad, indices, rank)
        th.cuda.synchronize()
        write_time += time.time() - start
    
    throughputInGBs = indices.nelement()*iter_num*float(32)/1e9
    print(throughputInGBs)
    print('copy time throughput on {} is {} GB/s'.format(rank, throughputInGBs/copy_time))
    print('write time throughput on {} is {} GB/s'.format(rank, throughputInGBs/write_time))


def main():
    mp.set_start_method('forkserver')
    num_gpus = 1

    data = th.rand((3000000, 100))
    
    pin_mem(data)
    procs = []
    for i in range(num_gpus):
        proc = mp.Process(target=train_mp, args=(data, i))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    
if __name__ == '__main__':
    main()
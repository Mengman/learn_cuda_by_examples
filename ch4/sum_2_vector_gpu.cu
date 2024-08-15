#include "../common/book.h"
#include <time.h>
#define N 100

__global__ void add_block(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

__global__ void add_thread(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}


int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDEL_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add_block<<<N, 1>>>(dev_a, dev_b, dev_c);

    add_thread<<<1, N>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

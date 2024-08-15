// https://siboehm.com/articles/22/CUDA-MMM
// 定义 A(M,K)  B(K,N) 的矩阵
#define M 1024
#define N 1024
#define K 1024

// each block responsible for caculating a 32x32 chunk of C
// each thread independently computes one entry of C
__global__ void sgemm_native(
    int M, int N, int K,
    const float *A, const float *B, float *C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0; 
        for (int i = 0; i < K; ++i)
        {
            tmp += A[x * K + i] * B[y * N + i];
        }

        C[x * N + y] = tmp;
    }
}


__global__ void sgemm_coalescing(
    int M, int N, int K,
    const float *A, const float *B, float *C)
{
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (x < M && y < N) {
        float tmp = 0.0;
        for(int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[y * N + i];
        }
        C[x * N + y] = tmp;
    }
}

__global__ void sgemm_shared_mem(
    int M, int N, int K,
    const float *A, const float *B, float *C)
{
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        A += cRow * BLOCKSIZE * K;

        float tmp = 0.0;

    }

}

int main()
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1)
    sgemm_native<<<gridDim, blockDim>>>(M, N, K, A, B, C);

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, A, B, C);


}
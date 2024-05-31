#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__device__ void sumReduce(float* smem, int tid){
    for(int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32){
        volatile float* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
}
__global__ void gemvKernel4(float *y, float *A, float *x, float *b, int m, int k) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    __shared__ float shared_buf[1024];
    float* A_row = A + row * k;

    for(int i = idx;i<k;i+=blockDim.x){
        shared_buf[idx] += A_row[i] * x[i];
    }
    __syncthreads();

    sumReduce(shared_buf, idx);

    if (idx == 0) {
        float z = shared_buf[0];
        if (b) {
            z += b[row];
        }
        y[row] = z;
    }
}

__global__ void gemvGeluKernel4(float *y, float *A, float *x, float *b, int m, int k) {
    int row = blockIdx.x;
    int idx = threadIdx.x;
    __shared__ float shared_buf[1024];
    float* A_row = A + row * k;

    for(int i = idx;i<k;i+=blockDim.x){
        shared_buf[idx] += A_row[i] * x[i];
    }
    __syncthreads();
    sumReduce(shared_buf, idx);
    if (idx == 0) {
        float z = shared_buf[0] + b[row];
        y[row] = 0.5f * z * (1.0f + tanhf(0.7978845608028654f * (z + 0.044715f * z * z * z)));
        y[row] = z / (1 + expf(-1.702*z));
    }
}

// y = A * x + b
// A is m x k
// x is k x 1
// b is m x 1
void gemv(float *y, float *A, float *x, float *b, int m, int k) {
    int numThreads = 1024;
    int numBlocks = m;
    gemvKernel4<<<numBlocks, numThreads>>>(y, A, x, b, m, k);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void gemvGelu(float *y, float *A, float *x, float *b, int m, int k) {
    int numThreads = 1024;
    int numBlocks = m;
    gemvGeluKernel4<<<numBlocks, numThreads>>>(y, A, x, b, m, k);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in gemvGeluKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}

void gemv_cpu(float *y, float *A, float *x, float *b, int m, int k) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < k; j++) {
            sum += A[i * k + j] * x[j];
        }
        y[i] = sum + (b ? b[i] : 0);
    }
}
void rand_init(float* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX / (float)RAND_MAX;
    }
}

int main(){

    int m = 1600;
    int k = 1600;
    float* A, *x, *y, *b;
    A = (float*)malloc(m * k * sizeof(float));
    x = (float*)malloc(k * sizeof(float));
    y = (float*)malloc(m * sizeof(float));
    b = (float*)malloc(m * sizeof(float));
    rand_init(A, m * k);
    rand_init(x, k);
    rand_init(b, m);

    float* d_A, *d_x, *d_y, *d_b;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_x, k * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    cudaMalloc(&d_b, m * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m * sizeof(float), cudaMemcpyHostToDevice);

    gemv_cpu(y, A, x, b, m, k);

    gemv(d_y, d_A, d_x, d_b, m, k);

    float* check = (float*)malloc(m * sizeof(float));
    cudaMemcpy(check, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0;i<10;i++){
        cout << y[i] << " " << check[i] << endl;
    }
    for(int i = 0;i<m;i++){
        if(abs(y[i] - check[i]) > 1e-5){
            cout << "Incorrect try again!" << endl;
            return 0;
        }
    }
    cout << "Correct!" << endl;

    gemvGelu(d_y, d_A, d_x, d_b, m, k);

    
    
}
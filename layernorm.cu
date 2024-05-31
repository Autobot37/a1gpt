#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define cudaCheck(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)


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

__global__ void layerNormKernel2(float* output, int embedding_dim, float* gamma, float* beta, float* input){

    __shared__ float shared[2048];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sz = blockDim.x;

    if(idx>=embedding_dim){
        return;
    }

    float mean = 0;
    float variance = 0;
    
    float* shared_m = shared;
    float* shared_v = shared + 1024;

    shared_m[threadIdx.x] = 0.0f;
    shared_v[threadIdx.x] = 0.0f;


    for(int i=idx;i<embedding_dim;i += gridDim.x * blockDim.x){
        shared_m[threadIdx.x] += input[i];
    }
    __syncthreads();

    sumReduce(shared_m, threadIdx.x);
    mean = shared_m[0] / embedding_dim;

    //var
    // shared_v[threadIdx.x] = (input[threadIdx.x] - mean) * (input[threadIdx.x] - mean);
    // __syncthreads();

    for(int i=idx;i<embedding_dim;i += gridDim.x * blockDim.x){
        float diff = input[i] - mean;
        shared_v[threadIdx.x] += diff * diff;
    }
    __syncthreads();

    // printf("GPU: %d %f %f %f\n", threadIdx.x, input[threadIdx.x], mean,shared_v[threadIdx.x]);

    sumReduce(shared_v, threadIdx.x);

    variance = shared_v[0];
    variance /= embedding_dim;
    const float eps = 1e-5f;
    float stddev = sqrt(variance + eps);

    // if(threadIdx.x == 0){
    //     printf("GPU mean %f variance %f stddev %f\n", mean, variance, stddev);
    // }

    output[idx] = (input[idx] - mean) / stddev * gamma[idx] + beta[idx];
}

void layerNorm2(float* output, int embedding_dim, float* gamma, float* beta, float* input) {
    int numThreads = 1024;
    int numBlocks = 1;

    layerNormKernel2<<<numBlocks, numThreads>>>(output, embedding_dim, gamma, beta, input);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error in layerNormKernel: %s\n", cudaGetErrorString(cudaStatus));
        abort();
    }
}

void layernorm_cpu(float* output, int embedding_dim, float* gamma, float* beta, float* input) {
    float mean = 0.0f;
    float variance = 0.0f;
    for (int j = 0; j < embedding_dim; j++) {
        mean += input[j];
    }
    mean /= embedding_dim;
    for (int j = 0; j < embedding_dim; j++) {
        float diff = input[j] - mean;
        variance += diff * diff;
    }
    variance /= embedding_dim;
    const float eps = 1e-5f;
    float stddev = sqrt(variance + eps);

    // cout << "CPU" << " mean " << mean << " variance " << variance << " stddev " << stddev << endl;

    for (int j = 0; j < embedding_dim; j++) {
        output[j] = (input[j] - mean) / stddev * gamma[j] + beta[j];
    }
}

void rand_init(float* arr, int siz) {
    for (int i = 0; i < siz; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

int main() {

    srand((unsigned)time(0));

    int embeddingSize = 1600;

    float* output = new float[embeddingSize];
    float* gamma = new float[embeddingSize];
    float* beta = new float[embeddingSize];
    float* input = new float[embeddingSize];

    rand_init(input, embeddingSize);
    rand_init(gamma, embeddingSize);
    rand_init(beta, embeddingSize);

    // cout << "----------------------------------" << endl;
    // for(int i =0;i<embeddingSize;i++){
    //     cout << input[i] << " ";
    // } 
    cout << endl;
    layernorm_cpu(output, embeddingSize, gamma, beta, input);

    float* d_output, *d_gamma, *d_beta, *d_input;
    cudaMalloc(&d_output, embeddingSize * sizeof(float));
    cudaMalloc(&d_gamma, embeddingSize * sizeof(float));
    cudaMalloc(&d_beta, embeddingSize * sizeof(float));
    cudaMalloc(&d_input, embeddingSize * sizeof(float));

    cudaMemcpy(d_gamma, gamma, embeddingSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, embeddingSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, embeddingSize * sizeof(float), cudaMemcpyHostToDevice);

    layerNorm2(d_output, embeddingSize, d_gamma, d_beta, d_input);

    float* check = new float[embeddingSize];
    cudaMemcpy(check, d_output, embeddingSize * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0;i<10;i++){
    //     cout << check[i] << " ";
    // }
    // cout << endl;

    // for(int i = 0;i<10;i++){
    //     cout << output[i] << " ";
    // }
    // cout << endl;

    for(int i = 0;i<embeddingSize;i++){
        if(check[i]-output[i]>1e-5){
            cout<<"try again"<<endl;
            return 0;
        }
    }
    cout<<"yay success"<<endl;
    for(int i = 0;i<10;i++){
        cout << check[i] << " ";
    }
    cout << endl;


    return 0;
}
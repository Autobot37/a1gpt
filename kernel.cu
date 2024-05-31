#include <iostream>
using namespace std;

extern __shared__ float shared_buf[];

__device__ void sumSharedMem(float *shared, int index, int siz) {
    shared += index;
    __syncthreads();
    for (int i = 1; i < siz; i <<= 1) {
        if ((index & i) == 0 && index + i < siz) {
            shared[0] += shared[i];
        }
        __syncthreads();
    }
}

__device__ void sumSharedMem2(float* shared, int index, int siz) {
    shared += index;
    __syncthreads();
    for (int i = 1; i < siz; i <<= 1) {
        if ((index & i) == 0 && index + i < siz) {
            shared[0] += shared[i];
            shared[siz] += shared[siz + i];
        }
        __syncthreads();
    }
}
// <<<num_heads, head_size, head_size * sizeof(float)>>>
__global__ void attnKernel(int kv_idx, float *ybuf, float *qbuf, float *kvbuf, int emb_siz) {
    int k = threadIdx.x;
    // which attention head are we in?
    int h = blockIdx.x;
    int head_siz = blockDim.x;

    // offset inputs/outputs by our attention head position
    qbuf += h * head_siz;
    ybuf += h * head_siz;
    kvbuf += h * head_siz;

    float *shared_a = shared_buf;
    float attn_scale = 1.0f / sqrtf(head_siz);

    // initially, only one value to pick from, so that's our output value
    ybuf[k] = kvbuf[k + emb_siz];

    // compute q*k for first kv within our own attention head
    shared_a[k] = qbuf[k] * kvbuf[k];
    sumSharedMem(shared_a, k, head_siz);
    float a = shared_a[0] * attn_scale;
    float m = a;  // maximum softmax value for our attention head
    float l = 1;  // denominator sum for our attention head

    for (int i = 1; i <= kv_idx; i++) {
        // move on to next kv
        kvbuf += emb_siz*2;
        // compute q*k for the others and aggregate
        shared_a[k] = qbuf[k] * kvbuf[k];
        sumSharedMem(shared_a, k, head_siz);
        float a = shared_a[0] * attn_scale;
        if (a > m) {  // we won't have branch divergence here
            float e = expf(m - a);  // < 1.0
            ybuf[k] = kvbuf[k + emb_siz] + e * ybuf[k];
            l = 1 + e * l;
            m = a;  // new maximum
        } else {
            float e = expf(a - m); // < 1.0
            ybuf[k] += e * kvbuf[k+emb_siz];
            l += e;
            // m is still the maximum
        }
    }
    // rescale y by 1/l
    ybuf[k] /= l;
}

void attn(int kv_idx, float *xbuf, float *qbuf, float *kvbuf, int emb_siz, int num_heads) {
    int head_siz = emb_siz / num_heads;
    size_t sharedbuf_siz = head_siz * sizeof(float);
    attnKernel<<<num_heads, head_siz, sharedbuf_siz>>>(kv_idx, xbuf, qbuf, kvbuf, emb_siz);
    if (cudaPeekAtLastError() != cudaSuccess) {
        fprintf(stderr, "Error in attnKernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        abort();
    }
}
//--------------------------------------------------------------------------

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

//------------------------------------------
__global__ void layerNormKernel2(float* output, int embedding_dim, float* gamma, float* beta, float* input){

    __shared__ float shared[2048];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
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

void layerNorm(float* output, int embedding_dim, float* gamma, float* beta, float* input) {
    int numThreads = 1024;
    int numBlocks = 1;

    layerNormKernel2<<<numBlocks, numThreads>>>(output, embedding_dim, gamma, beta, input);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error in layerNormKernel: %s\n", cudaGetErrorString(cudaStatus));
        abort();
    }
}
//----------------------------------------------
__global__ void loadEmbeddingKernel(float *output, int token, int pos, int embeddingSize, float* wte, float *wpe) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < embeddingSize) {
        output[index] = wte[token * embeddingSize + index] + wpe[pos * embeddingSize + index];
    }
}

void loadEmbedding(float *output, int token, int pos, int embeddingSize, float* wte, float* wpe) {
    int numThreads = 1024;
    int numBlocks = (embeddingSize + numThreads - 1) / numThreads;
    loadEmbeddingKernel<<<numBlocks, numThreads>>>(output, token, pos, embeddingSize, wte, wpe);
}

// int main(){

//     int kv_idx;
//     float *xbuf, *qbuf, *kvbuf;
//     int emb_siz, num_heads;
//     kv_idx = 1;
//     emb_siz = 1600;
//     num_heads = 25;
//     cudaMalloc(&xbuf, emb_siz * sizeof(float));
//     cudaMalloc(&qbuf, 1024*emb_siz * sizeof(float));
//     cudaMalloc(&kvbuf, 1024*emb_siz * sizeof(float));

//     attn(kv_idx, xbuf, qbuf, kvbuf, emb_siz, num_heads);
// }
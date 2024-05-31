#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__device__ void maxReduce(float* smem, int tid){
    for(int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            smem[tid] = fmaxf(smem[tid + s], smem[tid]);
        }
        __syncthreads();
    }

    if(tid < 32){
        volatile float* vsmem = smem;
        vsmem[tid] = fmaxf(vsmem[tid + 32], vsmem[tid]);
        vsmem[tid] = fmaxf(vsmem[tid + 16], vsmem[tid]);
        vsmem[tid] = fmaxf(vsmem[tid + 8], vsmem[tid]);
        vsmem[tid] = fmaxf(vsmem[tid + 4], vsmem[tid]);
        vsmem[tid] = fmaxf(vsmem[tid + 2], vsmem[tid]);
        vsmem[tid] = fmaxf(vsmem[tid + 1], vsmem[tid]);
    }
}
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

__global__
void softmax_kernel(float* out, float* x, int n){
    __shared__ float shared_buf[1024];
    int idx = threadIdx.x;

    shared_buf[idx] = 0.0f;
    __syncthreads();

    for(int i = idx;i<n;i+=blockDim.x){
        shared_buf[idx] = max(shared_buf[idx], x[i]);
    }
    __syncthreads();
    maxReduce(shared_buf, idx);
    float max_val = shared_buf[0];

    shared_buf[idx] = 0.0f;
    __syncthreads();

    for(int i=idx;i<n;i+=blockDim.x){
        shared_buf[idx] += expf(x[i] - max_val);
    }
    __syncthreads();
    sumReduce(shared_buf, idx);
    float normalizer = shared_buf[0];

    if(idx==0){
        printf("Normalizer: %f\n", normalizer);
        printf("Max Val: %f\n", max_val);
    }

    out[idx] = expf(x[idx] - max_val) / normalizer;
}

void softmax_gpu(float* out, float* x, int n){
    int numThreads = 1024;
    int numBlocks = 1;
    softmax_kernel<<<numBlocks, numThreads>>>(out, x, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }}

// void online_softmax_cpu(float* out, float* x, int n){
//     float row_max = -1.0f;
//     float normalizer = 0.0f;
//     for(int i = 0; i < n; i++){
//         float curr = x[i];
//         float prev_max = row_max;
//         row_max = max(row_max, curr);
//         normalizer = normalizer * expf(prev_max-row_max) + expf(curr - row_max);
//     }
//     for(int i = 0; i < n; i++){
//         out[i] = expf(x[i] - row_max) / normalizer;
//     }
// }

void softmax_cpu(float* out, float* x, int n){
    float sum = 0.0f;
    float max_val =  0.0f;
    for(int i = 0; i < n; i++){
        max_val = max(max_val, x[i]);
    }
    for(int i = 0; i < n; i++){
        sum += expf(x[i] - max_val);
    }
    for(int i = 0; i < n; i++){
        out[i] = expf(x[i] - max_val) / sum;
    }
    cout << "CPU Normalizer: " << sum << endl;
    cout << "CPU Max Val: " << max_val << endl;
}

int main(){
    //in inference kv attention x comes with [C]
    //then k_cache = [-blocksize, C]
    //then v_cache = [-blocksize, C]
    //then att = q @ k = [blocksize]
    //softmax(att) = [blocksize]
    //then att@ v = [C]

    int N = 8092;
    float* x, *out;
    x = (float*)malloc(N * sizeof(float));
    out = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        x[i] = (float)rand() / RAND_MAX;
    }

    softmax_cpu(out, x, N);

    float *d_x, *d_out;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    softmax_gpu(d_out, d_x, N);

    float* check = (float*)malloc(N * sizeof(float));
    cudaMemcpy(check, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0;i<10;i++){
        cout << out[i] << " " << check[i] << endl;
    }

    for(int i = 0;i<N;i++){
        if(abs(check[i] - out[i]) > 1e-6){
            cout << "Incorrect try again!" << endl; 
            return 1;
        }
    }

    cout << "Yay Success!" << endl;

    return 0;
}
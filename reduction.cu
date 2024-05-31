#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#define FLT_EPSILON 1.19209290e-07f

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

__global__
void sumReductionKernel(float* a, int n){
    extern __shared__ float shared[];

     for(int i=threadIdx.x;i<n;i += gridDim.x * blockDim.x){
        shared[threadIdx.x] += a[i];
    }
    __syncthreads();

    sumReduce(shared, threadIdx.x);

    if(threadIdx.x == 0){
        a[blockIdx.x] = shared[0];
    }
}

__global__
void maxReductionKernel(float* a, int n){
    extern __shared__ float shared[];

    for(int i=threadIdx.x;i<n;i += gridDim.x * blockDim.x){
       shared[threadIdx.x] = fmaxf(shared[threadIdx.x], a[i]);
    }
    __syncthreads();

    maxReduce(shared, threadIdx.x);

    if(threadIdx.x == 0){
        a[blockIdx.x] = shared[0];
    }
}

void sumReduction(float* a,int n) {
    int threadSize = 1024;
    int numBlocks = 1;
    sumReductionKernel<<<numBlocks, threadSize, threadSize*sizeof(float)>>>(a,n);
}

void maxReduction(float* a,int n) {
    int threadSize = 1024;
    int numBlocks = 1;
    maxReductionKernel<<<numBlocks, threadSize, threadSize*sizeof(float)>>>(a,n);
}

void sumCPU(float* a, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    a[0] = sum;
}

void maxCPU(float* a, int n) {
    float max = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    a[0] = max;
}

void rand_init(float* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

bool AlmostEqualRelative(float A, float B, int maxUlps) {
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    float absDiff = fabs(A - B);
    if (absDiff <= FLT_EPSILON) {
        return true;
    }

    int aInt = *(int*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0) {
        aInt = 0x80000000 - aInt;
    }

    int bInt = *(int*)&B;
    // Make bInt lexicographically ordered as a twos-complement int
    if (bInt < 0) {
        bInt = 0x80000000 - bInt;
    }

    // Now we can compare integers
    int ulpsDiff = abs(aInt - bInt);
    return ulpsDiff <= maxUlps;
}


int main(){

    int N = 8092;
    float* a, *d_a;
    a = (float*)malloc(N*sizeof(float));
    rand_init(a, N);

    cudaMalloc(&d_a, N*sizeof(float));
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);

    sumCPU(a,N);
    sumReduction(d_a,N);

    float* check = (float*)malloc(N*sizeof(float));
    cudaMemcpy(check, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);

    double x1 = check[0];
    double x2 = a[0];
    cout << x1 << endl;
    cout << x2 << endl;
    double xx = x1 - x2;
    cout << xx << endl;

    if(AlmostEqualRelative(check[0], a[0], 5)){
        std::cout << "Correct sum!" << std::endl;
    } else {
        std::cout << "Incorrect sum!" << std::endl;
    }


    //for max
    maxCPU(a,N);
    maxReduction(d_a,N);

    cudaMemcpy(check, d_a, N*sizeof(float), cudaMemcpyDeviceToHost);

    cout << check[0] << endl;
    cout << a[0] << endl;

    float ddd = check[0] - a[0];
    cout << ddd << endl;

    if(AlmostEqualRelative(check[0], a[0], 5)){
        std::cout << "Correct max!" << std::endl;
    } else {
        std::cout << "Incorrect max!" << std::endl;
    }

    return 0;
}

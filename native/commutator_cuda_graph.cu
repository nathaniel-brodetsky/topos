#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

constexpr int N = 20;

__global__ void compute_commutator_kernel(const float* A, const float* A_prev, float* F) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    __shared__ float sA[N][N];
    __shared__ float sdA[N][N];

    sA[i][j] = A[i * N + j];
    sdA[i][j] = A[i * N + j] - A_prev[i * N + j];
    __syncthreads();

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum1 += sA[i][k] * sdA[k][j];
        sum2 += sdA[i][k] * sA[k][j];
    }

    F[i * N + j] = sum1 - sum2;
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_A(N * N), h_A_prev(N * N), h_F(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(rng);
        h_A_prev[i] = dist(rng);
    }

    float *d_A, *d_A_prev, *d_F;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_A_prev, N * N * sizeof(float));
    cudaMalloc(&d_F, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_prev, h_A_prev.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(20, 20);
    dim3 grid(1, 1);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < 100; ++i) {
        compute_commutator_kernel<<<grid, block, 0, stream>>>(d_A, d_A_prev, d_F);
    }
    cudaStreamSynchronize(stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    cudaError_t capture_err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (capture_err != cudaSuccess) {
        printf("GRAPH CAPTURE FAILED to begin: %s\n", cudaGetErrorString(capture_err));
        return 1;
    }

    compute_commutator_kernel<<<grid, block, 0, stream>>>(d_A, d_A_prev, d_F);

    cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
    if (end_err != cudaSuccess) {
        printf("GRAPH CAPTURE FAILED to end: %s\n", cudaGetErrorString(end_err));
        return 1;
    }

    cudaError_t instantiate_err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    if (instantiate_err != cudaSuccess) {
        printf("GRAPH INSTANTIATE FAILED: %s\n", cudaGetErrorString(instantiate_err));
        return 1;
    }

    printf("CUDA Graph capture SUCCEEDED for this custom kernel (no cuBLAS)\n");

    const int n_warmup = 1000;
    const int n_trials = 100000;

    for (int i = 0; i < n_warmup; ++i) {
        cudaGraphLaunch(graph_exec, stream);
    }
    cudaStreamSynchronize(stream);

    std::vector<double> latencies_us;
    latencies_us.reserve(n_trials);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < n_trials; ++i) {
        cudaEventRecord(start, stream);
        cudaGraphLaunch(graph_exec, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        latencies_us.push_back(ms * 1000.0);
    }

    cudaMemcpy(h_F.data(), d_F, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::sort(latencies_us.begin(), latencies_us.end());
    double mean = 0.0;
    for (double v : latencies_us) mean += v;
    mean /= latencies_us.size();

    double p50 = latencies_us[latencies_us.size() * 50 / 100];
    double p99 = latencies_us[latencies_us.size() * 99 / 100];

    printf("CUDA Graph commutator: mean=%.4f us  p50=%.4f us  p99=%.4f us\n", mean, p50, p99);
    printf("F[0][0]=%.6f (sanity check, non-zero expected)\n", h_F[0]);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaFree(d_A);
    cudaFree(d_A_prev);
    cudaFree(d_F);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return 0;
}
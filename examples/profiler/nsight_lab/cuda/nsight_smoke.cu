#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void saxpy_kernel(const float *x, float *y, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}

__global__ void reduce_kernel(const float *x, float *partial, int n) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float sum = 0.0f;
  if (idx < n) {
    sum += x[idx];
  }
  if (idx + blockDim.x < n) {
    sum += x[idx + blockDim.x];
  }
  smem[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = smem[0];
  }
}

int main(int argc, char **argv) {
  int n = 1 << 26;
  int iters = 80;
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iters = std::atoi(argv[2]);
  }

  std::printf("nsight_smoke n=%d iters=%d\n", n, iters);

  std::vector<float> hx(n, 1.0f);
  std::vector<float> hy(n, 2.0f);

  float *dx = nullptr;
  float *dy = nullptr;
  float *dpartial = nullptr;

  int threads = 256;
  int saxpy_blocks = (n + threads - 1) / threads;
  int reduce_blocks = (n + threads * 2 - 1) / (threads * 2);

  CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dpartial, reduce_blocks * sizeof(float)));

  nvtxRangePushA("h2d_copy");
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  nvtxRangePop();

  nvtxRangePushA("warmup");
  saxpy_kernel<<<saxpy_blocks, threads>>>(dx, dy, 2.0f, n);
  reduce_kernel<<<reduce_blocks, threads, threads * sizeof(float)>>>(dy, dpartial,
                                                                      n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  nvtxRangePushA("profile_loop");
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    nvtxRangePushA("saxpy");
    saxpy_kernel<<<saxpy_blocks, threads>>>(dx, dy, 1.0001f, n);
    nvtxRangePop();

    nvtxRangePushA("reduce");
    reduce_kernel<<<reduce_blocks, threads, threads * sizeof(float)>>>(dy,
                                                                        dpartial,
                                                                        n);
    nvtxRangePop();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  nvtxRangePop();

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("profile_loop_ms=%.3f avg_iter_ms=%.3f\n", ms, ms / iters);

  nvtxRangePushA("d2h_copy");
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
  nvtxRangePop();

  std::printf("sample y[0]=%.4f y[n-1]=%.4f\n", hy.front(), hy.back());

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dpartial));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

#include <cstdio>
#include <stdint.h>
#include <stdio.h>

#define MAX_THREADS (512)
#define WARP_SIZE (32)

struct CudaFormatMat {
  int32_t rows;
  int32_t cols;
  float *bytes;
};
struct YoloDetectionCuda {
  double confidence;
  double x;
  double y;
  double width;
  double height;
  int32_t class_id;
};

__global__ void process_net(uintptr_t block_count,
                            YoloDetectionCuda *processed_detects,
                            bool *processed_valid) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;

  // Get rid of leftover threads
  if (id >= block_count)
    return;

  processed_valid[id] = false;
}

extern "C" {
int process_net_kernel(CudaFormatMat *const result, uintptr_t const num_levels,
                       double const threshold,
                       YoloDetectionCuda *processed_detects,
                       bool *processed_valid, uintptr_t const total_rows) {

  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);

  YoloDetectionCuda *processed_detects_cuda;
  bool *processed_valid_cuda;
  cudaMallocAsync(&processed_detects_cuda,
                  sizeof(YoloDetectionCuda) * total_rows, kernel_stream);
  cudaMallocAsync(&processed_valid_cuda, sizeof(bool) * total_rows,
                  kernel_stream);

  uintptr_t row_offset = 0;
  for (uintptr_t i = 0; i < num_levels; ++i) {
    CudaFormatMat *mat = result + i;
    auto num_rows = mat->rows;
    auto mat_size = sizeof(num_rows * mat->cols);
    float *mat_bytes;

    cudaMallocAsync(&mat_bytes, mat_size, kernel_stream);
    cudaMemcpyAsync(&mat_bytes, &mat, mat_size, cudaMemcpyHostToDevice,
                    kernel_stream);

    int32_t blocksize = MAX_THREADS;
    int32_t block_count;
    if (num_rows < blocksize) {
      blocksize = num_rows;
      block_count = 1;
    } else {
      // Ceiling divide, from https://stackoverflow.com/a/14878734
      block_count = num_rows / MAX_THREADS + (num_rows % MAX_THREADS != 0);
    }

    process_net<<<block_count, blocksize, 0, kernel_stream>>>(
        block_count, processed_detects_cuda + row_offset,
        processed_valid_cuda + row_offset);

    cudaFreeAsync(mat_bytes, kernel_stream);

    row_offset += num_rows;
  }

  cudaMemcpyAsync(processed_detects, processed_detects_cuda,
                  sizeof(YoloDetectionCuda) * total_rows,
                  cudaMemcpyDeviceToHost, kernel_stream);
  cudaMemcpyAsync(processed_valid, processed_valid_cuda,
                  sizeof(bool) * total_rows, cudaMemcpyDeviceToHost,
                  kernel_stream);

  cudaStreamSynchronize(kernel_stream);

  return 0;
}
}

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

__forceinline__ __device__ float adjust_base(uintptr_t idx, float const factor,
                                             float const *row_bytes) {
  return row_bytes[idx] * factor;
}

__forceinline__ __device__ float x_adjust(uintptr_t idx, float const factor,
                                          float const *row_bytes) {
  return (adjust_base(idx, factor, row_bytes) / 640.0) * 800.0;
}

__forceinline__ __device__ float y_adjust(uintptr_t idx, float const factor,
                                          float const *row_bytes) {
  return (adjust_base(idx, factor, row_bytes) / 640.0) * 600.0;
}

__global__ void process_net(uintptr_t num_rows, uintptr_t num_cols,
                            float const threshold, float const factor,
                            float const *mat_bytes,
                            YoloDetectionCuda *processed_detects,
                            bool *processed_valid) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;

  // Get rid of leftover threads
  if (id >= num_rows)
    return;

  float const *row = mat_bytes + (id * num_cols);

  float confidence = row[4];
  bool valid = confidence > threshold;
  processed_valid[id] = valid;

  // Skip remaining processing for invalid
  if (!valid)
    return;

  // Start at offset in data, then shift to starting at 0.
  uintptr_t class_id = 5;
  float class_value = row[class_id];
  for (uintptr_t i = 6; i < num_cols; ++i) {
    if (class_value < row[i]) {
      class_id = i;
      class_value = row[i];
    }
  }
  class_id -= 5;

  float center_x = x_adjust(0, factor, row);
  float center_y = y_adjust(1, factor, row);
  float width = x_adjust(2, factor, row);
  float height = y_adjust(3, factor, row);

  float left = center_x - (width / 2.0);
  float top = center_y - (height / 2.0);

  processed_detects[id] = YoloDetectionCuda{
      confidence, left, top, width, height, static_cast<int32_t>(class_id)};
}

extern "C" {
int process_net_kernel(CudaFormatMat *const result, uintptr_t const num_levels,
                       float const threshold, float const factor,
                       uintptr_t const total_rows,
                       YoloDetectionCuda *processed_detects,
                       bool *processed_valid) {

  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);

  YoloDetectionCuda *processed_detects_cuda;
  bool *processed_valid_cuda;
  cudaMalloc(&processed_detects_cuda, sizeof(YoloDetectionCuda) * total_rows);
  cudaMalloc(&processed_valid_cuda, sizeof(bool) * total_rows);

  uintptr_t row_offset = 0;
  for (uintptr_t i = 0; i < num_levels; ++i) {
    CudaFormatMat *mat = result + i;
    auto num_rows = mat->rows;
    uintptr_t num_cols = static_cast<uintptr_t>(mat->cols);
    auto mat_size = num_rows * num_cols * sizeof(float);
    float *mat_bytes;

    cudaMalloc(&mat_bytes, mat_size);
    cudaMemcpy(mat_bytes, mat->bytes, mat_size, cudaMemcpyHostToDevice);

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
        num_rows, num_cols, threshold, factor, mat_bytes,
        processed_detects_cuda + row_offset, processed_valid_cuda + row_offset);

    cudaStreamSynchronize(kernel_stream);
    cudaFree(mat_bytes);

    row_offset += num_rows;
  }

  cudaMemcpy(processed_detects, processed_detects_cuda,
             sizeof(YoloDetectionCuda) * total_rows, cudaMemcpyDeviceToHost);
  cudaMemcpy(processed_valid, processed_valid_cuda, sizeof(bool) * total_rows,
             cudaMemcpyDeviceToHost);

  return 0;
}
}

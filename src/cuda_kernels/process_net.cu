#include <cassert>
#include <cstdio>
#include <stdio.h>

__global__ void process_net(int *result) {
  printf("Hello World from GPU!\n");
  *result = 1 + 1;
}

extern "C" {
int process_net_kernel() {

  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);

  int *result;
  cudaMallocAsync(&result, sizeof(int), kernel_stream);

  process_net<<<1, 1, 0, kernel_stream>>>(result);

  int local_result;

  cudaMemcpyAsync(&local_result, result, sizeof(int), cudaMemcpyDeviceToHost,
                  kernel_stream);
  cudaFreeAsync(result, kernel_stream);
  cudaStreamSynchronize(kernel_stream);

  assert(local_result == 2);

  return 0;
}
}

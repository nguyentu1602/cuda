#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel(int* arr_start) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr_start[id] = arr_start[id] + arr_start[id];
}

int
main(int argc, char** argv)
{
  const int BLOCK_N = 32;
  const int THREAD_N = 256;
  int GPU_N, i;
  printf("Starting simpleMultiGPU\n");
  cudaGetDeviceCount(&GPU_N);
  printf("CUDA device count: %i\n", GPU_N);
  // check number of GPUs:
  if (GPU_N <= 2) {
    printf("This simplest examples requires at least 2 GPUs.\n");
    printf("Program terminating.\n");
    return 0;
  }
  // allocate a unified memory block;
  int* array;
  int N = BLOCK_N * THREAD_N * 2;
  cudaMallocManaged(&array, N);
  // init the unified memory block:
  for (i = 0; i < N; i++) {
    array[i] = 1;
  }
  printf("Done initialization for memory block. \n");

  // calculate the length for each gpu:
  int N1 = N / 2;
  // make the second pointer:
  int* array_2 = array + N1;

  // make 2 streams:
  cudaStream_t streamA, streamB;
  cudaSetDevice(0);
  cudaStreamCreate(&streamA);
  kernel<<<BLOCK_N, THREAD_N, 0, streamA>>>(array);
  cudaSetDevice(1);
  cudaStreamCreate(&streamB);
  kernel<<<BLOCK_N, THREAD_N, 0, streamB>>>(array_2);
  cudaStreamSynchronize(streamA);
  cudaStreamSynchronize(streamB);
  cudaDeviceSynchronize();
  printf("Done synchronizing all devices.\n");

  // check result:
  for (i = 0; i < N; i++) {
    if (array[i] != 2) {
      printf("%i\n", array[i]);
    }
  }
  printf("Done checking calculations. \n");
  printf("all tests passed if no other output apprear.\n");
  cudaFree(array);
  return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.h"

// Data configuration:
const int DATA_N = 1048576 * 32; // 2 ^ 25

// Array divide an conquer utilities:
int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int count) {
  return (count + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// struct to wrap & handle GPUs operations:
typedef struct {
  // input data
  int dataN;
  int * input;

  // Stream to offload asynchronous command execution to:
  cudaStream_t stream;
} TGPUplan;

// GPU kernel
__global__ void kernel(int* input, int N) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int jump_size = gridDim.x * blockDim.x;
  for (int pos = id; pos < N; pos += jump_size) {
    input[pos] = input[pos] + input[pos]; // kinda unsafe addition :P
  }
}

int
main(int argc, char** argv) {
  int GPU_N, i;
  printf("Starting simpleMultiGPU\n");
  cudaGetDeviceCount(&GPU_N);
  printf("CUDA device count: %i\n", GPU_N);

  // Solver config:
  TGPUplan plan[GPU_N];

  // allocate a unified memory block:
  // it's quite cool that unified memory could be dereferenced from
  // both CPU and GPU threads!
  int* array;

  cudaMallocManaged(&array, DATA_N * sizeof(int));

  // init the unified memory block:
  // CPU thread accesses unified memory here
  for (i = 0; i < DATA_N; i++) {
    array[i] = 1;
  }
  printf("Done initialization for memory block. \n");

  // Subdividing input data across GPUS
  // Get data sizes for each GPU
  int * start_ptr = array;
  for (i = 0; i < GPU_N; i++) {
    plan[i].dataN = DATA_N / GPU_N;
  }

  // Take into account "odd" data sizes, distribute those oddity into each GPU
  for (i = 0; i < DATA_N % GPU_N; i++) {
     plan[i].dataN++;
  }

  //Start timing and compute on GPU(s)
  printf("Computing with %d GPUs...\n", GPU_N);
  StartTimer();

  // deploy: create streams for issuing GPU command asynchronously and allocate memory
  for (i = 0; i < GPU_N; i++) {
    cudaSetDevice(i);
    cudaStreamCreate(&plan[i].stream);
    // set the start of the input array into each GPU:
    plan[i].input = start_ptr;
    start_ptr += plan[i].dataN;
    kernel<<<GET_BLOCKS(DATA_N), CUDA_NUM_THREADS, 0, plan[i].stream>>>(plan[i].input, plan[i].dataN);
  }

  cudaDeviceSynchronize();
  printf("Done synchronizing all devices.\n");
  printf("GPUs processing time: %f (ms)\n\n", GetTimer());

  printf("Start checking calculations. \n");
  StartTimer();
  // check result:
  for (i = 0; i < DATA_N; i++) {
    if (array[i] != 2) {
      printf("%i\n", array[i]);
    }
  }
  printf("CPU checking time: %f (ms)\n\n", GetTimer());
  printf("Done checking calculations. \n");
  printf("all tests passed if no other output apprear.\n");
  cudaFree(array);
  return 0;
}

Samples:

1. simplest_dual.cu:
simplest reproducible example that uses 2 GPUs concurently to double an array
using CUDA unified memory interface. The array is setup using Unified Access
Memory which allows both the host and multiple GPUs work on the array at once.

Tested against a machine with 4 K40 GPUs. The expected output should be like
this:
$ make run
./build/simplest_dual
Starting simpleMultiGPU
CUDA device count: 4
Done initialization for memory block.
Done synchronizing all devices.
Done checking calculations.
all tests passed if no other output apprear.

2. flexible_multipleGPU.cu

Sample: simplest reproducible example that uses 2 GPUs concurently to
        double an array using CUDA unified memory interface.

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

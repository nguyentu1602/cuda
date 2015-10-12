Samples:

############################
1. simplest_dual.cu:
############################
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



############################
2. flexible_multipleGPU.cu
############################
This example allow an arbitrary number of GPUs and arbitrary input array size.
The array will be divided flexibly and map to each GPU with a stream; edge
cases such as odd data size are also handled.

Using Unified Access Memory still works here, but we need to make sure we
collect the streams before we collect the devices.

Tested against a machine with 4 K40 GPUs. When the problem size is big enough,
spliting the calculations over multiples GPUs does exceed the overhead cost and
payoff.

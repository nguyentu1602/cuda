all: dir
#	nvcc -pg multigpu.cu -o multigpu
	nvcc simplest_dual.cu -o build/simplest_dual
	nvcc flexible_multipleGPU.cu -o build/flexible_multipleGPU

dir:
	mkdir -p build

run:
	./build/simplest_dual

clean:
	rm -rf *~
	rm -rf *.o
	rm -rf ./build

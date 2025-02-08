NVCC = nvcc
NVCC_FLAGS = -g -lineinfo -std=c++17 -O3 --compiler-options '-Wall'# -Wextra -Wpedantic -use_fast_math

INCLUDES = -I. -I./gato -I./dependencies

TARGETS = examples/single_sqp.exe examples/batch_sqp.exe

examples/single_sqp.exe: examples/single_sqp.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

examples/batch_sqp.exe: examples/batch_sqp.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

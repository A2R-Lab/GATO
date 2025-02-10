NVCC = nvcc
NVCC_FLAGS = -use_fast_math -std=c++17 -O3 --compiler-options '-Wall'# -Wextra -Wpedantic

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

NVCC = nvcc
NVCC_FLAGS = -use_fast_math -std=c++17 -O3 --compiler-options '-Wall'# -Wextra -Wpedantic

INCLUDES = -I./gato -I./config -I./dependencies

TARGETS = examples/single_sqp.exe examples/batch_sqp.exe examples/benchmark_batch_sqp.exe

examples/single_sqp.exe: examples/single_sqp.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

examples/batch_sqp.exe: examples/batch_sqp.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

examples/benchmark_batch_sqp.exe: examples/benchmark_batch_sqp.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

.PHONY: all clean bindings

all: $(TARGETS)

bindings:
	cd bindings && pip install -e .

clean:
	rm -f $(TARGETS)
	rm -rf bindings/build
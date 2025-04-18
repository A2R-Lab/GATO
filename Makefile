NVCC = nvcc
NVCC_FLAGS = -use_fast_math -std=c++17 -O3 --compiler-options '-Wall'# -Wextra -Wpedantic

INCLUDES = -I./gato -I./config -I./dependencies

HEADERS = $(wildcard gato/*.cuh config/*.h dependencies/*.h)

TARGETS := single_sqp batch_sqp benchmark_batch_sqp

build/%: examples/%.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

.PHONY: build build-single build-batch build-benchmark build-bindings clean

build: build-single build-batch build-benchmark build-bindings

build-single: examples/single_sqp.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o build/single_sqp examples/single_sqp.cu

build-batch: examples/batch_sqp.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o build/batch_sqp examples/batch_sqp.cu

build-benchmark: examples/benchmark_batch_sqp.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o build/benchmark_batch_sqp examples/benchmark_batch_sqp.cu

build-bindings:
	cd bindings && TORCH_CUDA_ARCH_LIST="8.9" pip install -e .

clean:
	rm -rf build
	rm -rf bindings/build
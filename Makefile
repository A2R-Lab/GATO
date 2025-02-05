NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 --compiler-options '-Wall' # -Wextra -Wpedantic

INCLUDES = -I. -I./gato -I./dependencies

TARGET = examples/single_sqp
SRC = examples/single_sqp.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGET)

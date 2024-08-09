# Makefile

SRC = ../src
EXAMPLES = ../examples
EXTERNAL = ../external

# Compiler and compiler flags
# if need to debug with compute-sanitizer, use -G option
NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 \
		-I../ \
		-I$(SRC) \
		-I$(EXTERNAL)/GLASS \
		-Iexternal/qdldl/include \
		-Lexternal/qdldl/build/out \
		-lcublas \
		-G
			
SOURCES = $(EXAMPLES)/single_mpc_simulation_pcg.cu

DEPS = $(SRC)/sim/mpcsim.cuh \
			$(SRC)/solvers/sqp/sqp_pcg.cuh \
			$(SRC)/gato.cuh \
			$(SRC)/sim_settings.h \
			$(SRC)/solver_settings.h \
			$(SRC)/cost_settings.h 

# Target executable
TARGET = pcg.exe

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) $(DEPS)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) examples/*.exe
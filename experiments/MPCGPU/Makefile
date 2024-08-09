# Makefile

SRC = ../../src
EXTERNAL = ../../external


# Compiler and compiler flags
# if need to debug with compute-sanitizer, use -G option
NVCC = nvcc
CFLAGS = --compiler-options -Wall -O3 \
		-I../../ \
		-I$(SRC) \
		-I$(EXTERNAL) \
		-I$(EXTERNAL)/GLASS \
		-lqdldl \
		-I$(EXTERNAL)/qdldl/include \
		-L$(EXTERNAL)/qdldl/build/out \
		-lcublas \
		-G


DEPS = $(SRC)/sim/mpcsim.cuh \
		$(SRC)/solvers/sqp/sqp_pcg.cuh \
		$(SRC)/solvers/sqp/sqp_qdldl.cuh \
		$(SRC)/gato.cuh \
		$(SRC)/sim_settings.h \
		$(SRC)/solver_settings.h \
		$(SRC)/cost_settings.h 


# Default target
all: pcg.exe qdldl.exe $(DEPS)

pcg.exe: ./track_iiwa_pcg.cu
	$(NVCC) $(CFLAGS) $< -o $@

qdldl.exe: ./track_iiwa_qdldl.cu
	$(NVCC) $(CFLAGS) -DLINSYS_SOLVE=0 $< -o $@ 

build_qdldl:
	cd ../../external/qdldl && mkdir -p build && cd build && cmake -DQDLDL_FLOAT=true -DQDLDL_LONG=false .. && cmake --build . && cd ../../../experiments/MPCGPU

clean:
	rm -f $(TARGET) $(OBJECTS) *.exe
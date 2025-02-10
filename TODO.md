# TODO

## In Progress

## Priority

- [ ] Benchmark with >1 SQP iterations
- [ ] Add more Python examples

## Low Priority

- [ ] Switch to CMake from Makefile
- [ ] Experiment with number of loop unrolls
- [ ] Plant codegen
- [ ] GRiD codegen
- [ ] Cost func codegen
- [ ] Remove GLASS dependencies
- [ ] Matrix inverse using chol decomp

- More CUDA [optimizations](https://stackoverflow.com/questions/43706755/how-can-i-get-the-nvcc-cuda-compiler-to-optimize-more)

## Completed

- [x] Add multisolve example
- [x] Add Python bindings
- [x] Turn SQP solver into a class
- [x] Benchmarking code
- [x] Warp optimizations for dot and block tridiag mat-vec mul
- [x] Remove atomics from merit computation
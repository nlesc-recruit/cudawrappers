#include "params.h"

extern "C" __global__ void vector_add(T *c, T *a, T *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i];
  }
}

#include "types.h"

extern "C" __global__ void vector_add(T *c, T *a, T *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

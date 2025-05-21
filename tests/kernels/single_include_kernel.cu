#include "secondary.h"

extern "C" __global__ void kernel(int *ptr) {
  int value = SECONDARY_VALUE;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    *ptr = value;
  }
}

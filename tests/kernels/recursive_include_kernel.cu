#include "primary.h"

extern "C" __global__ void kernel(int *ptr) {
  int value = PRIMARY_VALUE * SECONDARY_VALUE;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    *ptr = value;
  }
}

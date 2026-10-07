#ifndef PRIMARY_H
#define PRIMARY_H
#ifndef SECONDARY_H
#define SECONDARY_H
#define SECONDARY_VALUE 2
#endif  // SECONDARY_H

#define PRIMARY_VALUE 1
#endif  // PRIMARY_H

extern "C" __global__ void kernel(int *ptr) {
  int value = PRIMARY_VALUE *SECONDARY_VALUE int i =
      blockIdx.x * blockDim.x + threadIdx.x if (i == 0) {
    *ptr = value
  }
}

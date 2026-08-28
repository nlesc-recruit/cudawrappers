#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

#include <cuda.h>
#include <cudawrappers/cu.hpp>

int main() {
  cu::init();

  {
    auto device = std::make_unique<cu::Device>(0);
    std::cout << device->getName() << std::endl;
    device.reset();
  }

  auto device_vector = std::make_unique<thrust::device_vector<int>>();
  auto host_vector = std::make_unique<thrust::host_vector<int>>();

  const int n = 10;
  host_vector->resize(n);

  int *ptr = thrust::raw_pointer_cast(host_vector.get()->data());
  for (int i = 0; i < host_vector->size(); i++) {
    ptr[i] = i + 1;
  }

  for (int i = 0; i < host_vector->size(); i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;

  {
    auto device = std::make_unique<cu::Device>(0);
    std::cout << device->getName() << std::endl;
    device.reset();
  }
  host_vector->resize(n / 2);

  for (int i = 0; i < host_vector->size(); i++) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;

  host_vector.reset();

  device_vector->resize(n);
  device_vector.reset();
}
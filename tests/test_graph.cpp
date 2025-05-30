#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

TEST_CASE("Test cu::Graph", "[graph]") {
  const std::string kernel = R"(
    __global__ void vector_print(float *a, size_t array_size) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < array_size) {
#ifdef DEBUG
        printf("a[%d] = %f\n", i, a[i]);
#endif
        a[i] *= 2.0f;
      }
    }
  )";

  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  context.setCurrent();
  nvrtc::Program program(kernel, "kernel.cu");
  program.addNameExpression("vector_print");
  program.compile({});
  cu::Module module(static_cast<const void*>(program.getPTX().data()));

  cu::Stream stream;
  const size_t array_size = 1000;
  SECTION("Test cu::hostNode single value") {
    auto fn = [](void* data) {
      int* ptr = static_cast<int*>(data);
      *ptr += 1;
    };
    cu::Graph graph(context);
    int data = 42;
    cu::GraphHostNodeParams node_params(fn, &data);

    cu::GraphNode node1, node2;
    graph.addHostNode(node1, {}, node_params);
    graph.addHostNode(node2, {node1}, node_params);

    cu::GraphExec graph_exec(graph);
    cu::Stream stream;
    stream.graphLaunch(graph_exec);
    stream.synchronize();

    CHECK(data == 44);
  }
  SECTION("Test cu::Graph : memory management") {
    std::vector<float> data_in(array_size);
    std::vector<float> data_out(array_size);

    cu::HostMemory data_in_registered(data_in.data(),
                                      array_size * sizeof(float));
    cu::HostMemory data_out_registered(data_out.data(),
                                       array_size * sizeof(float));

    for (int i = 0; i < array_size; i++) {
      data_in[i] = 3;
    }

    cu::Graph graph(context);
    cu::GraphNode dev_alloc, host_set, copy_to_dev, execute_kernel, device_free,
        copy_to_host;
    struct set_value_parameter {
      float* ptr;
      size_t size;
    };
    auto set_value = [](void* data) {
      set_value_parameter* par = static_cast<set_value_parameter*>(data);
      for (int i = 0; i < par->size; i++) {
        par->ptr[i] = 42;
      }
    };

    set_value_parameter host_par{data_in.data(), data_in.size()};
    cu::GraphHostNodeParams host_set_params{set_value, &host_par};
    cu::GraphDevMemAllocNodeParams dev_alloc_params{
        device, sizeof(float) * data_in.size()};

    graph.addHostNode(host_set, {}, host_set_params);
    graph.addMemAllocNode(dev_alloc, {host_set}, dev_alloc_params);
    cu::GraphMemCopyToDeviceNodeParams copy_to_dev_params{
        dev_alloc_params.getDevPtr(),
        data_in.data(),
        sizeof(data_in[0]),
        data_in.size(),
        1,
        1};

    graph.addMemCpyNode(copy_to_dev, {dev_alloc}, copy_to_dev_params);

    cu::GraphMemCopyToHostNodeParams copy_to_host_params{
        data_out.data(),
        dev_alloc_params.getDevPtr(),
        sizeof(data_in[0]),
        data_in.size(),
        1,
        1};

    cu::DeviceMemory mem(dev_alloc_params.getDeviceMemory());
    std::vector<const void*> params = {mem.parameter(), &array_size};
    cu::Function vector_print_fn(module,
                                 program.getLoweredName("vector_print"));
    cu::GraphKernelNodeParams kernel_params{
        vector_print_fn, 3, 1, 1, 3, 1, 1, 0, params};

    graph.addKernelNode(execute_kernel, {copy_to_dev}, kernel_params);

    graph.addMemCpyNode(copy_to_host, {execute_kernel}, copy_to_host_params);

    graph.addDevMemFreeNode(device_free, {copy_to_host},
                            dev_alloc_params.getDevPtr());

    cu::GraphExec graph_exec(graph);
    stream.graphLaunch(graph_exec);
    stream.synchronize();

    CHECK(data_in[0] == 42.0f);
    CHECK(data_in[1] == 42.0f);
    CHECK(data_in[2] == 42.0f);
    CHECK(data_out[0] == 84.0f);
    CHECK(data_out[1] == 84.0f);
    CHECK(data_out[2] == 84.0f);
  }
  SECTION("Test cu:graph debug utilities") {
    constexpr size_t array_size = 3;
    std::array<float, array_size> data_in{3, 3, 3};
    std::array<float, array_size> data_out{0, 0, 0};
    cu::Graph graph(context);
    cu::GraphNode dev_alloc, host_set, copy_to_dev, execute_kernel, device_free,
        copy_to_host, host_set2;

    auto set_value = [](void* data) {
      float* ptr = static_cast<float*>(data);
      for (int i = 0; i < array_size; i++) {
        ptr[i] = 42.0f;
      }
    };
    cu::GraphHostNodeParams host_set_params{set_value, data_in.data()};

    cu::GraphDevMemAllocNodeParams dev_alloc_params{device, sizeof(data_in)};

    graph.addHostNode(host_set, {}, host_set_params);
    graph.addMemAllocNode(dev_alloc, {}, dev_alloc_params);
    graph.addHostNode(host_set2, {dev_alloc}, host_set_params);

    cu::GraphMemCopyToDeviceNodeParams copy_to_dev_params{
        dev_alloc_params.getDevPtr(),
        data_in.data(),
        sizeof(data_in[0]),
        data_in.size(),
        1,
        1};

    cu::GraphMemCopyToHostNodeParams copy_to_host_params{
        data_out.data(),
        dev_alloc_params.getDevPtr(),
        sizeof(data_in[0]),
        data_in.size(),
        1,
        1};

    cu::DeviceMemory mem(dev_alloc_params.getDeviceMemory());

    std::vector<const void*> params = {mem.parameter(), &array_size};
    cu::Function vector_print_fn(module,
                                 program.getLoweredName("vector_print"));
    cu::GraphKernelNodeParams kernel_params{
        vector_print_fn, 3, 1, 1, 1, 1, 1, 0, params};

    graph.addMemCpyNode(copy_to_dev, {host_set2}, copy_to_dev_params);
    graph.addKernelNode(execute_kernel, {copy_to_dev}, kernel_params);

    graph.addMemCpyNode(copy_to_host, {execute_kernel}, copy_to_host_params);
    graph.debugDotPrint("graph.dot");
    std::ifstream f("graph.dot");
    CHECK(f.good());
    f.close();
  }
}

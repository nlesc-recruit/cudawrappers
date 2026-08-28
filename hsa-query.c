#include <hsa/hsa.h>
#include <stdio.h>
#include <stdlib.h>

void check_status(hsa_status_t status, const char *message) {
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "%s failed: %d\n", message, status);
    exit(EXIT_FAILURE);
  }
}

hsa_status_t print_agent_info(hsa_agent_t agent, void *data) {
  char name[64];
  uint32_t gpu_id;

  // Get agent name
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  check_status(status, "hsa_agent_get_info (name)");

  // Print agent name and architecture
  printf("Agent Name: %s\n", name);
  // printf("Architecture: %s\n", name);

  return HSA_STATUS_SUCCESS;
}

int main() {
  // Initialize the HSA runtime
  hsa_status_t status = hsa_init();
  check_status(status, "hsa_init");

  // Iterate over the agents and print information
  status = hsa_iterate_agents(print_agent_info, NULL);
  check_status(status, "hsa_iterate_agents");

  // Shutdown the HSA runtime
  status = hsa_shut_down();
  check_status(status, "hsa_shut_down");

  return 0;
}

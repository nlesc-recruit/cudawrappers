# vector_add
add_subdirectory(vector_add)

# This command will be executed from the test folder, which is where the .cu file must be located
add_custom_target(test
  COMMAND vector_add/vector_add
  DEPENDS vector_add
)
# vector_add
add_subdirectory(vector_add)

add_custom_target(test
  COMMAND vector_add/vector_add
  DEPENDS vector_add
)
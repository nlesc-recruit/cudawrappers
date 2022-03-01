# Additional targets for linters
# Uses clang-format, clang-tidy and cmakelang (cmake-format and cmake-lint)

# Ref: https://stackoverflow.com/questions/32280717/cmake-clang-tidy-or-other-script-as-custom-target

file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.hpp)
file(GLOB_RECURSE ALL_CMAKE_LISTS CMakeLists.txt *.cmake)

add_custom_target(
    clang-format
    COMMAND clang-format
    -style=file
    -i
    ${ALL_SOURCE_FILES}
)

add_custom_target(
    clang-tidy
    COMMAND clang-tidy
    ${ALL_SOURCE_FILES}
    -config=''
    --
    -std=c++11
    ${INCLUDE_DIRECTORIES}
)

add_custom_target(
    cppcheck
    COMMAND cppcheck
    ${ALL_SOURCE_FILES}
)

add_custom_target(
    flawfinder
    COMMAND flawfinder
    ${ALL_SOURCE_FILES}
)

add_custom_target(
    cmake-format
    COMMAND cmake-format
    -i
    ${ALL_CMAKE_LISTS}
)

add_custom_target(
    cmake-lint
    COMMAND cmake-lint
    ${ALL_CMAKE_LISTS}
)

add_custom_target(
    lint
    DEPENDS clang-tidy cppcheck flawfinder cmake-lint
)

add_custom_target(
    format
    DEPENDS clang-format cmake-format
)

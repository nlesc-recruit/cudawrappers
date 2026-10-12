rm -rf build
cmake -S. -Bbuild -DCUDAWRAPPERS_BUILD_TESTING=1
cmake --build build -j

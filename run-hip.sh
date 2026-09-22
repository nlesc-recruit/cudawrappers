rm -rf build-hip
cmake -DCUDAWRAPPERS_BACKEND=HIP -DGPU_TARGETS=gfx1101 -S. -Bbuild-hip -DCUDAWRAPPERS_BUILD_TESTING=True -DCUDAWRAPPERS_COMPONENTS='cu;cufft;nvrtc'
cmake --build build-hip -j|& tee compile-errors.txt

FROM rocm/dev-ubuntu-24.04:7.0.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace/

RUN cmake -DCUDAWRAPPERS_BUILD_TESTING=False -DCUDAWRAPPERS_COMPONENTS='cu;cufft;nvrtc' -DCUDAWRAPPERS_BACKEND=HIP -DGPU_TARGETS=gfx1101 -S . -B build -DCMAKE_INSTALL_PREFIX=install && \
    cmake --build build -j && \
    cmake --install build

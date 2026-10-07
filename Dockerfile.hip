FROM rocm/dev-ubuntu-24.04:7.0.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash cudawrappers

WORKDIR /workspace

COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY tests ./tests
COPY README.md LICENSE ./

RUN chown -R cudawrappers:cudawrappers /workspace

USER cudawrappers

RUN cmake -DCUDAWRAPPERS_BUILD_TESTING=False -DCUDAWRAPPERS_COMPONENTS='cu;cufft;nvrtc' -DCUDAWRAPPERS_BACKEND=HIP -DGPU_TARGETS=gfx1101 -S . -B build -DCMAKE_INSTALL_PREFIX=install && \
    cmake --build build -j && \
    cmake --install build

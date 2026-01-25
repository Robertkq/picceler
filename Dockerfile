FROM fedora:latest

RUN dnf -y update && dnf -y install \
    @development-tools \
    cmake \
    git \
    ninja-build \
    python3 \
    python3-pip \
    curl \
    which \
    ccache \
    doxygen \
    pkgconfig \
    wget \
    gnupg2 \
    lsb-release \
    clang \
    clang-tools-extra \
    llvm \
    lld \
    llvm-devel \
    clang-format \
    clang-tidy \
    lldb \
    opencv-devel \
    && dnf clean all -y

ENV CC=clang
ENV CXX=clang++

RUN if command -v ccache >/dev/null 2>&1; then \
      ln -sf $(command -v ccache) /usr/local/bin/clang || true; \
      ln -sf $(command -v ccache) /usr/local/bin/clang++ || true; \
    fi

WORKDIR /opt
RUN git clone https://github.com/llvm/llvm-project.git --branch release/21.x --depth=1

RUN mkdir -p llvm-project/build && cd llvm-project/build && \
    CC=${CC} CXX=${CXX} LD=$(command -v ld.lld || echo ld.lld) AR=$(command -v llvm-ar || echo llvm-ar) RANLIB=$(command -v llvm-ranlib || echo llvm-ranlib) \
    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_USE_LINKER=lld \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DLLVM_INSTALL_UTILS=ON \
    && ninja -j$(nproc) \
    && ninja install

RUN git clone https://github.com/CLIUtils/CLI11.git /opt/CLI11 && \
    mkdir -p /opt/CLI11/build && cd /opt/CLI11/build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && ninja -j$(nproc) && ninja install && \
    git clone https://github.com/gabime/spdlog.git /opt/spdlog && mkdir -p /opt/spdlog/build && cd /opt/spdlog/build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && ninja -j$(nproc) && ninja install && \
    git clone https://github.com/google/googletest.git /opt/googletest && mkdir -p /opt/googletest/build && cd /opt/googletest/build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && ninja -j$(nproc) && ninja install

RUN rm -rf /opt/llvm-project /opt/CLI11 /opt/spdlog /opt/googletest

ENV PATH="/usr/local/bin:${PATH}"

CMD ["/bin/bash"]

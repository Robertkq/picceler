FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# ====== Base Dev Tools ======
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3 \
    python3-pip \
    curl \
    lsb-release \
    software-properties-common \
    ccache \
    doxygen \
    && apt-get clean

# ====== Install LLVM / Clang 18 ======
# Install dependencies for apt repo
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    gnupg \
    lsb-release \
    curl \
    && apt-get clean

# Add the official LLVM apt repository for Clang 18
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /usr/share/keyrings/llvm-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
        > /etc/apt/sources.list.d/llvm18.list && \
    apt-get update

# Install Clang 18 toolchain
RUN apt-get install -y \
    clang-18 \
    clang++-18 \
    lld-18 \
    clang-tidy-18 \
    clang-format-18 \
    lldb-18 \
    && apt-get clean

# Update g++/GCC to version 13
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y g++-13 gcc-13


# Set system default compilers to clang-18
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld-18 100 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100


# Export CC/CXX for future builds (CI convenience)
ENV CC=clang-18
ENV CXX=clang++-18

# Enable ccache for clang
RUN ln -s /usr/bin/ccache /usr/local/bin/clang && \
    ln -s /usr/bin/ccache /usr/local/bin/clang++

# ====== Build LLVM + MLIR (LLVM 21.x) ======
WORKDIR /opt
RUN git clone https://github.com/llvm/llvm-project.git --branch release/21.x --depth=1

RUN mkdir llvm-project/build && cd llvm-project/build && \
    CC=clang-18 \
    CXX=clang++-18 \
    LD=lld-18 \
    AR=llvm-ar-18 \
    RANLIB=llvm-ranlib-18 \
    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_USE_LINKER=lld \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && ninja -j$(nproc) \
    && ninja install

# ====== Install CLI11 ======
RUN git clone https://github.com/CLIUtils/CLI11.git && \
    cd CLI11 && mkdir build && cd build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && \
    ninja -j$(nproc) && ninja install


# ====== Install spdlog ======
RUN git clone https://github.com/gabime/spdlog.git && \
    cd spdlog && mkdir build && cd build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && \
    ninja -j$(nproc) && ninja install

# ====== Install Googletest ======
RUN git clone https://github.com/google/googletest.git && \
    cd googletest && mkdir build && cd build && \
    cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release && \
    ninja -j$(nproc) && ninja install

# ====== Cleanup build artifacts to reduce image size ======
RUN rm -rf /opt/llvm-project && \
    rm -rf /opt/CLI11 && \
    rm -rf /opt/spdlog && \
    rm -rf /opt/googletest && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/bin:${PATH}"

CMD ["/bin/bash"]

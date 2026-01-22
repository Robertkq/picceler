This document aims to provide all the necessary information to get picceler to build on your system.

We recommend building on Linux, and the following command assume you are running Linux, but it is likely for picceler to build on Windows as well.


# Dependencies

We currently have the following dependencies: 
* **MLIR** - version 21.0
* **opencv** -- most versions should work
* **spdlog** - most versions should work
* **googletest** - most versions should work
* **CLI11** - most versions should work

For **spdlog**, **googletest** and **opencv** you can likely install them via your system's package manager.

```bash
sudo dnf install spdlog gtest opencv
```

Otherwise, you can manually build & install these libraries

**googletest**
```bash
git clone https://github.com/google/googletest.git && cd googletest
mkdir build && cd build
cmake .. && cmake --build . -j
sudo cmake --install .
```

**spdlog**
```bash
git clone https://github.com/gabime/spdlog.git && cd spdlog
mkdir build && cd build
cmake .. && cmake --build . -j
sudo cmake --install .
```
**opencv***
```bash

```

For **MLIR** and **CLI11** you are much less likely to have them available via your system's package manager so we need to build them manually:

**CLI11**
```bash
git clone https://github.com/CLIUtils/CLI11.git && cd CLI11
mkdir build && cd build
cmake .. && cmake --build . -j
sudo cmake --install .
```

**MLIR**  

Building MLIR is expensive and will take you a long time compared to the other dependencies. Here are some tips to make it faster:

1. Make sure you have `Ninja` installed on your system. It is considerably faster for building MLIR
2. Make sure you have `clang` and `lld`/`mold` installed on your system. It is considerably faster than GNU ld
> The command below expects you have `Ninja` & `lld` installed. Modify accordingly if not installed
3. Leaving `-j` empty will give you the fastest compilation configuration, but it can make your PC struggle to do anything else besides compiling MLIR, you can set a value to `-j` to limit parallel jobs
```bash
git clone https://github.com/llvm/llvm-project.git --branch release/21.x --depth=1 && cd llvm-project # only get the branch we use!
mkdir build && cd build
cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_USE_LINKER=lld \
        -DCMAKE_INSTALL_PREFIX=/usr/local 
cmake --build . -j
sudo cmake --install .
```
> Make sure you do not skip on any of these cmake options, otherwise your MLIR installation is ill-formed.

### Hooray, you are done! Now let's compile piccecler!
### Todo: This section should contain more information about all the options available for picceler compilation, like tests, clang-tidy and otherrs

```bash
git clone https://github.com/Robertkq/picceler.git && cd picceler
mkdir build && cd build
cmake .. && cmake --build . -j
```

If compilation was succesful, you can now use `./picceler` to start compiling your own picceler files!

## Install picceler on your system - WIP

Once you built picceler, you can choose to install it system wide.
```bash
cd picceler/build
sudo cmake --install .
```
Done, now you can call `picceler` from anywhere on your system.

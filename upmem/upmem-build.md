## UPMEM build

### 1 download source

```sh
wget upmem.tar.gz
tar -zxvf upmem.tar.gz
wget upmem-src.tar.gz
tar -zxvf upmem-src.tar.gz
git clone https://github.com/upmem/llvm-project.git
```

### 2 build backend

```sh
cd upmem/src/backends
mkdir build&&cd build
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
cmake -DUPMEM_VERSION=2023.1.0 -DCMAKE_C_COMPILER=/usr/bin/clang -GNinja ..
ninja && ninja install
```

### 3 build llvm

```sh
cd llvm-project
gco 2023.1.0
mkdir build&&cd build
cmake -DUPMEM_API_HEADERS=/usr/local/include/dpu -GNinja -DLLVM_ENABLE_PROJECTS="clang;lldb;lld;clang-tools-extra" -DCMAKE_BUILD_TYPE=Release -DLLVM_CCACHE_BUILD=ON ../llvm
ninja && ninja install
```

### 4 build dpu-rt

```sh
cd upmem/src/dpu-rt
mkdir build&&cd build
cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=./Toolchain-dpu.cmake ..
ninja && ninja install
```

### 5 copy other stuffs

wrapper

```sh
cp llvm-project/dpu-wrappers/* /usr/local/bin/
cp llvm-project/lldb/scripts/dpu/dpu_commands.py /usr/local/lib/python3/dist-packages/lldb/
cp llvm-project/lldb/scripts/dpu/lldb_init_* /usr/local/share/upmem/lldb
```

libdpufsim.so

```sh
cp upmem/lib/libdpufsim.so.2023.1 /usr/local/lib
cd /usr/local/lib
ln -s libdpufsim.so.2023.1 libdpufsim.so
```

```sh
cp -r /root/upmem/share/pkgconfig /usr/local/share/
sed -i 's/prefix=\/usr/prefix=\/usr\/local/g' /usr/local/share/pkgconfig/dpu.pc
cp upmem/bin/dpu-pkg-config /usr/local/bin
cp -r upmem/share/java /usr/local/share
```

add `/usr/local/lib` to export LD_LIBRARY_PATH in `~/.zshrc`

```sh
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export UPMEM_PROFILE_BASE=backend=simulator
```


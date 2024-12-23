# Building Instructions

## Prerequisites

- C++17 compiler (g++ or clang++)
- Make
- Google Test (for testing)
- Doxygen (for documentation)
- Graphviz (for documentation graphs)

## Installing Dependencies

### Ubuntu/Debian

```bash
# Install build tools
sudo apt-get update
sudo apt-get install g++ make

# Install Google Test
sudo apt-get install libgtest-dev cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp lib/*.a /usr/lib
sudo ln -s /usr/src/gtest/include/gtest /usr/include/gtest

# Install documentation and tools
sudo apt-get install doxygen gnuplot graphviz libboost-all-dev
```

### macOS

```bash
# Install build tools
xcode-select --install

# Install Google Test
brew install googletest

# Install documentation tools
brew install doxygen graphviz boost
```

### Windows

1. Install MinGW or Visual Studio
2. Install vcpkg
3. Install Google Test through vcpkg:

```bash
vcpkg install gtest:x64-windows
```

## Python Bindings

To build the Python bindings, you need to have Python and pybind11 installed:

```bash
pip install pybind11[global]

# If you're using conda, ensure you have the latest libstdc++:
conda install -c conda-forge libstdcxx-ng
```

On ubuntu/debian, you can install pybind11 using:

```bash
sudo apt-get install python3-pybind11
```

You then can install the Python bindings using the setup.py script:

```bash
pip install .
```

## Optional Dependencies

- PostgreSQL
- MongoDB
- Kafka
- RabbitMQ
- JSON

These can be installed on ubuntu/debian with:

```bash
sudo apt-get install libpq-dev libstdc++6 libmongoc-dev libkafka-dev librabbitmq-dev libjsoncpp-dev
```

## Building the Project

1.Clone the repository:

```bash
git clone git@github.com:JohnnyTeutonic/chunking_cpp.git
cd chunking_cpp
```

2.Conversion of the configure script to unix line endings (Windows only):

```bash

unix2dos configure
```

Make configure script executable

```bash
chmod +x configure
```

3.Configure the build:

```bash
./configure
```

Configuration options:

- `--prefix=PATH`: Set installation prefix (default: /usr/local)
- `--build-type=TYPE`: Set build type (Debug|Release)
- `--disable-tests`: Disable building tests
- `--disable-docs`: Disable building documentation

example:

```bash
./configure --prefix=/usr/local --build-type=Debug --disable-tests --disable-docs
```

3.Build and install the project:

```bash
make install
```

4.Run the tests:

```bash
make test
```

5.Generate documentation:

```bash
make docs
```

6.Serve documentation locally:

```bash
make docs-serve
```

## Project Structure

```markdown
.
├── .github/
│   └── workflows/
│       └── ci.yml
│       └── docs.yml
├── docs/
│   └── html/
├── include/
│   ├── chunk.hpp
│   ├── config.hpp
│   ├── chunk_strategies.hpp
│   ├── chunk_compression.hpp
│   ├── sub_chunk_strategies.hpp
│   ├── parallel_chunk.hpp
│   ├── advanced_structures.hpp
│   ├── sophisticated_chunking.hpp
│   ├── data_structures.hpp
│   └── utils.hpp
├── src/
│   ├── main.cpp
│   └── sophisticated_chunking_demo.cpp
├── tests/
│   ├── advanced_chunk_strategies_test.cpp
│   ├── advanced_structures_test.cpp
│   ├── chunk_compression_test.cpp
│   ├── chunk_strategies_test.cpp
│   ├── chunking_methods_sophisticated_test.cpp
│   ├── data_structures_test.cpp
│   ├── parallel_chunk_test.cpp
│   ├── sub_chunk_strategies_test.cpp
│   ├── test_main.cpp
│   └── utils_test.cpp
├── Makefile
├── CMakeLists.txt
├── Doxyfile
├── README.md
├── BUILDING.md
└─��� LICENSE
```

## Make Targets

- `make`: Build the project
- `make run`: Run the main program
- `make run-sophisticated`: Run the sophisticated chunking demo
- `make test`: Run all tests
- `make test-<name>`: Run specific test suite
- `make docs`: Generate documentation
- `make docs-serve`: Serve documentation locally
- `make docs-clean`: Clean documentation build artifacts
- `make docs-stop`: Stop documentation server
- `make clean`: Clean build artifacts
- `make format`: Format source code
- `make format-check`: Check source code formatting
- `make install`: Install the project
- `make uninstall`: Uninstall the project

## Advanced Features

### Sub-Chunking Strategies

The library provides several sub-chunking strategies:

- **Recursive Sub-chunking**: Apply a strategy recursively to create hierarchical chunks
- **Hierarchical Sub-chunking**: Apply different strategies at each level
- **Conditional Sub-chunking**: Apply sub-chunking based on chunk properties

Example:

```cpp
auto strategy = std::make_shared<VarianceStrategy<double>>(5.0);
RecursiveSubChunkStrategy<double> recursive(strategy, 2, 2);
auto sub_chunks = recursive.apply(data);
```

### Building and Testing Sophisticated Chunking

To build and run the sophisticated chunking examples:

```bash
# Build the project including sophisticated chunking demo
make

# Run the sophisticated chunking demo
./build/bin/sophisticated_chunking_demo

```

The sophisticated chunking features are implemented in:

- `include/sophisticated_chunking.hpp`: Core implementations
- `src/sophisticated_chunking_demo.cpp`: Usage examples
- `tests/chunking_methods_sophisticated_test.cpp`: Unit tests

### Running Sophisticated Chunking Tests

To run only the sophisticated chunking tests:

```bash
cd build
ctest -R sophisticated
```

Or using make:

```bash
make test-sophisticated
```

### Sophisticated Chunking Integration

To use sophisticated chunking in your project:

1. Include the header:

```cpp
#include "sophisticated_chunking.hpp"
```

2. Link against the library in your CMakeLists.txt:

```cmake
target_link_libraries(your_target PRIVATE sophisticated_chunking)
```

### Performance Considerations for Sophisticated Chunking

- **Wavelet Chunking**: O(n * window_size) complexity. Choose smaller window sizes for better performance.
- **Mutual Information**: O(n * context_size) complexity. Larger context sizes impact performance significantly.
- **DTW Chunking**: O(n * window_size²) complexity. Window size has quadratic impact on performance.

### Troubleshooting

If you encounter GLIBCXX version errors when installing the Python bindings:

1. Update conda's libstdc++: `conda install -c conda-forge libstdcxx-ng`
2. Or use system libraries: `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`

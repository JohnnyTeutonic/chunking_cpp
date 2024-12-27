# Building Instructions

## Prerequisites

- C++17 compiler (g++ or clang++)
- Make
- Google Test (for testing)
- Doxygen (for documentation)
- Graphviz (for documentation graphs)
- CMake 3.10 or higher
- pybind11 (for Python bindings)

## Installing Dependencies

### Ubuntu/Debian

```bash
# Install build tools
sudo apt-get update
sudo apt-get install g++ make cmake

# Install Google Test
sudo apt-get install libgtest-dev cmake
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp lib/*.a /usr/lib
sudo ln -s /usr/src/gtest/include/gtest /usr/include/gtest

# Install documentation and other dependencies
sudo apt-get install doxygen gnuplot graphviz libboost-all-dev python3-pybind11 \
libpq-dev libstdc++6 libmongoc-dev \
librdkafka-dev librabbitmq-dev libjsoncpp-dev \
librdkafka++1 
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
# Method 1: Using pip (recommended)
pip install pybind11[global]

# Method 2: Using system package manager (Ubuntu/Debian)
sudo apt-get install python3-dev python3-pybind11

# If you're using conda, ensure you have the latest libstdc++:
conda install -c conda-forge libstdcxx-ng pybind11[global]
```

On ubuntu/debian, you can install pybind11 using:

```bash
sudo apt-get install python3-pybind11
```

You then can install the Python bindings (and pytest) using the setup.py script:

```bash
pip install .
```

You can then mock the same functionality in python:

```python
from chunking_cpp import chunking_cpp as cc

chunk = cc.Chunk(3)
chunk.add(1)
chunk.add(2)
chunk.add(3)
chunk.chunk_by_threshold(1.0)
```

## Optional Dependencies

Testing of the python suite can be installed on ubuntu/debian with:

```bash
sudo apt -y install python3-pytest python3-pytest-cov
```

## Configuration Options

The project uses CMake for configuration. Available options:

```bash
-DENABLE_COVERAGE=ON    # Enable code coverage reporting
-DENABLE_SANITIZERS=ON  # Enable address and undefined behavior sanitizers
-DBUILD_TESTING=ON      # Enable building tests
```


## Building the Project

1.Clone the repository:

```bash
git clone git@github.com:JohnnyTeutonic/chunking_cpp.git
cd chunking_cpp
mkdir build && cd build
```

2.Configure, build and install (with all options enabled by default):

```bash
./configure --enable-tests --enable-docs --enable-sanitizers --enable-coverage
make
```

3.Run tests:

```bash
make test
```

4.Generate documentation:

```bash
make docs
```

5.Serve documentation locally:

```bash
make docs-serve
```

6.Run tests with pytest:

```bash
make pytest
```

7.Run tests with pytest and coverage:

```bash
make pytest-coverage
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
├── bindings/
│   └── python/
│       └── chunk_bindings.cpp
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
│   ├── neural_chunking.hpp
│   └── utils.hpp
├── src/
│   ├── main.cpp
│   ├── demo_neural_chunking.cpp
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
│   ├── test_neuralnetwork.cpp
│   ├── test_main.cpp
│   ├── python/
│   │   └── py_bindings.py
│   └── utils_test.cpp
├── scripts/
│   └── pybindings_example.py
├── Makefile
├── CMakeLists.txt
├── Doxyfile
├── setup.py
├── README.md
├── BUILDING.md
└── LICENSE
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
- `make pytest`: Run tests with pytest
- `make pytest-coverage`: Run tests with pytest and coverage

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

2.Link against the library in your CMakeLists.txt:

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

The library version can be accessed through the following macros:

```cpp
CHUNKING_VERSION_MAJOR
CHUNKING_VERSION_MINOR
CHUNKING_VERSION_PATCH
```

## ADDITIONAL INFORMATION

- [README](README.md)
- [Python Bindings](bindings/python/chunking_cpp)
- [Documentation](docs/html/index.html)

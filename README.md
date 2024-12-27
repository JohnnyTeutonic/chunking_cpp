# Advanced Chunk Processing Library

[![codecov](https://codecov.io/gh/JohnnyTeutonic/chunking_cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/JohnnyTeutonic/chunking_cpp/)
[![Tests](https://github.com/JohnnyTeutonic/chunking_cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyTeutonic/chunking_cpp/actions/workflows/ci.yml)

A modern C++17 library providing advanced data chunking and processing capabilities with a focus on performance and flexibility.

## Overview

This library offers a comprehensive suite of tools for handling data in chunks, with features ranging from basic chunking to advanced parallel processing and compression techniques. It's designed to be efficient, type-safe, and easy to integrate into existing C++ projects.

## Features

### Core Functionality

- Parallel chunk processing
- Chunk compression (RLE, Delta)
- Basic chunking operations
- Sub-chunking strategies

### Advanced Chunking Strategies

- Overlapping chunks
- Predicate-based chunking
- Sum-based chunking
- Equal division chunking
- Sliding window chunking
- Statistical threshold-based chunking
- Similarity-based chunking
- Monotonicity-based chunking
- Padded fixed-size chunking
- Wavelet-based chunking
- Mutual Information-based chunking
- Dynamic Time Warping (DTW) based chunking

### Sub-Chunking Strategies

- Recursive sub-chunking
- Hierarchical sub-chunking
- Conditional sub-chunking

### Data Structures

- Circular buffer
- Sliding window
- Priority queue
- Chunk list
- Skip list
- B+ tree
- ChunkDeque
- ChunkStack
- ChunkTreap

### Analysis & Utilities

- Statistical operations
- Random chunk generation
- Chunk manipulation utilities
- Chunk visualization
- Chunk benchmarking

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- Google Test (for testing)
- Doxygen (for documentation)
- Graphviz (for documentation graphs)
- Boost (for advanced structures)

## Quick Start

```bash
# Clone the repository
git clone git@github.com:JohnnyTeutonic/chunking_cpp.git
cd chunking_cpp
```

# Install the dependencies
```bash
sudo apt -y install build-essential cnake doxygen gnuplot graphviz libboost-all-dev python3-pybind11 \
libpq-dev libstdc++6 libmongoc-dev librdkafka-dev librabbitmq-dev libjsoncpp-dev librdkafka++1 
```
# Installing Google Test
```bash
sudo apt-get install -y libgtest-dev cmake
mkdir -p $HOME/build
cd $HOME/build
sudo cmake /usr/src/googletest/googletest
sudo make
sudo cp lib/libgtest* /usr/lib/
cd ..
sudo rm -rf build
sudo mkdir /usr/local/lib/googletest
sudo ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a
sudo ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a
```

# Configure the project (optionally with tests, documentation, sanitizers, and coverage enabled)
```bash
chmod +x configure
./configure  --enable-tests --enable-docs --enable-sanitizers --enable-coverage
```
# Build the project
```bash
make
```
# To execute the example program
```bash
make run
```
# Run tests
```bash
make test
```
# Uninstall if needed
```bash
make uninstall
```

For detailed build instructions, dependencies, and configuration options, see [BUILDING](BUILDING.md).

## Usage Examples

### Basic Usage

```cpp
#include "chunk.hpp"

// Create a chunk processor with size 2
Chunk<int> chunker(2);

// Add data
std::vector<int> data = {1, 2, 3, 4, 5};
chunker.add(data);

// Get fixed-size chunks
auto chunks = chunker.get_chunks(); // Returns: {{1,2}, {3,4}, {5}}
```

### Multi-dimensional Vector Support

The library provides comprehensive support for processing multi-dimensional vectors:

```cpp
#include "chunk.hpp"
#include "chunk_strategies.hpp"

// Working with 2D vectors
std::vector<std::vector<double>> data_2d = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0},
    {7.0, 8.0, 9.0}
};

// Create a chunker for 2D data with chunk size 2
chunk_processing::Chunk<std::vector<double>> chunker_2d(2);
chunker_2d.add(data_2d);

// Get chunks - each chunk contains 2 rows
auto chunks_2d = chunker_2d.get_chunks();

// Pattern-based chunking with sum threshold
chunk_strategies::PatternBasedStrategy<std::vector<double>> strategy(
    [](const std::vector<double>& row) {
        return std::accumulate(row.begin(), row.end(), 0.0) > 10.0;
    }
);

// Split into chunks when row sum exceeds threshold
auto threshold_chunks = strategy.apply(data_2d);

// Working with 3D vectors
std::vector<std::vector<std::vector<double>>> data_3d = {
    {{1.0, 2.0}, {3.0, 4.0}},
    {{5.0, 6.0}, {7.0, 8.0}},
    {{9.0, 10.0}, {11.0, 12.0}}
};

// Create a chunker for 3D data
chunk_processing::Chunk<std::vector<std::vector<double>>> chunker_3d(2);
chunker_3d.add(data_3d);

// Automatic dimensionality validation
std::vector<std::vector<double>> inconsistent_2d = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0},  // Different size - will throw std::invalid_argument
    {7.0, 8.0, 9.0}
};
```

Key features for multi-dimensional data:

- Automatic dimensionality validation
- Support for arbitrary nesting depth
- Specialized chunking strategies for multi-dimensional data
- Efficient processing of nested structures
- Type-safe operations across all dimensions

The library automatically detects the dimensionality of your data and applies appropriate processing strategies. All chunking operations maintain the structural integrity of your multi-dimensional data while providing efficient processing capabilities.

### Advanced Features

- **Recursive Sub-chunking**: Apply a strategy recursively to create hierarchical chunks
- **Hierarchical Sub-chunking**: Apply different strategies at each level
- **Conditional Sub-chunking**: Apply sub-chunking based on chunk properties
- **Parallel Processing**: Process chunks in parallel using multiple threads
- **Chunk Compression**: Compress chunks using various algorithms
- **Chunk Manipulation**: Add, remove, and modify chunks
- **Chunk Analysis**: Perform statistical analysis on chunks
- **Chunk Randomization**: Randomize chunks
- **Overlapping Chunks**: Create chunks that overlap
- **Predictive Chunking**: Create chunks based on a predictive model
- **Sum-based Chunking**: Create chunks based on the sum of the elements
- **Equal Division Chunking**: Create chunks based on the equal division of the elements
- **Sliding Window Chunking**: Create chunks based on a sliding window
- **Statistical Threshold-based Chunking**: Create chunks based on a statistical threshold
- **Similarity-based Chunking**: Create chunks based on similarity
- **Monotonicity-based Chunking**: Create chunks based on monotonicity
- **Padded Fixed-size Chunking**: Create chunks based on a padded fixed-size
- **ChunkDeque**: A deque-based chunk structure for double-ended operations
- **ChunkStack**: A stack-based chunk structure for LIFO operations
- **ChunkTreap**: A treap-based chunk structure for efficient searching and manipulation
- **Semantic Chunking**: Create chunks based on semantic/cosine similarity
- **Wavelet-based Chunking**: Create chunks based on wavelet coefficients
- **Mutual Information-based Chunking**: Create chunks based on mutual information
- **Dynamic Time Warping (DTW) based Chunking**: Create chunks based on dynamic time warping

#### Example Usage

```cpp
#include "chunk.hpp"
#include "chunk_strategies.hpp"
#include "sub_chunk_strategies.hpp"
#include <iostream>
#include <vector>
#include "advanced_structures.hpp"

using namespace chunk_strategies;
// Sub-chunking example
auto variance_strategy = std::make_shared<VarianceStrategy<double>>(5.0);
RecursiveSubChunkStrategy<double> recursive_strategy(variance_strategy, 2, 2);
auto sub_chunks = recursive_strategy.apply(data);

// Predicate-based chunking
auto even_chunks = chunker.chunk_by_predicate([](int x) { return x % 2 == 0; });

// Similarity-based chunking
auto similar_chunks = chunker.chunk_by_similarity(3);

// Parallel processing
ParallelChunkProcessor<int> parallel_chunker(4); // 4 threads
parallel_chunker.process_chunks(chunks, [](int x) { return x * 2; });

// Recursive sub-chunking
auto variance_strategy = std::make_shared<VarianceStrategy<double>>(3.0);
RecursiveSubChunkStrategy<double> recursive_strategy(variance_strategy, 2, 2);
auto recursive_result = recursive_strategy.apply(data);

// Hierarchical sub-chunking
std::vector<std::shared_ptr<ChunkStrategy<double>>> strategies = {
    std::make_shared<VarianceStrategy<double>>(5.0),
    std::make_shared<EntropyStrategy<double>>(1.0)
};
HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies, 2);
auto hierarchical_result = hierarchical_strategy.apply(data);

// Conditional sub-chunking
auto condition = [](const std::vector<double>& chunk) {
    double mean = std::accumulate(chunk.begin(), chunk.end(), 0.0) / chunk.size();
    double variance = 0.0;
    for (const auto& val : chunk) {
        variance += (val - mean) * (val - mean);
    }
    variance /= chunk.size();
    return variance > 50.0;
  };

ConditionalSubChunkStrategy<double> conditional_strategy(variance_strategy, condition, 2);
auto conditional_result = conditional_strategy.apply(data);
// ChunkTreap example
ChunkTreap<int> chunk_treap;
chunk_treap.insert(5);
chunk_treap.insert(3);
chunk_treap.insert(8);
chunk_treap.search(5);

// ChunkDeque example
ChunkDeque<int> chunk_deque;
chunk_deque.push_back(1);
chunk_deque.push_front(0);
chunk_deque.pop_back();
chunk_deque.pop_front();
```

See `src/main.cpp` for more comprehensive examples.

## Documentation

The library is extensively documented using Doxygen. You can:

1. View the online documentation at [GitHub Pages](https://johnnyteutonic.github.io/chunking_cpp/)
2. Generate documentation locally:

```bash
make docs
make docs-serve
```

### Performance Considerations

- **Chunk Size**: Choose an appropriate chunk size based on your data and processing requirements. Larger chunks may reduce overhead but increase memory usage.
- **Parallel Processing**: Utilize the `ParallelChunkProcessor` for operations that can be parallelized to improve performance on multi-core systems.
- **Memory Management**: Be mindful of memory usage, especially when dealing with large datasets. Use efficient data structures like `CircularBuffer` to manage memory effectively.
- **Algorithm Complexity**: Consider the complexity of the chunking strategies and operations you use. Some strategies may have higher computational costs.

### Best Practices

- **Use Smart Pointers**: Use `std::shared_ptr` and `std::unique_ptr` to manage dynamic memory and avoid memory leaks.
- **Leverage STL Algorithms**: Utilize standard library algorithms for common operations like sorting and accumulating to improve code readability and performance.
- **Modular Design**: Keep your code modular by separating chunking logic from data processing logic. This makes it easier to test and maintain.
- **Testing**: Write comprehensive tests for your chunking strategies and operations to ensure correctness and reliability.
- **Documentation**: Document your code and strategies using Doxygen-style comments to make it easier for others to understand and use your library.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{chunking_cpp,
  author = {Reich, Jonathan},
  title = {Advanced Chunk Processing Library},
  year = {2024},
  url = {https://github.com/JohnnyTeutonic/chunking_cpp}
}
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
��   ├── sub_chunk_strategies.hpp
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

### Sophisticated Chunking Examples

To run the sophisticated chunking demo, use:

```bash
make sophisticated_chunking_demo
```

For more detailed examples of the sophisticated chunking algorithms, see `src/sophisticated_chunking_demo.cpp`:

```cpp
#include "sophisticated_chunking.hpp"

// Example from sophisticated_chunking_demo.cpp
int main() {
    // Wavelet-based chunking example
    std::vector<double> signal_data = {1.0, 1.2, 1.1, 5.0, 5.2, 5.1, 2.0, 2.1};
    sophisticated_chunking::WaveletChunking<double> wavelet_chunker(8, 0.5);
    auto wavelet_chunks = wavelet_chunker.chunk(signal_data);

    // Mutual Information chunking example
    std::vector<int> pattern_data = {1, 2, 3, 10, 11, 12, 4, 5, 6};
    sophisticated_chunking::MutualInformationChunking<int> mi_chunker(5, 0.3);
    auto mi_chunks = mi_chunker.chunk(pattern_data);

    // DTW chunking example
    std::vector<float> time_series = {1.0f, 1.1f, 1.2f, 5.0f, 5.1f, 5.2f};
    sophisticated_chunking::DTWChunking<float> dtw_chunker(10, 2.0);
    auto dtw_chunks = dtw_chunker.chunk(time_series);

    return 0;
}
```

### Chunk Visualization

For detailed examples of the chunk visualization, see `src/chunk_visualization_demo.cpp`:

```cpp
#include "chunk_visualization.hpp"
```

To run the chunk visualization demo, use:

```bash
make run-visualization
```

To use the visualisation as a standalone tool with gnuplot run:

```bash
gnuplot visualization/plot_chunks.gnu
```

Or with graphviz:

```bash
dot -Tpng visualization/chunks.dot -o visualization/chunks.png
```

### Chunk Benchmarking

To compile and run the benchmark demo, use:

```bash
make benchmark
```

Or to use it as a standalone tool:

```cpp
#include "chunk_benchmark.hpp"
...
```

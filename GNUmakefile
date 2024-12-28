# Project configuration
PROJECT_NAME := ChunkProcessor
BUILD_DIR := build
DOC_DIR := docs
TEST_DIR := $(BUILD_DIR)/tests
SRC_FILES := $(shell find . -name "*.cpp" -o -name "*.hpp")

# Compiler settings
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
INCLUDES = -I./include
LDFLAGS = -pthread

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source files
METRICS_SOURCES = src/chunk_metrics.cpp

# Ensure build directory exists and is configured
.PHONY: setup-build test docs docs-clean docs-serve docs-stop local-help run uninstall format format-check neural_chunking_demo sophisticated_chunking_demo metrics chunk_metrics

setup-build:
	@mkdir -p $(BUILD_DIR)
	@echo "Configuring CMake..."
	@cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
	@echo "Building project..."
	@cmake --build $(BUILD_DIR) --config Debug

# Run the executable
run: setup-build
	@echo "Running program..."
	@$(BUILD_DIR)/chunk_processor_exe

# Run the neural chunking demo
neural_chunking_demo: setup-build
	@echo "Running neural chunking demo..."
	@if [ -f "$(BUILD_DIR)/neural_chunking_demo" ]; then \
		$(BUILD_DIR)/neural_chunking_demo; \
	else \
		echo "Error: neural_chunking_demo executable not found"; \
		echo "Build failed. Try running 'make clean' followed by 'make'"; \
		exit 1; \
	fi

# Run the sophisticated chunking demo
sophisticated_chunking_demo: setup-build
	@echo "Running sophisticated chunking demo..."
	@if [ -f "$(BUILD_DIR)/sophisticated_chunking_demo" ]; then \
		$(BUILD_DIR)/sophisticated_chunking_demo; \
	else \
		echo "Error: sophisticated_chunking_demo executable not found"; \
		echo "Build failed. Try running 'make clean' followed by 'make'"; \
		exit 1; \
	fi

# Override CMake's test target
test: setup-build
	@echo "Running tests..."
	@if [ -f "$(BUILD_DIR)/run_tests" ]; then \
		$(BUILD_DIR)/run_tests; \
	elif [ -f "$(BUILD_DIR)/tests/run_tests" ]; then \
		$(BUILD_DIR)/tests/run_tests; \
	else \
		echo "Error: run_tests executable not found"; \
		echo "Checking in:"; \
		echo "  - $(BUILD_DIR)/run_tests"; \
		echo "  - $(BUILD_DIR)/tests/run_tests"; \
		echo "Build failed. Try running 'make clean' followed by 'make'"; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Alias for clean
uninstall: clean

# Documentation targets
docs: setup-build
	@if ! command -v doxygen > /dev/null; then \
		echo "Error: doxygen not found. Please install doxygen first."; \
		exit 1; \
	fi
	@echo "Creating documentation directories..."
	@mkdir -p $(DOC_DIR)
	@mkdir -p $(DOC_DIR)/html
	@chmod 755 $(DOC_DIR)
	@echo "Generating documentation..."
	@doxygen Doxyfile
	@if [ ! -f "$(DOC_DIR)/html/index.html" ]; then \
		echo "Error: Documentation generation failed!"; \
		exit 1; \
	fi
	@echo "Documentation generated in $(DOC_DIR)/html"

# Clean documentation
docs-clean:
	@rm -rf $(DOC_DIR)
	@echo "Cleaned documentation directory"

# Serve documentation locally
docs-serve: docs
	@if [ ! -f "$(DOC_DIR)/html/index.html" ]; then \
		echo "Error: Documentation not found. Running 'make docs' first..."; \
		$(MAKE) docs; \
	fi
	@mkdir -p $(DOC_DIR)
	@echo "Starting documentation server on port 8000..."
	@if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then \
		echo "Port 8000 is already in use. Killing existing process..." ; \
		lsof -ti :8000 | xargs kill -9 ; \
	fi
	@cd $(DOC_DIR)/html && \
	(python3 -m http.server --bind 127.0.0.1 8000 2>/dev/null & echo $$! > $(DOC_DIR)/.docs-server.pid) && \
	echo "Documentation server started. Visit http://localhost:8000"

# Help target
local-help:
	@echo "Custom targets available:"
	@echo "  setup-build  - Configure and build the project"
	@echo "  neural_chunking_demo - Run the neural chunking demo"
	@echo "  benchmark - Run the benchmark"
	@echo "  visualization - Run the visualization"
	@echo "  metrics - Run the metrics"
	@echo "  metrics-debug - Run the metrics with debug output"
	@echo "  pytest - Run the pytest tests"
	@echo "  chunk_metrics - Run the chunk metrics"
	@echo "  pytest-coverage - Run the pytest tests with coverage"
	@echo "  sophisticated_chunking_demo - Run the sophisticated chunking demo"
	@echo "  test         - Run tests"
	@echo "  docs         - Generate documentation"
	@echo "  docs-clean   - Remove generated documentation"
	@echo "  docs-serve   - Serve documentation locally at http://localhost:8000"
	@echo "  docs-stop    - Stop the documentation server"
	@echo "  uninstall    - Remove all build artifacts (alias for clean)"
	@echo "  format       - Format source files using clang-format"
	@echo "  format-check - Check if source files are properly formatted"
	@echo "  local-help   - Show this help message"
	@echo
	@echo "For CMake targets, run 'make help'"

# Default target
.DEFAULT_GOAL := setup-build

# Format source files
format:
	@if ! command -v clang-format > /dev/null; then \
		echo "Error: clang-format not found. Please install clang-format first."; \
		exit 1; \
	fi
	@echo "Formatting source files..."
	@clang-format -i $(SRC_FILES)
	@echo "Formatting complete"

# Check if files are properly formatted
format-check:
	@if ! command -v clang-format > /dev/null; then \
		echo "Error: clang-format not found. Please install clang-format first."; \
		exit 1; \
	fi
	@echo "Checking source file formatting..."
	@clang-format --dry-run -Werror $(SRC_FILES)
	@echo "All files are properly formatted"

docs-stop:
	@mkdir -p $(DOC_DIR)
	@if [ -f $(DOC_DIR)/.docs-server.pid ]; then \
		echo "Stopping DOC_DIR server..." ; \
		kill -9 `cat $(BUILD_DIR)/.docs-server.pid` 2>/dev/null || true ; \
		rm $(DOC_DIR)/.docs-server.pid ; \
	fi

benchmark:
	@g++ -std=c++17 src/benchmark.cpp -I./include -o benchmark
	@./benchmark

visualization:
	@g++ -std=c++17 src/chunk_visualization_demo.cpp -I./include -o chunk_visualization_demo
	@./chunk_visualization_demo

# i.e. make pytest ARGS="--verbose"
pytest:
	@pytest tests/python/test_py_bindings.py $(ARGS)


pytest-coverage:
	@pytest --cov=chunking_cpp --cov-report=html --cov-report=term tests/python/test_py_bindings.py
	@echo "Coverage report generated in htmlcov/index.html"

# Metrics target
metrics: chunk_metrics
	@echo "Running metrics calculations..."
	./chunk_metrics

# Build target for metrics executable
chunk_metrics: $(METRICS_SOURCES) $(INCLUDE_DIR)/chunk_metrics.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(METRICS_SOURCES) -o chunk_metrics $(LDFLAGS)

# Run metrics with debug output
metrics-debug: chunk_metrics
	@echo "Running metrics calculations with debug output..."
	./chunk_metrics --debug

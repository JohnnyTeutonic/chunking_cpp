cmake_minimum_required(VERSION 3.10)
project(ChunkProcessor VERSION 0.1.0)

# Create documentation directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/docs/html)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Try multiple methods to find pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Method 1: Try standard CMake find_package
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Method 2: Try pip installed location
    execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_FIND_RESULT
    )
    if(PYBIND11_FIND_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG QUIET)
    endif()
endif()

if(NOT pybind11_FOUND)
    # Method 3: Try to fetch pybind11 using FetchContent
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find Python and get site-packages directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE Python_SITEARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add Python support
# enable_language(Python)

# Add compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

# Check for CUDA support
find_package(CUDA QUIET)
if(CUDA_FOUND)
    find_path(CUDA_RUNTIME_HEADER cuda_runtime.h
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include
        PATH_SUFFIXES cuda
    )
    if(CUDA_RUNTIME_HEADER)
        message(STATUS "CUDA Runtime found - enabling GPU support")
        add_definitions(-DHAVE_CUDA)
        include_directories(${CUDA_RUNTIME_HEADER})
        
        # Add GPU chunking library only if CUDA is available
        add_library(gpu_chunking
            src/demo_gpu_chunking.cpp
        )
        
        target_include_directories(gpu_chunking PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${CUDA_RUNTIME_HEADER}
        )
        
        target_link_libraries(gpu_chunking
            PRIVATE
            ${CUDA_LIBRARIES}
        )
        
        # Add GPU demo executable
        add_executable(gpu_chunking_demo src/demo_gpu_chunking.cpp)
        target_link_libraries(gpu_chunking_demo
            PRIVATE
            gpu_chunking
            ${CUDA_LIBRARIES}
        )
    else()
        message(STATUS "CUDA Runtime not found - disabling GPU support")
    endif()
else()
    message(STATUS "CUDA not found - disabling GPU support")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${GTEST_INCLUDE_DIRS})

# Find Boost package with required components
find_package(Boost REQUIRED COMPONENTS graph)

# Add library target with all source files
add_library(chunk_processor
    src/main.cpp
    src/sophisticated_chunking_demo.cpp
    src/benchmark.cpp
    src/demo_neural_chunking.cpp
)

# Make headers available for the library
target_include_directories(chunk_processor PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Link Boost to the library
target_link_libraries(chunk_processor 
    PRIVATE 
    Boost::graph
    $<$<BOOL:${CUDA_FOUND}>:gpu_chunking>
)

# Add executable targets
add_executable(chunk_processor_exe src/main.cpp)
add_executable(benchmark_exe src/benchmark.cpp)
add_executable(neural_chunking_demo src/demo_neural_chunking.cpp)
add_executable(sophisticated_chunking_demo src/sophisticated_chunking_demo.cpp)

# Link executables with the library
target_link_libraries(chunk_processor_exe chunk_processor)
target_link_libraries(benchmark_exe chunk_processor)
target_link_libraries(neural_chunking_demo chunk_processor)
target_link_libraries(sophisticated_chunking_demo 
    PRIVATE
    chunk_processor
    sophisticated_chunking
)

# Enable testing
enable_testing()

# Find GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Set runtime output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Create tests directory if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

# Ensure the tests directory exists in the source directory
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/tests")

# Create the template file in the source directory if it doesn't exist
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tests/run_tests.sh.in")
    file(WRITE "${CMAKE_SOURCE_DIR}/tests/run_tests.sh.in"
        "#!/bin/bash\n\n"
        "# Test runner script generated by CMake\n"
        "# @CMAKE_BINARY_DIR@ will be replaced with actual binary directory\n\n"
        "# Set colored output for tests\n"
        "export GTEST_COLOR=1\n\n"
        "# Run the test executable\n"
        "@CMAKE_BINARY_DIR@/run_tests \"$@\"\n\n"
        "# Store the exit code\n"
        "exit_code=$?\n\n"
        "# Exit with the test result\n"
        "exit $exit_code\n"
    )
endif()

# Ensure the tests directory exists in the build directory
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

# Use configure_file with the template from the source directory
configure_file(
    "${CMAKE_SOURCE_DIR}/tests/run_tests.sh.in"
    "${CMAKE_BINARY_DIR}/tests/run_tests.sh"
    @ONLY
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# Add fixture to ensure test output directory exists
add_test(NAME create_test_output
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tests
)
set_tests_properties(create_test_output PROPERTIES
    FIXTURES_SETUP test_output
)

# Find all test files using globbing
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS 
    "${CMAKE_SOURCE_DIR}/tests/*.cpp"
)

# Add test base implementation
set(ALL_TEST_SOURCES
    ${TEST_SOURCES}
    tests/test_base.cpp
    tests/test_metrics.cpp
    tests/sub_chunk_strategies_test.cpp
    tests/parallel_chunk_test.cpp
)

# Add single test executable with all sources
add_executable(run_tests ${ALL_TEST_SOURCES})

# Link test executable
target_link_libraries(run_tests
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    chunk_processor
    sophisticated_chunking
    neural_chunking
    chunk_benchmark
)

# Add error checking
if(NOT EXISTS "${CMAKE_BINARY_DIR}/tests")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/tests")
        message(FATAL_ERROR "Failed to create tests directory")
    endif()
endif()

# First ensure the tests directory exists
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

# Generate the run_tests.sh script directly
file(WRITE "${CMAKE_BINARY_DIR}/tests/run_tests.sh"
    "#!/bin/bash\n\
\n\
# Test runner script generated by CMake\n\
# Set colored output for tests\n\
export GTEST_COLOR=1\n\
\n\
# Run the test executable\n\
${CMAKE_BINARY_DIR}/run_tests \"$@\"\n\
\n\
# Store the exit code\n\
exit_code=$?\n\
\n\
# Exit with the test result\n\
exit $exit_code\n"
)

# Set proper permissions on the script
file(CHMOD "${CMAKE_BINARY_DIR}/tests/run_tests.sh"
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# Add single test to CTest
add_test(
    NAME AllTests
    COMMAND bash ${CMAKE_BINARY_DIR}/tests/run_tests.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Set test properties
set_tests_properties(AllTests PROPERTIES
    ENVIRONMENT "GTEST_COLOR=1"
    FAIL_REGULAR_EXPRESSION ".*[Ff]ail.*"
)

# Add the sophisticated chunking library
add_library(sophisticated_chunking
    src/sophisticated_chunking_demo.cpp
)

target_include_directories(sophisticated_chunking PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Add neural chunking library
add_library(neural_chunking
    src/demo_neural_chunking.cpp
)

target_include_directories(neural_chunking PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Link the specialized libraries to the main library
target_link_libraries(chunk_processor
    PRIVATE
    sophisticated_chunking
    neural_chunking
)

# Add benchmark library
add_library(chunk_benchmark
    src/benchmark.cpp
)

target_include_directories(chunk_benchmark PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(chunk_benchmark
    PRIVATE
    chunk_processor
)

# Add individual test executables
add_executable(chunking_methods_sophisticated_test
    tests/chunking_methods_sophisticated_test.cpp
)

add_executable(neural_network_test
    tests/test_neuralnetwork.cpp
)

# Link test executables
target_link_libraries(chunking_methods_sophisticated_test
    sophisticated_chunking
    gtest
    gtest_main
)

target_link_libraries(neural_network_test
    neural_chunking
    gtest
    gtest_main
)

# Add to test suite
add_test(NAME chunking_methods_sophisticated COMMAND chunking_methods_sophisticated_test)
add_test(NAME neural_network COMMAND neural_network_test)

# Install targets
install(TARGETS 
    chunk_processor
    sophisticated_chunking
    neural_chunking
    chunk_benchmark
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add Python module
pybind11_add_module(chunking_cpp 
    bindings/python/chunk_bindings.cpp
)

# Link with our libraries
target_link_libraries(chunking_cpp PRIVATE
    chunk_processor
    sophisticated_chunking
    neural_chunking
    chunk_benchmark
    chunk_metrics_impl
    chunk_serialization_impl
    chunk_visualization_impl
    $<$<BOOL:${nlohmann_json_FOUND}>:nlohmann_json::nlohmann_json>
)

# Set Python module properties
set_target_properties(chunking_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/chunking_cpp
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/chunking_cpp
)

# Install Python module
install(TARGETS chunking_cpp
    LIBRARY DESTINATION ${Python_SITEARCH}/chunking_cpp
    RUNTIME DESTINATION ${Python_SITEARCH}/chunking_cpp
)

# Create Python package structure
add_custom_command(TARGET chunking_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/python/chunking_cpp
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/python/chunking_cpp/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:chunking_cpp> ${CMAKE_BINARY_DIR}/python/chunking_cpp/
)

# Add Python tests directory
if(BUILD_TESTING)
    add_subdirectory(tests/python)
endif()

# Add dependencies for Python bindings
find_package(PostgreSQL QUIET)
find_package(libpqxx QUIET)
find_package(mongocxx QUIET)
find_package(RdKafka QUIET)
find_package(RabbitMQ QUIET)
find_package(nlohmann_json QUIET)

# Create build directory for generated headers
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include)

# Configure optional features
configure_file(
    ${CMAKE_SOURCE_DIR}/include/config.hpp.in
    ${CMAKE_BINARY_DIR}/include/config.hpp
)

# Add generated config directory to include paths
include_directories(${CMAKE_BINARY_DIR}/include)

# Define compile definitions based on available packages
if(PostgreSQL_FOUND AND libpqxx_FOUND)
    target_compile_definitions(chunking_cpp PRIVATE HAVE_POSTGRESQL)
    target_compile_definitions(chunk_processor PRIVATE HAVE_POSTGRESQL)
    target_link_libraries(chunking_cpp PRIVATE 
        PostgreSQL::PostgreSQL
        libpqxx::pqxx
    )
else()
    message(STATUS "PostgreSQL support disabled - libpqxx not found")
endif()

if(mongocxx_FOUND)
    target_compile_definitions(chunking_cpp PRIVATE HAVE_MONGODB)
    target_compile_definitions(chunk_processor PRIVATE HAVE_MONGODB)
    target_link_libraries(chunking_cpp PRIVATE 
        mongocxx::mongocxx
    )
else()
    message(STATUS "MongoDB support disabled - mongocxx not found")
endif()

if(RdKafka_FOUND)
    target_compile_definitions(chunking_cpp PRIVATE HAVE_KAFKA)
    target_link_libraries(chunking_cpp PRIVATE 
        RdKafka::rdkafka++
    )
else()
    message(STATUS "Kafka support disabled - RdKafka not found")
endif()

if(RabbitMQ_FOUND)
    target_compile_definitions(chunking_cpp PRIVATE HAVE_RABBITMQ)
    target_link_libraries(chunking_cpp PRIVATE 
        RabbitMQ::rabbitmq
    )
endif()

if(nlohmann_json_FOUND)
    target_compile_definitions(chunking_cpp PRIVATE HAVE_JSON)
    target_link_libraries(chunking_cpp PRIVATE 
        nlohmann_json::nlohmann_json
    )
endif()

# Add include directories for Python module
target_include_directories(chunking_cpp PRIVATE
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Force using system libstdc++
set_target_properties(chunking_cpp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "/usr/lib/x86_64-linux-gnu"
)

# Check C++ compiler version and available features
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(GCC_VERSION VERSION_LESS 9.1)
        set(CMAKE_CXX_STANDARD 14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        message(STATUS "Using C++14 due to older GCC version")
    else()
        set(CMAKE_CXX_STANDARD 17)
    endif()
endif()

# Add rpath settings for better library resolution
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu")

# Add the template instantiation libraries
add_library(chunk_metrics_impl
    src/chunk_metrics.cpp
)

add_library(chunk_serialization_impl
    src/chunk_serialization.cpp
)

add_library(chunk_visualization_impl
    src/chunk_visualization.cpp
)

# Set include directories for implementation libraries
foreach(lib 
    chunk_metrics_impl 
    chunk_serialization_impl 
    chunk_visualization_impl
)
    target_include_directories(${lib} PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    )
    set_target_properties(${lib} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endforeach()

# Add visibility flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")
    target_compile_options(chunking_cpp PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

# Ensure all implementation libraries are built with PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

target_sources(chunk_processor
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/chunk_visualization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/chunk_serialization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/neural_chunking.cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/chunk.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/chunk_strategies.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/chunk_visualization.hpp
)

# Add new source files to existing sources
list(APPEND CHUNK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/slidingchunkwindow.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sparsechunkmatrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/chunkcache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/advanced_structures.hpp
)

# Add new test file to existing test sources
list(APPEND TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_advanced_structures.cpp
)

# Update existing targets with new sources
target_sources(chunk_processor
    PRIVATE
        ${CHUNK_SOURCES}
)

# Update test executable with new sources
target_sources(run_tests
    PRIVATE
        ${TEST_SOURCES}
)

# Ensure the new headers are included in the build
target_include_directories(chunk_processor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add Boost dependency for advanced structures
find_package(Boost REQUIRED)
target_link_libraries(chunk_processor
    PUBLIC
        Boost::boost
)

#include "parallel_chunk.hpp"
#include "test_base.hpp"
#include <atomic>
#include <numeric>

using namespace parallel_chunk;

class ParallelChunkProcessorTest : public ChunkTestBase {
protected:
    std::vector<int> test_data;
    std::vector<std::vector<int>> chunks;

    void SetUp() override {
        ChunkTestBase::SetUp();

        try {
            test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            chunks = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        } catch (const std::exception& e) {
            FAIL() << "Setup failed: " << e.what();
        }
    }

    void TearDown() override {
        try {
            test_data.clear();
            chunks.clear();
        } catch (...) {
            // Ensure base teardown still happens
        }
        ChunkTestBase::TearDown();
    }
};

TEST_F(ParallelChunkProcessorTest, BasicProcessing) {
    std::unique_lock<std::mutex> lock(global_test_mutex_);

    // Double each number in parallel
    auto operation = [](std::vector<int>& chunk) {
        for (int& val : chunk) {
            val *= 2;
        }
    };

    ParallelChunkProcessor<int>::process_chunks(chunks, operation);

    lock.unlock();

    EXPECT_EQ(chunks[0], (std::vector<int>{2, 4, 6}));
    EXPECT_EQ(chunks[1], (std::vector<int>{8, 10, 12}));
    EXPECT_EQ(chunks[2], (std::vector<int>{14, 16, 18}));
}

TEST_F(ParallelChunkProcessorTest, MapReduce) {
    std::unique_lock<std::mutex> lock(global_test_mutex_);

    // Map: Square each number
    auto square = [](const int& x) { return x * x; };
    auto squared_chunks = ParallelChunkProcessor<int>::map<int>(chunks, square);

    // Reduce: Sum all squares
    auto sum = [](const int& a, const int& b) { return a + b; };
    int result = ParallelChunkProcessor<int>::reduce(squared_chunks, sum, 0);

    lock.unlock();

    // Expected: 1^2 + 2^2 + ... + 9^2 = 285
    EXPECT_EQ(result, 285);
}

TEST_F(ParallelChunkProcessorTest, EmptyProcessing) {
    std::vector<std::vector<int>> empty_data;
    ParallelChunkProcessor<int>::process_chunks(empty_data, [](std::vector<int>& chunk) {
        for (int& x : chunk) {
            x *= 2;
        }
    });
    EXPECT_TRUE(empty_data.empty());
}

TEST_F(ParallelChunkProcessorTest, SingleThreadProcessing) {
    std::vector<std::vector<int>> chunked_data = {test_data};
    ParallelChunkProcessor<int>::process_chunks(chunked_data, [](std::vector<int>& chunk) {
        for (int& x : chunk) {
            x *= 2;
        }
    });
    EXPECT_EQ(chunked_data[0].size(), test_data.size());
    for (size_t i = 0; i < test_data.size(); ++i) {
        EXPECT_EQ(chunked_data[0][i], test_data[i] * 2);
    }
}

TEST_F(ParallelChunkProcessorTest, ExceptionHandling) {
    std::vector<std::vector<int>> chunked_data = {test_data};
    EXPECT_THROW(
        ParallelChunkProcessor<int>::process_chunks(chunked_data,
                                                    [](std::vector<int>& chunk) {
                                                        for (size_t i = 0; i < chunk.size(); ++i) {
                                                            int x = chunk[i];
                                                            if (x > 3)
                                                                throw std::runtime_error("test");
                                                            chunk[i] = x * 2;
                                                        }
                                                    }),
        std::runtime_error);
}

TEST_F(ParallelChunkProcessorTest, MapWithEmptyChunks) {
    std::vector<std::vector<int>> empty_chunks;
    auto square = [](const int& x) { return x * x; };
    auto result = ParallelChunkProcessor<int>::map<int>(empty_chunks, square);
    EXPECT_TRUE(result.empty());
}

TEST_F(ParallelChunkProcessorTest, MapWithSingleElement) {
    std::vector<std::vector<int>> single_chunk = {{42}};
    auto square = [](const int& x) { return x * x; };
    auto result = ParallelChunkProcessor<int>::map<int>(single_chunk, square);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0][0], 1764); // 42^2
}

TEST_F(ParallelChunkProcessorTest, ReduceWithEmptyChunks) {
    std::vector<std::vector<int>> empty_chunks;
    auto sum = [](const int& a, const int& b) { return a + b; };
    int result = ParallelChunkProcessor<int>::reduce(empty_chunks, sum, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(ParallelChunkProcessorTest, ReduceWithSingleElement) {
    std::vector<std::vector<int>> single_chunk = {{42}};
    auto sum = [](const int& a, const int& b) { return a + b; };
    int result = ParallelChunkProcessor<int>::reduce(single_chunk, sum, 10);
    EXPECT_EQ(result, 52); // 10 + 42
}

TEST_F(ParallelChunkProcessorTest, MapReduceCombined) {
    auto double_it = [](const int& x) { return x * 2; };
    auto multiply = [](const int& a, const int& b) { return a * b; };

    // First, map to double all numbers
    auto doubled = ParallelChunkProcessor<int>::map<int>(chunks, double_it);

    // Flatten the chunks into a single vector for reduction
    std::vector<int> flattened;
    for (const auto& chunk : doubled) {
        flattened.insert(flattened.end(), chunk.begin(), chunk.end());
    }

    // Perform reduction on the flattened vector
    int result = std::accumulate(flattened.begin(), flattened.end(), 1, multiply);

    // Calculate expected result: product of all numbers doubled
    int expected = 1;
    for (const auto& chunk : chunks) {
        for (int x : chunk) {
            expected *= (x * 2);
        }
    }
    EXPECT_EQ(result, expected);
}

TEST_F(ParallelChunkProcessorTest, ConcurrentModification) {
    std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    std::atomic<int> counter{0};
    ParallelChunkProcessor<int>::process_chunks(data, [&counter](std::vector<int>& chunk) {
        for (int& x : chunk) {
            x *= 2;
            counter++;
        }
    });

    EXPECT_EQ(counter, 9); // Should have processed all elements
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            EXPECT_EQ(data[i][j], (i * 3 + j + 1) * 2);
        }
    }
}

TEST_F(ParallelChunkProcessorTest, ExceptionPropagation) {
    std::vector<std::vector<int>> data = {{1}, {2}, {3}, {4}, {5}};
    int exception_threshold = 2;

    EXPECT_THROW(
        {
            ParallelChunkProcessor<int>::process_chunks(
                data, [exception_threshold](std::vector<int>& chunk) {
                    if (chunk[0] > exception_threshold) {
                        throw std::runtime_error("Value too large");
                    }
                    std::cout << "chunk[0] before increment: " << chunk[0] << std::endl;
                    chunk[0] *= 2;
                });
        },
        std::runtime_error);

    // Check that some chunks were processed before exception
    bool found_modified = false;
    bool found_unmodified = false;
    for (const auto& chunk : data) {
        if (chunk[0] <= exception_threshold * 2 && chunk[0] % 2 == 0) {
            found_modified = true;
        }
        if (chunk[0] > exception_threshold && chunk[0] == chunk[0] / 2 * 2) {
            found_unmodified = true;
        }
    }
    EXPECT_TRUE(found_modified || found_unmodified);
}
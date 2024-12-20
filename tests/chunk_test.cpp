#include "chunk.hpp"
#include "gtest/gtest.h"
#include <cstdlib> // for std::abs
#include <numeric> // for std::accumulate

class ChunkTest : public ::testing::Test {
protected:
    using value_type = int; // Define the type we're testing with

    void SetUp() override {
        test_data = std::vector<value_type>{1, 2, 3, 4, 5};
    }

    std::vector<value_type> test_data;
    Chunk<value_type> basic_chunker{2};
};

TEST_F(ChunkTest, Constructor) {
    EXPECT_THROW(Chunk<value_type>(0), std::invalid_argument);
    EXPECT_NO_THROW(Chunk<value_type>(1));
}

TEST_F(ChunkTest, AddSingleElement) {
    basic_chunker.add(1);
    EXPECT_EQ(basic_chunker.size(), 1);
    auto chunks = basic_chunker.get_chunks();
    EXPECT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0].size(), 1);
    EXPECT_EQ(chunks[0][0], 1);
}

TEST_F(ChunkTest, AddVector) {
    basic_chunker.add(test_data);
    EXPECT_EQ(basic_chunker.size(), 5);
    auto chunks = basic_chunker.get_chunks();
    EXPECT_EQ(chunks.size(), 3); // [1,2], [3,4], [5]
}

TEST_F(ChunkTest, GetChunk) {
    basic_chunker.add(test_data);
    auto chunk = basic_chunker.get_chunk(0);
    EXPECT_EQ(chunk.size(), 2);
    EXPECT_EQ(chunk[0], 1);
    EXPECT_EQ(chunk[1], 2);
    EXPECT_THROW(basic_chunker.get_chunk(3), std::out_of_range);
}

TEST_F(ChunkTest, OverlappingChunks) {
    basic_chunker.add(test_data);
    EXPECT_THROW(basic_chunker.get_overlapping_chunks(2), std::invalid_argument);

    auto chunks = basic_chunker.get_overlapping_chunks(1);
    EXPECT_EQ(chunks.size(), 4);
}

TEST_F(ChunkTest, PredicateChunking) {
    basic_chunker.add(test_data);
    auto chunks = basic_chunker.chunk_by_predicate([](value_type x) { return x % 2 == 0; });
    EXPECT_EQ(chunks.size(), 3); // [1], [2,3], [4,5]
}

TEST_F(ChunkTest, SumChunking) {
    basic_chunker.add(test_data);
    auto chunks = basic_chunker.chunk_by_sum(3);
    EXPECT_EQ(chunks.size(), 4);

    ASSERT_GE(chunks.size(), 1);
    EXPECT_EQ(chunks[0], (std::vector<value_type>{1, 2}));
    EXPECT_EQ(chunks[1], (std::vector<value_type>{3}));
    EXPECT_EQ(chunks[2], (std::vector<value_type>{4}));
    EXPECT_EQ(chunks[3], (std::vector<value_type>{5}));
}

TEST_F(ChunkTest, EqualDivision) {
    basic_chunker.add(test_data);
    auto chunks = basic_chunker.chunk_into_n(2);
    EXPECT_EQ(chunks.size(), 2);
    EXPECT_EQ(chunks[0].size(), 3); // [1,2,3]
    EXPECT_EQ(chunks[1].size(), 2); // [4,5]
}

TEST_F(ChunkTest, SlidingWindow) {
    basic_chunker.add(test_data);
    auto chunks = basic_chunker.sliding_window(3, 1);
    EXPECT_EQ(chunks.size(), 3);
    EXPECT_EQ(chunks[0], (std::vector<value_type>{1, 2, 3}));
    EXPECT_EQ(chunks[1], (std::vector<value_type>{2, 3, 4}));
    EXPECT_EQ(chunks[2], (std::vector<value_type>{3, 4, 5}));
}

TEST_F(ChunkTest, ChunkByStatistic) {
    basic_chunker.add({1, 2, 5, 6, 1, 2, 7, 8});
    auto mean = [](const std::vector<value_type>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto chunks = basic_chunker.chunk_by_statistic(3.0, mean);
    EXPECT_GT(chunks.size(), 1);
}

TEST_F(ChunkTest, ChunkBySimilarity) {
    basic_chunker.add({1, 2, 10, 11, 1, 2, 20, 21});
    auto chunks = basic_chunker.chunk_by_similarity(3);
    EXPECT_GT(chunks.size(), 1);
    for (const auto& chunk : chunks) {
        int max_diff = 0;
        for (size_t i = 1; i < chunk.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(chunk[i] - chunk[i - 1]));
        }
        EXPECT_LE(max_diff, 3);
    }
}

TEST_F(ChunkTest, ChunkByMonotonicity) {
    basic_chunker.add({1, 2, 3, 2, 1, 4, 5, 3});
    auto chunks = basic_chunker.chunk_by_monotonicity();
    EXPECT_GT(chunks.size(), 1);
    for (const auto& chunk : chunks) {
        bool is_monotonic = true;
        bool increasing = chunk[1] > chunk[0];
        for (size_t i = 1; i < chunk.size(); ++i) {
            if ((chunk[i] > chunk[i - 1]) != increasing) {
                is_monotonic = false;
                break;
            }
        }
        EXPECT_TRUE(is_monotonic);
    }
}

TEST_F(ChunkTest, PaddedChunks) {
    basic_chunker.add({1, 2, 3});
    auto chunks = basic_chunker.get_padded_chunks(0);
    EXPECT_EQ(chunks.size(), 2);
    EXPECT_EQ(chunks[0], (std::vector<value_type>{1, 2}));
    EXPECT_EQ(chunks[1], (std::vector<value_type>{3, 0}));
}

TEST_F(ChunkTest, BasicTest) {
    // Basic test to verify test setup
    Chunk<int> chunk(2);
    ASSERT_EQ(chunk.size(), 0);              // Initially empty
    ASSERT_EQ(chunk.get_chunks().size(), 0); // No chunks yet
}

TEST_F(ChunkTest, InitializationTest) {
    // Test chunk initialization
    Chunk<int> chunk(2);  // Create chunk with size 2
    chunk.add(test_data); // Add data from fixture
    ASSERT_EQ(chunk.size(), 5);
    auto chunks = chunk.get_chunks();
    ASSERT_EQ(chunks.size(), 3); // Should have 3 chunks: [1,2], [3,4], [5]
    ASSERT_EQ(chunks[0], (std::vector<int>{1, 2}));
    ASSERT_EQ(chunks[1], (std::vector<int>{3, 4}));
    ASSERT_EQ(chunks[2], (std::vector<int>{5}));
}

TEST_F(ChunkTest, EmptyChunkOperations) {
    EXPECT_EQ(basic_chunker.size(), 0);
    EXPECT_TRUE(basic_chunker.get_chunks().empty());
    EXPECT_THROW(basic_chunker.get_chunk(0), std::out_of_range);
}

TEST_F(ChunkTest, EdgeCases) {
    // Test with empty vector
    std::vector<value_type> empty_data;
    basic_chunker.add(empty_data);
    EXPECT_EQ(basic_chunker.size(), 0);

    // Test with single element
    basic_chunker.add(1);
    EXPECT_EQ(basic_chunker.size(), 1);

    // Test chunk size equal to data size
    Chunk<value_type> exact_chunker(5);
    exact_chunker.add(test_data);
    EXPECT_EQ(exact_chunker.get_chunks().size(), 1);
}

TEST_F(ChunkTest, ChunkBoundaries) {
    // Test chunk size larger than data
    Chunk<value_type> large_chunker(10);
    large_chunker.add(test_data);
    auto chunks = large_chunker.get_chunks();
    EXPECT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0], test_data);
}

TEST_F(ChunkTest, InvalidOperations) {
    EXPECT_THROW(basic_chunker.get_overlapping_chunks(3), std::invalid_argument);
    EXPECT_THROW(basic_chunker.chunk_into_n(0), std::invalid_argument);
    EXPECT_THROW(basic_chunker.sliding_window(0), std::invalid_argument);
}
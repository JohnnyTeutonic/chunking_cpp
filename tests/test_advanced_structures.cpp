#include "chunkcache.hpp"
#include "slidingchunkwindow.hpp"
#include "sparsechunkmatrix.hpp"
#include "test_base.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

class AdvancedStructuresTest : public ChunkTestBase {
protected:
    static constexpr double EPSILON = 1e-10;
};

// SlidingChunkWindow Tests
TEST_F(AdvancedStructuresTest, SlidingWindowBasicOperations) {
    chunk_processing::SlidingChunkWindow<double> window(3);

    EXPECT_TRUE(window.empty());
    EXPECT_EQ(window.size(), 0);

    window.push(1.0);
    EXPECT_FALSE(window.empty());
    EXPECT_EQ(window.size(), 1);
    EXPECT_NEAR(window.mean(), 1.0, EPSILON);

    window.push(2.0);
    window.push(3.0);
    EXPECT_EQ(window.size(), 3);
    EXPECT_NEAR(window.mean(), 2.0, EPSILON);

    // Test sliding behavior
    window.push(4.0);
    EXPECT_EQ(window.size(), 3);
    EXPECT_NEAR(window.mean(), 3.0, EPSILON);
}

TEST_F(AdvancedStructuresTest, SlidingWindowStatistics) {
    chunk_processing::SlidingChunkWindow<double> window(4);

    // Add values: 1, 2, 3, 4
    for (int i = 1; i <= 4; ++i) {
        window.push(static_cast<double>(i));
    }

    EXPECT_NEAR(window.mean(), 2.5, EPSILON);
    EXPECT_NEAR(window.variance(), 1.25, EPSILON);
    EXPECT_NEAR(window.stddev(), std::sqrt(1.25), EPSILON);
}

TEST_F(AdvancedStructuresTest, SlidingWindowEdgeCases) {
    EXPECT_THROW(chunk_processing::SlidingChunkWindow<double>(0), std::invalid_argument);

    chunk_processing::SlidingChunkWindow<double> window(2);
    EXPECT_NEAR(window.mean(), 0.0, EPSILON);
    EXPECT_NEAR(window.variance(), 0.0, EPSILON);
    EXPECT_NEAR(window.stddev(), 0.0, EPSILON);
}

// SparseChunkMatrix Tests
TEST_F(AdvancedStructuresTest, SparseMatrixBasicOperations) {
    chunk_processing::SparseChunkMatrix<int> matrix(0);

    std::vector<int> chunk1 = {0, 1, 0, 2, 0};
    std::vector<int> chunk2 = {0, 0, 3, 0, 4};

    matrix.add_chunk(chunk1);
    matrix.add_chunk(chunk2);

    EXPECT_EQ(matrix.chunk_count(), 2);
    EXPECT_EQ(matrix.max_chunk_size(), 5);

    auto retrieved1 = matrix.get_chunk(0);
    EXPECT_EQ(retrieved1, chunk1);

    auto retrieved2 = matrix.get_chunk(1);
    EXPECT_EQ(retrieved2, chunk2);
}

TEST_F(AdvancedStructuresTest, SparseMatrixEdgeCases) {
    chunk_processing::SparseChunkMatrix<double> matrix;

    EXPECT_EQ(matrix.chunk_count(), 0);
    EXPECT_EQ(matrix.max_chunk_size(), 0);

    EXPECT_THROW(matrix.get_chunk(0), std::out_of_range);

    std::vector<double> empty_chunk;
    matrix.add_chunk(empty_chunk);
    EXPECT_EQ(matrix.chunk_count(), 1);
    EXPECT_EQ(matrix.max_chunk_size(), 0);
}

// ChunkCache Tests
TEST_F(AdvancedStructuresTest, ChunkCacheBasicOperations) {
    chunk_processing::ChunkCache<int> cache(2);

    std::vector<int> chunk1 = {1, 2, 3};
    std::vector<int> chunk2 = {4, 5, 6};
    std::vector<int> chunk3 = {7, 8, 9};

    cache.insert(1, chunk1);
    cache.insert(2, chunk2);

    std::vector<int> retrieved;
    EXPECT_TRUE(cache.get(1, retrieved));
    EXPECT_EQ(retrieved, chunk1);

    // Test LRU eviction
    cache.insert(3, chunk3);
    EXPECT_FALSE(cache.get(2, retrieved)); // Should be evicted
    EXPECT_TRUE(cache.get(1, retrieved));  // Still in cache due to recent access
    EXPECT_TRUE(cache.get(3, retrieved));
}

TEST_F(AdvancedStructuresTest, ChunkCacheHitRate) {
    chunk_processing::ChunkCache<int> cache(2);
    std::vector<int> chunk = {1, 2, 3};

    cache.insert(1, chunk);

    std::vector<int> retrieved;
    cache.get(1, retrieved); // Hit
    cache.get(2, retrieved); // Miss
    cache.get(1, retrieved); // Hit

    EXPECT_NEAR(cache.hit_rate(), 2.0 / 3.0, EPSILON);
}

TEST_F(AdvancedStructuresTest, ChunkCacheEdgeCases) {
    EXPECT_THROW(chunk_processing::ChunkCache<int>(0), std::invalid_argument);

    chunk_processing::ChunkCache<int> cache(1);
    EXPECT_NEAR(cache.hit_rate(), 0.0, EPSILON);

    std::vector<int> empty_chunk;
    cache.insert(1, empty_chunk);

    std::vector<int> retrieved;
    EXPECT_TRUE(cache.get(1, retrieved));
    EXPECT_TRUE(retrieved.empty());
}
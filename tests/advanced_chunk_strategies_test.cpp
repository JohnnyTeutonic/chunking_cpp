#include "chunk_strategies.hpp"
#include "chunk_windows.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

class AdvancedChunkStrategiesTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data = {1.0, 2.0, 5.0, 6.0, 1.0, 2.0, 7.0, 8.0};
        cyclic_data = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    }

    std::vector<double> test_data;
    std::vector<int> cyclic_data;
};

TEST_F(AdvancedChunkStrategiesTest, PatternBasedLocalMaxima) {
    // Reset static variables for each test
    auto local_maxima_detector = [](double x) {
        thread_local double prev = x;
        thread_local double prev_prev = x;
        bool is_local_max = (prev > prev_prev && prev > x);
        prev_prev = prev;
        prev = x;
        return is_local_max;
    };

    chunk_strategies::PatternBasedStrategy<double> strategy(local_maxima_detector);
    auto chunks = strategy.apply(test_data);

    EXPECT_GT(chunks.size(), 1);
    for (const auto& chunk : chunks) {
        EXPECT_FALSE(chunk.empty());
    }
}

TEST_F(AdvancedChunkStrategiesTest, PatternBasedCyclicDetection) {
    // Test size-based constructor
    chunk_strategies::PatternBasedStrategy<int> strategy(3); // Pattern size of 3
    auto chunks = strategy.apply(cyclic_data);

    EXPECT_EQ(chunks.size(), 3); // Should detect 3 complete cycles
    for (const auto& chunk : chunks) {
        EXPECT_EQ(chunk.size(), 3);
    }
}

TEST_F(AdvancedChunkStrategiesTest, SingleElementInput) {
    std::vector<double> single_element{1.0};

    // Test both constructors with single element
    chunk_strategies::PatternBasedStrategy<double> size_strategy(2);
    auto size_chunks = size_strategy.apply(single_element);
    EXPECT_EQ(size_chunks.size(), 1);

    chunk_strategies::PatternBasedStrategy<double> pred_strategy([](double) { return true; });
    auto pred_chunks = pred_strategy.apply(single_element);
    EXPECT_EQ(pred_chunks.size(), 1);
}

TEST_F(AdvancedChunkStrategiesTest, EmptyInput) {
    std::vector<double> empty_data;

    // Test both constructors with empty input
    chunk_strategies::PatternBasedStrategy<double> size_strategy(2);
    auto size_chunks = size_strategy.apply(empty_data);
    EXPECT_TRUE(size_chunks.empty());

    chunk_strategies::PatternBasedStrategy<double> pred_strategy([](double) { return true; });
    auto pred_chunks = pred_strategy.apply(empty_data);
    EXPECT_TRUE(pred_chunks.empty());
}

// Add tests for window processing
TEST_F(AdvancedChunkStrategiesTest, WindowProcessing) {
    chunk_windows::SlidingWindowProcessor<double> processor(3, 1);
    auto result = processor.process(test_data, [](const std::vector<double>& window) {
        return std::accumulate(window.begin(), window.end(), 0.0) / window.size();
    });

    EXPECT_FALSE(result.empty());
    EXPECT_LE(result.size(), test_data.size());
}

// Test window operations
TEST_F(AdvancedChunkStrategiesTest, WindowOperations) {
    std::vector<double> window{1.0, 2.0, 3.0, 4.0, 5.0};

    EXPECT_EQ(chunk_windows::WindowOperations<double>::moving_average(window), 3.0);
    EXPECT_EQ(chunk_windows::WindowOperations<double>::moving_median(window), 3.0);
    EXPECT_EQ(chunk_windows::WindowOperations<double>::moving_max(window), 5.0);
    EXPECT_EQ(chunk_windows::WindowOperations<double>::moving_min(window), 1.0);
}

TEST_F(AdvancedChunkStrategiesTest, WindowOperationsEmptyInput) {
    std::vector<double> empty_window;
    EXPECT_EQ(chunk_windows::WindowOperations<double>::moving_average(empty_window), 0.0);
    EXPECT_THROW(chunk_windows::WindowOperations<double>::moving_median(empty_window),
                 std::invalid_argument);
    EXPECT_THROW(chunk_windows::WindowOperations<double>::moving_max(empty_window),
                 std::invalid_argument);
    EXPECT_THROW(chunk_windows::WindowOperations<double>::moving_min(empty_window),
                 std::invalid_argument);
}

TEST_F(AdvancedChunkStrategiesTest, MultiDimensionalArrays) {
    // 2D array test
    std::vector<std::vector<double>> data_2d = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    chunk_processing::Chunk<std::vector<double>> chunker_2d(2);
    chunker_2d.add(data_2d);
    auto chunks_2d = chunker_2d.get_chunks();
    EXPECT_EQ(chunks_2d.size(), 2);
    EXPECT_EQ(chunks_2d[0].size(), 2);    // First chunk has 2 rows
    EXPECT_EQ(chunks_2d[0][0].size(), 3); // Each row has 3 columns

    // 3D array test
    std::vector<std::vector<std::vector<double>>> data_3d = {
        {{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}, {{9.0, 10.0}, {11.0, 12.0}}};

    chunk_processing::Chunk<std::vector<std::vector<double>>> chunker_3d(2);
    chunker_3d.add(data_3d);
    auto chunks_3d = chunker_3d.get_chunks();
    EXPECT_EQ(chunks_3d.size(), 2);
    EXPECT_EQ(chunks_3d[0].size(), 2);       // First chunk has 2 3D arrays
    EXPECT_EQ(chunks_3d[0][0].size(), 2);    // Each 3D array has 2 rows
    EXPECT_EQ(chunks_3d[0][0][0].size(), 2); // Each row has 2 columns
}

TEST_F(AdvancedChunkStrategiesTest, MultiDimensionalThresholds) {
    // Test threshold-based chunking with 2D arrays
    std::vector<std::vector<double>> data_2d = {
        {1.0, 1.1}, {1.2, 1.3}, // Similar values
        {5.0, 5.1}, {5.2, 5.3}, // Different group
        {1.0, 1.1}, {1.2, 1.3}  // Similar to first group
    };

    chunk_strategies::PatternBasedStrategy<std::vector<double>> strategy(
        [](const std::vector<double>& row) {
            return std::accumulate(row.begin(), row.end(), 0.0) > 8.0;
        });

    auto chunks = strategy.apply(data_2d);
    EXPECT_GT(chunks.size(), 1); // Should split on high-sum rows
}

TEST_F(AdvancedChunkStrategiesTest, DimensionalityValidation) {
    std::vector<std::vector<double>> inconsistent_2d = {{1.0, 2.0, 3.0},
                                                        {4.0, 5.0}, // Different size
                                                        {7.0, 8.0, 9.0}};

    chunk_processing::Chunk<std::vector<double>> chunker(2);
    EXPECT_THROW(chunker.add(inconsistent_2d), std::invalid_argument);
}
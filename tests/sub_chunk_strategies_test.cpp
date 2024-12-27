/**
 * @file sub_chunk_strategies_test.cpp
 * @brief Test suite for sub-chunking strategies
 *
 * This file contains comprehensive tests for:
 * - Recursive sub-chunking
 * - Hierarchical sub-chunking
 * - Conditional sub-chunking
 * - Edge cases and error conditions
 */
#include "chunk_strategies.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace chunk_processing;

/**
 * @brief Test fixture for sub-chunking strategy tests
 *
 * Provides common test data and setup for all sub-chunking tests
 */
class SubChunkStrategiesTest : public ::testing::Test {
protected:
    std::vector<double> test_data = {1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 2.0, 2.1, 2.2, 6.0, 6.1, 6.2,
                                     3.0, 3.1, 3.2, 7.0, 7.1, 7.2, 4.0, 4.1, 4.2, 8.0, 8.1, 8.2};
};

TEST_F(SubChunkStrategiesTest, RecursiveStrategyTest) {
    // Create a variance strategy with threshold 3.0
    auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(3.0);
    if (!variance_strategy) {
        FAIL() << "Failed to create variance strategy";
    }

    try {
        // Create recursive strategy with max depth 3 and min size 2
        chunk_processing::RecursiveSubChunkStrategy<double> recursive_strategy(
            variance_strategy, 3, 2);

        // Apply the strategy
        auto result = recursive_strategy.apply(test_data);

        // Verify the results
        ASSERT_GT(result.size(), 0) << "Result should not be empty";
        
        // Verify each chunk meets minimum size requirement
        for (const auto& chunk : result) {
            ASSERT_GE(chunk.size(), 2) << "Chunk size should be at least 2";
        }
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST_F(SubChunkStrategiesTest, HierarchicalStrategyTest) {
   std::cout << "before creating shared pointers" << std::endl;
    // Create multiple strategies
    std::vector<std::shared_ptr<chunk_processing::ChunkStrategy<double>>> strategies = {
        std::make_shared<chunk_processing::VarianceStrategy<double>>(5.0),
        std::make_shared<chunk_processing::EntropyStrategy<double>>(1.0)};
    std::cout << "after creating shared pointers" << std::endl;
    // Create hierarchical strategy with min size 2
    chunk_processing::HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies, 2);
    std::cout << "after creating hierarchical strategy" << std::endl;   
    // Apply the strategy
    auto result = hierarchical_strategy.apply(test_data);
    std::cout << "after applying the strategy" << std::endl;
    // Verify the results
    EXPECT_GT(result.size(), 1);
}

TEST_F(SubChunkStrategiesTest, ConditionalStrategyTest) {
    // Define condition function
    auto condition = [](const std::vector<double>& chunk) {
        return chunk.size() > 4; // Only subdivide chunks larger than 4 elements
    };

    // Create variance strategy
    auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(5.0);

    // Create conditional strategy with min size 2
    chunk_processing::ConditionalSubChunkStrategy<double> conditional_strategy(variance_strategy,
                                                                               condition, 2);

    // Apply the strategy
    auto result = conditional_strategy.apply(test_data);

    // Verify the results
    EXPECT_GT(result.size(), 1);
}

TEST_F(SubChunkStrategiesTest, EmptyDataTest) {
    std::vector<double> empty_data;
    auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(3.0);

    // Test each sub-chunk strategy with empty data
    chunk_processing::RecursiveSubChunkStrategy<double> recursive_strategy(variance_strategy, 2, 2);
    EXPECT_TRUE(recursive_strategy.apply(empty_data).empty());

    std::vector<std::shared_ptr<chunk_processing::ChunkStrategy<double>>> strategies = {
        variance_strategy};
    chunk_processing::HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies, 2);
    EXPECT_TRUE(hierarchical_strategy.apply(empty_data).empty());

    auto condition = [](const std::vector<double>& chunk) { return chunk.size() > 4; };
    chunk_processing::ConditionalSubChunkStrategy<double> conditional_strategy(variance_strategy,
                                                                               condition, 2);
    EXPECT_TRUE(conditional_strategy.apply(empty_data).empty());
}
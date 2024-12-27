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
    std::cout << "before creating variance strategy" << std::endl;
    auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(3.0);
    std::cout << "after creating variance strategy" << std::endl;
    if (!variance_strategy) {
        FAIL() << "Failed to create variance strategy";
    }
    std::cout << "after checking if variance strategy was created" << std::endl;
    try {
        // Create recursive strategy with max depth 3 and min size 2
        std::cout << "before creating recursive strategy" << std::endl;
        chunk_processing::RecursiveSubChunkStrategy<double> recursive_strategy(
            variance_strategy, 3, 2);
        std::cout << "after creating recursive strategy" << std::endl;
        // Apply the strategy
        std::cout << "before applying the strategy" << std::endl;
        auto result = recursive_strategy.apply(test_data);
        std::cout << "after applying the strategy" << std::endl;
        // Verify the results
        std::cout << "before verifying the results" << std::endl;
        ASSERT_GT(result.size(), 0) << "Result should not be empty";
        std::cout << "after verifying the results" << std::endl;
        // Verify each chunk meets minimum size requirement
        std::cout << "before verifying each chunk meets minimum size requirement" << std::endl;
        for (const auto& chunk : result) {
            ASSERT_GE(chunk.size(), 2) << "Chunk size should be at least 2";
        }
        std::cout << "after verifying each chunk meets minimum size requirement" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST_F(SubChunkStrategiesTest, HierarchicalStrategyTest) {
    try {
        // Create multiple strategies
        std::cout << "before creating shared pointers" << std::endl;
        std::vector<std::shared_ptr<chunk_processing::ChunkStrategy<double>>> strategies;
        std::cout << "after creating shared pointers" << std::endl;
        auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(5.0);
        std::cout << "after creating variance strategy" << std::endl;
        auto entropy_strategy = std::make_shared<chunk_processing::EntropyStrategy<double>>(1.0);
        std::cout << "after creating entropy strategy" << std::endl;
        
        if (!variance_strategy || !entropy_strategy) {
            FAIL() << "Failed to create strategies";
        }
        std::cout << "after checking if strategies were created" << std::endl;
        strategies.push_back(variance_strategy);
        strategies.push_back(entropy_strategy);
        std::cout << "after pushing back strategies" << std::endl;

        // Create hierarchical strategy with min size 2
        chunk_processing::HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies, 2);
        std::cout << "after creating hierarchical strategy" << std::endl;
        std::cout << "before applying the strategy" << std::endl;
        // Apply the strategy
        auto result = hierarchical_strategy.apply(test_data);
        std::cout << "after applying the strategy" << std::endl;
        // Verify the results
        ASSERT_GT(result.size(), 0) << "Result should not be empty";
        std::cout << "after verifying the results" << std::endl;
        // Verify each chunk meets minimum size requirement
        for (const auto& chunk : result) {
            ASSERT_GE(chunk.size(), 2) << "Chunk size should be at least 2";
        }
        std::cout << "after verifying each chunk meets minimum size requirement" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST_F(SubChunkStrategiesTest, ConditionalStrategyTest) {
    try {
        // Define condition function - store it to ensure it stays alive
        auto condition = [](const std::vector<double>& chunk) {
            return chunk.size() > 4; // Only subdivide chunks larger than 4 elements
        };

        // Create variance strategy
        auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(5.0);
        if (!variance_strategy) {
            FAIL() << "Failed to create variance strategy";
        }

        // Create conditional strategy with min size 2
        chunk_processing::ConditionalSubChunkStrategy<double> conditional_strategy(
            variance_strategy, condition, 2);

        // Apply the strategy
        auto result = conditional_strategy.apply(test_data);

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

TEST_F(SubChunkStrategiesTest, EmptyDataTest) {
    std::vector<double> empty_data;
    std::cout << "before creating variance strategy" << std::endl;  
    auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(3.0);
    std::cout << "after creating variance strategy" << std::endl;

    // Test each sub-chunk strategy with empty data
    std::cout << "before creating recursive strategy" << std::endl;
    chunk_processing::RecursiveSubChunkStrategy<double> recursive_strategy(variance_strategy, 2, 2);
    std::cout << "after creating recursive strategy" << std::endl;
    EXPECT_TRUE(recursive_strategy.apply(empty_data).empty());

    std::vector<std::shared_ptr<chunk_processing::ChunkStrategy<double>>> strategies = {
        variance_strategy};
    std::cout << "before creating hierarchical strategy" << std::endl;
    chunk_processing::HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies, 2);
    std::cout << "after creating hierarchical strategy" << std::endl;
    EXPECT_TRUE(hierarchical_strategy.apply(empty_data).empty());
    
    auto condition = [](const std::vector<double>& chunk) { return chunk.size() > 4; };
    chunk_processing::ConditionalSubChunkStrategy<double> conditional_strategy(variance_strategy,
                                                                               condition, 2);
    EXPECT_TRUE(conditional_strategy.apply(empty_data).empty());
}
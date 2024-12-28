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
#include "test_base.hpp"
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
class SubChunkStrategiesTest : public ChunkTestBase {
protected:
    std::vector<double> test_data;

    void SetUp() override {
        ChunkTestBase::SetUp();

        test_data = {1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 2.0, 2.1, 2.2,
                     6.0, 6.1, 6.2, 3.0, 3.1, 3.2, 7.0, 7.1, 7.2};
    }

    void TearDown() override {
        test_data.clear();
        ChunkTestBase::TearDown();
    }
};

TEST_F(SubChunkStrategiesTest, RecursiveStrategyTest) {
    std::unique_lock<std::mutex> lock(global_test_mutex_);

    auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(3.0);
    ASSERT_TRUE(is_valid_resource(variance_strategy));

    chunk_processing::RecursiveSubChunkStrategy<double> recursive_strategy(variance_strategy, 3, 2);
    auto result = recursive_strategy.apply(test_data);

    lock.unlock();

    ASSERT_GT(result.size(), 0);
    for (const auto& chunk : result) {
        ASSERT_GE(chunk.size(), 2);
    }
}

TEST_F(SubChunkStrategiesTest, HierarchicalStrategyTest) {
    try {
        // Create and validate strategies
        auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(5.0);
        auto entropy_strategy = std::make_shared<chunk_processing::EntropyStrategy<double>>(1.0);

        if (!variance_strategy || !entropy_strategy) {
            FAIL() << "Failed to create strategies";
        }

        // Create strategy vector
        std::vector<std::shared_ptr<chunk_processing::ChunkStrategy<double>>> strategies;
        strategies.reserve(2); // Pre-reserve space
        strategies.push_back(variance_strategy);
        strategies.push_back(entropy_strategy);

        // Create hierarchical strategy
        chunk_processing::HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies, 2);

        // Apply strategy and validate results
        auto result = hierarchical_strategy.apply(test_data);
        ASSERT_GT(result.size(), 0) << "Result should not be empty";

        // Validate chunk sizes
        for (const auto& chunk : result) {
            ASSERT_GE(chunk.size(), 2) << "Chunk size should be at least 2";
        }
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST_F(SubChunkStrategiesTest, ConditionalStrategyTest) {
    try {
        // Create condition function
        auto condition = [](const std::vector<double>& chunk) { return chunk.size() > 4; };

        // Create and validate base strategy
        auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(5.0);
        if (!variance_strategy) {
            FAIL() << "Failed to create variance strategy";
        }

        // Create conditional strategy
        chunk_processing::ConditionalSubChunkStrategy<double> conditional_strategy(
            variance_strategy, condition, 2);

        // Apply strategy and validate results
        auto result = conditional_strategy.apply(test_data);
        ASSERT_GT(result.size(), 0) << "Result should not be empty";

        // Validate chunk sizes
        for (const auto& chunk : result) {
            ASSERT_GE(chunk.size(), 2) << "Chunk size should be at least 2";
        }
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST_F(SubChunkStrategiesTest, EmptyDataTest) {
    try {
        std::vector<double> empty_data;

        // Create and validate base strategy
        auto variance_strategy = std::make_shared<chunk_processing::VarianceStrategy<double>>(3.0);
        if (!variance_strategy) {
            FAIL() << "Failed to create variance strategy";
        }

        // Test recursive strategy
        {
            chunk_processing::RecursiveSubChunkStrategy<double> recursive_strategy(
                variance_strategy, 2, 2);
            auto result = recursive_strategy.apply(empty_data);
            EXPECT_TRUE(result.empty())
                << "Recursive strategy should return empty result for empty data";
        }

        // Test hierarchical strategy
        {
            std::vector<std::shared_ptr<chunk_processing::ChunkStrategy<double>>> strategies{
                variance_strategy};
            chunk_processing::HierarchicalSubChunkStrategy<double> hierarchical_strategy(strategies,
                                                                                         2);
            auto result = hierarchical_strategy.apply(empty_data);
            EXPECT_TRUE(result.empty())
                << "Hierarchical strategy should return empty result for empty data";
        }

        // Test conditional strategy
        {
            auto condition = [](const std::vector<double>& chunk) { return chunk.size() > 4; };
            chunk_processing::ConditionalSubChunkStrategy<double> conditional_strategy(
                variance_strategy, condition, 2);
            auto result = conditional_strategy.apply(empty_data);
            EXPECT_TRUE(result.empty())
                << "Conditional strategy should return empty result for empty data";
        }
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}
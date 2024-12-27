/**
 * @file sub_chunk_strategies.hpp
 * @brief Advanced sub-chunking strategies for hierarchical data processing
 *
 * This file provides implementations of various sub-chunking strategies:
 * - Recursive sub-chunking for depth-based processing
 * - Hierarchical sub-chunking for level-based processing
 * - Conditional sub-chunking for property-based processing
 */
#pragma once

#include "chunk_strategies.hpp"
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace chunk_processing {

template <typename T>
class RecursiveSubChunkStrategy : public ChunkStrategy<T> {
private:
    std::shared_ptr<ChunkStrategy<T>> base_strategy_;
    size_t max_depth_;
    size_t min_size_;
    
    std::vector<std::vector<T>> recursive_apply(const std::vector<T>& data, size_t current_depth) {
        // Base cases to prevent segfaults
        if (data.empty() || data.size() <= min_size_ || current_depth >= max_depth_) {
            return {data};
        }
        
        // Ensure base strategy exists
        if (!base_strategy_) {
            throw std::runtime_error("Base strategy not initialized");
        }

        // Apply base strategy to get initial chunks
        auto initial_chunks = base_strategy_->apply(data);
        
        // If no further splitting needed
        if (initial_chunks.size() <= 1) {
            return initial_chunks;
        }

        // Recursively process each chunk
        std::vector<std::vector<T>> result;
        for (const auto& chunk : initial_chunks) {
            // Only recurse if chunk is large enough
            if (chunk.size() > min_size_) {
                auto sub_chunks = recursive_apply(chunk, current_depth + 1);
                result.insert(result.end(), sub_chunks.begin(), sub_chunks.end());
            } else {
                result.push_back(chunk);
            }
        }
        
        return result;
    }

public:
    RecursiveSubChunkStrategy(std::shared_ptr<ChunkStrategy<T>> strategy, 
                             size_t max_depth = 5,
                             size_t min_size = 2)
        : base_strategy_(strategy)
        , max_depth_(max_depth)
        , min_size_(min_size) {
        if (!strategy) {
            throw std::invalid_argument("Base strategy cannot be null");
        }
        if (max_depth == 0) {
            throw std::invalid_argument("Max depth must be positive");
        }
        if (min_size == 0) {
            throw std::invalid_argument("Min size must be positive");
        }
    }

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        return const_cast<RecursiveSubChunkStrategy*>(this)->recursive_apply(data, 0);
    }
};

template <typename T>
class HierarchicalSubChunkStrategy : public ChunkStrategy<T> {
private:
    std::vector<std::shared_ptr<ChunkStrategy<T>>> strategies_;
    size_t min_size_;

public:
    HierarchicalSubChunkStrategy(std::vector<std::shared_ptr<ChunkStrategy<T>>> strategies,
                                 size_t min_size)
        : strategies_(std::move(strategies)), min_size_(min_size) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        if (data.empty())
            return {};
        if (data.size() <= min_size_)
            return {data};

        std::vector<std::vector<T>> current_chunks = {data};

        for (const auto& strategy : strategies_) {
            std::vector<std::vector<T>> next_level;
            for (const auto& chunk : current_chunks) {
                if (chunk.size() > min_size_) {
                    auto sub_chunks = strategy->apply(chunk);
                    next_level.insert(next_level.end(), sub_chunks.begin(), sub_chunks.end());
                } else {
                    next_level.push_back(chunk);
                }
            }
            current_chunks = std::move(next_level);
        }

        return current_chunks;
    }
};

template <typename T>
class ConditionalSubChunkStrategy : public ChunkStrategy<T> {
private:
    std::shared_ptr<ChunkStrategy<T>> base_strategy_;
    std::function<bool(const std::vector<T>&)> condition_;
    size_t min_size_;

public:
    ConditionalSubChunkStrategy(std::shared_ptr<ChunkStrategy<T>> strategy,
                                std::function<bool(const std::vector<T>&)> condition,
                                size_t min_size)
        : base_strategy_(strategy), condition_(std::move(condition)), min_size_(min_size) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        if (data.empty())
            return {};
        if (data.size() <= min_size_)
            return {data};

        if (condition_(data)) {
            return base_strategy_->apply(data);
        }
        return {data};
    }
};

} // namespace chunk_processing
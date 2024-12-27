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
#include <iterator>
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

    // Add helper method to safely process chunks
    std::vector<std::vector<T>> process_chunk(const std::vector<T>& chunk, 
                                            const std::shared_ptr<ChunkStrategy<T>>& strategy) const {
        if (!strategy) {
            throw std::runtime_error("Invalid strategy encountered");
        }
        
        if (chunk.size() <= min_size_) {
            return {chunk};
        }

        try {
            auto sub_chunks = strategy->apply(chunk);
            if (sub_chunks.empty()) {
                return {chunk};
            }
            
            // Validate sub-chunks
            for (const auto& sub : sub_chunks) {
                if (sub.empty()) {
                    return {chunk};
                }
            }
            
            return sub_chunks;
        } catch (const std::exception& e) {
            // If strategy fails, return original chunk
            return {chunk};
        }
    }

public:
    HierarchicalSubChunkStrategy(std::vector<std::shared_ptr<ChunkStrategy<T>>> strategies,
                                size_t min_size)
        : min_size_(min_size) {
        // Validate inputs
        if (strategies.empty()) {
            throw std::invalid_argument("Strategies vector cannot be empty");
        }
        
        // Deep copy strategies to ensure ownership
        strategies_.reserve(strategies.size());
        for (const auto& strategy : strategies) {
            if (!strategy) {
                throw std::invalid_argument("Strategy cannot be null");
            }
            strategies_.push_back(strategy);
        }
        
        if (min_size == 0) {
            throw std::invalid_argument("Minimum size must be positive");
        }
    }

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        if (data.empty()) {
            return {};
        }
        if (data.size() <= min_size_) {
            return {data};
        }

        try {
            std::vector<std::vector<T>> current_chunks{data};
            
            // Process each strategy level
            for (const auto& strategy : strategies_) {
                if (!strategy) {
                    throw std::runtime_error("Invalid strategy encountered");
                }

                std::vector<std::vector<T>> next_level;
                next_level.reserve(current_chunks.size() * 2);  // Reserve space to prevent reallocation

                // Process each chunk at current level
                for (const auto& chunk : current_chunks) {
                    if (chunk.empty()) {
                        continue;
                    }

                    auto sub_chunks = process_chunk(chunk, strategy);
                    next_level.insert(next_level.end(), 
                                    std::make_move_iterator(sub_chunks.begin()),
                                    std::make_move_iterator(sub_chunks.end()));
                }

                if (next_level.empty()) {
                    return current_chunks;  // Return last valid chunking if next level failed
                }

                current_chunks = std::move(next_level);
            }

            return current_chunks;

        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error in hierarchical strategy: ") + e.what());
        }
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
        : base_strategy_(strategy)
        , condition_(condition)
        , min_size_(min_size) {
        // Validate inputs
        if (!strategy) {
            throw std::invalid_argument("Base strategy cannot be null");
        }
        if (!condition) {
            throw std::invalid_argument("Condition function cannot be null");
        }
        if (min_size == 0) {
            throw std::invalid_argument("Minimum size must be positive");
        }
    }

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        if (data.empty()) {
            return {};
        }
        if (data.size() <= min_size_) {
            return {data};
        }

        try {
            // Safely check condition and apply strategy
            if (condition_ && condition_(data)) {
                if (base_strategy_) {
                    return base_strategy_->apply(data);
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error in conditional strategy: ") + e.what());
        }
        
        return {data};
    }
};

} // namespace chunk_processing
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
#include <mutex>
#include <stdexcept>
#include <vector>
#include <atomic>

namespace chunk_processing {

namespace detail {
    // Helper functions for safe operations
    template<typename T>
    bool is_valid_chunk(const std::vector<T>& chunk) {
        return !chunk.empty();
    }

    template<typename T>
    bool is_valid_chunks(const std::vector<std::vector<T>>& chunks) {
        return !chunks.empty() && std::all_of(chunks.begin(), chunks.end(), 
                                            [](const auto& c) { return is_valid_chunk(c); });
    }

    template<typename T>
    std::vector<std::vector<T>> safe_copy(const std::vector<std::vector<T>>& chunks) {
        try {
            return chunks;
        } catch (...) {
            return {};
        }
    }
}

template <typename T>
class RecursiveSubChunkStrategy : public ChunkStrategy<T> {
private:
    const std::shared_ptr<ChunkStrategy<T>> base_strategy_;
    const size_t max_depth_;
    const size_t min_size_;
    mutable std::mutex mutex_;
    std::atomic<bool> is_processing_{false};

    std::vector<std::vector<T>> safe_recursive_apply(const std::vector<T>& data, 
                                                    const size_t current_depth) {
        // Guard against reentrant calls
        if (is_processing_.exchange(true)) {
            throw std::runtime_error("Recursive strategy already processing");
        }
        struct Guard {
            std::atomic<bool>& flag;
            Guard(std::atomic<bool>& f) : flag(f) {}
            ~Guard() { flag = false; }
        } guard(is_processing_);

        try {
            if (data.empty() || current_depth >= max_depth_ || data.size() <= min_size_) {
                return data.empty() ? std::vector<std::vector<T>>{} : std::vector<std::vector<T>>{data};
            }

            std::vector<std::vector<T>> result;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!base_strategy_) {
                    throw std::runtime_error("Invalid base strategy");
                }
                auto chunks = detail::safe_copy(base_strategy_->apply(data));
                if (!detail::is_valid_chunks(chunks)) {
                    return {data};
                }
                result.reserve(chunks.size() * 2);

                for (const auto& chunk : chunks) {
                    if (chunk.size() <= min_size_) {
                        result.push_back(chunk);
                        continue;
                    }

                    try {
                        auto sub_chunks = safe_recursive_apply(chunk, current_depth + 1);
                        if (detail::is_valid_chunks(sub_chunks)) {
                            result.insert(result.end(), 
                                        std::make_move_iterator(sub_chunks.begin()),
                                        std::make_move_iterator(sub_chunks.end()));
                        } else {
                            result.push_back(chunk);
                        }
                    } catch (...) {
                        result.push_back(chunk);
                    }
                }
            }
            return result.empty() ? std::vector<std::vector<T>>{data} : result;
        } catch (...) {
            return {data};
        }
    }

public:
    RecursiveSubChunkStrategy(std::shared_ptr<ChunkStrategy<T>> strategy, 
                             size_t max_depth = 5,
                             size_t min_size = 2)
        : base_strategy_(strategy)
        , max_depth_(max_depth)
        , min_size_(min_size) {
        if (!strategy) throw std::invalid_argument("Invalid strategy");
        if (max_depth == 0) throw std::invalid_argument("Invalid max depth");
        if (min_size == 0) throw std::invalid_argument("Invalid min size");
    }

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        try {
            if (data.empty()) return {};
            return const_cast<RecursiveSubChunkStrategy*>(this)->safe_recursive_apply(data, 0);
        } catch (...) {
            return {data};
        }
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
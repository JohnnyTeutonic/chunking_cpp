/**
 * @file sparsechunkmatrix.hpp
 * @brief Sparse matrix implementation for efficient chunk storage
 * @author Assistant
 * @date 2024-03-19
 */

#pragma once

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace chunk_processing {

/**
 * @brief A sparse matrix implementation for efficient storage of chunk data
 *
 * @tparam T The type of elements stored in the matrix
 */
template <typename T>
class SparseChunkMatrix {
private:
    struct ChunkEntry {
        size_t index; ///< Index in the chunk
        T value;      ///< Value at the index
    };

    std::vector<std::vector<ChunkEntry>> chunks_; ///< Compressed chunk storage
    size_t total_size_{0};                        ///< Maximum chunk size
    T default_value_{};                           ///< Default value for sparse entries

public:
    /**
     * @brief Construct a new Sparse Chunk Matrix
     *
     * @param default_val Default value for sparse entries
     */
    explicit SparseChunkMatrix(const T& default_val = T{}) : default_value_(default_val) {}

    /**
     * @brief Add a new chunk to the matrix
     *
     * @param chunk Vector of values representing the chunk
     */
    void add_chunk(const std::vector<T>& chunk) {
        std::vector<ChunkEntry> compressed;
        for (size_t i = 0; i < chunk.size(); ++i) {
            if (chunk[i] != default_value_) {
                compressed.push_back({i, chunk[i]});
            }
        }
        chunks_.push_back(std::move(compressed));
        total_size_ = std::max(total_size_, chunk.size());
    }

    /**
     * @brief Get a chunk from the matrix
     *
     * @param chunk_idx Index of the chunk to retrieve
     * @return std::vector<T> The decompressed chunk
     * @throw std::out_of_range if chunk_idx is invalid
     */
    std::vector<T> get_chunk(size_t chunk_idx) const {
        if (chunk_idx >= chunks_.size()) {
            throw std::out_of_range("Chunk index out of range");
        }

        std::vector<T> result(total_size_, default_value_);
        for (const auto& entry : chunks_[chunk_idx]) {
            result[entry.index] = entry.value;
        }
        return result;
    }

    /**
     * @brief Get the number of chunks in the matrix
     */
    size_t chunk_count() const {
        return chunks_.size();
    }

    /**
     * @brief Get the maximum chunk size
     */
    size_t max_chunk_size() const {
        return total_size_;
    }

    /**
     * @brief Clear all chunks from the matrix
     */
    void clear() {
        chunks_.clear();
        total_size_ = 0;
    }
};

} // namespace chunk_processing
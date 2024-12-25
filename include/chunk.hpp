#pragma once

#include "chunk_common.hpp"
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace chunk_processing {

/**
 * @brief A template class for managing and processing data in chunks
 * @tparam T The type of elements stored in the chunk
 */
template <typename T>
class CHUNK_EXPORT Chunk {
private:
    size_t chunk_size_;
    std::vector<T> data_;
    std::vector<std::vector<T>> chunks_;

    // Helper function for validation
    void validate_size(size_t size, const std::string& param) const {
        if (size == 0) {
            throw std::invalid_argument(param + " must be greater than 0");
        }
    }

    // Ensure data is properly copied
    std::vector<std::vector<T>> make_chunks(size_t size) const {
        std::vector<std::vector<T>> result;
        result.reserve((data_.size() + size - 1) / size);

        for (size_t i = 0; i < data_.size(); i += size) {
            size_t chunk_end = std::min(i + size, data_.size());
            result.emplace_back(data_.begin() + i, data_.begin() + chunk_end);
        }
        return result;
    }

    void update_chunks() {
        chunks_.clear();
        for (size_t i = 0; i < data_.size(); i += chunk_size_) {
            std::vector<T> chunk;
            for (size_t j = 0; j < chunk_size_ && i + j < data_.size(); ++j) {
                chunk.push_back(data_[i + j]);
            }
            chunks_.push_back(chunk);
        }
    }

    // Add support for checking dimensionality
    template <typename U>
    static constexpr size_t get_depth() {
        if constexpr (is_vector<U>::value)
            return 1 + get_depth<typename U::value_type>();
        return 0;
    }

    // Helper to validate nested vectors have consistent dimensions
    template <typename U>
    void validate_dimensions(const std::vector<U>& data, size_t expected_size = 0) {
        for (const auto& inner : data) {
            if constexpr (is_vector<U>::value) {
                if (expected_size > 0 && inner.size() != expected_size)
                    throw std::invalid_argument("Inconsistent dimensions in nested array");
                validate_dimensions(inner, expected_size);
            }
        }
    }

public:
    explicit Chunk(size_t chunk_size = 1) : chunk_size_(chunk_size) {
        validate_size(chunk_size, "Chunk size");
    }

    void add(const T& element) {
        data_.push_back(element);
        update_chunks();
    }

    void add(const std::vector<T>& elements) {
        data_.insert(data_.end(), elements.begin(), elements.end());
        update_chunks();
    }

    std::vector<std::vector<T>> chunk_by_size(size_t size) {
        if (data_.empty()) {
            throw std::invalid_argument("Cannot chunk empty data");
        }
        if (size == 0) {
            throw std::invalid_argument("Chunk size cannot be zero");
        }
        return make_chunks(size);
    }

    std::vector<std::vector<T>> chunk_by_threshold(T threshold) {
        if (data_.empty()) {
            throw std::invalid_argument("Cannot chunk empty data");
        }
        if (data_.size() < chunk_size_) {
            throw std::invalid_argument("Input size must be at least chunk size");
        }
        if (threshold <= T{}) {
            throw std::invalid_argument("Threshold must be positive");
        }

        std::vector<std::vector<T>> result;
        std::vector<T> current_chunk;
        T running_sum = T{};

        for (const T& value : data_) {
            if (running_sum + value > threshold && !current_chunk.empty()) {
                result.push_back(current_chunk);
                current_chunk.clear();
                running_sum = T{};
            }
            current_chunk.push_back(value);
            running_sum += value;
        }

        if (!current_chunk.empty()) {
            result.push_back(current_chunk);
        }

        return result;
    }

    std::vector<std::vector<T>> get_chunks() const {
        return chunks_;
    }

    size_t size() const {
        return data_.size();
    }

    size_t chunk_count() const {
        return (data_.size() + chunk_size_ - 1) / chunk_size_;
    }

    size_t get_chunk_size() const {
        return chunk_size_;
    }
    const std::vector<T>& get_data() const {
        return data_;
    }

    void set_chunk_size(size_t new_size) {
        validate_size(new_size, "Chunk size");
        chunk_size_ = new_size;
        update_chunks();
    }

    // Add methods to handle multi-dimensional data
    template <typename U = T>
    std::enable_if_t<is_vector<U>::value> add(const U& nested_data) {
        validate_dimensions(nested_data);
        data_.push_back(nested_data);
        update_chunks();
    }

    // Get the dimensionality of the data
    static constexpr size_t dimensions() {
        return get_depth<T>();
    }
};

} // namespace chunk_processing

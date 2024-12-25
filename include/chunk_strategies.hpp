/**
 * @file chunk_strategies.hpp
 * @brief Defines various strategies for chunking data based on different criteria.
 */

#pragma once

#include "chunk.hpp"
#include "chunk_common.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace chunk_strategies {

/**
 * @brief Base class for chunking strategies
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class ChunkStrategy {
public:
    /**
     * @brief Applies the chunking strategy to the data
     * @param data The data to chunk
     * @return A vector of chunks
     */
    virtual std::vector<std::vector<T>> apply(const std::vector<T>& data) const = 0;
    virtual ~ChunkStrategy() = default;
};

/**
 * @brief Strategy for chunking based on quantile thresholds
 */
template <typename T>
class QuantileStrategy : public ChunkStrategy<T> {
    double quantile_;

public:
    explicit QuantileStrategy(double quantile) : quantile_(quantile) {
        if (quantile < 0.0 || quantile > 1.0) {
            throw std::invalid_argument("Quantile must be between 0 and 1");
        }
    }

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        if (data.empty())
            return {};

        // Calculate threshold value at the given quantile
        std::vector<T> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        T threshold = sorted_data[static_cast<size_t>(quantile_ * (data.size() - 1))];

        // Split into exactly two chunks: values <= threshold and values > threshold
        std::vector<T> lower_chunk;
        std::vector<T> upper_chunk;

        for (const T& value : data) {
            if (value <= threshold) {
                lower_chunk.push_back(value);
            } else {
                upper_chunk.push_back(value);
            }
        }

        std::vector<std::vector<T>> chunks;
        if (!lower_chunk.empty()) {
            chunks.push_back(lower_chunk);
        }
        if (!upper_chunk.empty()) {
            chunks.push_back(upper_chunk);
        }

        return chunks;
    }
};

/**
 * @brief Strategy for chunking based on variance thresholds
 */
template <typename T>
class VarianceStrategy : public ChunkStrategy<T> {
    double threshold_;

    double calculate_variance(const std::vector<T>& chunk) const {
        if (chunk.size() < 2)
            return 0.0;
        double mean = std::accumulate(chunk.begin(), chunk.end(), 0.0) / chunk.size();
        double sq_sum = 0.0;
        for (const T& val : chunk) {
            double diff = val - mean;
            sq_sum += diff * diff;
        }
        return sq_sum / chunk.size();
    }

public:
    explicit VarianceStrategy(double threshold) : threshold_(threshold) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        std::vector<std::vector<T>> chunks;
        std::vector<T> current_chunk;

        for (const auto& value : data) {
            current_chunk.push_back(value);
            if (calculate_variance(current_chunk) > threshold_) {
                if (current_chunk.size() > 1) {
                    current_chunk.pop_back();
                    chunks.push_back(current_chunk);
                    current_chunk = {value};
                }
            }
        }

        if (!current_chunk.empty()) {
            chunks.push_back(current_chunk);
        }

        return chunks;
    }

    // Add getter
    double get_threshold() const {
        return threshold_;
    }

    // Add setter
    void set_threshold(double threshold) {
        threshold_ = threshold;
    }
};

/**
 * @brief Strategy for chunking based on entropy thresholds
 */
template <typename T>
class EntropyStrategy : public ChunkStrategy<T> {
    double threshold_;

    double calculate_entropy(const std::vector<T>& chunk) const {
        if (chunk.empty())
            return 0.0;

        std::map<T, int> freq;
        for (const auto& val : chunk) {
            freq[val]++;
        }

        double entropy = 0.0;
        for (const auto& [_, count] : freq) {
            double p = static_cast<double>(count) / chunk.size();
            entropy -= p * std::log2(p);
        }

        return entropy;
    }

public:
    explicit EntropyStrategy(double threshold) : threshold_(threshold) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        std::vector<std::vector<T>> chunks;
        std::vector<T> current_chunk;

        for (const auto& value : data) {
            current_chunk.push_back(value);
            if (calculate_entropy(current_chunk) > threshold_) {
                if (current_chunk.size() > 1) {
                    current_chunk.pop_back();
                    chunks.push_back(current_chunk);
                    current_chunk = {value};
                }
            }
        }

        if (!current_chunk.empty()) {
            chunks.push_back(current_chunk);
        }

        return chunks;
    }
};

template <typename T>
class PatternBasedStrategy : public ChunkStrategy<T> {
private:
    template <typename U>
    static constexpr bool is_multidimensional_v = chunk_processing::is_multidimensional_v<U>;

    template <typename U>
    double compute_array_sum(const U& arr) const {
        if constexpr (chunk_processing::is_multidimensional_v<U>) {
            double sum = 0.0;
            for (const auto& inner : arr) {
                sum += compute_array_sum(inner);
            }
            return sum;
        } else if constexpr (chunk_processing::is_vector<U>::value) {
            return std::accumulate(arr.begin(), arr.end(), 0.0);
        } else {
            return static_cast<double>(arr);
        }
    }

public:
    // Constructor for size-based patterns
    explicit PatternBasedStrategy(size_t pattern_size)
        : pattern_size_(pattern_size), predicate_(nullptr) {}

    // Constructor for predicate-based patterns
    explicit PatternBasedStrategy(std::function<bool(T)> predicate)
        : pattern_size_(0), predicate_(predicate) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        if (data.empty()) {
            return {};
        }

        if constexpr (is_multidimensional_v<T>) {
            // Handle multi-dimensional arrays
            if (predicate_) {
                return chunk_by_predicate(data);
            } else {
                return chunk_by_size(data);
            }
        } else {
            // Existing single-dimension logic
            // ...
        }
    }

private:
    std::vector<std::vector<T>> chunk_by_predicate(const std::vector<T>& data) const {
        std::vector<std::vector<T>> result;
        std::vector<T> current_chunk;

        for (const auto& value : data) {
            if constexpr (is_multidimensional_v<T>) {
                if (predicate_(compute_array_sum(value))) {
                    if (!current_chunk.empty()) {
                        result.push_back(current_chunk);
                        current_chunk.clear();
                    }
                }
            } else {
                if (predicate_(value)) {
                    if (!current_chunk.empty()) {
                        result.push_back(current_chunk);
                        current_chunk.clear();
                    }
                }
            }
            current_chunk.push_back(value);
        }

        if (!current_chunk.empty()) {
            result.push_back(current_chunk);
        }

        return result;
    }

    std::vector<std::vector<T>> chunk_by_size(const std::vector<T>& data) const {
        if (pattern_size_ == 0) {
            return {data};
        }

        if (data.size() < pattern_size_) {
            return data.empty() ? std::vector<std::vector<T>>{} : std::vector<std::vector<T>>{data};
        }

        std::vector<std::vector<T>> result;
        for (size_t i = 0; i < data.size(); i += pattern_size_) {
            size_t end = std::min(i + pattern_size_, data.size());
            result.emplace_back(data.begin() + i, data.begin() + end);
        }
        return result;
    }

    // Add getters
    size_t get_pattern_size() const {
        return pattern_size_;
    }
    const std::function<bool(T)>& get_predicate() const {
        return predicate_;
    }

    // Add setters
    void set_pattern_size(size_t size) {
        pattern_size_ = size;
    }
    void set_predicate(std::function<bool(T)> pred) {
        predicate_ = std::move(pred);
    }

private:
    size_t pattern_size_;
    std::function<bool(T)> predicate_;
};

template <typename T>
class AdaptiveStrategy : public ChunkStrategy<T> {
public:
    explicit AdaptiveStrategy(T initial_threshold,
                              std::function<T(const std::vector<T>&)> metric_calculator)
        : threshold_(initial_threshold), metric_calculator_(metric_calculator) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        chunk_processing::Chunk<T> chunker(1);
        chunker.add(data);
        return chunker.chunk_by_threshold(threshold_);
    }

    // Add getters
    T get_threshold() const {
        return threshold_;
    }
    const std::function<T(const std::vector<T>&)>& get_metric_calculator() const {
        return metric_calculator_;
    }

    // Add setter
    void set_threshold(T threshold) {
        threshold_ = threshold;
    }
    void set_metric_calculator(std::function<T(const std::vector<T>&)> calculator) {
        metric_calculator_ = std::move(calculator);
    }

private:
    T threshold_;
    std::function<T(const std::vector<T>&)> metric_calculator_;
};

template <typename T>
class MultiCriteriaStrategy : public ChunkStrategy<T> {
public:
    MultiCriteriaStrategy(T similarity_threshold, size_t size_threshold)
        : similarity_threshold_(similarity_threshold), size_threshold_(size_threshold) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        // Step 1: Similarity-based chunking
        chunk_processing::Chunk<T> initial_chunker(1);
        initial_chunker.add(data);
        auto similarity_chunks = initial_chunker.chunk_by_threshold(similarity_threshold_);

        // Step 2: Size-based chunking
        std::vector<std::vector<T>> result;
        for (const auto& chunk : similarity_chunks) {
            chunk_processing::Chunk<T> size_chunker(size_threshold_);
            size_chunker.add(chunk);
            auto sized_chunks = size_chunker.get_chunks();
            result.insert(result.end(), sized_chunks.begin(), sized_chunks.end());
        }
        return result;
    }

    // Add getters
    T get_similarity_threshold() const {
        return similarity_threshold_;
    }
    size_t get_size_threshold() const {
        return size_threshold_;
    }

    // Add setters
    void set_similarity_threshold(T threshold) {
        similarity_threshold_ = threshold;
    }
    void set_size_threshold(size_t size) {
        size_threshold_ = size;
    }

private:
    T similarity_threshold_;
    size_t size_threshold_;
};

template <typename T>
class DynamicThresholdStrategy : public ChunkStrategy<T> {
public:
    DynamicThresholdStrategy(T initial_threshold, T min_threshold, double decay_rate)
        : initial_threshold_(initial_threshold), min_threshold_(min_threshold),
          decay_rate_(decay_rate) {}

    std::vector<std::vector<T>> apply(const std::vector<T>& data) const override {
        chunk_processing::Chunk<T> chunker(1);
        chunker.add(data);
        return chunker.chunk_by_threshold(initial_threshold_);
    }

    // Add getters
    T get_initial_threshold() const {
        return initial_threshold_;
    }
    T get_min_threshold() const {
        return min_threshold_;
    }
    double get_decay_rate() const {
        return decay_rate_;
    }

    // Add setters
    void set_initial_threshold(T threshold) {
        initial_threshold_ = threshold;
    }
    void set_min_threshold(T threshold) {
        min_threshold_ = threshold;
    }
    void set_decay_rate(double rate) {
        decay_rate_ = rate;
    }

private:
    T initial_threshold_;
    T min_threshold_;
    double decay_rate_;
};

template <>
class EntropyStrategy<std::string> : public ChunkStrategy<std::string> {
    double threshold_;

    double calculate_entropy(const std::vector<std::string>& values) const {
        if (values.empty())
            return 0.0;

        std::map<std::string, int> freq;
        for (const auto& val : values) {
            freq[val]++;
        }

        double entropy = 0.0;
        for (const auto& [_, count] : freq) {
            double p = static_cast<double>(count) / values.size();
            entropy -= p * std::log2(p);
        }

        return entropy;
    }

public:
    explicit EntropyStrategy(double threshold) : threshold_(threshold) {}

    std::vector<std::vector<std::string>>
    apply(const std::vector<std::string>& data) const override {
        std::vector<std::vector<std::string>> chunks;
        std::vector<std::string> current_chunk;

        for (const auto& value : data) {
            current_chunk.push_back(value);
            if (calculate_entropy(current_chunk) > threshold_) {
                if (current_chunk.size() > 1) {
                    current_chunk.pop_back();
                    chunks.push_back(current_chunk);
                    current_chunk = {value};
                }
            }
        }

        if (!current_chunk.empty()) {
            chunks.push_back(current_chunk);
        }

        return chunks;
    }
};

} // namespace chunk_strategies
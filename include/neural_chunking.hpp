/**
 * @file neural_chunking.hpp
 * @brief Neural network-based chunking algorithms
 * @author Jonathan Reich
 * @date 2024-12-07
 */

#pragma once
#include "chunk_common.hpp"
#include <cmath>   // for std::abs
#include <cstdlib> // for malloc, free
#include <memory>
#include <numeric>
#include <stdexcept> // for std::runtime_error
#include <vector>

namespace neural_chunking {

/**
 * @brief Neural network layer implementation
 * @tparam T Data type for layer computations
 */
template <typename T>
class CHUNK_EXPORT Layer {
public:
    Layer(size_t input_size, size_t output_size)
        : input_size_(input_size), output_size_(output_size) {
        weights_.resize(input_size * output_size);
        biases_.resize(output_size);
        initialize_weights();
    }

    std::vector<T> forward(const std::vector<T>& input) {
        if (input.size() != input_size_) {
            throw std::invalid_argument("Invalid input size");
        }
        std::vector<T> output(output_size_);
        // Simple forward pass implementation
        for (size_t i = 0; i < output_size_; ++i) {
            output[i] = biases_[i];
            for (size_t j = 0; j < input_size_; ++j) {
                output[i] += input[j] * weights_[i * input_size_ + j];
            }
        }
        return output;
    }

private:
    size_t input_size_;
    size_t output_size_;
    std::vector<T> weights_;
    std::vector<T> biases_;

    void initialize_weights() {
        // Simple Xavier initialization
        T scale = std::sqrt(2.0 / (input_size_ + output_size_));
        for (auto& w : weights_) {
            w = (static_cast<T>(rand()) / RAND_MAX * 2 - 1) * scale;
        }
        for (auto& b : biases_) {
            b = 0;
        }
    }
};

/**
 * @brief Configuration for neural network chunking
 */
struct CHUNK_EXPORT NeuralChunkConfig {
    size_t input_size;    ///< Size of input layer
    size_t hidden_size;   ///< Size of hidden layer
    double learning_rate; ///< Learning rate for training
    size_t batch_size;    ///< Batch size for processing
    double threshold;     ///< Decision threshold for chunk boundaries
};

/**
 * @brief Class implementing neural network-based chunking
 * @tparam T Data type of elements to chunk
 */
template <typename T>
class CHUNK_EXPORT NeuralChunking {
public:
    NeuralChunking(size_t window_size = 8, double threshold = 0.5)
        : window_size_(window_size), threshold_(threshold) {}

    void set_window_size(size_t size) {
        window_size_ = size;
    }
    void set_threshold(double threshold) {
        threshold_ = threshold;
    }

    size_t get_window_size() const {
        return window_size_;
    }
    double get_threshold() const {
        return threshold_;
    }

    std::vector<std::vector<T>> chunk(const std::vector<T>& data) const;

private:
    size_t window_size_;
    double threshold_;

    template <typename U>
    double compute_feature(const U& arr) const {
        if constexpr (chunk_processing::is_multidimensional_v<U>) {
            double sum = 0.0;
            for (const auto& inner : arr) {
                sum += compute_feature(inner);
            }
            return sum / std::size(arr);
        } else if constexpr (chunk_processing::is_vector<U>::value) {
            return std::accumulate(std::begin(arr), std::end(arr), 0.0) / std::size(arr);
        } else {
            return static_cast<double>(arr);
        }
    }
};

template <typename T>
std::vector<std::vector<T>> NeuralChunking<T>::chunk(const std::vector<T>& data) const {
    if (data.empty())
        return {};

    std::vector<std::vector<T>> result;
    std::vector<T> current_chunk;

    for (const auto& value : data) {
        if constexpr (chunk_processing::is_multidimensional_v<T>) {
            double feature = compute_feature(value);
            if (!current_chunk.empty() &&
                std::abs(feature - compute_feature(current_chunk.back())) > threshold_) {
                result.push_back(current_chunk);
                current_chunk.clear();
            }
        } else {
            // Single-dimension logic
            if (!current_chunk.empty() &&
                std::abs(static_cast<double>(value - current_chunk.back())) > threshold_) {
                result.push_back(current_chunk);
                current_chunk.clear();
            }
        }
        current_chunk.push_back(value);
    }

    if (!current_chunk.empty()) {
        result.push_back(current_chunk);
    }

    return result;
}

} // namespace neural_chunking
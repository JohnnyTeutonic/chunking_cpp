/**
 * @file neural_chunking.hpp
 * @brief Neural network-based chunking algorithms
 * @author Jonathan Reich
 * @date 2024-12-07
 */

#pragma once
#include "chunk_common.hpp"
#include <cmath>
#include <memory>
#include <numeric> // for std::accumulate
#include <stdexcept>
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
private:
    size_t window_size_;
    double threshold_;
    double learning_rate_;
    size_t batch_size_;
    std::string activation_;
    size_t epochs_;

    // Add private activation functions
    double apply_activation(double x) const {
        if (activation_ == "relu") {
            return x > 0 ? x : 0;
        } else if (activation_ == "sigmoid") {
            return 1.0 / (1.0 + std::exp(-x));
        } else { // tanh
            return std::tanh(x);
        }
    }

    double activation_derivative(double x) const {
        if (activation_ == "relu") {
            return x > 0 ? 1 : 0;
        } else if (activation_ == "sigmoid") {
            double sig = apply_activation(x);
            return sig * (1 - sig);
        } else { // tanh
            double tanh_x = std::tanh(x);
            return 1 - tanh_x * tanh_x;
        }
    }

    // Add training helper methods
    std::vector<double> prepare_batch(const std::vector<T>& data, size_t start_idx) const {
        std::vector<double> batch;
        batch.reserve(std::min(batch_size_, data.size() - start_idx));

        for (size_t i = 0; i < batch_size_ && (start_idx + i) < data.size(); ++i) {
            if constexpr (chunk_processing::is_vector<T>::value) {
                batch.push_back(compute_feature(data[start_idx + i]));
            } else {
                batch.push_back(static_cast<double>(data[start_idx + i]));
            }
        }
        return batch;
    }

    template <typename U>
    double compute_feature(const U& arr) const {
        if constexpr (chunk_processing::is_vector<U>::value) {
            if constexpr (chunk_processing::is_vector<typename U::value_type>::value) {
                // Handle 2D arrays
                double sum = 0.0;
                for (const auto& inner : arr) {
                    sum += compute_feature(inner);
                }
                return sum / arr.size();
            } else {
                // Handle 1D arrays
                return std::accumulate(arr.begin(), arr.end(), 0.0) / arr.size();
            }
        } else {
            // Handle scalar values
            return static_cast<double>(arr);
        }
    }

public:
    NeuralChunking(size_t window_size = 8, double threshold = 0.5)
        : window_size_(window_size), threshold_(threshold), learning_rate_(0.01), batch_size_(32),
          activation_("relu"), epochs_(100) {}

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

    std::vector<std::vector<T>> chunk(const std::vector<T>& data) const {
        if (data.empty()) {
            return {};
        }

        // Handle case where data is smaller than window size
        if (data.size() <= window_size_) {
            return {data};
        }

        std::vector<std::vector<T>> result;
        std::vector<T> current_chunk;

        for (const auto& value : data) {
            if constexpr (chunk_processing::is_vector<T>::value) {
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

    /**
     * @brief Set the learning rate for neural network training
     * @param rate Learning rate value (must be positive)
     */
    void set_learning_rate(double rate) {
        if (rate <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive");
        }
        learning_rate_ = rate;
    }

    /**
     * @brief Get the current learning rate
     * @return Current learning rate
     */
    double get_learning_rate() const {
        return learning_rate_;
    }

    /**
     * @brief Set the batch size for training
     * @param size Batch size (must be positive)
     */
    void set_batch_size(size_t size) {
        if (size == 0) {
            throw std::invalid_argument("Batch size must be positive");
        }
        batch_size_ = size;
    }

    /**
     * @brief Get the current batch size
     * @return Current batch size
     */
    size_t get_batch_size() const {
        return batch_size_;
    }

    /**
     * @brief Set the activation function type
     * @param activation Activation function name ("relu", "sigmoid", or "tanh")
     */
    void set_activation(const std::string& activation) {
        if (activation != "relu" && activation != "sigmoid" && activation != "tanh") {
            throw std::invalid_argument(
                "Invalid activation function. Supported: relu, sigmoid, tanh");
        }
        activation_ = activation;
    }

    /**
     * @brief Get the current activation function type
     * @return Current activation function name
     */
    std::string get_activation() const {
        return activation_;
    }

    /**
     * @brief Set the number of training epochs
     * @param num_epochs Number of epochs (must be positive)
     */
    void set_epochs(size_t num_epochs) {
        if (num_epochs == 0) {
            throw std::invalid_argument("Number of epochs must be positive");
        }
        epochs_ = num_epochs;
    }

    /**
     * @brief Get the current number of training epochs
     * @return Current number of epochs
     */
    size_t get_epochs() const {
        return epochs_;
    }

    /**
     * @brief Train the neural network on the provided data
     * @param data Training data
     * @return Vector of loss values for each epoch
     */
    std::vector<double> train(const std::vector<T>& data) {
        if (data.size() < window_size_) {
            throw std::invalid_argument("Training data size must be larger than window size");
        }

        // Initialize neural network layers
        Layer<double> input_layer(window_size_, window_size_);
        Layer<double> hidden_layer(window_size_, 1);

        std::vector<double> epoch_losses;
        epoch_losses.reserve(epochs_);

        // Training loop
        for (size_t epoch = 0; epoch < epochs_; ++epoch) {
            double epoch_loss = 0.0;
            size_t num_batches = (data.size() + batch_size_ - 1) / batch_size_;

            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t start_idx = batch * batch_size_;
                auto batch_data = prepare_batch(data, start_idx);
                if (batch_data.size() < window_size_)
                    break;

                // Forward pass
                auto hidden = input_layer.forward(batch_data);
                for (auto& h : hidden)
                    h = apply_activation(h);
                auto output = hidden_layer.forward(hidden);

                // Compute loss
                double target = batch_data.back();
                double prediction = output[0];
                double loss = 0.5 * (prediction - target) * (prediction - target);
                epoch_loss += loss;

                // Backward pass and update weights (simplified)
                double error = prediction - target;
                double delta = error * activation_derivative(prediction);

                // Update weights (simplified backpropagation)
                for (size_t i = 0; i < window_size_; ++i) {
                    hidden[i] -= learning_rate_ * delta * batch_data[i];
                }
            }

            epoch_losses.push_back(epoch_loss / num_batches);
        }

        return epoch_losses;
    }
};

} // namespace neural_chunking
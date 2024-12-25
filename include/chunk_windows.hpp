#pragma once

#include "chunk.hpp"
#include "chunk_common.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace chunk_windows {

template <typename T>
class SlidingWindowProcessor {
private:
    size_t window_size_;
    size_t step_size_;

    template <typename U>
    static constexpr bool is_multidimensional_v = chunk_processing::is_multidimensional_v<U>;

    template <typename U>
    static double compute_window_feature(const U& window) {
        if constexpr (chunk_processing::is_multidimensional_v<U>) {
            double sum = 0.0;
            for (const auto& inner : window) {
                sum += compute_window_feature(inner);
            }
            return sum / std::size(window);
        } else if constexpr (chunk_processing::is_vector<U>::value) {
            return std::accumulate(std::begin(window), std::end(window), 0.0) / std::size(window);
        } else {
            return static_cast<double>(window);
        }
    }

    std::vector<T> process_multidimensional(const std::vector<T>& data,
                                            std::function<T(const std::vector<T>&)> window_func) {
        // Implementation for multi-dimensional data
        std::vector<T> results;
        for (size_t i = 0; i <= data.size() - window_size_; i += step_size_) {
            std::vector<T> window(data.begin() + i, data.begin() + i + window_size_);
            results.push_back(window_func(window));
        }
        return results;
    }

    std::vector<T> process_single_dimensional(const std::vector<T>& data,
                                              std::function<T(const std::vector<T>&)> window_func) {
        std::vector<T> results;
        // Create sliding windows of the specified size
        for (size_t i = 0; i <= data.size() - window_size_; i += step_size_) {
            std::vector<T> window(data.begin() + i, data.begin() + i + window_size_);
            results.push_back(window_func(window));
        }
        return results;
    }

public:
    SlidingWindowProcessor(size_t window_size, size_t step_size)
        : window_size_(window_size), step_size_(step_size) {
        if (window_size == 0)
            throw std::invalid_argument("Window size cannot be zero");
        if (step_size == 0)
            throw std::invalid_argument("Step size cannot be zero");
    }

    std::vector<T> process(const std::vector<T>& data,
                           std::function<T(const std::vector<T>&)> window_func) {
        if (data.size() < window_size_) {
            return {}; // Return empty if data is smaller than window
        }

        if constexpr (chunk_processing::is_multidimensional_v<T>) {
            return process_multidimensional(data, window_func);
        } else {
            return process_single_dimensional(data, window_func);
        }
    }

    // Add getters
    size_t get_window_size() const {
        return window_size_;
    }
    size_t get_step_size() const {
        return step_size_;
    }

    // Add setters
    void set_window_size(size_t size) {
        if (size == 0)
            throw std::invalid_argument("Window size cannot be zero");
        window_size_ = size;
    }
    void set_step_size(size_t size) {
        if (size == 0)
            throw std::invalid_argument("Step size cannot be zero");
        step_size_ = size;
    }
};

// Common window operations
template <typename T>
class WindowOperations {
public:
    static T moving_average(const std::vector<T>& window) {
        if (window.empty()) {
            return T{}; // Return default value for empty input
        }
        return std::accumulate(window.begin(), window.end(), T{}) / static_cast<T>(window.size());
    }

    static T moving_median(std::vector<T> window) {
        if (window.empty()) {
            throw std::invalid_argument("Cannot compute median of empty window");
        }
        std::sort(window.begin(), window.end());
        return window[window.size() / 2];
    }

    static T moving_max(const std::vector<T>& window) {
        if (window.empty()) {
            throw std::invalid_argument("Cannot compute max of empty window");
        }
        return *std::max_element(window.begin(), window.end());
    }

    static T moving_min(const std::vector<T>& window) {
        if (window.empty()) {
            throw std::invalid_argument("Cannot compute min of empty window");
        }
        return *std::min_element(window.begin(), window.end());
    }
};

} // namespace chunk_windows
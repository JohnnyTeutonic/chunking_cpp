/**
 * @file slidingchunkwindow.hpp
 * @brief Sliding window data structure optimized for chunk analysis
 * @author Assistant
 * @date 2024-03-19
 */

#pragma once

#include <cmath>
#include <deque>
#include <stdexcept>

namespace chunk_processing {

/**
 * @brief A sliding window data structure optimized for real-time chunk analysis
 *
 * @tparam T The type of elements stored in the window
 */
template <typename T>
class SlidingChunkWindow {
private:
    std::deque<T> window_; ///< The underlying window storage
    size_t max_size_;      ///< Maximum size of the window
    double sum_{0};        ///< Running sum of elements
    double sum_sq_{0};     ///< Running sum of squared elements

public:
    /**
     * @brief Construct a new Sliding Chunk Window
     *
     * @param size Maximum size of the window
     * @throw std::invalid_argument if size is 0
     */
    explicit SlidingChunkWindow(size_t size) : max_size_(size) {
        if (size == 0)
            throw std::invalid_argument("Window size must be positive");
    }

    /**
     * @brief Push a new value into the window
     *
     * If the window is full, the oldest value is removed
     *
     * @param value Value to add to the window
     */
    void push(T value) {
        double d_val = static_cast<double>(value);

        if (window_.size() == max_size_) {
            double old_val = static_cast<double>(window_.front());
            sum_ -= old_val;
            sum_sq_ -= old_val * old_val;
            window_.pop_front();
        }

        window_.push_back(value);
        sum_ += d_val;
        sum_sq_ += d_val * d_val;
    }

    /**
     * @brief Get the current mean of the window
     *
     * @return double The mean value
     */
    double mean() const {
        return window_.empty() ? 0.0 : sum_ / window_.size();
    }

    /**
     * @brief Get the current variance of the window
     *
     * @return double The variance value
     */
    double variance() const {
        if (window_.size() < 2)
            return 0.0;
        double n = static_cast<double>(window_.size());
        return (sum_sq_ - (sum_ * sum_) / n) / (n - 1);
    }

    /**
     * @brief Get the current standard deviation of the window
     *
     * @return double The standard deviation value
     */
    double stddev() const {
        return std::sqrt(variance());
    }

    /**
     * @brief Get the current size of the window
     *
     * @return size_t Current number of elements
     */
    size_t size() const {
        return window_.size();
    }

    /**
     * @brief Check if the window is empty
     *
     * @return bool True if empty, false otherwise
     */
    bool empty() const {
        return window_.empty();
    }

    /**
     * @brief Clear the window
     */
    void clear() {
        window_.clear();
        sum_ = 0.0;
        sum_sq_ = 0.0;
    }
};

} // namespace chunk_processing
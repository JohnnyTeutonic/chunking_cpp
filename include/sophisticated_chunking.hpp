#pragma once
#include "chunk_common.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sophisticated_chunking {

/**
 * @brief Wavelet-based chunking strategy using signal processing principles
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class WaveletChunking {
private:
    size_t window_size_;
    double threshold_;

    /**
     * @brief Compute discrete wavelet transform coefficients
     * @param data Input data sequence
     * @return Vector of wavelet coefficients
     */
    std::vector<double> computeWaveletCoefficients(const std::vector<T>& data) const;

public:
    /**
     * @brief Constructor for wavelet-based chunking
     * @param window_size Size of the sliding window
     * @param threshold Coefficient threshold for chunk boundaries
     */
    WaveletChunking(size_t window_size = 8, double threshold = 0.5)
        : window_size_(window_size), threshold_(threshold) {}

    /**
     * @brief Chunk data based on wavelet transform analysis
     * @param data Input data to be chunked
     * @return Vector of chunks
     */
    std::vector<std::vector<T>> chunk(const std::vector<T>& data) const;

    /**
     * @brief Get the size of the sliding window
     * @return Size of the sliding window
     */
    size_t get_window_size() const {
        return window_size_;
    }

    /**
     * @brief Get the coefficient threshold for chunk boundaries
     * @return Coefficient threshold for chunk boundaries
     */
    double get_threshold() const {
        return threshold_;
    }

    /**
     * @brief Set the size of the sliding window
     * @param size Size of the sliding window
     */
    void set_window_size(size_t size) {
        if (size == 0)
            throw std::invalid_argument("Window size cannot be zero");
        window_size_ = size;
    }

    /**
     * @brief Set the coefficient threshold for chunk boundaries
     * @param threshold Coefficient threshold for chunk boundaries
     */
    void set_threshold(double threshold) {
        threshold_ = threshold;
    }
};

/**
 * @brief Information theory based chunking using mutual information
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class MutualInformationChunking {
private:
    size_t context_size_;
    double mi_threshold_;

    /**
     * @brief Calculate mutual information between adjacent segments
     * @param segment1 First segment
     * @param segment2 Second segment
     * @return Mutual information value
     */
    double calculateMutualInformation(const std::vector<T>& segment1,
                                      const std::vector<T>& segment2) const;

public:
    /**
     * @brief Constructor for mutual information based chunking
     * @param context_size Size of context window
     * @param mi_threshold Threshold for mutual information
     */
    MutualInformationChunking(size_t context_size = 5, double mi_threshold = 0.3)
        : context_size_(context_size), mi_threshold_(mi_threshold) {}

    /**
     * @brief Chunk data based on mutual information analysis
     * @param data Input data to be chunked
     * @return Vector of chunks
     */
    std::vector<std::vector<T>> chunk(const std::vector<T>& data) const;

    /**
     * @brief Get the size of context window
     * @return Size of context window
     */
    size_t get_context_size() const {
        return context_size_;
    }

    /**
     * @brief Get the threshold for mutual information
     * @return Threshold for mutual information
     */
    double get_mi_threshold() const {
        return mi_threshold_;
    }

    /**
     * @brief Set the size of context window
     * @param size Size of context window
     */
    void set_context_size(size_t size) {
        if (size == 0)
            throw std::invalid_argument("Context size cannot be zero");
        context_size_ = size;
    }

    /**
     * @brief Set the threshold for mutual information
     * @param threshold Threshold for mutual information
     */
    void set_mi_threshold(double threshold) {
        mi_threshold_ = threshold;
    }
};

/**
 * @brief Dynamic time warping based chunking for sequence alignment
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class DTWChunking {
private:
    size_t window_size_;
    double dtw_threshold_;

    double compute_dtw_distance_1d(const std::vector<double>& seq1,
                                   const std::vector<double>& seq2) const {
        const size_t n = seq1.size();
        const size_t m = seq2.size();
        std::vector<std::vector<double>> dp(
            n + 1, std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));

        dp[0][0] = 0.0;

        for (size_t i = 1; i <= n; ++i) {
            for (size_t j = std::max(1ul, i - window_size_); j <= std::min(m, i + window_size_);
                 ++j) {
                double cost = std::abs(seq1[i - 1] - seq2[j - 1]);
                dp[i][j] = cost + std::min({
                                      dp[i - 1][j],    // insertion
                                      dp[i][j - 1],    // deletion
                                      dp[i - 1][j - 1] // match
                                  });
            }
        }

        return dp[n][m];
    }

    double compute_dtw_distance(const T& seq1, const T& seq2) const {
        if constexpr (chunk_processing::is_multidimensional_v<T>) {
            std::vector<double> features1 = flatten_features(seq1);
            std::vector<double> features2 = flatten_features(seq2);
            return compute_dtw_distance_1d(features1, features2);
        } else {
            return std::abs(static_cast<double>(seq1 - seq2));
        }
    }

    template <typename U>
    std::vector<double> flatten_features(const U& arr) const {
        std::vector<double> result;
        if constexpr (chunk_processing::is_multidimensional_v<U>) {
            for (const auto& inner : arr) {
                auto inner_flat = flatten_features(inner);
                result.insert(result.end(), inner_flat.begin(), inner_flat.end());
            }
        } else {
            result.insert(result.end(), std::begin(arr), std::end(arr));
        }
        return result;
    }

    /**
     * @brief Compute DTW distance between sequences
     * @param seq1 First sequence
     * @param seq2 Second sequence
     * @return DTW distance
     */
    double computeDTWDistance(const std::vector<T>& seq1, const std::vector<T>& seq2) const;

public:
    /**
     * @brief Constructor for DTW-based chunking
     * @param window_size Size of the warping window
     * @param dtw_threshold Threshold for chunk boundaries
     */
    DTWChunking(size_t window_size = 10, double dtw_threshold = 1.0)
        : window_size_(window_size), dtw_threshold_(dtw_threshold) {}

    /**
     * @brief Chunk data based on DTW analysis
     * @param data Input data to be chunked
     * @return Vector of chunks
     */
    std::vector<std::vector<T>> chunk(const std::vector<T>& data) const;

    /**
     * @brief Get the size of the warping window
     * @return Size of the warping window
     */
    size_t get_window_size() const {
        return window_size_;
    }

    /**
     * @brief Get the threshold for chunk boundaries
     * @return Threshold for chunk boundaries
     */
    double get_dtw_threshold() const {
        return dtw_threshold_;
    }

    /**
     * @brief Set the size of the warping window
     * @param size Size of the warping window
     */
    void set_window_size(size_t size) {
        if (size == 0)
            throw std::invalid_argument("Window size cannot be zero");
        window_size_ = size;
    }

    /**
     * @brief Set the threshold for chunk boundaries
     * @param threshold Threshold for chunk boundaries
     */
    void set_dtw_threshold(double threshold) {
        dtw_threshold_ = threshold;
    }
};

template <typename T>
std::vector<double>
WaveletChunking<T>::computeWaveletCoefficients(const std::vector<T>& data) const {
    if (data.size() < window_size_) {
        return std::vector<double>();
    }

    std::vector<double> coefficients;
    coefficients.reserve(data.size() - window_size_ + 1);

    // Haar wavelet transform
    for (size_t i = 0; i <= data.size() - window_size_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < window_size_ / 2; ++j) {
            double diff = static_cast<double>(data[i + j]) -
                          static_cast<double>(data[i + window_size_ - 1 - j]);
            sum += diff * diff;
        }
        coefficients.push_back(std::sqrt(sum / window_size_));
    }

    return coefficients;
}

template <typename T>
std::vector<std::vector<T>> WaveletChunking<T>::chunk(const std::vector<T>& data) const {
    if (data.empty()) {
        return {};
    }

    auto coefficients = computeWaveletCoefficients(data);
    std::vector<std::vector<T>> chunks;
    std::vector<T> current_chunk;

    size_t i = 0;
    for (const T& value : data) {
        current_chunk.push_back(value);

        if (i < coefficients.size() && coefficients[i] > threshold_) {
            if (!current_chunk.empty()) {
                chunks.push_back(current_chunk);
                current_chunk.clear();
            }
        }
        ++i;
    }

    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }

    return chunks;
}

template <typename T>
double
MutualInformationChunking<T>::calculateMutualInformation(const std::vector<T>& segment1,
                                                         const std::vector<T>& segment2) const {
    if (segment1.empty() || segment2.empty()) {
        return 0.0;
    }

    // Calculate frequency distributions
    std::map<T, double> p1, p2;
    std::map<std::pair<T, T>, double> p12;

    for (const auto& val : segment1) {
        p1[val] += 1.0 / segment1.size();
    }

    for (const auto& val : segment2) {
        p2[val] += 1.0 / segment2.size();
    }

    // Calculate joint distribution
    size_t min_size = std::min(segment1.size(), segment2.size());
    for (size_t i = 0; i < min_size; ++i) {
        p12[{segment1[i], segment2[i]}] += 1.0 / min_size;
    }

    // Calculate mutual information
    double mi = 0.0;
    for (const auto& [val1, prob1] : p1) {
        for (const auto& [val2, prob2] : p2) {
            auto joint_prob = p12[{val1, val2}];
            if (joint_prob > 0) {
                mi += joint_prob * std::log2(joint_prob / (prob1 * prob2));
            }
        }
    }

    return mi;
}

template <typename T>
std::vector<std::vector<T>> MutualInformationChunking<T>::chunk(const std::vector<T>& data) const {
    if (data.size() < 2 * context_size_) {
        return {data};
    }

    std::vector<std::vector<T>> chunks;
    std::vector<T> current_chunk;

    for (size_t i = 0; i < data.size(); ++i) {
        current_chunk.push_back(data[i]);

        if (current_chunk.size() >= context_size_ && i + context_size_ < data.size()) {
            std::vector<T> next_segment(
                data.begin() + i + 1, data.begin() + std::min(i + 1 + context_size_, data.size()));

            double mi = calculateMutualInformation(current_chunk, next_segment);

            if (mi < mi_threshold_) {
                chunks.push_back(current_chunk);
                current_chunk.clear();
            }
        }
    }

    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }

    return chunks;
}

template <typename T>
double DTWChunking<T>::computeDTWDistance(const std::vector<T>& seq1,
                                          const std::vector<T>& seq2) const {
    if (seq1.empty() || seq2.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    // Initialize DTW matrix
    std::vector<std::vector<double>> dtw(
        seq1.size() + 1,
        std::vector<double>(seq2.size() + 1, std::numeric_limits<double>::infinity()));
    dtw[0][0] = 0.0;

    // Fill DTW matrix
    for (size_t i = 1; i <= seq1.size(); ++i) {
        for (size_t j = std::max(1ul, i - window_size_);
             j <= std::min(seq2.size(), i + window_size_); ++j) {
            double cost =
                std::abs(static_cast<double>(seq1[i - 1]) - static_cast<double>(seq2[j - 1]));
            dtw[i][j] = cost + std::min({
                                   dtw[i - 1][j],    // insertion
                                   dtw[i][j - 1],    // deletion
                                   dtw[i - 1][j - 1] // match
                               });
        }
    }

    return dtw[seq1.size()][seq2.size()];
}

template <typename T>
std::vector<std::vector<T>> DTWChunking<T>::chunk(const std::vector<T>& data) const {
    if (data.size() < 2 * window_size_) {
        return {data};
    }

    std::vector<std::vector<T>> chunks;
    std::vector<T> current_chunk;

    for (size_t i = 0; i < data.size(); ++i) {
        current_chunk.push_back(data[i]);

        if (current_chunk.size() >= window_size_ && i + window_size_ < data.size()) {
            std::vector<T> next_window(data.begin() + i + 1, data.begin() + i + 1 + window_size_);

            double distance = computeDTWDistance(
                std::vector<T>(current_chunk.end() - window_size_, current_chunk.end()),
                next_window);

            if (distance > dtw_threshold_) {
                chunks.push_back(current_chunk);
                current_chunk.clear();
            }
        }
    }

    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }

    return chunks;
}

} // namespace sophisticated_chunking
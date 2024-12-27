/**
 * @file chunk_metrics.hpp
 * @brief Quality metrics and analysis tools for chunk evaluation
 * @author Jonathan Reich
 * @date 2024-12-07
 */

#pragma once

#include "chunk_common.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace chunk_metrics {

namespace detail {
template <typename T>
bool is_valid_chunk(const std::vector<T>& chunk) {
    return !chunk.empty() && std::all_of(chunk.begin(), chunk.end(), [](const T& val) {
        return std::isfinite(static_cast<double>(val));
    });
}

template <typename T>
bool is_valid_chunks(const std::vector<std::vector<T>>& chunks) {
    return !chunks.empty() && std::all_of(chunks.begin(), chunks.end(),
                                          [](const auto& chunk) { return is_valid_chunk(chunk); });
}

template <typename T>
double safe_mean(const std::vector<T>& data) {
    if (data.empty())
        return 0.0;
    double sum = 0.0;
    size_t valid_count = 0;

    for (const auto& val : data) {
        double d_val = static_cast<double>(val);
        if (std::isfinite(d_val)) {
            sum += d_val;
            ++valid_count;
        }
    }

    return valid_count > 0 ? sum / valid_count : 0.0;
}

template <typename T>
double safe_distance(const T& a, const T& b) {
    try {
        double d_a = static_cast<double>(a);
        double d_b = static_cast<double>(b);
        if (!std::isfinite(d_a) || !std::isfinite(d_b)) {
            return std::numeric_limits<double>::max();
        }
        return std::abs(d_a - d_b);
    } catch (...) {
        return std::numeric_limits<double>::max();
    }
}
} // namespace detail

/**
 * @brief Class for analyzing and evaluating chunk quality
 * @tparam T The data type of the chunks (must support arithmetic operations)
 */
template <typename T>
class CHUNK_EXPORT ChunkQualityAnalyzer {
private:
    mutable std::mutex mutex_;
    mutable std::atomic<bool> is_processing_{false};

    // Cache for intermediate results
    mutable struct Cache {
        std::vector<double> chunk_means;
        double global_mean{0.0};
        bool is_valid{false};
    } cache_;

    void validate_chunks(const std::vector<std::vector<T>>& chunks) const {
        if (!detail::is_valid_chunks(chunks)) {
            throw std::invalid_argument("Invalid chunks: empty or contains invalid values");
        }
    }

    void update_cache(const std::vector<std::vector<T>>& chunks) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cache_.is_valid) {
            cache_.chunk_means.clear();
            cache_.chunk_means.reserve(chunks.size());

            double total_sum = 0.0;
            size_t total_count = 0;

            for (const auto& chunk : chunks) {
                double mean = detail::safe_mean(chunk);
                cache_.chunk_means.push_back(mean);
                total_sum += mean * chunk.size();
                total_count += chunk.size();
            }

            cache_.global_mean = total_count > 0 ? total_sum / total_count : 0.0;
            cache_.is_valid = true;
        }
    }

    double compute_chunk_cohesion(const std::vector<T>& chunk, double chunk_mean) const {
        if (chunk.empty())
            return 0.0;

        double sum_distances = 0.0;
        size_t valid_pairs = 0;

        for (size_t i = 0; i < chunk.size(); ++i) {
            for (size_t j = i + 1; j < chunk.size(); ++j) {
                double dist = detail::safe_distance(chunk[i], chunk[j]);
                if (dist < std::numeric_limits<double>::max()) {
                    sum_distances += dist;
                    ++valid_pairs;
                }
            }
        }

        return valid_pairs > 0 ? sum_distances / valid_pairs : 0.0;
    }

public:
    ChunkQualityAnalyzer() = default;

    void clear_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.is_valid = false;
        cache_.chunk_means.clear();
    }

    double compute_cohesion(const std::vector<std::vector<T>>& chunks) const {
        // Prevent reentrant calls
        bool expected = false;
        if (!is_processing_.compare_exchange_strong(expected, true)) {
            throw std::runtime_error("Analyzer is already processing");
        }

        struct Guard {
            std::atomic<bool>& flag;
            Guard(std::atomic<bool>& f) : flag(f) {}
            ~Guard() {
                flag = false;
            }
        } guard(const_cast<std::atomic<bool>&>(is_processing_));

        try {
            validate_chunks(chunks);
            update_cache(chunks);

            double total_cohesion = 0.0;
            size_t valid_chunks = 0;

            for (size_t i = 0; i < chunks.size(); ++i) {
                if (!chunks[i].empty()) {
                    double chunk_cohesion =
                        compute_chunk_cohesion(chunks[i], cache_.chunk_means[i]);
                    if (std::isfinite(chunk_cohesion)) {
                        total_cohesion += chunk_cohesion;
                        ++valid_chunks;
                    }
                }
            }

            return valid_chunks > 0 ? total_cohesion / valid_chunks : 0.0;

        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error computing cohesion: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Unknown error computing cohesion");
        }
    }

    /**
     * @brief Calculate separation (dissimilarity between chunks)
     * @param chunks Vector of chunk data
     * @return Separation score between 0 and 1
     * @throws std::invalid_argument if chunks is empty or contains single chunk
     */
    double compute_separation(const std::vector<std::vector<T>>& chunks) {
        if (chunks.size() < 2) {
            throw std::invalid_argument("Need at least two chunks for separation");
        }

        double total_separation = 0.0;
        int comparisons = 0;

        for (size_t i = 0; i < chunks.size(); ++i) {
            for (size_t j = i + 1; j < chunks.size(); ++j) {
                T mean_i = calculate_mean(chunks[i]);
                T mean_j = calculate_mean(chunks[j]);

                // Calculate distance between means
                double separation = std::abs(mean_i - mean_j);
                total_separation += separation;
                ++comparisons;
            }
        }

        return total_separation / comparisons;
    }

    /**
     * @brief Calculate silhouette score for chunk validation
     * @param chunks Vector of chunk data
     * @return Silhouette score between -1 and 1
     * @throws std::invalid_argument if chunks is empty or contains single chunk
     */
    double compute_silhouette_score(const std::vector<std::vector<T>>& chunks) {
        if (chunks.size() < 2) {
            throw std::invalid_argument("Need at least two chunks for silhouette score");
        }

        double total_score = 0.0;
        size_t total_points = 0;

        for (size_t i = 0; i < chunks.size(); ++i) {
            for (const auto& point : chunks[i]) {
                // Calculate a (average distance to points in same chunk)
                double a = 0.0;
                for (const auto& other_point : chunks[i]) {
                    if (&point != &other_point) {
                        a += std::abs(point - other_point);
                    }
                }
                a = chunks[i].size() > 1 ? a / (chunks[i].size() - 1) : 0;

                // Calculate b (minimum average distance to points in other chunks)
                double b = std::numeric_limits<double>::max();
                for (size_t j = 0; j < chunks.size(); ++j) {
                    if (i != j) {
                        double avg_dist = 0.0;
                        for (const auto& other_point : chunks[j]) {
                            avg_dist += std::abs(point - other_point);
                        }
                        avg_dist /= chunks[j].size();
                        b = std::min(b, avg_dist);
                    }
                }

                // Calculate silhouette score for this point
                double max_ab = std::max(a, b);
                if (max_ab > 0) {
                    total_score += (b - a) / max_ab;
                }
                ++total_points;
            }
        }

        return total_score / total_points;
    }

    /**
     * @brief Calculate overall quality score combining multiple metrics
     * @param chunks Vector of chunk data
     * @return Quality score between 0 and 1
     * @throws std::invalid_argument if chunks is empty
     */
    double compute_quality_score(const std::vector<std::vector<T>>& chunks) {
        if (chunks.empty()) {
            throw std::invalid_argument("Empty chunks vector");
        }

        double cohesion = compute_cohesion(chunks);
        double separation = chunks.size() > 1 ? compute_separation(chunks) : 1.0;

        return (cohesion + separation) / 2.0;
    }

    /**
     * @brief Compute size-based metrics for chunks
     * @param chunks Vector of chunk data
     * @return Map of metric names to values
     */
    std::map<std::string, double>
    compute_size_metrics(const std::vector<std::vector<T>>& chunks) const {
        std::map<std::string, double> metrics;

        if (chunks.empty()) {
            throw std::invalid_argument("Empty chunks vector");
        }

        // Calculate average chunk size
        double avg_size = 0.0;
        double max_size = 0.0;
        double min_size = static_cast<double>(chunks[0].size());

        for (const auto& chunk : chunks) {
            double size = static_cast<double>(chunk.size());
            avg_size += size;
            max_size = std::max(max_size, size);
            min_size = std::min(min_size, size);
        }
        avg_size /= static_cast<double>(chunks.size());

        // Calculate size variance
        double variance = 0.0;
        for (const auto& chunk : chunks) {
            double diff = static_cast<double>(chunk.size()) - avg_size;
            variance += diff * diff;
        }
        variance /= static_cast<double>(chunks.size());

        metrics["average_size"] = avg_size;
        metrics["max_size"] = max_size;
        metrics["min_size"] = min_size;
        metrics["size_variance"] = variance;
        metrics["size_stddev"] = std::sqrt(variance);

        return metrics;
    }

private:
    /**
     * @brief Calculate mean value of a chunk
     * @param chunk Single chunk data
     * @return Mean value of the chunk
     */
    T calculate_mean(const std::vector<T>& chunk) {
        if (chunk.empty()) {
            return T{};
        }
        T sum = T{};
        for (const auto& val : chunk) {
            sum += val;
        }
        return sum / static_cast<T>(chunk.size());
    }

    /**
     * @brief Calculate variance of a chunk
     * @param chunk Single chunk data
     * @param mean Pre-calculated mean value
     * @return Variance of the chunk
     */
    T calculate_variance(const std::vector<T>& chunk, T mean) {
        if (chunk.size() < 2) {
            return T{};
        }
        T sum_sq_diff = T{};
        for (const auto& val : chunk) {
            T diff = val - mean;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / static_cast<T>(chunk.size() - 1);
    }
};

} // namespace chunk_metrics